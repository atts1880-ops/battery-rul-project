"""Two-pass strict OOF refit for the V1.0 NASA-transfer B_stats model.

The completed nested-OOF run is the selection pass.  This entrypoint verifies
its splits and chosen epochs, recreates the NASA/target initialization, and
then fits the target TCN and B_stats head on all outer-training devices for
exactly those epoch counts.  The outer fold remains unseen until prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_positive_transfer import snapshot
from train_bhump_v10_bstats_oof import (
    FORMAL_SEEDS, OUTER_FOLDS, audit_oof, inner_split, maximum_lifetime_without_test,
    stratified_outer_folds, subset, summarize_oof,
)
from train_bhump_v10_history_ablation import (
    HistoryAblationModel, HistoryConfig, assert_causal, bundle_tensors,
    fit_statistics, knee_class_weights, make_bundle, model_loss, move_batch,
    predict_bundle, regression_metrics, seed_all, soh_metrics, statistic_names,
)
from train_bhump_v10_rul_multitask import (
    DynamicsTCN, build_config, dynamics_loader, initialize_states,
    make_multitask_windows, multitask_loss, prepare_data, set_dynamics_trainable,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_SELECTION = ROOT / "bhump_v10_bstats_oof_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_bstats_refit_oof_runs"
BASELINE_RUL_MAE = 5.0395032373770565
BASELINE_SOH_MAE = 0.011627460907369845
BASELINE_WORST_DEVICE_MAE = 18.058352528965514
BASELINE_EARLY_MAE = 8.4515
BASELINE_LONG_LIFE_BIAS = -3.5138


def fit_multitask_fixed_epochs(initial_state: dict[str, torch.Tensor],
                               train: pd.DataFrame, features: list[str],
                               target_state: Any, maximum_lifetime: float,
                               config: Any, seed: int, epochs: int,
                               device: torch.device) -> DynamicsTCN:
    """Reinitialize target heads and train on all outer-train units."""
    if epochs < 1 or epochs > config.adapt_epochs:
        raise ValueError(f"Invalid fixed TCN epoch count: {epochs}")
    train_x, train_y, _ = make_multitask_windows(train, features, target_state, config)
    seed_all(seed)
    model = DynamicsTCN(len(features), config)
    model.load_state_dict(copy.deepcopy(initial_state))
    model = model.to(device)
    model.reset_dynamics_heads()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.shared_projection.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.soh_head.parameters(), "lr": config.head_learning_rate},
        {"params": model.pre_rate_head.parameters(), "lr": config.head_learning_rate},
        {"params": model.post_delta_head.parameters(), "lr": config.head_learning_rate},
        {"params": model.knee_time_head.parameters(), "lr": config.head_learning_rate},
        {"params": model.knee_probability_head.parameters(), "lr": config.head_learning_rate},
    ], weight_decay=config.weight_decay)
    batches = dynamics_loader(train_x, train_y, config.batch_size, seed + 101)
    for epoch in range(1, epochs + 1):
        set_dynamics_trainable(model, epoch, config)
        model.train()
        for values, labels in batches:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.dynamics(values)
            cap = torch.clamp(maximum_lifetime - labels[:, 6], min=0.0)
            loss, _ = multitask_loss(outputs, labels, cap)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
    return model


def fit_bstats_fixed_epochs(train: Any, maximum_lifetime: float, seed: int,
                            epochs: int, config: HistoryConfig,
                            device: torch.device) -> HistoryAblationModel:
    """Fit B_stats on every outer-train unit without validation/early stop."""
    if epochs < 1 or epochs > config.epochs:
        raise ValueError(f"Invalid fixed B_stats epoch count: {epochs}")
    seed_all(seed + 1000)
    model = HistoryAblationModel("B_stats", config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        bundle_tensors(train), batch_size=config.batch_devices, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 71), num_workers=0,
    )
    weights = knee_class_weights(train, device)
    for _epoch in range(1, epochs + 1):
        model.train()
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            raw, local, local_soh, stats, times = batch[:5]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(raw, local, local_soh, stats, times, maximum_lifetime)
            loss, _ = model_loss(outputs, batch, "B_stats", config, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def selection_row(selection: pd.DataFrame, seed: int, fold: int) -> pd.Series:
    rows = selection.loc[selection.seed.eq(seed) & selection.outer_fold.eq(fold)]
    if len(rows) != 1:
        raise ValueError(f"Expected one selection record for seed={seed}, fold={fold}")
    return rows.iloc[0]


def validate_selection_split(selection_dir: Path, seed: int, fold: int,
                             outer_train: list[str], outer_test: list[str],
                             inner_train_units: list[str],
                             inner_validation_units: list[str]) -> None:
    path = selection_dir / f"split_seed_{seed}_fold_{fold}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "outer_train_units": outer_train,
        "outer_test_units": outer_test,
        "inner_train_units": inner_train_units,
        "inner_validation_units": inner_validation_units,
    }
    for name, values in expected.items():
        if list(map(str, payload[name])) != list(map(str, values)):
            raise AssertionError(f"Selection split mismatch for {path.name}: {name}")


def save_refit_checkpoints(output_dir: Path, seed: int, fold: int,
                           tcn: DynamicsTCN, head: HistoryAblationModel,
                           target_state: Any, features: list[str], config: Any,
                           history_config: HistoryConfig, maximum_lifetime: float,
                           outer_train: list[str], outer_test: list[str],
                           tcn_epoch: int, head_epoch: int) -> None:
    common = {
        "seed": seed, "outer_fold": fold,
        "refit_train_units": outer_train, "outer_test_units": outer_test,
        "refit_supervised_device_count": len(outer_train),
        "selection_tcn_best_epoch": tcn_epoch,
        "selection_bstats_best_epoch": head_epoch,
        "maximum_training_lifetime": maximum_lifetime,
        "features": features, "nasa_supervised_pretraining": True,
        "outer_test_used_for_training": False,
    }
    torch.save({
        **common, "model_state": snapshot(tcn), "configuration": asdict(config),
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "target_scaler_median": target_state.median, "target_scaler_iqr": target_state.iqr,
    }, output_dir / f"checkpoint_tcn_refit_seed_{seed}_fold_{fold}.pt")
    torch.save({
        **common, "model_state": snapshot(head), "variant": "B_stats",
        "history_configuration": asdict(history_config),
    }, output_dir / f"checkpoint_bstats_refit_seed_{seed}_fold_{fold}.pt")


def run_fold(seed: int, fold: int, source: pd.DataFrame, target: pd.DataFrame,
             target_knees: pd.DataFrame, features: list[str], report: dict[str, Any],
             reference: dict[str, Any], fold_map: dict[str, int], selection_seed: int,
             selection_dir: Path, selection: pd.DataFrame,
             history_config: HistoryConfig, smoke: bool, output_dir: Path,
             device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    outer_test_units = sorted(unit for unit, value in fold_map.items() if value == fold)
    outer_train_units = sorted(set(fold_map) - set(outer_test_units))
    inner_train_units, inner_validation_units = inner_split(
        outer_train_units, target, selection_seed, fold,
        validation_fraction=0.20 if smoke else 0.10,
    )
    validate_selection_split(
        selection_dir, seed, fold, outer_train_units, outer_test_units,
        inner_train_units, inner_validation_units,
    )
    chosen = selection_row(selection, seed, fold)
    if int(chosen.outer_train_devices) != len(outer_train_units):
        raise AssertionError("Selection outer-training count changed")
    tcn_epoch = int(chosen.tcn_best_epoch)
    head_epoch = int(chosen.bstats_best_epoch)
    outer_train = subset(target, outer_train_units)
    outer_test = subset(target, outer_test_units)
    if set(outer_train.unit_id.astype(str)) & set(outer_test.unit_id.astype(str)):
        raise AssertionError("Outer train/test overlap")

    config = build_config(reference, 24, smoke)
    maximum_lifetime = maximum_lifetime_without_test(outer_train)
    _ssl_state, nasa_state, target_state, _, _ = initialize_states(
        source, outer_train, features, report, config, seed + 100 * fold, device,
    )
    if set(map(str, target_state.fit_units)) != set(outer_train_units):
        raise AssertionError("Target normalization/SSL fit set is not the outer-train set")
    tcn = fit_multitask_fixed_epochs(
        nasa_state, outer_train, features, target_state, maximum_lifetime,
        config, seed + 100 * fold, tcn_epoch, device,
    )
    for parameter in tcn.parameters():
        parameter.requires_grad = False

    train_bundle = make_bundle(
        outer_train, target_knees, features, target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, outer_train_units,
    )
    test_bundle = make_bundle(
        outer_test, target_knees, features, target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, outer_test_units,
    )
    stats_median, stats_iqr = fit_statistics(train_bundle, test_bundle)
    head = fit_bstats_fixed_epochs(
        train_bundle, maximum_lifetime, seed + 100 * fold,
        head_epoch, history_config, device,
    )
    assert_causal(head, test_bundle, maximum_lifetime, device)
    prediction = predict_bundle(
        head, test_bundle, maximum_lifetime, device, history_config.batch_devices,
    )
    prediction.insert(0, "outer_fold", fold)
    prediction.insert(0, "seed", seed)
    metrics = regression_metrics(prediction, "predicted_rul_raw")
    row = {
        "seed": seed, "outer_fold": fold,
        "selection_inner_train_devices": len(inner_train_units),
        "selection_inner_validation_devices": len(inner_validation_units),
        "refit_supervised_devices": len(outer_train_units),
        "outer_test_devices": len(outer_test_units), "outer_test_samples": len(prediction),
        "selection_tcn_best_epoch": tcn_epoch,
        "selection_bstats_best_epoch": head_epoch,
        "selection_inner_validation_rul_mae": float(chosen.inner_validation_rul_mae),
        **metrics, **soh_metrics(prediction),
        "maximum_lifetime_from_outer_train": maximum_lifetime,
    }
    scaler_rows = [{
        "seed": seed, "outer_fold": fold, "feature": name,
        "median": float(stats_median[index]), "iqr": float(stats_iqr[index]),
        "fit_device_count": len(outer_train_units), "fit_stage": "refit_outer_train",
    } for index, name in enumerate(statistic_names(features))]
    split_row = {
        "seed": seed, "outer_fold": fold,
        "outer_train_units": outer_train_units,
        "inner_train_units": inner_train_units,
        "inner_validation_units": inner_validation_units,
        "outer_test_units": outer_test_units,
        "selection_train_units": inner_train_units,
        "selection_validation_units": inner_validation_units,
        "refit_supervised_units": outer_train_units,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "tcn_refit_epochs": tcn_epoch, "bstats_refit_epochs": head_epoch,
    }
    (output_dir / f"split_seed_{seed}_fold_{fold}.json").write_text(
        json.dumps(split_row, indent=2), encoding="utf-8",
    )
    save_refit_checkpoints(
        output_dir, seed, fold, tcn, head, target_state, features, config,
        history_config, maximum_lifetime, outer_train_units, outer_test_units,
        tcn_epoch, head_epoch,
    )
    return prediction, row, scaler_rows


def grouped_diagnostics(ensemble: pd.DataFrame, knees: pd.DataFrame) -> pd.DataFrame:
    working = ensemble.copy()
    working["current_time_bin"] = pd.cut(
        working.time, [-np.inf, 15, 30, 45, 60, np.inf],
        labels=["7-15", "15-30", "30-45", "45-60", ">60"],
    ).astype(str)
    working["true_rul_bin"] = pd.cut(
        working.true_rul_cycles, [-np.inf, 10, 20, 30, 45, 60, np.inf],
        labels=["<=10", "10-20", "20-30", "30-45", "45-60", ">60"],
    ).astype(str)
    knee_map = knees.set_index(knees.unit_id.astype(str)).has_knee.astype(bool).to_dict()
    working["knee_type"] = working.unit_id.astype(str).map(knee_map).map(
        {True: "knee", False: "no_knee"}
    )
    life = working.groupby("unit_id", as_index=False).true_eol_cycle.max()
    life["life_quartile"] = pd.qcut(
        life.true_eol_cycle.rank(method="first"), 4,
        labels=["Q1_short", "Q2", "Q3", "Q4_long"],
    ).astype(str)
    working = working.merge(life[["unit_id", "life_quartile"]], on="unit_id", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for group_type, column in (
        ("current_time", "current_time_bin"), ("true_rul", "true_rul_bin"),
        ("knee", "knee_type"), ("lifetime", "life_quartile"),
    ):
        for label, group in working.groupby(column, sort=False):
            rows.append({
                "group_type": group_type, "group": str(label),
                "samples": len(group), "devices": int(group.unit_id.nunique()),
                **regression_metrics(group, "predicted_rul_raw"), **soh_metrics(group),
            })
    return pd.DataFrame(rows)


def refit_audit(base: dict[str, Any], diagnostics: pd.DataFrame,
                output_dir: Path, seeds: tuple[int, ...], folds: int,
                expected_refit_devices: int) -> dict[str, Any]:
    for seed in seeds:
        for fold in range(folds):
            split = json.loads((output_dir / f"split_seed_{seed}_fold_{fold}.json").read_text(encoding="utf-8"))
            outer_train = set(split["outer_train_units"])
            outer_test = set(split["outer_test_units"])
            refit = set(split["refit_supervised_units"])
            scaler = set(split["target_scaler_fit_units"])
            if len(refit) != expected_refit_devices or refit != outer_train or scaler != outer_train:
                raise AssertionError("Refit did not use every allowed outer-training device")
            if refit & outer_test:
                raise AssertionError("Outer-test device entered refit")
    early = diagnostics.loc[
        diagnostics.group_type.eq("current_time") & diagnostics.group.eq("7-15")
    ].iloc[0]
    long_life = diagnostics.loc[
        diagnostics.group_type.eq("lifetime") & diagnostics.group.eq("Q4_long")
    ].iloc[0]
    audit = {
        **base,
        "selection_pass_reused_and_verified": True,
        "refit_supervised_devices_per_outer_fold": expected_refit_devices,
        "all_refit_models_use_full_outer_train": True,
        "baseline_rul_mae": BASELINE_RUL_MAE,
        "rul_mae_change_vs_selection_only": float(base["rul_mae"] - BASELINE_RUL_MAE),
        "early_7_15_rul_mae": float(early.rul_mae),
        "early_7_15_improvement_fraction": float((BASELINE_EARLY_MAE - early.rul_mae) / BASELINE_EARLY_MAE),
        "long_life_rul_bias": float(long_life.rul_bias),
        "long_life_bias_absolute_improvement_fraction": float(
            (abs(BASELINE_LONG_LIFE_BIAS) - abs(long_life.rul_bias)) / abs(BASELINE_LONG_LIFE_BIAS)
        ),
        "soh_mae_within_5_percent": bool(base["soh_mae"] <= 1.05 * BASELINE_SOH_MAE),
        "worst_device_within_10_percent": bool(base["worst_device_mae"] <= 1.10 * BASELINE_WORST_DEVICE_MAE),
        "mixture_experts_required": bool(base["rul_mae"] > 5.0),
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
    }
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    source, target, _validation, features, report, target_knees, _ = prepare_data(args.data_root)
    if len(features) != 16:
        raise ValueError("Refit OOF requires the frozen 16-dimensional contract")
    forbidden = [name for name in features if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        raise ValueError(f"Forbidden inputs: {forbidden}")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if args.smoke:
        keep = sorted(target.unit_id.astype(str).unique())[:args.smoke_units]
        target = subset(target, keep)
        target_knees = target_knees.loc[target_knees.unit_id.astype(str).isin(keep)].copy()
        folds = 2
    else:
        if seeds != FORMAL_SEEDS:
            raise ValueError("Formal refit OOF requires seeds 42,43,44")
        folds = OUTER_FOLDS
    selection = pd.read_csv(args.selection_run / "oof_results_by_fold_seed.csv")
    selection = selection.loc[selection.seed.isin(seeds) & selection.outer_fold.lt(folds)].copy()
    if len(selection) != len(seeds) * folds:
        raise ValueError("Selection pass does not cover every requested seed/fold")
    fold_map = stratified_outer_folds(target, folds)
    history_config = HistoryConfig(
        epochs=3 if args.smoke else args.epochs,
        patience=2 if args.smoke else args.patience,
        batch_devices=args.batch_devices,
    )
    predictions_all: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    scalers: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in range(folds):
            prediction, result, scaler = run_fold(
                seed, fold, source, target, target_knees, features, report, reference,
                fold_map, args.selection_seed, args.selection_run, selection,
                history_config, args.smoke, args.output_dir, device,
            )
            predictions_all.append(prediction)
            results.append(result)
            scalers.extend(scaler)
            pd.concat(predictions_all, ignore_index=True).to_csv(
                args.output_dir / "oof_predictions_by_seed.csv", index=False,
            )
            pd.DataFrame(results).to_csv(args.output_dir / "oof_results_by_fold_seed.csv", index=False)
            print(json.dumps(result), flush=True)
    predictions = pd.concat(predictions_all, ignore_index=True)
    ensemble, metrics, folds_frame = summarize_oof(predictions)
    metrics["variant"] = "B_stats_refit_strict_oof"
    diagnostics = grouped_diagnostics(ensemble, target_knees)
    ensemble.to_csv(args.output_dir / "oof_ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "oof_metrics.csv", index=False)
    folds_frame.to_csv(args.output_dir / "oof_results_by_fold.csv", index=False)
    diagnostics.to_csv(args.output_dir / "oof_grouped_diagnostics.csv", index=False)
    pd.DataFrame(scalers).to_csv(args.output_dir / "oof_history_scalers.csv", index=False)
    if args.smoke:
        base_audit = {
            "passed": True, "mode": "smoke", "devices": int(ensemble.unit_id.nunique()),
            **regression_metrics(ensemble, "predicted_rul_raw"), **soh_metrics(ensemble),
        }
        expected_refit = args.smoke_units // 2
    else:
        base_audit = audit_oof(predictions, ensemble, folds_frame, target, seeds, args.output_dir)
        expected_refit = 256
    audit = refit_audit(base_audit, diagnostics, args.output_dir, seeds, folds, expected_refit)
    (args.output_dir / "oof_refit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "strict two-pass device OOF refit for NASA+B_stats",
        "mode": "smoke" if args.smoke else "formal",
        "data_version": "Basilisk V1.0 unchanged",
        "selection_run": str(args.selection_run.resolve()),
        "selection_records_reused": True,
        "features": features, "feature_count": len(features),
        "local_tcn_window": 24, "history_statistics": 38,
        "outer_folds": folds, "seeds": list(seeds), "selection_seed": args.selection_seed,
        "history_configuration": asdict(history_config),
        "refit_policy": "inner-selected epochs; reinitialize; fit all outer-train units; no early stop",
        "audit": audit,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--selection-seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-devices", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-units", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
