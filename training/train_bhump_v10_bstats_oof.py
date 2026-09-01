"""Strict device-level OOF validation for the V1.0 NASA+B_stats model.

Each outer fold is completely unseen by target-supervised TCN adaptation,
history-head training, early stopping and all normalization fits.  The outer
training devices are split again into inner-train and inner-validation groups.
The held-out outer fold is used once for prediction after both stages stop.
No existing full320 target-supervised checkpoint and no sealed file is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_v10_history_ablation import (
    HistoryAblationModel, HistoryConfig, assert_causal, fit_statistics,
    make_bundle, predict_bundle, regression_metrics, seed_all, soh_metrics,
    statistic_names, summarize_predictions, train_variant,
)
from train_bhump_v10_rul_multitask import (
    build_config, initialize_states, maximum_training_lifetime, prepare_data,
    train_configuration,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_bstats_oof_runs"
FORMAL_SEEDS = (42, 43, 44)
OUTER_FOLDS = 5


def device_lifetimes(frame: pd.DataFrame) -> pd.DataFrame:
    life = frame.groupby("unit_id", as_index=False).agg(
        eol_cycle=("true_eol_cycle", "max"), observations=("time", "size"),
    )
    life["unit_id"] = life.unit_id.astype(str)
    return life.sort_values(["eol_cycle", "unit_id"], ignore_index=True)


def stratified_outer_folds(frame: pd.DataFrame, folds: int = OUTER_FOLDS) -> dict[str, int]:
    """Round-robin sorted lifetimes, reversing alternate blocks for balance."""
    life = device_lifetimes(frame)
    assignment: dict[str, int] = {}
    for block_start in range(0, len(life), folds):
        block = life.iloc[block_start:block_start + folds]
        fold_order = list(range(len(block)))
        if (block_start // folds) % 2:
            fold_order.reverse()
        for row_index, fold in enumerate(fold_order):
            assignment[str(block.iloc[row_index].unit_id)] = int(fold)
    counts = pd.Series(assignment).value_counts()
    if len(assignment) != frame.unit_id.nunique() or counts.max() - counts.min() > 1:
        raise RuntimeError("Outer fold construction is incomplete or imbalanced")
    return assignment


def deterministic_score(unit_id: str, selection_seed: int, outer_fold: int) -> int:
    value = f"{selection_seed}:{outer_fold}:{unit_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def inner_split(outer_train_units: list[str], frame: pd.DataFrame,
                selection_seed: int, outer_fold: int,
                validation_fraction: float = 0.10) -> tuple[list[str], list[str]]:
    """Select a deterministic, lifetime-stratified inner validation set."""
    life = device_lifetimes(frame.loc[frame.unit_id.astype(str).isin(outer_train_units)]).copy()
    bin_count = min(10, max(2, len(life) // 12))
    life["life_bin"] = pd.qcut(life.eol_cycle.rank(method="first"), bin_count,
                               labels=False, duplicates="drop")
    selected: list[str] = []
    for _, group in life.groupby("life_bin", sort=True):
        ordered = sorted(group.unit_id.astype(str),
                         key=lambda unit: deterministic_score(unit, selection_seed, outer_fold))
        count = max(1, int(round(len(group) * validation_fraction)))
        selected.extend(ordered[:count])
    desired = max(1, int(round(len(outer_train_units) * validation_fraction)))
    if len(selected) > desired:
        selected = sorted(selected,
                          key=lambda unit: deterministic_score(unit, selection_seed + 1, outer_fold))[:desired]
    elif len(selected) < desired:
        remaining = sorted(set(outer_train_units) - set(selected),
                           key=lambda unit: deterministic_score(unit, selection_seed + 2, outer_fold))
        selected.extend(remaining[:desired - len(selected)])
    inner_validation = sorted(set(selected))
    inner_train = sorted(set(outer_train_units) - set(inner_validation))
    if set(inner_train) & set(inner_validation) or not inner_train or not inner_validation:
        raise RuntimeError("Invalid inner split")
    return inner_train, inner_validation


def subset(frame: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    return frame.loc[frame.unit_id.astype(str).isin(units)].copy()


def maximum_lifetime_without_test(outer_train: pd.DataFrame) -> float:
    value = maximum_training_lifetime(outer_train)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Invalid outer-training lifetime cap")
    return value


def save_tcn_checkpoint(path: Path, model: torch.nn.Module, target_state: Any,
                        config: Any, seed: int, fold: int, inner_train: list[str],
                        inner_validation: list[str], outer_test: list[str],
                        best_epoch: int, features: list[str], maximum_lifetime: float) -> None:
    torch.save({
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "seed": seed, "outer_fold": fold,
        "inner_train_units": inner_train,
        "inner_validation_units": inner_validation,
        "outer_test_units": outer_test,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "target_scaler_median": target_state.median,
        "target_scaler_iqr": target_state.iqr,
        "configuration": asdict(config), "best_epoch": best_epoch,
        "features": features, "maximum_training_lifetime": maximum_lifetime,
        "nasa_supervised_pretraining": True,
    }, path)


def run_fold(seed: int, fold: int, source: pd.DataFrame, target: pd.DataFrame,
             target_knees: pd.DataFrame, features: list[str], report: dict[str, Any],
             reference: dict[str, Any], fold_map: dict[str, int], selection_seed: int,
             history_config: HistoryConfig, smoke: bool, output_dir: Path,
             device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    outer_test_units = sorted(unit for unit, value in fold_map.items() if value == fold)
    outer_train_units = sorted(set(fold_map) - set(outer_test_units))
    inner_train_units, inner_validation_units = inner_split(
        outer_train_units, target, selection_seed, fold,
        validation_fraction=0.20 if smoke else 0.10,
    )
    outer_train = subset(target, outer_train_units)
    inner_train_frame = subset(target, inner_train_units)
    inner_validation_frame = subset(target, inner_validation_units)
    outer_test = subset(target, outer_test_units)
    if set(outer_test.unit_id.astype(str)) & set(outer_train.unit_id.astype(str)):
        raise AssertionError("Outer train/test overlap")

    config = build_config(reference, 24, smoke)
    maximum_lifetime = maximum_lifetime_without_test(outer_train)
    ssl_state, nasa_state, target_state, _, _ = initialize_states(
        source, outer_train, features, report, config, seed + 100 * fold, device,
    )
    del ssl_state
    tcn, tcn_epoch, _, _ = train_configuration(
        "nasa_pretrain_finetune", inner_train_frame, inner_validation_frame,
        features, target_state, nasa_state, maximum_lifetime, config,
        seed + 100 * fold, device,
    )
    for parameter in tcn.parameters():
        parameter.requires_grad = False
    tcn_path = output_dir / f"checkpoint_tcn_seed_{seed}_fold_{fold}.pt"
    save_tcn_checkpoint(
        tcn_path, tcn, target_state, config, seed, fold, inner_train_units,
        inner_validation_units, outer_test_units, tcn_epoch, features, maximum_lifetime,
    )

    train_bundle = make_bundle(
        inner_train_frame, target_knees, features, target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, inner_train_units,
    )
    validation_bundle = make_bundle(
        inner_validation_frame, target_knees, features, target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, inner_validation_units,
    )
    test_bundle = make_bundle(
        outer_test, target_knees, features, target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, outer_test_units,
    )
    stats_median, stats_iqr = fit_statistics(train_bundle, validation_bundle)
    test_bundle.stats = ((test_bundle.stats - stats_median) / stats_iqr).astype(np.float32)
    head_path = output_dir / f"checkpoint_bstats_seed_{seed}_fold_{fold}.pt"
    head, head_epoch, inner_mae = train_variant(
        "B_stats", train_bundle, validation_bundle, maximum_lifetime,
        seed + 100 * fold, history_config, device, head_path,
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
        "outer_train_devices": len(outer_train_units),
        "inner_train_devices": len(inner_train_units),
        "inner_validation_devices": len(inner_validation_units),
        "outer_test_devices": len(outer_test_units),
        "outer_test_samples": len(prediction),
        "tcn_best_epoch": tcn_epoch, "bstats_best_epoch": head_epoch,
        "inner_validation_rul_mae": inner_mae,
        **metrics, **soh_metrics(prediction),
        "maximum_lifetime_from_outer_train": maximum_lifetime,
    }
    scaler_rows = [{
        "seed": seed, "outer_fold": fold, "feature": name,
        "median": float(stats_median[index]), "iqr": float(stats_iqr[index]),
        "fit_device_count": len(inner_train_units),
    } for index, name in enumerate(statistic_names(features))]
    split_row = {
        "seed": seed, "outer_fold": fold,
        "outer_train_units": outer_train_units,
        "inner_train_units": inner_train_units,
        "inner_validation_units": inner_validation_units,
        "outer_test_units": outer_test_units,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
    }
    (output_dir / f"split_seed_{seed}_fold_{fold}.json").write_text(
        json.dumps(split_row, indent=2), encoding="utf-8",
    )
    return prediction, row, scaler_rows


def summarize_oof(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle", "outer_fold"]
    ensemble = predictions.groupby(keys, as_index=False).agg(
        predicted_soh=("predicted_soh", "mean"),
        predicted_rul_raw=("predicted_rul_raw", "mean"),
        local_tcn_soh=("local_tcn_soh", "mean"),
        soh_ensemble_std=("predicted_soh", lambda values: float(np.std(values, ddof=0))),
        rul_ensemble_std=("predicted_rul_raw", lambda values: float(np.std(values, ddof=0))),
        seed_count=("seed", "nunique"),
    )
    ensemble["predicted_rul_calibrated"] = ensemble.predicted_rul_raw
    for_metrics = ensemble.copy()
    for_metrics["variant"] = "B_stats_strict_oof"
    metrics = summarize_predictions(for_metrics)
    fold_rows: list[dict[str, Any]] = []
    for fold, group in ensemble.groupby("outer_fold", sort=True):
        fold_rows.append({"outer_fold": int(fold), "samples": len(group),
                          "devices": int(group.unit_id.nunique()),
                          **regression_metrics(group, "predicted_rul_raw"), **soh_metrics(group)})
    return ensemble, metrics, pd.DataFrame(fold_rows)


def audit_oof(predictions: pd.DataFrame, ensemble: pd.DataFrame,
              fold_rows: pd.DataFrame, target: pd.DataFrame, seeds: tuple[int, ...],
              output_dir: Path) -> dict[str, Any]:
    expected_keys = target.loc[target.time.ge(7.0), ["unit_id", "time"]].copy()
    if expected_keys.duplicated().any():
        raise AssertionError("Target keys are not unique")
    expected_count = len(expected_keys)
    per_seed = predictions.groupby("seed").size()
    if set(per_seed.index.astype(int)) != set(seeds) or not per_seed.eq(expected_count).all():
        raise AssertionError("Every seed must predict every eligible target sample exactly once")
    occurrence = predictions.groupby(["seed", "unit_id", "time"]).size()
    if not occurrence.eq(1).all():
        raise AssertionError("Duplicate or missing OOF key")
    if ensemble.unit_id.nunique() != 320 or len(ensemble) != expected_count:
        raise AssertionError("OOF ensemble does not cover all 320 devices")
    if not ensemble.seed_count.eq(len(seeds)).all():
        raise AssertionError("OOF seed ensemble is incomplete")
    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Non-finite OOF output")
    split_files = sorted(output_dir.glob("split_seed_*_fold_*.json"))
    if len(split_files) != len(seeds) * OUTER_FOLDS:
        raise AssertionError("Missing split audits")
    for path in split_files:
        split = json.loads(path.read_text(encoding="utf-8"))
        outer_test = set(split["outer_test_units"])
        outer_train = set(split["outer_train_units"])
        inner_train = set(split["inner_train_units"])
        inner_validation = set(split["inner_validation_units"])
        scaler = set(split["target_scaler_fit_units"])
        if outer_test & outer_train or inner_train & inner_validation:
            raise AssertionError(f"Device leakage in {path.name}")
        if inner_train | inner_validation != outer_train:
            raise AssertionError(f"Inner split does not reconstruct outer train in {path.name}")
        if scaler != outer_train or scaler & outer_test:
            raise AssertionError(f"Target scaler leakage in {path.name}")
    overall = regression_metrics(ensemble, "predicted_rul_raw")
    audit = {
        "passed": True,
        "outer_folds": OUTER_FOLDS, "seeds": list(seeds),
        "devices": int(ensemble.unit_id.nunique()), "samples": len(ensemble),
        **overall, **soh_metrics(ensemble),
        "fold_mae_mean": float(fold_rows.rul_mae.mean()),
        "fold_mae_std": float(fold_rows.rul_mae.std(ddof=0)),
        "fold_mae_min": float(fold_rows.rul_mae.min()),
        "fold_mae_max": float(fold_rows.rul_mae.max()),
        "mae_in_target_4p5_to_5p0": bool(4.5 <= overall["rul_mae"] <= 5.0),
        "mae_at_most_5": bool(overall["rul_mae"] <= 5.0),
        "all_folds_at_most_6": bool(fold_rows.rul_mae.le(6.0).all()),
        "fold_std_at_most_0p5": bool(fold_rows.rul_mae.std(ddof=0) <= 0.5),
        "outer_test_never_used_for_training_or_scaling": True,
        "existing_full320_supervised_checkpoints_reused": False,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
    }
    (output_dir / "oof_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.selection_seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    source, target, _validation, features, report, target_knees, _ = prepare_data(args.data_root)
    if len(features) != 16:
        raise ValueError("Strict OOF requires the frozen 16-dimensional contract")
    forbidden = [feature for feature in features
                 if any(token in feature.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        raise ValueError(f"Forbidden inputs: {forbidden}")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not args.smoke and seeds != FORMAL_SEEDS:
        raise ValueError("Formal strict OOF requires seeds 42,43,44")
    if args.smoke:
        keep = sorted(target.unit_id.astype(str).unique())[:args.smoke_units]
        target = subset(target, keep)
        target_knees = target_knees.loc[target_knees.unit_id.astype(str).isin(keep)].copy()
        folds = 2
    else:
        folds = OUTER_FOLDS
    fold_map = stratified_outer_folds(target, folds)
    history_config = HistoryConfig(
        epochs=3 if args.smoke else args.epochs,
        patience=2 if args.smoke else args.patience,
        batch_devices=args.batch_devices,
    )
    prediction_frames: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in range(folds):
            prediction, result, scalers = run_fold(
                seed, fold, source, target, target_knees, features, report, reference,
                fold_map, args.selection_seed, history_config, args.smoke,
                args.output_dir, device,
            )
            prediction_frames.append(prediction)
            result_rows.append(result)
            scaler_rows.extend(scalers)
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                args.output_dir / "oof_predictions_by_seed.csv", index=False,
            )
            pd.DataFrame(result_rows).to_csv(args.output_dir / "oof_results_by_fold_seed.csv", index=False)
            print(json.dumps(result), flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble, metrics, fold_rows = summarize_oof(predictions)
    ensemble.to_csv(args.output_dir / "oof_ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "oof_metrics.csv", index=False)
    fold_rows.to_csv(args.output_dir / "oof_results_by_fold.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(args.output_dir / "oof_history_scalers.csv", index=False)
    if args.smoke:
        audit = {"passed": True, "mode": "smoke", "devices": int(ensemble.unit_id.nunique()),
                 "samples": len(ensemble), "sealed_features_accessed": False,
                 "sealed_labels_accessed": False}
    else:
        audit = audit_oof(predictions, ensemble, fold_rows, target, seeds, args.output_dir)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "strict nested device OOF for NASA+B_stats",
        "mode": "smoke" if args.smoke else "formal",
        "data_version": "Basilisk V1.0 unchanged",
        "features": features, "feature_count": len(features),
        "outer_folds": folds, "inner_validation_fraction": 0.20 if args.smoke else 0.10,
        "seeds": list(seeds), "selection_seed": args.selection_seed,
        "local_tcn_window": 24, "history_statistics": 38,
        "target_supervised_tcn_retrained_per_fold": True,
        "nasa_supervised_pretraining_retrained_per_fold": True,
        "existing_full320_supervised_checkpoints_reused": False,
        "history_configuration": asdict(history_config),
        "audit": audit,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
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
