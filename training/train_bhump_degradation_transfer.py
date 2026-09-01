"""Second-stage transfer: align degradation laws instead of marginal domains."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_bhump_positive_transfer import (
    Config,
    PositiveTransferTCN,
    adapt_fit,
    causal_smooth,
    choose_nested_units,
    loader,
    make_windows,
    per_unit_metrics,
    predict,
    regression_metrics,
    robust_fit,
    rul_evaluation,
    seed_all,
    set_adaptation_trainable,
    snapshot,
    ssl_fit,
    unlabeled_view,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_data"
DEFAULT_OUTPUT = ROOT / "bhump_degradation_transfer_runs"
METHODS = ("target_only", "target_ssl", "law_calibrated_finetune", "law_pcgrad", "law_gated_teacher")


def load_data(data_root: Path, feature_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    contracts = json.loads((data_root / "feature_contracts.json").read_text(encoding="utf-8"))
    report = json.loads((data_root / "degradation_contract_report.json").read_text(encoding="utf-8"))
    invariant = list(contracts["bhump_degradation_invariant"])
    if invariant != report["features"]:
        raise ValueError("Degradation contract/report mismatch")
    features = invariant if feature_mode == "invariant" else list(contracts["bhump_compact_v2"])
    source = pd.read_csv(data_root / "nasa_source_rich.csv")
    target = pd.read_csv(data_root / "basilisk_train_rich.csv")
    validation = pd.read_csv(data_root / "basilisk_validation_rich.csv")
    if set(target.unit_id.astype(str)) & set(validation.unit_id.astype(str)):
        raise ValueError("Target train/validation leakage")
    return source, target, validation, features, report


def calibrate_source_labels(source: pd.DataFrame, target_initial: float) -> pd.DataFrame:
    output = source.copy()
    calibrated = np.empty(len(output), dtype=float)
    for _, indices in output.groupby("unit_id", sort=False).groups.items():
        ordered = output.loc[indices].sort_values("time")
        initial = max(float(ordered.target_soh.iloc[0]), 0.8001)
        progress = (ordered.target_soh.to_numpy(float) - 0.80) / (initial - 0.80)
        calibrated[ordered.index.to_numpy()] = 0.80 + progress * (target_initial - 0.80)
    output["original_source_soh"] = output.target_soh
    output["target_soh"] = np.clip(calibrated, 0.0, 1.10)
    return output


def calibrated_source_arrays(source: pd.DataFrame, features: list[str], report: dict[str, Any],
                             config: Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # B0018 was frozen by the previous source-selection experiment and has the
    # closest degradation rate to Basilisk.  No validation/sealed result is used here.
    source = source.loc[source.unit_id.eq("B0018")].copy().reset_index(drop=True)
    source = calibrate_source_labels(source, float(report["target_initial_soh_median"]))
    state = robust_fit(source, features, "NASA:B0018")
    x, y, metadata = make_windows(source, features, state, config.window, True)
    # In full-masked mode, inconsistent source channels are exactly zero after
    # robust scaling.  Target inputs retain all compact-v2 channels.
    ratios = np.asarray([
        report["source_feature_scale_ratios"].get(name, 0.0) for name in features
    ], dtype=np.float32)
    x = x * ratios[None, None, :]
    assert y is not None
    return x, y, metadata


def l2sp_source_fit(model: PositiveTransferTCN, source_x: np.ndarray, source_y: np.ndarray,
                    reference: dict[str, torch.Tensor], l2sp_weight: float, seed: int,
                    config: Config, device: torch.device) -> PositiveTransferTCN:
    seed_all(seed)
    model = model.to(device)
    model.reset_soh_head()
    parameters = list(model.encoder.parameters()) + list(model.shared_projection.parameters()) + list(model.soh_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.source_learning_rate, weight_decay=config.weight_decay)
    train_loader = loader(source_x, source_y, config.batch_size, seed)
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    reference = {name: value.to(device) for name, value in reference.items() if name.startswith(("encoder.", "shared_projection."))}
    for _ in range(config.source_epochs):
        model.train()
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values, "source")[0]
            anchor_terms = [
                (parameter - reference[name]).pow(2).mean()
                for name, parameter in model.named_parameters() if name in reference
            ]
            anchor = torch.stack(anchor_terms).mean()
            loss = loss_fn(prediction, labels) + l2sp_weight * anchor
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    return model


def pcgrad_adapt(model: PositiveTransferTCN, target_x: np.ndarray, target_y: np.ndarray,
                 source_x: np.ndarray, source_y: np.ndarray, validation_x: np.ndarray,
                 validation_y: np.ndarray, source_weight: float, seed: int,
                 config: Config, device: torch.device) -> tuple[PositiveTransferTCN, int, float]:
    seed_all(seed)
    model = model.to(device)
    model.reset_soh_head()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.shared_projection.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.soh_head.parameters(), "lr": config.head_learning_rate},
    ], weight_decay=config.weight_decay)
    target_loader = loader(target_x, target_y, config.batch_size, seed + 11)
    source_loader = loader(source_x, source_y, config.batch_size, seed + 22)
    source_iterator = itertools.cycle(source_loader)
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    best_state, best_mae, best_epoch, stale = snapshot(model), float("inf"), 0, 0
    positive_cosines: list[float] = []
    for epoch in range(1, config.adapt_epochs + 1):
        set_adaptation_trainable(model, epoch, config)
        model.train()
        for target_values, target_labels in target_loader:
            target_values, target_labels = target_values.to(device), target_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            parameters = [parameter for parameter in model.deployment_parameters() if parameter.requires_grad]
            target_loss = loss_fn(model(target_values, "target")[0], target_labels)
            target_grads = torch.autograd.grad(target_loss, parameters, retain_graph=False, allow_unused=True)
            use_source = epoch <= math.ceil(config.adapt_epochs * 2 / 3)
            if use_source:
                source_values, source_labels = next(source_iterator)
                source_values, source_labels = source_values.to(device), source_labels.to(device)
                source_loss = loss_fn(model(source_values, "source")[0], source_labels)
                source_grads = torch.autograd.grad(source_loss, parameters, retain_graph=False, allow_unused=True)
                pairs = [(gt, gs) for gt, gs in zip(target_grads, source_grads) if gt is not None and gs is not None]
                dot = sum((gt * gs).sum() for gt, gs in pairs)
                target_norm = torch.sqrt(sum((gt * gt).sum() for gt, _ in pairs) + 1e-12)
                source_norm = torch.sqrt(sum((gs * gs).sum() for _, gs in pairs) + 1e-12)
                cosine = float((dot / (target_norm * source_norm)).detach().cpu())
                positive_cosines.append(max(cosine, 0.0))
                scale = source_weight * max(cosine, 0.0)
            else:
                source_grads, scale = [None] * len(parameters), 0.0
            for parameter, target_grad, source_grad in zip(parameters, target_grads, source_grads):
                if target_grad is None:
                    continue
                parameter.grad = target_grad.detach().clone()
                if source_grad is not None and scale > 0.0:
                    parameter.grad.add_(source_grad.detach(), alpha=scale)
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        estimate = predict(model, validation_x, config, device)
        mae = float(np.mean(np.abs(estimate - validation_y)))
        if mae < best_mae - 1e-7:
            best_state, best_mae, best_epoch, stale = snapshot(model), mae, epoch, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, float(np.mean(positive_cosines)) if positive_cosines else 0.0


def fit_gate(target_model: PositiveTransferTCN, teacher: PositiveTransferTCN,
             target_x: np.ndarray, target_y: np.ndarray, config: Config,
             device: torch.device) -> float:
    target_prediction = predict(target_model, target_x, config, device)
    teacher_prediction = predict(teacher, target_x, config, device)
    direction = teacher_prediction - target_prediction
    denominator = float(direction @ direction)
    if denominator <= 1e-12:
        return 0.0
    gate = float(((target_y - target_prediction) @ direction) / denominator)
    return float(np.clip(gate, 0.0, 0.50))


def gated_predict(target_model: PositiveTransferTCN, teacher: PositiveTransferTCN,
                  values: np.ndarray, gate: float, config: Config,
                  device: torch.device) -> np.ndarray:
    target_prediction = predict(target_model, values, config, device)
    teacher_prediction = predict(teacher, values, config, device)
    return np.clip(target_prediction + gate * (teacher_prediction - target_prediction), 0.0, 1.10)


def tune(source_x: np.ndarray, source_y: np.ndarray, target: pd.DataFrame,
         validation: pd.DataFrame, target_unlabeled_x: np.ndarray, target_state: Any,
         features: list[str], config: Config, seed: int,
         device: torch.device) -> tuple[float, float, pd.DataFrame]:
    selected = choose_nested_units(target, [10], seed)[10]
    subset = target.loc[target.unit_id.astype(str).isin(selected)].copy()
    target_x, target_y, _ = make_windows(subset, features, target_state, config.window, True)
    validation_x, validation_y, _ = make_windows(validation, features, target_state, config.window, True)
    assert target_y is not None and validation_y is not None
    base = PositiveTransferTCN(len(features), config)
    base = ssl_fit(base, [("target", target_unlabeled_x)], seed, config, device)
    base_state = snapshot(base)
    records = []
    best_l2, best_mae = 0.01, float("inf")
    for weight in (0.001, 0.01, 0.1):
        teacher = PositiveTransferTCN(len(features), config)
        teacher.load_state_dict(copy.deepcopy(base_state))
        teacher = l2sp_source_fit(teacher, source_x, source_y, base_state, weight, seed + 1, config, device)
        model = PositiveTransferTCN(len(features), config)
        model.load_state_dict(snapshot(teacher))
        model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, config, device)
        mae = float(np.mean(np.abs(predict(model, validation_x, config, device) - validation_y)))
        records.append({"stage": "l2sp", "candidate": weight, "validation_mae": mae, "best_epoch": epoch, "positive_gradient_cosine": np.nan})
        if mae < best_mae:
            best_l2, best_mae = weight, mae
    best_pc, best_mae = 0.10, float("inf")
    for weight in (0.05, 0.10, 0.20):
        model = PositiveTransferTCN(len(features), config)
        model.load_state_dict(copy.deepcopy(base_state))
        model, epoch, cosine = pcgrad_adapt(
            model, target_x, target_y, source_x, source_y, validation_x, validation_y,
            weight, seed, config, device,
        )
        mae = float(np.mean(np.abs(predict(model, validation_x, config, device) - validation_y)))
        records.append({"stage": "pcgrad", "candidate": weight, "validation_mae": mae, "best_epoch": epoch, "positive_gradient_cosine": cosine})
        if mae < best_mae:
            best_pc, best_mae = weight, mae
    return best_l2, best_pc, pd.DataFrame(records)


def run(args: argparse.Namespace) -> None:
    config = Config(ssl_epochs=args.ssl_epochs, source_epochs=args.source_epochs,
                    adapt_epochs=args.adapt_epochs, patience=args.patience)
    if args.feature_mode == "full_masked":
        config = replace(config, reconstruction_weight=0.10, order_weight=0.05)
    methods = args.methods.split(",")
    if unknown := set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    sizes = [int(value) for value in args.subset_sizes.split(",")]
    seeds = [int(value) for value in args.formal_seeds.split(",")]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    source, target, validation, features, contract_report = load_data(args.data_root, args.feature_mode)
    if args.smoke:
        config = replace(config, ssl_epochs=1, source_epochs=2, adapt_epochs=3, patience=2, batch_size=64)
        sizes, seeds = [5], [42]
        target = target.loc[target.unit_id.astype(str).isin(sorted(target.unit_id.astype(str).unique())[:12])].copy()
        validation = validation.loc[validation.unit_id.astype(str).isin(sorted(validation.unit_id.astype(str).unique())[:4])].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_unlabeled = unlabeled_view(target, features)
    target_state = robust_fit(target_unlabeled, features, "Basilisk:all_unlabeled_train")
    target_unlabeled_x, _, _ = make_windows(target_unlabeled, features, target_state, config.window, False)
    validation_x, validation_y, validation_meta = make_windows(validation, features, target_state, config.window, True)
    source_x, source_y, _ = calibrated_source_arrays(source, features, contract_report, config)
    assert validation_y is not None

    if args.skip_tuning or args.smoke:
        l2sp_weight, pcgrad_weight = 0.01, 0.10
        tuning = pd.DataFrame([{"stage": "skipped", "candidate": "defaults", "validation_mae": np.nan, "best_epoch": 0, "positive_gradient_cosine": np.nan}])
    else:
        l2sp_weight, pcgrad_weight, tuning = tune(
            source_x, source_y, target, validation, target_unlabeled_x,
            target_state, features, config, args.tuning_seed, device,
        )
    tuning.to_csv(args.output_dir / "tuning_results.csv", index=False)

    result_rows, unit_rows, prediction_rows, selected_rows = [], [], [], []
    for seed in seeds:
        nested = choose_nested_units(target, sizes, seed)
        ssl_model = PositiveTransferTCN(len(features), config)
        ssl_model = ssl_fit(ssl_model, [("target", target_unlabeled_x)], seed, config, device)
        ssl_state = snapshot(ssl_model)
        teacher = PositiveTransferTCN(len(features), config)
        teacher.load_state_dict(copy.deepcopy(ssl_state))
        teacher = l2sp_source_fit(
            teacher, source_x, source_y, ssl_state, l2sp_weight, seed + 1, config, device,
        )
        teacher_state = snapshot(teacher)
        for size in sizes:
            selected_units = nested[size]
            selected_rows.extend({"seed": seed, "subset_size": size, "unit_id": unit} for unit in selected_units)
            subset = target.loc[target.unit_id.astype(str).isin(selected_units)].copy()
            target_x, target_y, _ = make_windows(subset, features, target_state, config.window, True)
            assert target_y is not None
            fitted: dict[str, tuple[PositiveTransferTCN, int, float, np.ndarray]] = {}
            if "target_only" in methods:
                model = PositiveTransferTCN(len(features), config)
                model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, config, device)
                fitted["target_only"] = (model, epoch, 0.0, predict(model, validation_x, config, device))
            target_model = PositiveTransferTCN(len(features), config)
            target_model.load_state_dict(copy.deepcopy(ssl_state))
            target_model, target_epoch = adapt_fit(target_model, target_x, target_y, validation_x, validation_y, seed, config, device)
            if "target_ssl" in methods:
                fitted["target_ssl"] = (target_model, target_epoch, 0.0, predict(target_model, validation_x, config, device))
            if "law_calibrated_finetune" in methods:
                model = PositiveTransferTCN(len(features), config)
                model.load_state_dict(copy.deepcopy(teacher_state))
                model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, config, device)
                fitted["law_calibrated_finetune"] = (model, epoch, 0.0, predict(model, validation_x, config, device))
            if "law_pcgrad" in methods:
                model = PositiveTransferTCN(len(features), config)
                model.load_state_dict(copy.deepcopy(ssl_state))
                model, epoch, cosine = pcgrad_adapt(
                    model, target_x, target_y, source_x, source_y, validation_x, validation_y,
                    pcgrad_weight, seed, config, device,
                )
                fitted["law_pcgrad"] = (model, epoch, cosine, predict(model, validation_x, config, device))
            if "law_gated_teacher" in methods:
                gate = fit_gate(target_model, teacher, target_x, target_y, config, device)
                estimate = gated_predict(target_model, teacher, validation_x, gate, config, device)
                fitted["law_gated_teacher"] = (target_model, target_epoch, gate, estimate)

            for method in methods:
                model, epoch, auxiliary, estimate = fitted[method]
                aggregate = regression_metrics(validation_y, estimate)
                by_unit = per_unit_metrics(validation_meta, estimate)
                result_rows.append({
                    "method": method, "seed": seed, "subset_size": size, "best_epoch": epoch,
                    "target_labeled_units": len(selected_units), "target_labeled_windows": len(target_x),
                    "target_unlabeled_units": target.unit_id.nunique(), "auxiliary_value": auxiliary,
                    **aggregate, "device_mae_std": float(by_unit.soh_mae.std(ddof=0)),
                    "worst_device_mae": float(by_unit.soh_mae.max()),
                })
                unit_rows.extend({"method": method, "seed": seed, "subset_size": size, **row} for row in by_unit.to_dict("records"))
                prediction_rows.extend({"method": method, "seed": seed, "subset_size": size, **meta, "predicted_soh": float(value)} for meta, value in zip(validation_meta.to_dict("records"), estimate))
                artifact = {
                    "method": method, "seed": seed, "subset_size": size, "features": features,
                    "target_units": selected_units, "model_state": snapshot(model),
                    "teacher_state": teacher_state if method == "law_gated_teacher" else None,
                    "auxiliary_value": auxiliary, "configuration": asdict(config),
                }
                torch.save(artifact, args.output_dir / f"checkpoint_{method}_{size}_{seed}.pt")
                print(json.dumps(result_rows[-1]), flush=True)

    results = pd.DataFrame(result_rows)
    units = pd.DataFrame(unit_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows).drop_duplicates()
    results.to_csv(args.output_dir / "results_by_seed.csv", index=False)
    units.to_csv(args.output_dir / "results_per_unit.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    selected.to_csv(args.output_dir / "selected_target_units.csv", index=False)
    baseline = results.loc[results.method.eq("target_ssl"), ["seed", "subset_size", "soh_mae", "worst_device_mae"]].rename(columns={
        "soh_mae": "baseline_mae", "worst_device_mae": "baseline_worst_device_mae",
    })
    gains = results.merge(baseline, on=["seed", "subset_size"], how="left")
    gains["relative_gain"] = (gains.baseline_mae - gains.soh_mae) / gains.baseline_mae
    gains["worst_device_change"] = (gains.worst_device_mae - gains.baseline_worst_device_mae) / gains.baseline_worst_device_mae
    gains["positive_seed"] = gains.relative_gain.gt(0)
    gains.to_csv(args.output_dir / "transfer_gains_by_seed.csv", index=False)
    summary = gains.groupby(["method", "subset_size"], as_index=False).agg(
        soh_mae_mean=("soh_mae", "mean"), soh_mae_std=("soh_mae", "std"),
        relative_gain_mean=("relative_gain", "mean"), positive_seed_count=("positive_seed", "sum"),
        worst_device_mae_mean=("worst_device_mae", "mean"),
        baseline_worst_device_mae_mean=("baseline_worst_device_mae", "mean"),
        device_mae_std_mean=("device_mae_std", "mean"),
    )
    summary["worst_device_change"] = (summary.worst_device_mae_mean - summary.baseline_worst_device_mae_mean) / summary.baseline_worst_device_mae_mean
    summary["subset_success"] = summary.relative_gain_mean.ge(0.05) & summary.positive_seed_count.ge(2) & summary.worst_device_change.le(0.10)
    stability = summary.loc[summary.method.str.startswith("law_")].groupby("method", as_index=False).agg(
        successful_subset_count=("subset_success", "sum"), mean_relative_gain=("relative_gain_mean", "mean"),
        maximum_worst_device_change=("worst_device_change", "max"),
    )
    stability["stable_positive_transfer"] = stability.successful_subset_count.ge(2)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    stability.to_csv(args.output_dir / "transfer_stability.csv", index=False)

    stable = stability.loc[stability.stable_positive_transfer, "method"].tolist()
    rul_info: dict[str, Any] = {"executed": False, "reason": "no stable positive transfer evidence"}
    if stable:
        choice = summary.loc[summary.method.isin(stable) & summary.subset_success].sort_values(["soh_mae_mean", "subset_size"], ascending=[True, False]).iloc[0]
        method, size = str(choice.method), int(choice.subset_size)
        seed_table = results.loc[results.method.eq(method) & results.subset_size.eq(size)].sort_values("soh_mae")
        chosen_seed = int(seed_table.iloc[len(seed_table) // 2].seed)
        chosen = predictions.loc[predictions.method.eq(method) & predictions.subset_size.eq(size) & predictions.seed.eq(chosen_seed)].copy()
        alpha_table = []
        for alpha in (0.1, 0.2, 0.3, 0.5):
            ordered = chosen.sort_values(["unit_id", "time"])
            smooth = np.concatenate([causal_smooth(group.predicted_soh.to_numpy(float), alpha) for _, group in ordered.groupby("unit_id")])
            truth = np.concatenate([group.target_soh.to_numpy(float) for _, group in ordered.groupby("unit_id")])
            alpha_table.append({"alpha": alpha, "soh_mae": float(np.mean(np.abs(smooth - truth)))})
        alpha_frame = pd.DataFrame(alpha_table).sort_values(["soh_mae", "alpha"])
        alpha_frame.to_csv(args.output_dir / "smoothing_selection.csv", index=False)
        chosen_units = selected.loc[selected.seed.eq(chosen_seed) & selected.subset_size.eq(size), "unit_id"].astype(str)
        slope_train = target.loc[target.unit_id.astype(str).isin(chosen_units)].copy()
        detail, rul_metrics = rul_evaluation(chosen, slope_train, float(alpha_frame.iloc[0].alpha))
        detail.to_csv(args.output_dir / "validation_rul_predictions.csv", index=False)
        rul_metrics.to_csv(args.output_dir / "rul_metrics.csv", index=False)
        rul_info = {"executed": True, "method": method, "subset_size": size, "seed": chosen_seed}

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_contract": (
            "bhump_degradation_invariant" if args.feature_mode == "invariant"
            else "bhump_compact_v2_with_source_inconsistent_channels_masked"
        ),
        "feature_mode": args.feature_mode, "features": features,
        "source_unit": "B0018", "source_labels_calibrated": True,
        "l2sp_weight": l2sp_weight, "pcgrad_weight": pcgrad_weight,
        "configuration": asdict(config), "methods": methods, "seeds": seeds, "subset_sizes": sizes,
        "stable_positive_transfer_found": bool(stable), "stable_methods": stable,
        "rul_evaluation": rul_info,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "conclusion": "stable positive transfer found" if stable else "no stable positive transfer evidence",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset-sizes", default="5,10,20")
    parser.add_argument("--tuning-seed", type=int, default=41)
    parser.add_argument("--formal-seeds", default="42,43,44")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--feature-mode", choices=("invariant", "full_masked"), default="invariant")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ssl-epochs", type=int, default=16)
    parser.add_argument("--source-epochs", type=int, default=30)
    parser.add_argument("--adapt-epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
