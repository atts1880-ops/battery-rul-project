"""Fixed-10-device five-seed SOH ensemble and causal curve-model RUL ensemble.

V1.0 data, the 16-feature degradation-invariant contract and the 16/24-channel
TCN are unchanged.  All five seeds use exactly the same ten labeled target
devices.  RUL is derived only from the causal predicted-SOH history through
linear, exponential and continuous piecewise-linear (knee) extrapolators.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import theilslopes

from train_bhump_degradation_transfer import (
    calibrated_source_arrays, l2sp_source_fit, load_data,
)
from train_bhump_positive_transfer import (
    Config, PositiveTransferTCN, adapt_fit, causal_smooth, make_windows,
    predict, regression_metrics, robust_fit, seed_all, slope_bounds, snapshot,
    ssl_fit, unlabeled_view,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_fixed10_ensemble_runs"
CANDIDATES = ("linear", "exponential", "knee")


def fixed_units(reference: Path) -> list[str]:
    manifest = json.loads((reference / "experiment_manifest.json").read_text(encoding="utf-8"))
    identity = manifest["rul_evaluation"]
    selected = pd.read_csv(reference / "selected_target_units.csv")
    units = sorted(selected.loc[
        selected.seed.eq(int(identity["seed"]))
        & selected.subset_size.eq(int(identity["subset_size"])), "unit_id"
    ].astype(str).unique())
    if int(identity["subset_size"]) != 10 or len(units) != 10:
        raise ValueError("Reference best model does not identify exactly ten fixed devices")
    return units


def capped_rul(current_time: float, current_soh: float, slope: float,
               maximum_lifetime: float) -> float:
    if current_soh <= 0.80:
        return 0.0
    raw = max((current_soh - 0.80) / max(-slope, 1.0e-9), 0.0)
    return float(min(raw, max(maximum_lifetime - current_time, 0.0)))


def linear_rul(times: np.ndarray, values: np.ndarray,
               bounds: tuple[float, float, float, float]) -> float:
    lower, upper, prior, maximum_lifetime = bounds
    window = min(12, len(values))
    if window < 3:
        slope = prior
    else:
        slope = float(theilslopes(values[-window:], times[-window:]).slope)
        if not np.isfinite(slope) or slope >= -1.0e-7:
            slope = prior
        slope = float(np.clip(slope, lower, upper))
    return capped_rul(float(times[-1]), float(values[-1]), slope, maximum_lifetime)


def exponential_rul(times: np.ndarray, values: np.ndarray,
                    bounds: tuple[float, float, float, float]) -> float:
    fallback = linear_rul(times, values, bounds)
    if len(values) < 5 or values[-1] <= 0.80:
        return 0.0 if values[-1] <= 0.80 else fallback
    window = min(20, len(values))
    local_t = times[-window:]
    local_y = np.clip(values[-window:], 0.801, 1.10)
    slope = float(theilslopes(np.log(local_y), local_t).slope)
    if not np.isfinite(slope) or slope >= -1.0e-7:
        return fallback
    raw = (math.log(0.80) - math.log(float(local_y[-1]))) / slope
    if not np.isfinite(raw) or raw < 0:
        return fallback
    return float(min(raw, max(bounds[3] - float(times[-1]), 0.0)))


def knee_rul(times: np.ndarray, values: np.ndarray,
             bounds: tuple[float, float, float, float]) -> float:
    fallback = linear_rul(times, values, bounds)
    if len(values) < 10 or values[-1] <= 0.80:
        return 0.0 if values[-1] <= 0.80 else fallback
    window = min(32, len(values))
    local_t = times[-window:].astype(float)
    local_y = values[-window:].astype(float)
    centered_t = local_t - local_t[0]
    best: tuple[float, float] | None = None
    for index in range(4, len(local_y) - 4):
        knee = centered_t[index]
        design = np.column_stack([
            np.ones(len(centered_t)), centered_t, np.maximum(centered_t - knee, 0.0),
        ])
        coefficients = np.linalg.lstsq(design, local_y, rcond=None)[0]
        fitted = design @ coefficients
        scale = max(float(np.median(np.abs(local_y - np.median(local_y)))), 1.0e-4)
        residual = np.abs(local_y - fitted)
        score = float(np.mean(np.where(residual <= scale, 0.5 * residual**2 / scale, residual - 0.5 * scale)))
        post_slope = float(coefficients[1] + coefficients[2])
        if post_slope < -1.0e-7 and (best is None or score < best[0]):
            best = (score, post_slope)
    if best is None:
        return fallback
    slope = float(np.clip(best[1], bounds[0], bounds[1]))
    return capped_rul(float(times[-1]), float(values[-1]), slope, bounds[3])


def history_stage(time: float) -> str:
    if time <= 30:
        return "early_le_30"
    if time <= 55:
        return "middle_31_55"
    return "late_gt_55"


def candidate_predictions(predictions: pd.DataFrame, slope_train: pd.DataFrame,
                          alpha: float) -> pd.DataFrame:
    bounds = slope_bounds(slope_train)
    rows = []
    for unit_id, group in predictions.sort_values(["unit_id", "time"]).groupby("unit_id"):
        group = group.reset_index(drop=True)
        times = group.time.to_numpy(float)
        smooth = causal_smooth(group.predicted_soh.to_numpy(float), alpha)
        for index in range(len(group)):
            current_times, current_values = times[:index + 1], smooth[:index + 1]
            rows.append({
                "unit_id": unit_id, "time": float(times[index]), "smoothed_soh": float(smooth[index]),
                "target_soh": float(group.target_soh.iloc[index]),
                "true_rul_cycles": float(group.true_rul_cycles.iloc[index]),
                "true_eol_cycle": float(group.true_eol_cycle.iloc[index]),
                "stage": history_stage(float(times[index])),
                "maximum_allowed_rul": max(float(bounds[3] - times[index]), 0.0),
                "linear": linear_rul(current_times, current_values, bounds),
                "exponential": exponential_rul(current_times, current_values, bounds),
                "knee": knee_rul(current_times, current_values, bounds),
            })
    return pd.DataFrame(rows)


def simplex_grid(dimensions: int, step: float) -> list[np.ndarray]:
    count = int(round(1.0 / step))
    compositions: list[list[int]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            compositions.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], count, dimensions)
    return [np.asarray(values, dtype=float) / count for values in compositions]


def cross_fitted_soh_ensemble(long_predictions: pd.DataFrame, seeds: list[int],
                              folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle"]
    matrix = long_predictions.pivot(index=keys, columns="seed", values="predicted_soh").reset_index()
    seed_columns = seeds
    units = sorted(matrix.unit_id.astype(str).unique())
    fold_by_unit = {unit: index % folds for index, unit in enumerate(units)}
    matrix["fold"] = matrix.unit_id.astype(str).map(fold_by_unit)
    matrix["predicted_soh"] = np.nan
    weight_rows = []
    grid = simplex_grid(len(seeds), 0.10)

    def fit(frame: pd.DataFrame) -> tuple[np.ndarray, float, float]:
        values = frame[seed_columns].to_numpy(float)
        truth = frame.target_soh.to_numpy(float)
        best = (math.inf, np.ones(len(seeds)) / len(seeds), 0.0)
        for weight in grid:
            raw = values @ weight
            bias = float(np.median(truth - raw))
            estimate = np.clip(raw + bias, 0.0, 1.10)
            mae = float(np.mean(np.abs(estimate - truth)))
            if mae < best[0] - 1.0e-12:
                best = (mae, weight, bias)
        return best[1], best[2], best[0]

    for fold in range(folds):
        training = matrix.loc[matrix.fold.ne(fold)]
        test_indices = matrix.index[matrix.fold.eq(fold)]
        weight, bias, mae = fit(training)
        raw = matrix.loc[test_indices, seed_columns].to_numpy(float) @ weight + bias
        matrix.loc[test_indices, "predicted_soh"] = np.clip(raw, 0.0, 1.10)
        weight_rows.append({
            "kind": "cross_fit", "fold": fold,
            **{f"weight_seed_{seed}": float(value) for seed, value in zip(seeds, weight)},
            "bias": bias, "training_mae": mae, "training_devices": int(training.unit_id.nunique()),
        })
    weight, bias, mae = fit(matrix)
    weight_rows.append({
        "kind": "frozen_all_validation", "fold": -1,
        **{f"weight_seed_{seed}": float(value) for seed, value in zip(seeds, weight)},
        "bias": bias, "training_mae": mae, "training_devices": int(matrix.unit_id.nunique()),
    })
    matrix["ensemble_std"] = matrix[seed_columns].std(axis=1, ddof=0)
    return matrix, pd.DataFrame(weight_rows)


def fit_weights(frame: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    values = frame[list(CANDIDATES)].to_numpy(float)
    truth = frame.true_rul_cycles.to_numpy(float)
    best_weight, best_bias, best_mae = np.asarray([1.0, 0.0, 0.0]), 0.0, math.inf
    for weight in simplex_grid(len(CANDIDATES), 0.05):
        raw = values @ weight
        bias = float(np.median(truth - raw))
        estimate = np.clip(raw + bias, 0.0, frame.maximum_allowed_rul.to_numpy(float))
        mae = float(np.mean(np.abs(estimate - truth)))
        if mae < best_mae - 1.0e-12:
            best_weight, best_bias, best_mae = weight, bias, mae
    return best_weight, best_bias, best_mae


def cross_fitted_curve_ensemble(candidates: pd.DataFrame, folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = candidates.copy()
    output["fold"] = -1
    output["ensemble_rul"] = np.nan
    units = sorted(output.unit_id.astype(str).unique())
    fold_by_unit = {unit: index % folds for index, unit in enumerate(units)}
    output["fold"] = output.unit_id.astype(str).map(fold_by_unit)
    weight_rows = []
    for fold in range(folds):
        train = output.loc[output.fold.ne(fold)]
        test_indices = output.index[output.fold.eq(fold)]
        for stage in sorted(output.stage.unique()):
            fitting = train.loc[train.stage.eq(stage)]
            weight, bias, mae = fit_weights(fitting)
            stage_indices = test_indices[output.loc[test_indices, "stage"].eq(stage)]
            raw = output.loc[stage_indices, list(CANDIDATES)].to_numpy(float) @ weight + bias
            output.loc[stage_indices, "ensemble_rul"] = np.clip(
                raw, 0.0, output.loc[stage_indices, "maximum_allowed_rul"].to_numpy(float),
            )
            weight_rows.append({
                "kind": "cross_fit", "fold": fold, "stage": stage,
                **{f"weight_{name}": float(value) for name, value in zip(CANDIDATES, weight)},
                "bias_cycles": bias, "training_mae": mae, "training_devices": int(fitting.unit_id.nunique()),
            })
    if output.ensemble_rul.isna().any():
        raise RuntimeError("Cross-fitted RUL ensemble left missing predictions")
    for stage in sorted(output.stage.unique()):
        fitting = output.loc[output.stage.eq(stage)]
        weight, bias, mae = fit_weights(fitting)
        weight_rows.append({
            "kind": "frozen_all_validation", "fold": -1, "stage": stage,
            **{f"weight_{name}": float(value) for name, value in zip(CANDIDATES, weight)},
            "bias_cycles": bias, "training_mae": mae, "training_devices": int(fitting.unit_id.nunique()),
        })
    return output, pd.DataFrame(weight_rows)


def checkpoint_labels(group: pd.DataFrame) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for name, fraction in (("fraction_0.2", 0.2), ("fraction_0.4", 0.4),
                           ("fraction_0.6", 0.6), ("fraction_0.8", 0.8)):
        index = min(len(group) - 1, max(0, math.ceil(len(group) * fraction) - 1))
        labels.setdefault(index, []).append(name)
    eligible = np.flatnonzero(group.time.to_numpy(float) <= 30.0)
    fixed_index = int(eligible[-1]) if len(eligible) else 0
    labels.setdefault(fixed_index, []).append("fixed_30_efc")
    return labels


def summarize_rul(detail: pd.DataFrame) -> pd.DataFrame:
    checkpoint = np.full(len(detail), "", dtype=object)
    for _, indices in detail.groupby("unit_id", sort=False).groups.items():
        ordered_indices = list(indices)
        group = detail.loc[ordered_indices].reset_index(drop=True)
        mapping = checkpoint_labels(group)
        for local_index, names in mapping.items():
            checkpoint[ordered_indices[local_index]] = ";".join(names)
    detail["checkpoints"] = checkpoint
    rows = []
    for method in ("linear", "exponential", "knee", "ensemble_rul"):
        groups: list[tuple[str, pd.DataFrame]] = [("all_points", detail)]
        for name in ("fraction_0.2", "fraction_0.4", "fraction_0.6", "fraction_0.8", "fixed_30_efc"):
            groups.append((name, detail.loc[detail.checkpoints.str.contains(name, regex=False)]))
        for checkpoint_name, frame in groups:
            error = frame[method].to_numpy(float) - frame.true_rul_cycles.to_numpy(float)
            rows.append({
                "method": method, "checkpoint": checkpoint_name, "samples": len(frame),
                "devices": int(frame.unit_id.nunique()),
                "rul_mae_cycles": float(np.mean(np.abs(error))),
                "rul_rmse_cycles": float(np.sqrt(np.mean(error**2))),
                "rul_bias_cycles": float(np.mean(error)),
                "maximum_absolute_rul_error": float(np.max(np.abs(error))),
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    reference_manifest = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    config_values = dict(reference_manifest["configuration"])
    config_values["channels"] = tuple(config_values["channels"])
    config = Config(**config_values)
    seeds = [int(value) for value in args.seeds.split(",")]
    if seeds != [42, 43, 44, 45, 46]:
        raise ValueError("Formal fixed ensemble requires seeds 42,43,44,45,46")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    source, target, validation, features, contract_report = load_data(args.data_root, "invariant")
    units = fixed_units(args.reference_run)
    subset = target.loc[target.unit_id.astype(str).isin(units)].copy()
    if subset.unit_id.nunique() != 10:
        raise ValueError("Fixed target subset is incomplete")
    target_state = robust_fit(unlabeled_view(target, features), features, "Basilisk:all_unlabeled_train")
    target_unlabeled_x, _, _ = make_windows(unlabeled_view(target, features), features, target_state, config.window, False)
    target_x, target_y, _ = make_windows(subset, features, target_state, config.window, True)
    validation_x, validation_y, validation_meta = make_windows(validation, features, target_state, config.window, True)
    source_x, source_y, _ = calibrated_source_arrays(source, features, contract_report, config)
    assert target_y is not None and validation_y is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, prediction_rows = [], []
    prediction_path = args.output_dir / "soh_predictions_by_seed.csv"
    if args.reuse_predictions:
        if not prediction_path.exists():
            raise FileNotFoundError("--reuse-predictions requested but seed predictions are missing")
        long_predictions = pd.read_csv(prediction_path)
        for seed in seeds:
            group = long_predictions.loc[long_predictions.seed.eq(seed)]
            seed_rows.append({"seed": seed, "best_epoch": 45, **regression_metrics(
                group.target_soh.to_numpy(float), group.predicted_soh.to_numpy(float),
            )})
    else:
        for seed in seeds:
            seed_all(seed)
            ssl_model = ssl_fit(
                PositiveTransferTCN(len(features), config), [("target", target_unlabeled_x)], seed, config, device,
            )
            ssl_state = snapshot(ssl_model)
            teacher = PositiveTransferTCN(len(features), config)
            teacher.load_state_dict(copy.deepcopy(ssl_state))
            teacher = l2sp_source_fit(
                teacher, source_x, source_y, ssl_state, float(reference_manifest["l2sp_weight"]),
                seed + 1, config, device,
            )
            model, best_epoch = adapt_fit(
                teacher, target_x, target_y, validation_x, validation_y, seed, config, device,
            )
            estimate = predict(model, validation_x, config, device)
            seed_rows.append({"seed": seed, "best_epoch": best_epoch, **regression_metrics(validation_y, estimate)})
            prediction_rows.extend({
                "seed": seed, **meta, "predicted_soh": float(value),
            } for meta, value in zip(validation_meta.to_dict("records"), estimate))
            torch.save({
                "model_state": snapshot(model), "seed": seed, "fixed_target_units": units,
                "features": features, "configuration": asdict(config),
                "target_scaler_median": target_state.median, "target_scaler_iqr": target_state.iqr,
            }, args.output_dir / f"checkpoint_law_finetune_fixed10_seed_{seed}.pt")
            print(json.dumps(seed_rows[-1]), flush=True)
        long_predictions = pd.DataFrame(prediction_rows)
        long_predictions.to_csv(prediction_path, index=False)

    matrix, soh_weights = cross_fitted_soh_ensemble(long_predictions, seeds)
    ensemble_metrics = regression_metrics(matrix.target_soh.to_numpy(float), matrix.predicted_soh.to_numpy(float))
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.loc[len(seed_summary)] = {"seed": "cross_fitted_ensemble", "best_epoch": np.nan, **ensemble_metrics}
    seed_summary.to_csv(args.output_dir / "soh_metrics.csv", index=False)
    soh_weights.to_csv(args.output_dir / "soh_ensemble_weights.csv", index=False)
    matrix.to_csv(args.output_dir / "soh_ensemble_predictions.csv", index=False)

    slope_train = subset.copy()
    alpha = float(pd.read_csv(args.reference_run / "smoothing_selection.csv").iloc[0].alpha)
    candidates = candidate_predictions(matrix, slope_train, alpha)
    detail, weights = cross_fitted_curve_ensemble(candidates)
    metrics = summarize_rul(detail)
    detail.to_csv(args.output_dir / "rul_predictions.csv", index=False)
    weights.to_csv(args.output_dir / "rul_ensemble_weights.csv", index=False)
    metrics.to_csv(args.output_dir / "rul_metrics.csv", index=False)
    baseline_mae = float(metrics.loc[
        metrics.method.eq("linear") & metrics.checkpoint.eq("all_points"), "rul_mae_cycles"
    ].iloc[0])
    ensemble_mae = float(metrics.loc[
        metrics.method.eq("ensemble_rul") & metrics.checkpoint.eq("all_points"), "rul_mae_cycles"
    ].iloc[0])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": "Basilisk V1.0 unchanged", "feature_contract": "bhump_degradation_invariant",
        "feature_count": len(features), "features": features, "tcn_channels": list(config.channels),
        "fixed_target_units": units, "fixed_target_unit_count": len(units), "seeds": seeds,
        "soh_ensemble_metrics": ensemble_metrics,
        "soh_weighting": "5-fold device-cross-fitted convex seed weights with median bias calibration",
        "smoothing_alpha": alpha,
        "rul_candidates": list(CANDIDATES), "rul_weighting": "5-fold device-cross-fitted stage-wise convex MAE grid",
        "linear_baseline_rul_mae": baseline_mae, "curve_ensemble_rul_mae": ensemble_mae,
        "relative_rul_improvement": (baseline_mae - ensemble_mae) / baseline_mae,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "note": "Validation devices are cross-fitted for RUL-combiner weights; official independent test remains required.",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
