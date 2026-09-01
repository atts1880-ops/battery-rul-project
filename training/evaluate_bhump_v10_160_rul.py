"""Causal linear+knee RUL evaluation for the frozen V1.0 160-unit SOH ensemble."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from train_bhump_positive_transfer import causal_smooth, slope_bounds
from train_bhump_v10_fixed_ensemble import capped_rul, checkpoint_labels, history_stage
from train_bhump_degradation_transfer import load_data


ROOT = Path(__file__).resolve().parent
DEFAULT_SOH_RUN = ROOT / "bhump_v10_160_five_seed_runs"
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_OUTPUT = ROOT / "bhump_v10_160_rul_runs"
ALPHAS = (0.1, 0.2, 0.3, 0.5)
LINEAR_WINDOWS = (8, 12, 20)
KNEE_WINDOWS = (20, 32, 48)


def alpha_tag(alpha: float) -> str:
    return str(int(round(alpha * 10)))


def candidate_column(family: str, alpha: float, window: int) -> str:
    return f"{family}_a{alpha_tag(alpha)}_w{window}"


def linear_rul_window(times: np.ndarray, values: np.ndarray,
                      bounds: tuple[float, float, float, float], window: int) -> float:
    lower, upper, prior, maximum_lifetime = bounds
    count = min(window, len(values))
    if count < 3:
        slope = prior
    else:
        slope = float(theilslopes(values[-count:], times[-count:]).slope)
        if not np.isfinite(slope) or slope >= -1.0e-7:
            slope = prior
        slope = float(np.clip(slope, lower, upper))
    return capped_rul(float(times[-1]), float(values[-1]), slope, maximum_lifetime)


def knee_rul_window(times: np.ndarray, values: np.ndarray,
                    bounds: tuple[float, float, float, float], window: int) -> float:
    fallback = linear_rul_window(times, values, bounds, min(12, window))
    if len(values) < 10 or values[-1] <= 0.80:
        return 0.0 if values[-1] <= 0.80 else fallback
    count = min(window, len(values))
    local_t = times[-count:].astype(float)
    local_y = values[-count:].astype(float)
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
        score = float(np.mean(np.where(
            residual <= scale, 0.5 * residual**2 / scale, residual - 0.5 * scale,
        )))
        post_slope = float(coefficients[1] + coefficients[2])
        if post_slope < -1.0e-7 and (best is None or score < best[0]):
            best = (score, post_slope)
    if best is None:
        return fallback
    slope = float(np.clip(best[1], bounds[0], bounds[1]))
    return capped_rul(float(times[-1]), float(values[-1]), slope, bounds[3])


def build_candidates(predictions: pd.DataFrame, slope_train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    bounds = slope_bounds(slope_train)
    rows: list[dict[str, Any]] = []
    for unit_id, group in predictions.sort_values(["unit_id", "time"]).groupby("unit_id"):
        group = group.reset_index(drop=True)
        times = group.time.to_numpy(float)
        smoothed = {
            alpha: causal_smooth(group.predicted_soh.to_numpy(float), alpha) for alpha in ALPHAS
        }
        for index in range(len(group)):
            row: dict[str, Any] = {
                "unit_id": str(unit_id), "time": float(times[index]),
                "target_soh": float(group.target_soh.iloc[index]),
                "predicted_soh": float(group.predicted_soh.iloc[index]),
                "true_rul_cycles": float(group.true_rul_cycles.iloc[index]),
                "true_eol_cycle": float(group.true_eol_cycle.iloc[index]),
                "stage": history_stage(float(times[index])),
                "maximum_allowed_rul": max(float(bounds[3] - times[index]), 0.0),
            }
            current_times = times[:index + 1]
            for alpha, values in smoothed.items():
                current_values = values[:index + 1]
                row[f"smoothed_soh_a{alpha_tag(alpha)}"] = float(values[index])
                for window in LINEAR_WINDOWS:
                    row[candidate_column("linear", alpha, window)] = linear_rul_window(
                        current_times, current_values, bounds, window,
                    )
                for window in KNEE_WINDOWS:
                    row[candidate_column("knee", alpha, window)] = knee_rul_window(
                        current_times, current_values, bounds, window,
                    )
            rows.append(row)
    report = {
        "slope_lower": bounds[0], "slope_upper": bounds[1],
        "slope_prior": bounds[2], "maximum_training_lifetime": bounds[3],
    }
    return pd.DataFrame(rows), report


def clipped_mae(raw: np.ndarray, truth: np.ndarray, cap: np.ndarray) -> tuple[float, float]:
    bias = float(np.median(truth - raw))
    estimate = np.clip(raw + bias, 0.0, cap)
    return float(np.mean(np.abs(estimate - truth))), bias


def fit_single(frame: pd.DataFrame, family: str) -> dict[str, Any]:
    windows = LINEAR_WINDOWS if family == "linear" else KNEE_WINDOWS
    truth = frame.true_rul_cycles.to_numpy(float)
    cap = frame.maximum_allowed_rul.to_numpy(float)
    best: dict[str, Any] | None = None
    for alpha in ALPHAS:
        for window in windows:
            column = candidate_column(family, alpha, window)
            mae, bias = clipped_mae(frame[column].to_numpy(float), truth, cap)
            candidate = {"family": family, "alpha": alpha, "window": window,
                         "column": column, "bias": bias, "training_mae": mae}
            if best is None or mae < best["training_mae"] - 1.0e-12:
                best = candidate
    assert best is not None
    return best


def fit_ensemble(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame.true_rul_cycles.to_numpy(float)
    cap = frame.maximum_allowed_rul.to_numpy(float)
    best: dict[str, Any] | None = None
    weights = np.linspace(0.0, 1.0, 51)
    for alpha in ALPHAS:
        for linear_window in LINEAR_WINDOWS:
            linear_column = candidate_column("linear", alpha, linear_window)
            linear = frame[linear_column].to_numpy(float)
            for knee_window in KNEE_WINDOWS:
                knee_column = candidate_column("knee", alpha, knee_window)
                knee = frame[knee_column].to_numpy(float)
                for linear_weight in weights:
                    raw = linear_weight * linear + (1.0 - linear_weight) * knee
                    mae, bias = clipped_mae(raw, truth, cap)
                    candidate = {
                        "family": "ensemble", "alpha": alpha,
                        "linear_window": linear_window, "knee_window": knee_window,
                        "linear_column": linear_column, "knee_column": knee_column,
                        "linear_weight": float(linear_weight),
                        "knee_weight": float(1.0 - linear_weight),
                        "bias": bias, "training_mae": mae,
                    }
                    if best is None or mae < best["training_mae"] - 1.0e-12:
                        best = candidate
    assert best is not None
    return best


def apply_single(frame: pd.DataFrame, setting: dict[str, Any]) -> np.ndarray:
    raw = frame[setting["column"]].to_numpy(float) + float(setting["bias"])
    return np.clip(raw, 0.0, frame.maximum_allowed_rul.to_numpy(float))


def apply_ensemble(frame: pd.DataFrame, setting: dict[str, Any]) -> np.ndarray:
    raw = (
        float(setting["linear_weight"]) * frame[setting["linear_column"]].to_numpy(float)
        + float(setting["knee_weight"]) * frame[setting["knee_column"]].to_numpy(float)
        + float(setting["bias"])
    )
    return np.clip(raw, 0.0, frame.maximum_allowed_rul.to_numpy(float))


def cross_fitted_optimize(candidates: pd.DataFrame, folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = candidates.copy()
    units = sorted(output.unit_id.astype(str).unique())
    fold_by_unit = {unit: index % folds for index, unit in enumerate(units)}
    output["fold"] = output.unit_id.astype(str).map(fold_by_unit)
    for column in ("linear_rul", "knee_rul", "ensemble_rul"):
        output[column] = np.nan
    setting_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = output.loc[output.fold.ne(fold)]
        for stage in sorted(output.stage.unique()):
            fitting = train.loc[train.stage.eq(stage)]
            indices = output.index[output.fold.eq(fold) & output.stage.eq(stage)]
            training_units = sorted(fitting.unit_id.astype(str).unique())
            heldout_units = sorted(output.loc[indices, "unit_id"].astype(str).unique())
            linear_setting = fit_single(fitting, "linear")
            knee_setting = fit_single(fitting, "knee")
            ensemble_setting = fit_ensemble(fitting)
            output.loc[indices, "linear_rul"] = apply_single(output.loc[indices], linear_setting)
            output.loc[indices, "knee_rul"] = apply_single(output.loc[indices], knee_setting)
            output.loc[indices, "ensemble_rul"] = apply_ensemble(output.loc[indices], ensemble_setting)
            for setting in (linear_setting, knee_setting, ensemble_setting):
                setting_rows.append({
                    "kind": "cross_fit", "fold": fold, "stage": stage,
                    "training_devices": len(training_units),
                    "training_unit_ids": ";".join(training_units),
                    "heldout_unit_ids": ";".join(heldout_units), **setting,
                })
    if output[["linear_rul", "knee_rul", "ensemble_rul"]].isna().any().any():
        raise RuntimeError("Cross-fitted RUL optimization left missing predictions")
    for stage in sorted(output.stage.unique()):
        fitting = output.loc[output.stage.eq(stage)]
        for setting in (fit_single(fitting, "linear"), fit_single(fitting, "knee"), fit_ensemble(fitting)):
            setting_rows.append({
                "kind": "frozen_all_validation", "fold": -1, "stage": stage,
                "training_devices": int(fitting.unit_id.nunique()),
                "training_unit_ids": ";".join(sorted(fitting.unit_id.astype(str).unique())),
                "heldout_unit_ids": "", **setting,
            })
    return output, pd.DataFrame(setting_rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    checkpoint = np.full(len(detail), "", dtype=object)
    for _, indices in detail.groupby("unit_id", sort=False).groups.items():
        ordered = list(indices)
        group = detail.loc[ordered].reset_index(drop=True)
        for local_index, names in checkpoint_labels(group).items():
            checkpoint[ordered[local_index]] = ";".join(names)
    detail["checkpoints"] = checkpoint
    rows = []
    for method in ("linear_rul", "knee_rul", "ensemble_rul"):
        groups: list[tuple[str, pd.DataFrame]] = [("all_points", detail)]
        for name in ("fraction_0.2", "fraction_0.4", "fraction_0.6", "fraction_0.8", "fixed_30_efc"):
            groups.append((name, detail.loc[detail.checkpoints.str.contains(name, regex=False)]))
        for name, frame in groups:
            error = frame[method].to_numpy(float) - frame.true_rul_cycles.to_numpy(float)
            rows.append({
                "method": method, "checkpoint": name, "samples": len(frame),
                "devices": int(frame.unit_id.nunique()),
                "rul_mae_cycles": float(np.mean(np.abs(error))),
                "rul_rmse_cycles": float(np.sqrt(np.mean(error**2))),
                "rul_bias_cycles": float(np.mean(error)),
                "maximum_absolute_rul_error": float(np.max(np.abs(error))),
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    source_manifest = json.loads((args.soh_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("experiment_mode") != "fixed_160_five_seed":
        raise ValueError("SOH input is not the frozen 160-unit five-seed experiment")
    predictions = pd.read_csv(args.soh_run / "nasa_equal_ensemble_predictions.csv")
    if len(predictions) != 5632 or predictions.unit_id.nunique() != 80:
        raise ValueError("Unexpected SOH ensemble validation shape")
    _, target, validation, _, _ = load_data(args.data_root, "invariant")
    selected = pd.read_csv(args.soh_run / "selected_target_units.csv")
    slope_units = selected.loc[selected.seed.eq(args.slope_reference_seed), "unit_id"].astype(str).unique()
    slope_train = target.loc[target.unit_id.astype(str).isin(slope_units)].copy()
    if len(slope_units) != 160 or slope_train.unit_id.nunique() != 160:
        raise ValueError("Slope prior must use exactly 160 labeled training devices")
    if set(map(str, slope_units)) & set(validation.unit_id.astype(str).unique()):
        raise ValueError("Slope-prior training and validation devices overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, slope_report = build_candidates(predictions, slope_train)
    detail, settings = cross_fitted_optimize(candidates)
    metrics = summarize(detail)
    detail.to_csv(args.output_dir / "rul_predictions.csv", index=False)
    settings.to_csv(args.output_dir / "rul_crossfit_settings.csv", index=False)
    metrics.to_csv(args.output_dir / "rul_metrics.csv", index=False)
    candidates.to_csv(args.output_dir / "rul_candidate_predictions.csv", index=False)
    primary = metrics.loc[
        metrics.method.eq("ensemble_rul") & metrics.checkpoint.eq("all_points")
    ].iloc[0]
    linear = metrics.loc[
        metrics.method.eq("linear_rul") & metrics.checkpoint.eq("all_points")
    ].iloc[0]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": "Basilisk V1.0 unchanged",
        "soh_source": str(args.soh_run),
        "soh_ensemble_mae": source_manifest["nasa_equal_ensemble_metrics"]["soh_mae"],
        "target_labeled_unit_budget": 160,
        "slope_reference_seed": args.slope_reference_seed,
        "slope_reference_units": list(map(str, slope_units)),
        "slope_report": slope_report,
        "candidate_families": ["linear", "knee"],
        "alphas": list(ALPHAS), "linear_windows": list(LINEAR_WINDOWS),
        "knee_windows": list(KNEE_WINDOWS), "weight_step": 0.02,
        "optimization": "5-fold device-cross-fitted stage-wise alpha/window/weight/median-bias MAE",
        "validation_samples": len(detail), "validation_devices": int(detail.unit_id.nunique()),
        "rul_mae_cycles": float(primary.rul_mae_cycles),
        "rul_rmse_cycles": float(primary.rul_rmse_cycles),
        "rul_bias_cycles": float(primary.rul_bias_cycles),
        "linear_rul_mae_cycles": float(linear.rul_mae_cycles),
        "relative_improvement_vs_linear": float(
            (linear.rul_mae_cycles - primary.rul_mae_cycles) / linear.rul_mae_cycles
        ),
        "previous_fixed10_rul_mae_cycles": 11.418359452116057,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "note": "Reported validation metrics are device-cross-fitted; frozen_all_validation settings are for later sealed inference only.",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soh-run", type=Path, default=DEFAULT_SOH_RUN)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slope-reference-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
