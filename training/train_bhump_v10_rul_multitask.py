"""Dual-track V1.0 multi-task TCN for causal SOH and physical RUL prediction.

The feature contract is frozen at the 16 causal V1.0 features.  ``fixed160``
uses one device sample selected by seed 42 for every model seed; ``full320``
is the labeled-data performance ceiling.  Sealed files are deliberately not
referenced by this module.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import theilslopes
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from bhump_common import FORBIDDEN_INPUT_TOKENS
from evaluate_bhump_v10_160_rul import build_candidates, cross_fitted_optimize
from train_bhump_degradation_transfer import calibrated_source_arrays, l2sp_source_fit, load_data
from train_bhump_positive_transfer import (
    Config, PositiveTransferTCN, choose_nested_units, make_windows, robust_fit,
    seed_all, snapshot, ssl_fit, unlabeled_view,
)
from train_bhump_v10_fixed_ensemble import checkpoint_labels, history_stage


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_rul_multitask_runs"
METHODS = ("target_ssl", "nasa_pretrain_finetune")
BUDGETS = ("fixed160", "full320")
FORMAL_SEEDS = (42, 43, 44, 45, 46)
WINDOW_SPECS: dict[int, tuple[int, ...]] = {
    8: (16, 24),
    16: (16, 24, 24),
    24: (16, 24, 24),
    32: (16, 24, 24, 24),
}
EOL_SOH = 0.80
KNEE_HORIZON = 24.0


class DynamicsTCN(PositiveTransferTCN):
    """Compact TCN with training-time dynamics heads and a physical RUL map."""

    def __init__(self, input_size: int, config: Config) -> None:
        super().__init__(input_size, config)
        projection = config.projection
        self.pre_rate_head = nn.Linear(projection, 1)
        self.post_delta_head = nn.Linear(projection, 1)
        self.knee_time_head = nn.Linear(projection, 1)
        self.knee_probability_head = nn.Linear(projection, 1)
        self.reset_dynamics_heads()

    def reset_dynamics_heads(self) -> None:
        self.reset_soh_head()
        for head in (self.pre_rate_head, self.post_delta_head,
                     self.knee_time_head, self.knee_probability_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
        # softplus(-6) ~= 0.00248 SOH/EFC, a neutral V1.0 degradation prior.
        nn.init.constant_(self.pre_rate_head.bias, -6.0)
        nn.init.constant_(self.post_delta_head.bias, -7.0)
        nn.init.constant_(self.knee_time_head.bias, 0.0)
        nn.init.constant_(self.knee_probability_head.bias, -1.0)

    def dynamics(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        soh, shared, _ = self(values, "target")
        pre_rate = F.softplus(self.pre_rate_head(shared).squeeze(-1)) + 1.0e-6
        post_rate = pre_rate + F.softplus(self.post_delta_head(shared).squeeze(-1))
        knee_time = KNEE_HORIZON * torch.sigmoid(self.knee_time_head(shared).squeeze(-1))
        knee_probability = torch.sigmoid(self.knee_probability_head(shared).squeeze(-1))
        return {
            "soh": soh, "pre_rate": pre_rate, "post_rate": post_rate,
            "knee_time": knee_time, "knee_probability": knee_probability,
        }

    @staticmethod
    def physical_rul_from_outputs(outputs: dict[str, torch.Tensor],
                                  maximum_allowed: torch.Tensor) -> torch.Tensor:
        soh_margin = torch.clamp(outputs["soh"] - EOL_SOH, min=0.0)
        pre = torch.clamp(outputs["pre_rate"], min=1.0e-6)
        post = torch.clamp(outputs["post_rate"], min=pre)
        tau = torch.clamp(outputs["knee_time"], 0.0, KNEE_HORIZON)
        linear = soh_margin / pre
        knee_before_eol = pre * tau < soh_margin
        knee = tau + torch.clamp(soh_margin - pre * tau, min=0.0) / post
        knee = torch.where(knee_before_eol, knee, linear)
        mixed = (1.0 - outputs["knee_probability"]) * linear + outputs["knee_probability"] * knee
        return torch.minimum(torch.clamp(mixed, min=0.0), torch.clamp(maximum_allowed, min=0.0))

    def deployment_parameters(self) -> Iterable[nn.Parameter]:
        return (*super().deployment_parameters(), *self.pre_rate_head.parameters(),
                *self.post_delta_head.parameters(), *self.knee_time_head.parameters(),
                *self.knee_probability_head.parameters())


def huber_mean(residual: np.ndarray, scale: float) -> float:
    absolute = np.abs(residual)
    loss = np.where(absolute <= scale, 0.5 * absolute**2 / scale, absolute - 0.5 * scale)
    return float(np.mean(loss))


def fit_unit_dynamics(unit: pd.DataFrame) -> dict[str, float | bool]:
    """Fit an offline continuous two-line degradation label for one train unit."""
    ordered = unit.sort_values("time")
    times = ordered.time.to_numpy(float)
    soh = ordered.target_soh.to_numpy(float)
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("Dynamics labels require at least three strictly ordered observations")
    global_slope = float(theilslopes(soh, times).slope)
    global_rate = max(-global_slope, 1.0e-5)
    best: dict[str, float] | None = None
    if len(times) >= 20:
        scale = max(float(np.median(np.abs(soh - np.median(soh)))), 1.0e-4)
        for index in range(10, len(times) - 9):
            knee = float(times[index])
            design = np.column_stack([np.ones(len(times)), times, np.maximum(times - knee, 0.0)])
            coefficients = np.linalg.lstsq(design, soh, rcond=None)[0]
            pre_slope = float(coefficients[1])
            post_slope = float(coefficients[1] + coefficients[2])
            score = huber_mean(soh - design @ coefficients, scale)
            candidate = {"score": score, "knee_cycle": knee,
                         "pre_slope": pre_slope, "post_slope": post_slope}
            if best is None or score < best["score"]:
                best = candidate
    valid = bool(
        best is not None and best["pre_slope"] < 0.0 and best["post_slope"] < 0.0
        and abs(best["post_slope"]) >= 1.25 * abs(best["pre_slope"])
    )
    if not valid:
        return {"has_knee": False, "knee_cycle": float(times[-1] + KNEE_HORIZON),
                "pre_rate": global_rate, "post_rate": global_rate,
                "fit_score": float("nan") if best is None else best["score"]}
    assert best is not None
    return {"has_knee": True, "knee_cycle": best["knee_cycle"],
            "pre_rate": max(-best["pre_slope"], 1.0e-5),
            "post_rate": max(-best["post_slope"], 1.0e-5),
            "fit_score": best["score"]}


def attach_dynamics_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    reports: list[dict[str, Any]] = []
    for unit_id, unit in output.groupby("unit_id", sort=True):
        fit = fit_unit_dynamics(unit)
        mask = output.unit_id.eq(unit_id)
        output.loc[mask, "label_pre_rate"] = float(fit["pre_rate"])
        output.loc[mask, "label_post_rate"] = float(fit["post_rate"])
        output.loc[mask, "label_knee_probability"] = float(bool(fit["has_knee"]))
        output.loc[mask, "label_knee_time"] = np.clip(
            float(fit["knee_cycle"]) - output.loc[mask, "time"].to_numpy(float),
            0.0, KNEE_HORIZON,
        )
        reports.append({"unit_id": str(unit_id), **fit})
    required = ["label_pre_rate", "label_post_rate", "label_knee_time", "label_knee_probability"]
    if not np.isfinite(output[required].to_numpy(float)).all():
        raise ValueError("Non-finite dynamics supervision label")
    return output, pd.DataFrame(reports)


def make_multitask_windows(frame: pd.DataFrame, features: list[str], state: Any,
                           config: Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Every candidate is evaluated on the exact historical support of the
    # original 8-cycle baseline (5632 validation samples). Longer windows are
    # causally left-padded with the unit's first observation; no future sample
    # is used and no early validation point is silently discarded.
    xs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    metadata = ["unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle",
                "label_pre_rate", "label_post_rate", "label_knee_time",
                "label_knee_probability"]
    has_sample_weights = "sample_weight" in frame.columns
    if has_sample_weights:
        metadata.append("sample_weight")
    for _, unit in frame.sort_values(["unit_id", "time"]).groupby("unit_id", sort=True):
        unit = unit.reset_index(drop=True)
        values = state.transform(unit[features].to_numpy(float))
        for end in range(7, len(unit)):
            start = end - config.window + 1
            if start < 0:
                window_values = np.concatenate([
                    np.repeat(values[[0]], -start, axis=0), values[:end + 1],
                ], axis=0)
            else:
                window_values = values[start:end + 1]
            if len(window_values) != config.window:
                raise RuntimeError("Causal padded window has an unexpected length")
            xs.append(window_values)
            rows.append(unit.loc[end, metadata].to_dict())
    if not xs:
        raise ValueError("No multi-task windows could be built")
    merged = pd.DataFrame(rows)
    soh = merged.target_soh.to_numpy(np.float32)
    targets = [
        soh,
        merged.label_pre_rate.to_numpy(np.float32),
        merged.label_post_rate.to_numpy(np.float32),
        merged.label_knee_time.to_numpy(np.float32),
        merged.label_knee_probability.to_numpy(np.float32),
        merged.true_rul_cycles.to_numpy(np.float32),
        merged.time.to_numpy(np.float32),
    ]
    if has_sample_weights:
        targets.append(merged.sample_weight.to_numpy(np.float32))
    y = np.column_stack(targets).astype(np.float32)
    return np.asarray(xs, np.float32), y, merged


def dynamics_loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    sampler = None
    shuffle = True
    if y.shape[1] > 7:
        weights = np.maximum(y[:, 7].astype(np.float64), 1.0e-12)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)), shuffle=shuffle, sampler=sampler,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def set_dynamics_trainable(model: DynamicsTCN, epoch: int, config: Config) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.soh_head, model.pre_rate_head, model.post_delta_head,
                   model.knee_time_head, model.knee_probability_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if epoch > config.head_only_epochs:
        for parameter in model.shared_projection.parameters():
            parameter.requires_grad = True
        for parameter in model.encoder.network[-1].parameters():
            parameter.requires_grad = True
    if epoch > config.head_only_epochs + config.last_block_epochs:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True


def multitask_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor,
                   maximum_allowed: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels[:, :7]
    soh, pre, post, tau, knee_probability, true_rul, _current_time = labels.unbind(1)
    # Rates are expressed as percentage points/EFC only inside their loss so
    # their numerical scale is comparable with SOH. Predictions remain SOH/EFC.
    parts = {
        "soh": F.smooth_l1_loss(outputs["soh"], soh, beta=0.01),
        "pre_rate": F.smooth_l1_loss(100.0 * outputs["pre_rate"], 100.0 * pre, beta=0.01),
        "post_rate": F.smooth_l1_loss(100.0 * outputs["post_rate"], 100.0 * post, beta=0.01),
        "knee_time": F.smooth_l1_loss(outputs["knee_time"] / KNEE_HORIZON,
                                      tau / KNEE_HORIZON, beta=0.01),
        "knee_probability": F.binary_cross_entropy(outputs["knee_probability"], knee_probability),
    }
    predicted_rul = DynamicsTCN.physical_rul_from_outputs(outputs, maximum_allowed)
    parts["log_rul"] = F.smooth_l1_loss(torch.log1p(predicted_rul), torch.log1p(true_rul), beta=0.01)
    total = (parts["soh"] + 0.2 * (parts["pre_rate"] + parts["post_rate"])
             + 0.1 * parts["knee_time"] + 0.1 * parts["knee_probability"]
             + 0.2 * parts["log_rul"])
    return total, parts


def predict_dynamics(model: DynamicsTCN, x: np.ndarray, meta: pd.DataFrame,
                     maximum_lifetime: float, config: Config,
                     device: torch.device) -> pd.DataFrame:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in
        ("predicted_soh", "predicted_pre_rate", "predicted_post_rate",
         "predicted_knee_time", "predicted_knee_probability", "physical_rul")}
    with torch.no_grad():
        for start in range(0, len(x), config.batch_size):
            batch = torch.from_numpy(x[start:start + config.batch_size]).to(device)
            outputs = model.dynamics(batch)
            times = torch.from_numpy(meta.time.iloc[start:start + len(batch)].to_numpy(np.float32)).to(device)
            cap = torch.clamp(torch.tensor(maximum_lifetime, device=device) - times, min=0.0)
            physical = model.physical_rul_from_outputs(outputs, cap)
            mapping = {
                "predicted_soh": outputs["soh"], "predicted_pre_rate": outputs["pre_rate"],
                "predicted_post_rate": outputs["post_rate"],
                "predicted_knee_time": outputs["knee_time"],
                "predicted_knee_probability": outputs["knee_probability"],
                "physical_rul": physical,
            }
            for name, value in mapping.items():
                chunks[name].append(value.detach().cpu().numpy())
    result = meta.copy()
    for name, values in chunks.items():
        result[name] = np.concatenate(values)
    return result


def fit_multitask(model: DynamicsTCN, train_x: np.ndarray, train_y: np.ndarray,
                  validation_x: np.ndarray, validation_y: np.ndarray,
                  validation_meta: pd.DataFrame, maximum_lifetime: float,
                  seed: int, config: Config, device: torch.device) -> tuple[DynamicsTCN, int]:
    seed_all(seed)
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
    best_state, best_mae, best_epoch, stale = snapshot(model), float("inf"), 0, 0
    for epoch in range(1, config.adapt_epochs + 1):
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
        validation_prediction = predict_dynamics(
            model, validation_x, validation_meta, maximum_lifetime, config, device,
        )
        mae = float(np.mean(np.abs(
            validation_prediction.physical_rul.to_numpy(float) - validation_y[:, 5]
        )))
        if mae < best_mae - 1.0e-7:
            best_state, best_mae, best_epoch, stale = snapshot(model), mae, epoch, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def fold_map(frame: pd.DataFrame, folds: int = 5) -> dict[str, int]:
    units = sorted(frame.unit_id.astype(str).unique())
    return {unit: index % folds for index, unit in enumerate(units)}


def crossfit_calibrate(frame: pd.DataFrame, source_column: str,
                       output_column: str) -> pd.DataFrame:
    output = frame.copy()
    output["fold"] = output.unit_id.astype(str).map(fold_map(output))
    output[output_column] = np.nan
    for fold in range(5):
        for stage in sorted(output.stage.unique()):
            train = output.loc[output.fold.ne(fold) & output.stage.eq(stage)]
            test = output.loc[output.fold.eq(fold) & output.stage.eq(stage)]
            bias = float(np.median(
                train.true_rul_cycles.to_numpy(float) - train[source_column].to_numpy(float)
            ))
            output.loc[test.index, output_column] = np.clip(
                test[source_column].to_numpy(float) + bias, 0.0,
                test.maximum_allowed_rul.to_numpy(float),
            )
    if output[output_column].isna().any():
        raise RuntimeError(f"Cross-fit calibration left missing {output_column}")
    return output


def crossfit_blend(frame: pd.DataFrame, physical_column: str,
                   curve_column: str = "ensemble_rul") -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    if "fold" not in output:
        output["fold"] = output.unit_id.astype(str).map(fold_map(output))
    output["combined_rul"] = np.nan
    settings: list[dict[str, Any]] = []
    for fold in range(5):
        for stage in sorted(output.stage.unique()):
            train = output.loc[output.fold.ne(fold) & output.stage.eq(stage)]
            test = output.loc[output.fold.eq(fold) & output.stage.eq(stage)]
            best: tuple[float, float, float] | None = None
            truth = train.true_rul_cycles.to_numpy(float)
            cap = train.maximum_allowed_rul.to_numpy(float)
            physical = train[physical_column].to_numpy(float)
            curve = train[curve_column].to_numpy(float)
            for physical_weight in np.linspace(0.0, 1.0, 51):
                raw = physical_weight * physical + (1.0 - physical_weight) * curve
                bias = float(np.median(truth - raw))
                mae = float(np.mean(np.abs(np.clip(raw + bias, 0.0, cap) - truth)))
                candidate = (mae, float(physical_weight), bias)
                if best is None or candidate[0] < best[0] - 1.0e-12:
                    best = candidate
            assert best is not None
            raw_test = (best[1] * test[physical_column].to_numpy(float)
                        + (1.0 - best[1]) * test[curve_column].to_numpy(float) + best[2])
            output.loc[test.index, "combined_rul"] = np.clip(
                raw_test, 0.0, test.maximum_allowed_rul.to_numpy(float),
            )
            settings.append({"fold": fold, "stage": stage, "training_mae": best[0],
                             "physical_weight": best[1], "curve_weight": 1.0 - best[1],
                             "bias": best[2], "training_devices": int(train.unit_id.nunique()),
                             "heldout_devices": int(test.unit_id.nunique())})
    if output.combined_rul.isna().any():
        raise RuntimeError("Cross-fitted blend left missing predictions")
    return output, pd.DataFrame(settings)


def add_stages_and_caps(frame: pd.DataFrame, maximum_lifetime: float) -> pd.DataFrame:
    output = frame.copy()
    output["stage"] = output.time.map(lambda value: history_stage(float(value)))
    output["maximum_allowed_rul"] = np.maximum(maximum_lifetime - output.time.to_numpy(float), 0.0)
    return output


def regression_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 1.0e-12 else 0.0,
        "bias": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
    }


def worst_device_mae(frame: pd.DataFrame, prediction_column: str,
                     truth_column: str) -> float:
    values = frame.assign(_error=np.abs(
        frame[prediction_column].to_numpy(float) - frame[truth_column].to_numpy(float)
    )).groupby("unit_id")._error.mean()
    return float(values.max())


def checkpoint_mask(frame: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(frame), "", dtype=object)
    for _, indices in frame.groupby("unit_id", sort=False).groups.items():
        ordered = list(indices)
        group = frame.loc[ordered].reset_index(drop=True)
        for local_index, names in checkpoint_labels(group).items():
            labels[ordered[local_index]] = ";".join(names)
    return labels


def summarize_rul(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    output["checkpoints"] = checkpoint_mask(output)
    rows: list[dict[str, Any]] = []
    for column in columns:
        groups: list[tuple[str, pd.DataFrame]] = [("all_points", output)]
        for name in ("fraction_0.2", "fraction_0.4", "fraction_0.6", "fraction_0.8", "fixed_30_efc"):
            groups.append((name, output.loc[output.checkpoints.str.contains(name, regex=False)]))
        for checkpoint, group in groups:
            metrics = regression_row(group.true_rul_cycles.to_numpy(float), group[column].to_numpy(float))
            rows.append({"method": column, "checkpoint": checkpoint, "samples": len(group),
                         "devices": int(group.unit_id.nunique()), **{f"rul_{key}": value for key, value in metrics.items()},
                         "worst_device_mae": worst_device_mae(group, column, "true_rul_cycles")})
    return pd.DataFrame(rows)


def build_config(reference: dict[str, Any], window: int, smoke: bool = False) -> Config:
    values = dict(reference["configuration"])
    values["channels"] = WINDOW_SPECS[window]
    values["window"] = window
    if smoke:
        values.update({"ssl_epochs": 1, "source_epochs": 2, "adapt_epochs": 3,
                       "patience": 2, "batch_size": 128})
    return Config(**values)


def maximum_training_lifetime(frame: pd.DataFrame) -> float:
    if "true_eol_cycle" in frame:
        return float(frame.true_eol_cycle.max())
    return float(np.max(frame.time.to_numpy(float) + frame.true_rul_cycles.to_numpy(float)))


def parameter_counts(model: DynamicsTCN) -> tuple[int, int]:
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    deployment = sum(parameter.numel() for parameter in model.deployment_parameters())
    return all_parameters, deployment


def prepare_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    source, target, validation, features, report = load_data(data_root, "invariant")
    if len(features) != 16:
        raise ValueError("V1.0 multi-task experiment requires exactly 16 frozen features")
    leaked = [feature for feature in features
              if any(token in feature.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if leaked:
        raise ValueError(f"Forbidden model features: {leaked}")
    if target.unit_id.nunique() != 320 or validation.unit_id.nunique() != 80:
        raise ValueError("Unexpected V1.0 device split")
    if set(target.unit_id.astype(str)) & set(validation.unit_id.astype(str)):
        raise ValueError("Target training and validation devices overlap")
    target, target_knees = attach_dynamics_labels(target)
    validation, validation_knees = attach_dynamics_labels(validation)
    return source, target, validation, features, report, target_knees, validation_knees


def initialize_states(source: pd.DataFrame, target: pd.DataFrame, features: list[str],
                      report: dict[str, Any], config: Config, seed: int,
                      device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor],
                                                     Any, np.ndarray, np.ndarray]:
    target_unlabeled = unlabeled_view(target, features)
    target_state = robust_fit(target_unlabeled, features, "Basilisk:all_320_unlabeled_train")
    target_unlabeled_x, _, _ = make_windows(target_unlabeled, features, target_state, config.window, False)
    ssl_model = ssl_fit(DynamicsTCN(len(features), config), [("target", target_unlabeled_x)],
                        seed, config, device)
    ssl_state = snapshot(ssl_model)
    source_x, source_y, _ = calibrated_source_arrays(source, features, report, config)
    source_model = DynamicsTCN(len(features), config)
    source_model.load_state_dict(copy.deepcopy(ssl_state))
    source_model = l2sp_source_fit(source_model, source_x, source_y, ssl_state, 0.0,
                                  seed + 1, config, device)
    return ssl_state, snapshot(source_model), target_state, source_x, source_y


def train_configuration(method: str, selected: pd.DataFrame, validation: pd.DataFrame,
                        features: list[str], target_state: Any,
                        initial_state: dict[str, torch.Tensor], maximum_lifetime: float,
                        config: Config, seed: int, device: torch.device) -> tuple[DynamicsTCN, int, pd.DataFrame, dict[str, float]]:
    train_x, train_y, _ = make_multitask_windows(selected, features, target_state, config)
    validation_x, validation_y, validation_meta = make_multitask_windows(
        validation, features, target_state, config,
    )
    model = DynamicsTCN(len(features), config)
    model.load_state_dict(copy.deepcopy(initial_state))
    model, best_epoch = fit_multitask(
        model, train_x, train_y, validation_x, validation_y, validation_meta,
        maximum_lifetime, seed, config, device,
    )
    prediction = predict_dynamics(
        model, validation_x, validation_meta, maximum_lifetime, config, device,
    )
    soh = regression_row(prediction.target_soh.to_numpy(float), prediction.predicted_soh.to_numpy(float))
    rul = regression_row(prediction.true_rul_cycles.to_numpy(float), prediction.physical_rul.to_numpy(float))
    metrics = {**{f"soh_{key}": value for key, value in soh.items()},
               **{f"rul_{key}": value for key, value in rul.items()},
               "soh_worst_device_mae": worst_device_mae(prediction, "predicted_soh", "target_soh"),
               "rul_worst_device_mae": worst_device_mae(prediction, "physical_rul", "true_rul_cycles")}
    return model, best_epoch, prediction, metrics


def window_tuning(args: argparse.Namespace, reference: dict[str, Any], source: pd.DataFrame,
                  target: pd.DataFrame, validation: pd.DataFrame, features: list[str],
                  report: dict[str, Any], device: torch.device) -> tuple[int, pd.DataFrame]:
    selected_units = choose_nested_units(target, [160], args.selection_seed)[160]
    selected = target.loc[target.unit_id.astype(str).isin(selected_units)].copy()
    rows: list[dict[str, Any]] = []
    windows = tuple(int(value) for value in args.windows.split(","))
    for window in windows:
        if window not in WINDOW_SPECS:
            raise ValueError(f"Unknown frozen window candidate: {window}")
        config = build_config(reference, window, args.smoke)
        ssl_state, source_state, target_state, _, _ = initialize_states(
            source, target, features, report, config, args.tuning_seed, device,
        )
        maximum_lifetime = maximum_training_lifetime(selected)
        model, epoch, prediction, metrics = train_configuration(
            "nasa_pretrain_finetune", selected, validation, features, target_state,
            source_state, maximum_lifetime, config, args.tuning_seed, device,
        )
        calibrated = crossfit_calibrate(
            add_stages_and_caps(prediction, maximum_lifetime), "physical_rul", "crossfit_physical_rul",
        )
        crossfit_mae = float(np.mean(np.abs(
            calibrated.crossfit_physical_rul - calibrated.true_rul_cycles
        )))
        training_parameters, deployment_parameters = parameter_counts(model)
        row = {"window": window, "channels": "/".join(map(str, config.channels)),
               "dilations": "/".join(str(2**index) for index in range(len(config.channels))),
               "best_epoch": epoch, "crossfit_rul_mae": crossfit_mae,
               "training_parameters": training_parameters,
               "deployment_parameters": deployment_parameters, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    tuning = pd.DataFrame(rows).sort_values(
        ["crossfit_rul_mae", "deployment_parameters", "window"], ignore_index=True,
    )
    minimum = float(tuning.crossfit_rul_mae.min())
    winner = tuning.loc[tuning.crossfit_rul_mae.le(minimum + 0.1)].sort_values(
        ["deployment_parameters", "window", "crossfit_rul_mae"]
    ).iloc[0]
    return int(winner.window), tuning


def numpy_physical_rul(frame: pd.DataFrame, maximum_lifetime: float) -> np.ndarray:
    margin = np.maximum(frame.predicted_soh.to_numpy(float) - EOL_SOH, 0.0)
    pre = np.maximum(frame.predicted_pre_rate.to_numpy(float), 1.0e-6)
    post = np.maximum(frame.predicted_post_rate.to_numpy(float), pre)
    tau = np.clip(frame.predicted_knee_time.to_numpy(float), 0.0, KNEE_HORIZON)
    probability = np.clip(frame.predicted_knee_probability.to_numpy(float), 0.0, 1.0)
    linear = margin / pre
    knee = tau + np.maximum(margin - pre * tau, 0.0) / post
    knee = np.where(pre * tau < margin, knee, linear)
    mixed = (1.0 - probability) * linear + probability * knee
    return np.clip(mixed, 0.0, np.maximum(maximum_lifetime - frame.time.to_numpy(float), 0.0))


def build_equal_ensemble(predictions: pd.DataFrame, maximum_lifetime: float) -> pd.DataFrame:
    keys = ["unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle"]
    base = predictions.groupby(keys, as_index=False).agg(
        predicted_soh=("predicted_soh", "mean"),
        predicted_knee_time=("predicted_knee_time", "mean"),
        predicted_knee_probability=("predicted_knee_probability", "mean"),
        soh_ensemble_std=("predicted_soh", lambda values: float(np.std(values, ddof=0))),
        seed_count=("seed", "nunique"),
    )
    rate = predictions.assign(
        _log_pre=np.log(np.maximum(predictions.predicted_pre_rate.to_numpy(float), 1.0e-8)),
        _log_post=np.log(np.maximum(predictions.predicted_post_rate.to_numpy(float), 1.0e-8)),
    ).groupby(keys, as_index=False).agg(_log_pre=("_log_pre", "mean"), _log_post=("_log_post", "mean"))
    base = base.merge(rate, on=keys, how="left", validate="one_to_one")
    base["predicted_pre_rate"] = np.exp(base.pop("_log_pre"))
    base["predicted_post_rate"] = np.maximum(np.exp(base.pop("_log_post")), base.predicted_pre_rate)
    base["physical_rul"] = numpy_physical_rul(base, maximum_lifetime)
    return base


def postprocess_formal(predictions: pd.DataFrame, target: pd.DataFrame,
                       selected_by_budget: dict[str, list[str]], output_dir: Path,
                       seed_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_details: list[pd.DataFrame] = []
    all_settings: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for budget in BUDGETS:
        train = target.loc[target.unit_id.astype(str).isin(selected_by_budget[budget])].copy()
        maximum_lifetime = maximum_training_lifetime(train)
        for method in METHODS:
            subset = predictions.loc[
                predictions.budget.eq(budget) & predictions.method.eq(method)
            ].copy()
            ensemble = build_equal_ensemble(subset, maximum_lifetime)
            if not ensemble.seed_count.eq(seed_count).all():
                raise RuntimeError(f"Incomplete equal ensemble: {budget}/{method}")
            curve_candidates, _ = build_candidates(ensemble, train)
            curve_detail, curve_settings = cross_fitted_optimize(curve_candidates)
            detail = curve_detail.merge(
                ensemble[["unit_id", "time", "physical_rul", "predicted_pre_rate",
                          "predicted_post_rate", "predicted_knee_time",
                          "predicted_knee_probability", "soh_ensemble_std", "seed_count"]],
                on=["unit_id", "time"], how="left", validate="one_to_one",
            )
            detail = crossfit_calibrate(detail, "physical_rul", "physical_rul_calibrated")
            detail, blend_settings = crossfit_blend(detail, "physical_rul_calibrated")
            metrics = summarize_rul(
                detail, ("ensemble_rul", "physical_rul", "physical_rul_calibrated", "combined_rul"),
            )
            for frame in (detail, curve_settings, blend_settings, metrics):
                frame.insert(0, "method_group", method)
                frame.insert(0, "budget", budget)
            all_details.append(detail)
            all_settings.extend([curve_settings.assign(setting_type="curve"),
                                 blend_settings.assign(setting_type="physical_curve_blend")])
            all_metrics.append(metrics)
            ensemble.to_csv(output_dir / f"ensemble_outputs_{budget}_{method}.csv", index=False)
            detail.to_csv(output_dir / f"rul_predictions_{budget}_{method}.csv", index=False)
            print(json.dumps({"budget": budget, "method": method,
                              "best_all_point_rul_mae": float(metrics.loc[
                                  metrics.checkpoint.eq("all_points"), "rul_mae"
                              ].min())}), flush=True)
    detail_table = pd.concat(all_details, ignore_index=True)
    setting_table = pd.concat(all_settings, ignore_index=True)
    metric_table = pd.concat(all_metrics, ignore_index=True)
    return detail_table, setting_table, metric_table


def transfer_acceptance(seed_results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    corrected_baseline_soh = 0.011625391617417335
    corrected_baseline_rul = 7.340312695552595
    for budget in BUDGETS:
        target_seed = seed_results.loc[
            seed_results.budget.eq(budget) & seed_results.method.eq("target_ssl")
        ].set_index("seed")
        nasa_seed = seed_results.loc[
            seed_results.budget.eq(budget) & seed_results.method.eq("nasa_pretrain_finetune")
        ].set_index("seed")
        seed_gain = (target_seed.rul_mae - nasa_seed.rul_mae) / target_seed.rul_mae
        all_points = metrics.loc[
            metrics.budget.eq(budget) & metrics.checkpoint.eq("all_points")
        ]
        chosen: dict[str, pd.Series] = {}
        for method in METHODS:
            chosen[method] = all_points.loc[all_points.method_group.eq(method)].sort_values(
                ["rul_mae", "worst_device_mae"]
            ).iloc[0]
        nasa = chosen["nasa_pretrain_finetune"]
        control = chosen["target_ssl"]
        nasa_seed_mean = float(nasa_seed.rul_mae.mean())
        target_seed_mean = float(target_seed.rul_mae.mean())
        mean_rul_positive = nasa_seed_mean < target_seed_mean
        positive_seed_count = int((seed_gain > 0.0).sum())
        nasa_worst_mean = float(nasa_seed.rul_worst_device_mae.mean())
        target_worst_mean = float(target_seed.rul_worst_device_mae.mean())
        worst_change = (nasa_worst_mean - target_worst_mean) / target_worst_mean
        nasa_soh_mean = float(nasa_seed.soh_mae.mean())
        soh_change = (nasa_soh_mean - corrected_baseline_soh) / corrected_baseline_soh
        accepted = bool(mean_rul_positive and positive_seed_count >= 4
                        and worst_change <= 0.10 and soh_change <= 0.05)
        rows.append({
            "budget": budget,
            "nasa_selected_rul_method": str(nasa.method),
            "target_selected_rul_method": str(control.method),
            "nasa_rul_mae": float(nasa.rul_mae), "target_rul_mae": float(control.rul_mae),
            "nasa_five_seed_mean_rul_mae": nasa_seed_mean,
            "target_five_seed_mean_rul_mae": target_seed_mean,
            "mean_rul_gain_vs_target_ssl": (target_seed_mean - nasa_seed_mean) / target_seed_mean,
            "ensemble_rul_gain_vs_target_ssl": (
                float(control.rul_mae) - float(nasa.rul_mae)
            ) / float(control.rul_mae),
            "positive_seed_count": positive_seed_count,
            "worst_device_change_vs_target_ssl": worst_change,
            "nasa_soh_mae_mean": nasa_soh_mean,
            "soh_change_vs_corrected_fixed160_baseline": soh_change,
            "improvement_vs_corrected_fixed160_rul": (corrected_baseline_rul - float(nasa.rul_mae)) / corrected_baseline_rul,
            "rul_at_most_6": float(nasa.rul_mae) <= 6.0,
            "stable_positive_transfer": accepted,
        })
    return pd.DataFrame(rows)


def finalize_existing(output_dir: Path) -> dict[str, Any]:
    results = pd.read_csv(output_dir / "results_by_seed.csv")
    metrics = pd.read_csv(output_dir / "rul_metrics.csv")
    acceptance = transfer_acceptance(results, metrics)
    acceptance.to_csv(output_dir / "transfer_acceptance.csv", index=False)
    eligible = acceptance.loc[acceptance.stable_positive_transfer]
    frozen = None if eligible.empty else eligible.sort_values(
        ["nasa_rul_mae", "worst_device_change_vs_target_ssl"]
    ).iloc[0].to_dict()
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_best_positive_transfer"] = frozen
    manifest["sealed_evaluation_permitted"] = frozen is not None
    manifest["transfer_acceptance_basis"] = (
        "five-seed mean physical-RUL gain, >=4/5 positive seeds, mean worst-device change <=10%, "
        "and SOH change <=5%; final ensemble comparison is reported separately"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        print(json.dumps(finalize_existing(args.output_dir), indent=2), flush=True)
        return
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    source, target, validation, features, report, target_knees, validation_knees = prepare_data(args.data_root)
    target_knees.to_csv(args.output_dir / "target_knee_labels.csv", index=False)
    validation_knees.to_csv(args.output_dir / "validation_knee_labels.csv", index=False)

    selection_path = args.output_dir / "window_selection.json"
    if args.formal_only:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_window = int(selection["selected_window"])
        tuning = pd.read_csv(args.output_dir / "window_tuning.csv")
    else:
        selected_window, tuning = window_tuning(
            args, reference, source, target, validation, features, report, device,
        )
        tuning.to_csv(args.output_dir / "window_tuning.csv", index=False)
        selection_path.write_text(json.dumps({
            "selected_window": selected_window,
            "channels": list(WINDOW_SPECS[selected_window]),
            "tie_rule": "within 0.1 EFC choose fewer parameters then shorter window",
            "tuning_seed": args.tuning_seed,
        }, indent=2), encoding="utf-8")
    if args.tune_only:
        print(selection_path.read_text(encoding="utf-8"))
        return

    config = build_config(reference, selected_window, args.smoke)
    seeds = (42,) if args.smoke else tuple(int(value) for value in args.formal_seeds.split(","))
    if not args.smoke and seeds != FORMAL_SEEDS:
        raise ValueError("Formal experiment requires frozen seeds 42-46")
    fixed_units = choose_nested_units(target, [160], args.selection_seed)[160]
    full_units = sorted(target.unit_id.astype(str).unique())
    selected_by_budget = {"fixed160": fixed_units, "full320": full_units}
    budgets = ("fixed160",) if args.smoke else BUDGETS
    methods = ("nasa_pretrain_finetune",) if args.smoke else METHODS
    selection_rows = [{"budget": budget, "unit_id": unit}
                      for budget in budgets for unit in selected_by_budget[budget]]
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "selected_target_units.csv", index=False)

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        ssl_state, source_state, target_state, _, _ = initialize_states(
            source, target, features, report, config, seed, device,
        )
        for budget in budgets:
            units = selected_by_budget[budget]
            selected = target.loc[target.unit_id.astype(str).isin(units)].copy()
            maximum_lifetime = maximum_training_lifetime(selected)
            for method in methods:
                checkpoint = args.output_dir / f"checkpoint_{budget}_{method}_{seed}.pt"
                if args.resume and checkpoint.exists():
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if list(map(str, payload["target_units"])) != list(map(str, units)):
                        raise RuntimeError(f"Checkpoint device set mismatch: {checkpoint}")
                    model = DynamicsTCN(len(features), config).to(device)
                    model.load_state_dict(payload["model_state"])
                    best_epoch = int(payload["best_epoch"])
                    validation_x, _, validation_meta = make_multitask_windows(
                        validation, features, target_state, config,
                    )
                    prediction = predict_dynamics(
                        model, validation_x, validation_meta, maximum_lifetime, config, device,
                    )
                    soh_metrics = regression_row(prediction.target_soh.to_numpy(float), prediction.predicted_soh.to_numpy(float))
                    rul_metrics = regression_row(prediction.true_rul_cycles.to_numpy(float), prediction.physical_rul.to_numpy(float))
                    metrics_row = {**{f"soh_{key}": value for key, value in soh_metrics.items()},
                                   **{f"rul_{key}": value for key, value in rul_metrics.items()},
                                   "soh_worst_device_mae": worst_device_mae(prediction, "predicted_soh", "target_soh"),
                                   "rul_worst_device_mae": worst_device_mae(prediction, "physical_rul", "true_rul_cycles")}
                else:
                    initial = ssl_state if method == "target_ssl" else source_state
                    model, best_epoch, prediction, metrics_row = train_configuration(
                        method, selected, validation, features, target_state, initial,
                        maximum_lifetime, config, seed, device,
                    )
                    torch.save({
                        "model_state": snapshot(model), "budget": budget, "method": method,
                        "seed": seed, "selection_seed": args.selection_seed,
                        "target_units": units, "features": features,
                        "configuration": asdict(config), "best_epoch": best_epoch,
                        "maximum_training_lifetime": maximum_lifetime,
                        "nasa_source_unit": "B0018",
                        "nasa_supervised_pretraining": method == "nasa_pretrain_finetune",
                        "target_scaler_median": target_state.median,
                        "target_scaler_iqr": target_state.iqr,
                    }, checkpoint)
                row = {"budget": budget, "method": method, "seed": seed,
                       "target_labeled_units": len(units), "best_epoch": best_epoch,
                       "maximum_training_lifetime": maximum_lifetime, **metrics_row}
                result_rows.append(row)
                prediction.insert(0, "seed", seed)
                prediction.insert(0, "method", method)
                prediction.insert(0, "budget", budget)
                prediction_frames.append(prediction)
                pd.DataFrame(result_rows).to_csv(args.output_dir / "results_by_seed.csv", index=False)
                pd.concat(prediction_frames, ignore_index=True).to_csv(
                    args.output_dir / "validation_predictions_by_seed.csv", index=False,
                )
                print(json.dumps(row), flush=True)

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if args.smoke:
        manifest = {"experiment_mode": "smoke", "selected_window": selected_window,
                    "sealed_features_accessed": False, "sealed_labels_accessed": False}
        (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return

    detail, settings, rul_metrics = postprocess_formal(
        predictions, target, selected_by_budget, args.output_dir, len(seeds),
    )
    detail.to_csv(args.output_dir / "rul_predictions_all.csv", index=False)
    settings.to_csv(args.output_dir / "rul_crossfit_settings.csv", index=False)
    rul_metrics.to_csv(args.output_dir / "rul_metrics.csv", index=False)
    acceptance = transfer_acceptance(results, rul_metrics)
    acceptance.to_csv(args.output_dir / "transfer_acceptance.csv", index=False)
    eligible = acceptance.loc[acceptance.stable_positive_transfer]
    frozen = None if eligible.empty else eligible.sort_values(
        ["nasa_rul_mae", "worst_device_change_vs_target_ssl"]
    ).iloc[0].to_dict()
    training_parameters, deployment_parameters = parameter_counts(DynamicsTCN(len(features), config))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_mode": "dual_track_rul_multitask",
        "data_version": "Basilisk V1.0 unchanged",
        "feature_contract": "bhump_degradation_invariant", "features": features,
        "feature_count": len(features), "selected_window": selected_window,
        "channels": list(config.channels),
        "dilations": [2**index for index in range(len(config.channels))],
        "selection_seed": args.selection_seed, "formal_seeds": list(seeds),
        "fixed160_same_units_across_seeds": True,
        "fixed160_target_unit_union_count": len(set(fixed_units)),
        "budgets": list(BUDGETS), "methods": list(METHODS),
        "training_parameters": training_parameters,
        "deployment_parameters": deployment_parameters,
        "loss_weights": {"soh": 1.0, "pre_rate": 0.2, "post_rate": 0.2,
                         "knee_time": 0.1, "knee_probability": 0.1, "log_rul": 0.2},
        "nasa_source_unit": "B0018", "nasa_supervised_pretraining_required": True,
        "corrected_fixed160_baseline_rul_mae": 7.340312695552595,
        "previous_308_union_experimental_rul_mae": 7.210572589167677,
        "frozen_best_positive_transfer": frozen,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "sealed_evaluation_permitted": frozen is not None,
        "configuration": asdict(config),
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
    parser.add_argument("--budgets", default=",".join(BUDGETS),
                        help="Frozen interface; formal execution always uses fixed160,full320.")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--tuning-seed", type=int, default=41)
    parser.add_argument("--formal-seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--windows", default=",".join(map(str, WINDOW_SPECS)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--formal-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Recompute acceptance and the frozen choice from completed outputs.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
