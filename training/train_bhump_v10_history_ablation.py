"""Causal full-history ablation for the frozen V1.0 NASA-transfer TCN.

The experiment keeps the existing 24-cycle, 16-feature TCN checkpoints fixed
and compares five increasingly informative causal heads:

* A: local TCN representation only;
* B: A plus train-fitted full-history trend summaries;
* C: B plus a recurrent state over all observations from cycle zero;
* D: C plus multi-horizon SOH trajectory supervision;
* E: D plus knee-risk supervision.

No sealed path is referenced.  Future SOH and offline knee annotations are
training labels only and never enter a model input.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_positive_transfer import Config
from train_bhump_v10_fixed_ensemble import checkpoint_labels
from train_bhump_v10_rul_multitask import DynamicsTCN, prepare_data


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_TCN_RUN = ROOT / "bhump_v10_rul_multitask_runs"
DEFAULT_REFERENCE = ROOT / "bhump_v10_pssm_rul_runs" / "pssm_rul_predictions.csv"
DEFAULT_OUTPUT = ROOT / "bhump_v10_history_ablation_runs"
VARIANTS = ("A_local", "B_stats", "C_gru", "D_trajectory", "E_knee")
FORMAL_SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (6, 12, 24, 48)
EOL_SOH = 0.80


@dataclass(frozen=True)
class HistoryConfig:
    local_projection: int = 16
    statistics: int = 38
    gru_hidden: int = 32
    fusion_hidden: int = 48
    dropout: float = 0.10
    batch_devices: int = 32
    epochs: int = 45
    patience: int = 7
    learning_rate: float = 7.5e-4
    weight_decay: float = 1.0e-4
    soh_weight: float = 1.0
    log_rul_weight: float = 0.5
    trajectory_weight: float = 0.3
    knee_weight: float = 0.1


@dataclass
class SequenceBundle:
    unit_ids: list[str]
    raw: np.ndarray
    local: np.ndarray
    local_soh: np.ndarray
    stats: np.ndarray
    times: np.ndarray
    target_soh: np.ndarray
    true_rul: np.ndarray
    true_eol: np.ndarray
    future_soh: np.ndarray
    future_mask: np.ndarray
    knee_class: np.ndarray
    mask: np.ndarray
    eval_mask: np.ndarray
    unit_weight: np.ndarray
    sample_weight: np.ndarray | None = None


class HistoryAblationModel(nn.Module):
    """Small causal head over frozen local TCN features and optional history."""

    def __init__(self, variant: str, config: HistoryConfig) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ablation variant: {variant}")
        self.variant = variant
        self.rank = VARIANTS.index(variant)
        if self.rank >= 2:
            self.gru = nn.GRU(17, config.gru_hidden, batch_first=True)
        input_size = config.local_projection + 1
        if self.rank >= 1:
            input_size += config.statistics
        if self.rank >= 2:
            input_size += config.gru_hidden
        self.fusion = nn.Sequential(
            nn.Linear(input_size, config.fusion_hidden), nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, config.fusion_hidden), nn.SiLU(),
        )
        self.soh_residual_head = nn.Linear(config.fusion_hidden, 1)
        self.rul_fraction_head = nn.Linear(config.fusion_hidden, 1)
        if self.rank >= 3:
            self.future_rate_head = nn.Linear(config.fusion_hidden, len(HORIZONS))
        if self.rank >= 4:
            self.knee_risk_head = nn.Linear(config.fusion_hidden, 6)
        self._reset()

    def _reset(self) -> None:
        nn.init.zeros_(self.soh_residual_head.weight)
        nn.init.zeros_(self.soh_residual_head.bias)
        nn.init.zeros_(self.rul_fraction_head.weight)
        nn.init.zeros_(self.rul_fraction_head.bias)
        if hasattr(self, "future_rate_head"):
            nn.init.normal_(self.future_rate_head.weight, std=0.01)
            nn.init.constant_(self.future_rate_head.bias, -6.0)
        if hasattr(self, "knee_risk_head"):
            nn.init.normal_(self.knee_risk_head.weight, std=0.01)
            nn.init.zeros_(self.knee_risk_head.bias)

    def forward(self, raw: torch.Tensor, local: torch.Tensor,
                local_soh: torch.Tensor, stats: torch.Tensor,
                times: torch.Tensor, maximum_lifetime: float) -> dict[str, torch.Tensor]:
        pieces = [local, local_soh.unsqueeze(-1)]
        if self.rank >= 1:
            pieces.append(stats)
        if self.rank >= 2:
            memory_input = torch.cat([raw, local_soh.unsqueeze(-1)], dim=-1)
            memory, _ = self.gru(memory_input)
            pieces.append(memory)
        hidden = self.fusion(torch.cat(pieces, dim=-1))
        soh = torch.clamp(
            local_soh + 0.05 * torch.tanh(self.soh_residual_head(hidden).squeeze(-1)),
            0.0, 1.10,
        )
        cap = torch.clamp(torch.as_tensor(maximum_lifetime, device=times.device) - times, min=0.0)
        rul = cap * torch.sigmoid(self.rul_fraction_head(hidden).squeeze(-1))
        output = {"soh": soh, "rul": rul}
        if self.rank >= 3:
            rates = F.softplus(self.future_rate_head(hidden)) + 1.0e-6
            widths = torch.as_tensor((6.0, 6.0, 12.0, 24.0), device=times.device)
            output["future_soh"] = torch.clamp(
                soh.unsqueeze(-1) - torch.cumsum(rates * widths, dim=-1), 0.0, 1.10,
            )
        if self.rank >= 4:
            output["knee_logits"] = self.knee_risk_head(hidden)
        return output


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tcn_config(payload: dict[str, Any]) -> Config:
    values = dict(payload["configuration"])
    values["channels"] = tuple(values["channels"])
    return Config(**values)


def local_windows(values: np.ndarray, window: int) -> np.ndarray:
    windows: list[np.ndarray] = []
    for end in range(len(values)):
        start = end - window + 1
        if start < 0:
            item = np.concatenate([np.repeat(values[[0]], -start, axis=0), values[:end + 1]])
        else:
            item = values[start:end + 1]
        windows.append(item)
    return np.asarray(windows, np.float32)


def encode_local(model: DynamicsTCN, windows: np.ndarray, device: torch.device,
                 batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values = torch.from_numpy(windows[start:start + batch_size]).to(device)
            soh, shared, _ = model(values, "target")
            embeddings.append(shared.cpu().numpy())
            predictions.append(soh.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def safe_slope(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vector OLS slope for one or many signals, returning zero for one point."""
    if len(times) < 2:
        return np.zeros(values.shape[1:] or (), dtype=float)
    centered = times - times.mean()
    denominator = float(np.sum(centered**2))
    if denominator < 1.0e-12:
        return np.zeros(values.shape[1:] or (), dtype=float)
    return np.sum(centered.reshape((-1,) + (1,) * (values.ndim - 1)) * values, axis=0) / denominator


def causal_statistics(raw: np.ndarray, local_soh: np.ndarray,
                      times: np.ndarray, maximum_lifetime: float) -> np.ndarray:
    rows: list[np.ndarray] = []
    ewma = float(local_soh[0])
    for end in range(len(raw)):
        history_t = times[:end + 1]
        history_x = raw[:end + 1]
        history_soh = local_soh[:end + 1]
        ewma = 0.2 * float(local_soh[end]) + 0.8 * ewma if end else float(local_soh[0])
        recent_start = max(0, end - 11)
        long_soh_slope = float(safe_slope(history_t, history_soh))
        recent_soh_slope = float(safe_slope(
            times[recent_start:end + 1], local_soh[recent_start:end + 1],
        ))
        row = np.concatenate([
            raw[end] - raw[0],
            np.asarray(safe_slope(history_t, history_x), dtype=float),
            np.asarray([
                long_soh_slope,
                recent_soh_slope,
                recent_soh_slope - long_soh_slope,
                float(local_soh[end]) - ewma,
                float(times[end]) / maximum_lifetime,
                math.log1p(end + 1) / math.log1p(maximum_lifetime + 1.0),
            ]),
        ])
        rows.append(row)
    output = np.asarray(rows, np.float32)
    if output.shape[1] != 38:
        raise RuntimeError(f"Expected 38 causal statistics, received {output.shape[1]}")
    return output


def statistic_names(features: list[str]) -> list[str]:
    return (
        [f"delta_from_first__{feature}" for feature in features]
        + [f"full_history_slope__{feature}" for feature in features]
        + [
            "local_tcn_soh_full_history_slope",
            "local_tcn_soh_recent12_slope",
            "local_tcn_soh_slope_acceleration",
            "local_tcn_soh_minus_ewma",
            "current_efc_scaled",
            "observed_history_length_log_scaled",
        ]
    )


def knee_classes(times: np.ndarray, report: dict[str, Any]) -> np.ndarray:
    result = np.full(len(times), 5, np.int64)
    if not bool(report["has_knee"]):
        return result
    remaining = float(report["knee_cycle"]) - times
    result[remaining <= 0.0] = 0
    result[(remaining > 0.0) & (remaining <= 6.0)] = 1
    result[(remaining > 6.0) & (remaining <= 12.0)] = 2
    result[(remaining > 12.0) & (remaining <= 24.0)] = 3
    result[(remaining > 24.0) & (remaining <= 48.0)] = 4
    return result


def make_bundle(frame: pd.DataFrame, knee_report: pd.DataFrame,
                features: list[str], median: np.ndarray, iqr: np.ndarray,
                model: DynamicsTCN, config: Config, maximum_lifetime: float,
                device: torch.device, allowed_units: Iterable[str] | None = None) -> SequenceBundle:
    allowed = None if allowed_units is None else set(map(str, allowed_units))
    working = frame.copy()
    if allowed is not None:
        working = working.loc[working.unit_id.astype(str).isin(allowed)].copy()
    groups = [(str(unit), group.sort_values("time").reset_index(drop=True))
              for unit, group in working.groupby("unit_id", sort=True)]
    if not groups:
        raise ValueError("No units available for history bundle")
    maximum_steps = max(len(group) for _, group in groups)
    count, feature_count = len(groups), len(features)
    shapes = (count, maximum_steps)
    raw = np.zeros((*shapes, feature_count), np.float32)
    local = np.zeros((*shapes, 16), np.float32)
    local_soh = np.zeros(shapes, np.float32)
    stats = np.zeros((*shapes, 38), np.float32)
    times = np.zeros(shapes, np.float32)
    target_soh = np.zeros(shapes, np.float32)
    true_rul = np.zeros(shapes, np.float32)
    true_eol = np.zeros(shapes, np.float32)
    future_soh = np.zeros((*shapes, len(HORIZONS)), np.float32)
    future_mask = np.zeros((*shapes, len(HORIZONS)), bool)
    knee_class = np.full(shapes, 5, np.int64)
    mask = np.zeros(shapes, bool)
    has_sample_weight = "sample_weight" in working.columns
    sample_weight = np.zeros(shapes, np.float32) if has_sample_weight else None
    reports = knee_report.set_index(knee_report.unit_id.astype(str)).to_dict("index")
    unit_ids: list[str] = []
    for row, (unit_id, unit) in enumerate(groups):
        unit_ids.append(unit_id)
        length = len(unit)
        unit_raw = ((unit[features].to_numpy(float) - median) / iqr).astype(np.float32)
        embeddings, predictions = encode_local(
            model, local_windows(unit_raw, config.window), device,
        )
        unit_times = unit.time.to_numpy(float)
        raw[row, :length] = unit_raw
        local[row, :length] = embeddings
        local_soh[row, :length] = predictions
        stats[row, :length] = causal_statistics(unit_raw, predictions, unit_times, maximum_lifetime)
        times[row, :length] = unit_times
        target_soh[row, :length] = unit.target_soh.to_numpy(float)
        true_rul[row, :length] = unit.true_rul_cycles.to_numpy(float)
        true_eol[row, :length] = unit.true_eol_cycle.to_numpy(float)
        knee_class[row, :length] = knee_classes(unit_times, reports[unit_id])
        if sample_weight is not None:
            weights = unit.sample_weight.to_numpy(float)
            if not np.isfinite(weights).all() or np.any(weights <= 0.0):
                raise ValueError(f"Invalid sample weights for {unit_id}")
            sample_weight[row, :length] = weights.astype(np.float32)
        mask[row, :length] = True
        time_lookup = {float(value): index for index, value in enumerate(unit_times)}
        for index, current_time in enumerate(unit_times):
            for horizon_index, horizon in enumerate(HORIZONS):
                future_index = time_lookup.get(float(current_time + horizon))
                if future_index is not None:
                    future_soh[row, index, horizon_index] = float(unit.target_soh.iloc[future_index])
                    future_mask[row, index, horizon_index] = True
    eval_mask = mask & (times >= 7.0)
    unit_weight = np.asarray([
        float(group.unit_weight.iloc[0]) if "unit_weight" in group.columns else 1.0
        for _, group in groups
    ], np.float32)
    return SequenceBundle(
        unit_ids, raw, local, local_soh, stats, times, target_soh, true_rul,
        true_eol, future_soh, future_mask, knee_class, mask, eval_mask, unit_weight,
        sample_weight,
    )


def fit_statistics(train: SequenceBundle, validation: SequenceBundle) -> tuple[np.ndarray, np.ndarray]:
    values = train.stats[train.eval_mask]
    median = np.median(values, axis=0)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    iqr[iqr < 1.0e-6] = 1.0
    train.stats = ((train.stats - median) / iqr).astype(np.float32)
    validation.stats = ((validation.stats - median) / iqr).astype(np.float32)
    return median, iqr


def bundle_tensors(bundle: SequenceBundle) -> TensorDataset:
    arrays = (
        bundle.raw, bundle.local, bundle.local_soh, bundle.stats, bundle.times,
        bundle.target_soh, bundle.true_rul, bundle.future_soh,
        bundle.future_mask, bundle.knee_class, bundle.mask, bundle.eval_mask,
        bundle.unit_weight,
    )
    return TensorDataset(*(torch.from_numpy(value) for value in arrays))


def model_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...],
               variant: str, config: HistoryConfig,
               knee_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    (_raw, _local, _local_soh, _stats, _times, true_soh, true_rul,
     future_soh, future_mask, knee_class, _mask, eval_mask, unit_weight) = batch
    mask = eval_mask.bool()
    def device_balanced(losses: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weighted = losses * valid.to(losses.dtype)
        per_device = weighted.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(losses.dtype)
        weights = unit_weight / unit_weight.sum().clamp_min(1.0e-8)
        return torch.sum(per_device * weights)
    parts: dict[str, torch.Tensor] = {
        "soh": device_balanced(F.smooth_l1_loss(
            outputs["soh"], true_soh, beta=0.01, reduction="none",
        ), mask),
        "log_rul": device_balanced(F.smooth_l1_loss(
            torch.log1p(outputs["rul"]), torch.log1p(true_rul), beta=0.10, reduction="none",
        ), mask),
    }
    total = config.soh_weight * parts["soh"] + config.log_rul_weight * parts["log_rul"]
    rank = VARIANTS.index(variant)
    if rank >= 3:
        valid_future = future_mask.bool() & eval_mask.unsqueeze(-1)
        parts["trajectory"] = F.smooth_l1_loss(
            outputs["future_soh"][valid_future], future_soh[valid_future], beta=0.01,
        )
        total = total + config.trajectory_weight * parts["trajectory"]
    if rank >= 4:
        parts["knee"] = F.cross_entropy(
            outputs["knee_logits"][mask], knee_class[mask], weight=knee_weights,
        )
        total = total + config.knee_weight * parts["knee"]
    return total, {name: float(value.detach().cpu()) for name, value in parts.items()}


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def predict_bundle(model: HistoryAblationModel, bundle: SequenceBundle,
                   maximum_lifetime: float, device: torch.device,
                   batch_devices: int) -> pd.DataFrame:
    model.eval()
    rows: list[pd.DataFrame] = []
    dataset = bundle_tensors(bundle)
    loader = DataLoader(dataset, batch_size=batch_devices, shuffle=False, num_workers=0)
    unit_offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            raw, local, local_soh, stats, times = batch[:5]
            outputs = model(raw, local, local_soh, stats, times, maximum_lifetime)
            count = len(raw)
            for local_index in range(count):
                source_index = unit_offset + local_index
                valid = bundle.eval_mask[source_index]
                data: dict[str, Any] = {
                    "unit_id": bundle.unit_ids[source_index],
                    "time": bundle.times[source_index, valid],
                    "target_soh": bundle.target_soh[source_index, valid],
                    "true_rul_cycles": bundle.true_rul[source_index, valid],
                    "true_eol_cycle": bundle.true_eol[source_index, valid],
                    "local_tcn_soh": bundle.local_soh[source_index, valid],
                    "predicted_soh": outputs["soh"][local_index].cpu().numpy()[valid],
                    "predicted_rul_raw": outputs["rul"][local_index].cpu().numpy()[valid],
                }
                if "future_soh" in outputs:
                    future = outputs["future_soh"][local_index].cpu().numpy()[valid]
                    for column, horizon in enumerate(HORIZONS):
                        data[f"predicted_soh_plus_{horizon}"] = future[:, column]
                if "knee_logits" in outputs:
                    probabilities = torch.softmax(outputs["knee_logits"][local_index], -1).cpu().numpy()[valid]
                    for column in range(6):
                        data[f"knee_risk_class_{column}"] = probabilities[:, column]
                rows.append(pd.DataFrame(data))
            unit_offset += count
    return pd.concat(rows, ignore_index=True)


def knee_class_weights(bundle: SequenceBundle, device: torch.device) -> torch.Tensor:
    values = bundle.knee_class[bundle.eval_mask]
    counts = np.bincount(values, minlength=6).astype(float)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = np.clip(weights / weights.mean(), 0.25, 4.0)
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def train_variant(variant: str, train: SequenceBundle, validation: SequenceBundle,
                  maximum_lifetime: float, seed: int, config: HistoryConfig,
                  device: torch.device, checkpoint: Path) -> tuple[HistoryAblationModel, int, float]:
    seed_all(seed + 1000 * VARIANTS.index(variant))
    model = HistoryAblationModel(variant, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(
        bundle_tensors(train), batch_size=config.batch_devices, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 71), num_workers=0,
    )
    weights = knee_class_weights(train, device)
    best_state, best_mae, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            raw, local, local_soh, stats, times = batch[:5]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(raw, local, local_soh, stats, times, maximum_lifetime)
            loss, _ = model_loss(outputs, batch, variant, config, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_bundle(model, validation, maximum_lifetime, device, config.batch_devices)
        mae = float(np.mean(np.abs(prediction.predicted_rul_raw - prediction.true_rul_cycles)))
        if mae < best_mae - 1.0e-6:
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            best_mae, best_epoch, stale = mae, epoch, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state, "variant": variant, "seed": seed,
        "history_configuration": asdict(config), "best_epoch": best_epoch,
        "validation_raw_rul_mae": best_mae,
    }, checkpoint)
    return model, best_epoch, best_mae


def regression_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    error = frame[prediction].to_numpy(float) - frame.true_rul_cycles.to_numpy(float)
    denominator = float(np.sum((frame.true_rul_cycles - frame.true_rul_cycles.mean()) ** 2))
    device_mae = frame.assign(_error=np.abs(error)).groupby("unit_id")._error.mean()
    return {
        "rul_mae": float(np.mean(np.abs(error))),
        "rul_rmse": float(np.sqrt(np.mean(error**2))),
        "rul_r2": float(1.0 - np.sum(error**2) / denominator),
        "rul_bias": float(np.mean(error)),
        "rul_p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
        "worst_device_mae": float(device_mae.max()),
    }


def soh_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame.predicted_soh.to_numpy(float) - frame.target_soh.to_numpy(float)
    return {"soh_mae": float(np.mean(np.abs(error))),
            "soh_rmse": float(np.sqrt(np.mean(error**2))),
            "soh_bias": float(np.mean(error))}


def add_checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(["unit_id", "time"]).reset_index(drop=True).copy()
    labels = np.full(len(output), "", object)
    for _, indices in output.groupby("unit_id", sort=False).groups.items():
        ordered = list(indices)
        mapping = checkpoint_labels(output.loc[ordered].reset_index(drop=True))
        for index, names in mapping.items():
            labels[ordered[index]] = ";".join(names)
    output["checkpoints"] = labels
    return output


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, group in predictions.groupby("variant", sort=False):
        group = add_checkpoints(group)
        for column in ("predicted_rul_raw", "predicted_rul_calibrated"):
            for checkpoint in ("all_points", "fraction_0.2", "fraction_0.4",
                               "fraction_0.6", "fraction_0.8", "fixed_30_efc"):
                subset = group if checkpoint == "all_points" else group.loc[
                    group.checkpoints.str.contains(checkpoint, regex=False)
                ]
                rows.append({"variant": variant, "prediction": column,
                             "checkpoint": checkpoint, "samples": len(subset),
                             "devices": int(subset.unit_id.nunique()),
                             **regression_metrics(subset, column), **soh_metrics(subset)})
    return pd.DataFrame(rows)


def load_reference(path: Path, validation_units: set[str]) -> pd.DataFrame:
    reference = pd.read_csv(path)
    reference = reference.loc[reference.unit_id.astype(str).isin(validation_units)].copy()
    columns = ["unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle",
               "predicted_soh", "combined_rul"]
    output = reference[columns].rename(columns={"combined_rul": "predicted_rul_raw"})
    output["predicted_rul_calibrated"] = output.predicted_rul_raw
    output["variant"] = "A_reference_pipeline"
    return output


def assert_causal(model: HistoryAblationModel, bundle: SequenceBundle,
                  maximum_lifetime: float, device: torch.device) -> None:
    model.eval()
    index = int(np.flatnonzero(bundle.eval_mask[0])[-1] // 2)
    raw = torch.from_numpy(bundle.raw[[0]]).to(device)
    local = torch.from_numpy(bundle.local[[0]]).to(device)
    local_soh = torch.from_numpy(bundle.local_soh[[0]]).to(device)
    stats = torch.from_numpy(bundle.stats[[0]]).to(device)
    times = torch.from_numpy(bundle.times[[0]]).to(device)
    with torch.no_grad():
        before = model(raw, local, local_soh, stats, times, maximum_lifetime)
        changed_raw = raw.clone()
        changed_raw[:, index + 1:] += 100.0
        after = model(changed_raw, local, local_soh, stats, times, maximum_lifetime)
    for key in before:
        if not torch.allclose(before[key][:, :index + 1], after[key][:, :index + 1], atol=1.0e-6):
            raise AssertionError(f"Future mutation changed causal output {key}")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not set(variants).issubset(VARIANTS):
        raise ValueError(f"Variants must be a subset of {VARIANTS}")
    if not args.smoke and not args.development and seeds != FORMAL_SEEDS:
        raise ValueError("Formal history ablation requires seeds 42,43,44,45,46")
    _source, target, validation, features, _report, target_knees, validation_knees = prepare_data(args.data_root)
    if len(features) != 16:
        raise ValueError("The frozen V1.0 contract must contain exactly 16 features")
    forbidden = [name for name in features if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        raise ValueError(f"Forbidden TCN inputs: {forbidden}")
    if set(target.unit_id.astype(str)) & set(validation.unit_id.astype(str)):
        raise ValueError("Train and validation devices overlap")
    if args.smoke:
        train_units = sorted(target.unit_id.astype(str).unique())[:args.smoke_train_units]
        validation_units = sorted(validation.unit_id.astype(str).unique())[:args.smoke_validation_units]
    else:
        train_units = sorted(target.unit_id.astype(str).unique())
        validation_units = sorted(validation.unit_id.astype(str).unique())
    maximum_lifetime = float(target.loc[target.unit_id.astype(str).isin(train_units)].true_eol_cycle.max())
    history_config = HistoryConfig(
        epochs=3 if args.smoke else args.epochs,
        patience=2 if args.smoke else args.patience,
        batch_devices=min(args.batch_devices, len(train_units)),
    )
    prediction_frames: list[pd.DataFrame] = []
    seed_rows: list[dict[str, Any]] = []
    statistic_rows: list[dict[str, Any]] = []
    for seed in seeds:
        tcn_checkpoint = args.tcn_run / f"checkpoint_full320_{args.tcn_method}_{seed}.pt"
        payload = torch.load(tcn_checkpoint, map_location="cpu", weights_only=False)
        expected_nasa = args.tcn_method == "nasa_pretrain_finetune"
        if bool(payload.get("nasa_supervised_pretraining", False)) != expected_nasa:
            raise ValueError(f"Checkpoint is not NASA-pretrained: {tcn_checkpoint}")
        if len(set(map(str, payload["target_units"]))) != 320:
            raise ValueError("Expected the frozen full320 NASA-transfer checkpoint")
        config = tcn_config(payload)
        if config.window != 24:
            raise ValueError("History ablation requires the frozen 24-cycle local TCN")
        tcn = DynamicsTCN(len(features), config).to(device)
        tcn.load_state_dict(payload["model_state"])
        for parameter in tcn.parameters():
            parameter.requires_grad = False
        median = np.asarray(payload["target_scaler_median"], float)
        iqr = np.asarray(payload["target_scaler_iqr"], float)
        train_bundle = make_bundle(
            target, target_knees, features, median, iqr, tcn, config,
            maximum_lifetime, device, train_units,
        )
        validation_bundle = make_bundle(
            validation, validation_knees, features, median, iqr, tcn, config,
            maximum_lifetime, device, validation_units,
        )
        stats_median, stats_iqr = fit_statistics(train_bundle, validation_bundle)
        if args.statistics_mode == "history_only":
            train_bundle.stats[..., -2:] = 0.0
            validation_bundle.stats[..., -2:] = 0.0
        elif args.statistics_mode == "position_only":
            train_bundle.stats[..., :-2] = 0.0
            validation_bundle.stats[..., :-2] = 0.0
        for index, name in enumerate(statistic_names(features)):
            statistic_rows.append({"seed": seed, "feature": name,
                                   "median": stats_median[index], "iqr": stats_iqr[index],
                                   "fit_units": len(train_units)})
        for variant in variants:
            checkpoint = args.output_dir / f"checkpoint_{variant}_seed_{seed}.pt"
            model, best_epoch, best_mae = train_variant(
                variant, train_bundle, validation_bundle, maximum_lifetime,
                seed, history_config, device, checkpoint,
            )
            assert_causal(model, validation_bundle, maximum_lifetime, device)
            train_prediction = predict_bundle(
                model, train_bundle, maximum_lifetime, device, history_config.batch_devices,
            )
            validation_prediction = predict_bundle(
                model, validation_bundle, maximum_lifetime, device, history_config.batch_devices,
            )
            bias = float(np.median(
                train_prediction.true_rul_cycles - train_prediction.predicted_rul_raw
            ))
            validation_prediction["predicted_rul_calibrated"] = np.clip(
                validation_prediction.predicted_rul_raw + bias, 0.0,
                maximum_lifetime - validation_prediction.time,
            )
            validation_prediction.insert(0, "seed", seed)
            validation_prediction.insert(0, "variant", variant)
            prediction_frames.append(validation_prediction)
            raw = regression_metrics(validation_prediction, "predicted_rul_raw")
            calibrated = regression_metrics(validation_prediction, "predicted_rul_calibrated")
            row = {"variant": variant, "seed": seed, "best_epoch": best_epoch,
                   "early_stop_raw_rul_mae": best_mae, "train_bias_calibration": bias,
                   **{f"raw_{key}": value for key, value in raw.items()},
                   **{f"calibrated_{key}": value for key, value in calibrated.items()},
                   **soh_metrics(validation_prediction),
                   "parameters": sum(parameter.numel() for parameter in model.parameters())}
            seed_rows.append(row)
            print(json.dumps(row), flush=True)
            pd.DataFrame(seed_rows).to_csv(args.output_dir / "results_by_seed.csv", index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                args.output_dir / "validation_predictions_by_seed.csv", index=False,
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    keys = ["variant", "unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle"]
    ensemble = predictions.groupby(keys, as_index=False).agg(
        predicted_soh=("predicted_soh", "mean"),
        predicted_rul_raw=("predicted_rul_raw", "mean"),
        predicted_rul_calibrated=("predicted_rul_calibrated", "mean"),
        soh_ensemble_std=("predicted_soh", lambda values: float(np.std(values, ddof=0))),
        rul_ensemble_std=("predicted_rul_calibrated", lambda values: float(np.std(values, ddof=0))),
        seed_count=("seed", "nunique"),
    )
    reference = load_reference(args.reference_predictions, set(validation_units))
    combined = pd.concat([reference, ensemble], ignore_index=True, sort=False)
    metrics = summarize_predictions(combined)
    ensemble.to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "ensemble_metrics.csv", index=False)
    pd.DataFrame(statistic_rows).to_csv(args.output_dir / "history_statistics_scalers.csv", index=False)
    best = metrics.loc[metrics.checkpoint.eq("all_points")].sort_values(
        ["rul_mae", "worst_device_mae"]
    ).iloc[0]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V1.0 causal full-history ablation",
        "mode": "smoke" if args.smoke else ("development" if args.development else "formal_five_seed"),
        "data_version": "Basilisk V1.0 unchanged",
        "feature_contract": "frozen 16-d bhump_degradation_invariant",
        "features": features,
        "local_tcn_window": 24,
        "tcn_initialization_method": args.tcn_method,
        "nasa_transfer_checkpoint_required": expected_nasa,
        "train_devices": len(train_units), "validation_devices": len(validation_units),
        "validation_samples": int(ensemble.groupby("variant").size().min()),
        "seeds": list(seeds), "variants": list(variants),
        "history_statistics": 38, "gru_hidden": history_config.gru_hidden,
        "statistics_mode": args.statistics_mode,
        "future_soh_horizons": list(HORIZONS),
        "knee_classes": ["already", "within_6", "6_to_12", "12_to_24", "24_to_48", "later_or_none"],
        "best_variant": str(best.variant), "best_prediction": str(best.prediction),
        "best_rul_mae": float(best.rul_mae),
        "reference_rul_mae": 6.653362908399209,
        "future_labels_are_inputs": False,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "configuration": asdict(history_config),
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    leakage = {
        "model_input_features": features,
        "forbidden_input_tokens": list(FORBIDDEN_INPUT_TOKENS),
        "input_feature_leaks": forbidden,
        "time_used_only_as_current_causal_metadata": True,
        "future_soh_used_only_as_training_label": True,
        "knee_annotation_used_only_as_training_label": True,
        "statistics_fit_devices": len(train_units),
        "validation_statistics_used_for_fit": False,
        "sealed_paths_referenced": False,
    }
    (args.output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tcn-run", type=Path, default=DEFAULT_TCN_RUN)
    parser.add_argument("--reference-predictions", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tcn-method", choices=("nasa_pretrain_finetune", "target_ssl"),
                        default="nasa_pretrain_finetune")
    parser.add_argument("--statistics-mode", choices=("full", "history_only", "position_only"),
                        default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-devices", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--development", action="store_true",
                        help="Allow a seed subset on all train/validation devices before the frozen formal run.")
    parser.add_argument("--smoke-train-units", type=int, default=16)
    parser.add_argument("--smoke-validation-units", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
