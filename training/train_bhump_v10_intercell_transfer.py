"""Strict OOF explicit NASA inter-cell transfer for V1.0 battery RUL.

The target-only backbone is the frozen target-SSL TCN24 + B_stats model.  A
small source-conditioned head predicts the log-EOL difference between a
target device and synchronized run-to-failure reference batteries.  All
selection happens on an inner device split; every outer fold is evaluated
once after fixed-epoch refitting on all outer-training devices.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
try:
    from sklearn.isotonic import IsotonicRegression
except ModuleNotFoundError:
    class IsotonicRegression:  # Minimal PAVA fallback for the packaged CPU runtime.
        def __init__(self, increasing: bool = True, out_of_bounds: str = "clip") -> None:
            self.increasing = bool(increasing)
            self.out_of_bounds = out_of_bounds

        def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            order = np.argsort(np.asarray(x, dtype=float), kind="stable")
            values = np.asarray(y, dtype=float)[order]
            signed = values if self.increasing else -values
            levels: list[float] = []
            weights: list[int] = []
            starts: list[int] = []
            for index, value in enumerate(signed):
                levels.append(float(value)); weights.append(1); starts.append(index)
                while len(levels) >= 2 and levels[-2] > levels[-1]:
                    total = weights[-2] + weights[-1]
                    levels[-2] = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total
                    weights[-2] = total
                    levels.pop(); weights.pop(); starts.pop()
            fitted = np.empty(len(values), dtype=float)
            for block, start in enumerate(starts):
                stop = starts[block + 1] if block + 1 < len(starts) else len(values)
                fitted[start:stop] = levels[block]
            if not self.increasing:
                fitted = -fitted
            output = np.empty_like(fitted)
            output[order] = fitted
            return output
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_positive_transfer import (
    make_windows, robust_fit, snapshot, ssl_fit, unlabeled_view,
)
from train_bhump_v10_bstats_oof import (
    FORMAL_SEEDS, inner_split, maximum_lifetime_without_test,
    stratified_outer_folds, subset,
)
from train_bhump_v10_bstats_refit_oof import (
    fit_bstats_fixed_epochs, fit_multitask_fixed_epochs,
)
from train_bhump_v10_history_ablation import (
    HistoryConfig, assert_causal, causal_statistics, encode_local,
    fit_statistics, local_windows, make_bundle, predict_bundle,
    regression_metrics, seed_all, soh_metrics, statistic_names, train_variant,
)
from train_bhump_v10_rul_multitask import (
    DynamicsTCN, build_config, fit_unit_dynamics, prepare_data,
    train_configuration,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_SEQUENTIAL = ROOT / "bhump_v10_bstats_refit_oof_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_intercell_transfer_runs"
SOURCE_UNITS = ("B0005", "B0018", "B0033", "B0043", "B0044")
METHODS = (
    "target_ssl_bstats", "sequential_nasa_bstats",
    "target_reference_control", "nasa_intercell",
)
EOL_SOH = 0.80
REPRESENTATION_SIZE = 55
BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class IntercellConfig:
    progress_knots: int = 32
    projection: int = 16
    hidden: int = 32
    epochs: int = 40
    patience: int = 7
    batch_size: int = 256
    learning_rate: float = 7.5e-4
    weight_decay: float = 1.0e-4
    log_eol_weight: float = 1.0
    log_rul_weight: float = 0.5
    ranking_weight: float = 0.1
    consistency_weight: float = 0.05
    difficult_sample_weight: float = 1.5
    minimum_blend_improvement: float = 0.1


@dataclass(frozen=True)
class VectorState:
    median: np.ndarray
    iqr: np.ndarray
    fit_units: tuple[str, ...]
    domain: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.median) / self.iqr).astype(np.float32)


@dataclass
class FlatTarget:
    frame: pd.DataFrame
    representation: np.ndarray


@dataclass
class ReferenceBank:
    unit_ids: list[str]
    knots: np.ndarray
    representation: np.ndarray
    eol: np.ndarray
    metadata: pd.DataFrame
    domain: str


class IntercellLifeHead(nn.Module):
    """Shared target/reference projection followed by a log-life-ratio head."""

    def __init__(self, config: IntercellConfig) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(REPRESENTATION_SIZE, config.projection), nn.SiLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(4 * config.projection + 1, config.hidden), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(config.hidden, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, target: torch.Tensor, reference: torch.Tensor,
                progress: torch.Tensor) -> torch.Tensor:
        target_z = self.projection(target)
        reference_z = self.projection(reference)
        expanded = target_z.unsqueeze(1).expand_as(reference_z)
        progress_column = progress[:, None, None].expand(
            len(target), reference.shape[1], 1,
        )
        pair = torch.cat([
            expanded, reference_z, expanded - reference_z,
            torch.abs(expanded - reference_z), progress_column,
        ], dim=-1)
        return 1.5 * torch.tanh(self.delta_head(pair).squeeze(-1))


def robust_vector_fit(values: np.ndarray, units: Iterable[str], domain: str) -> VectorState:
    if values.ndim != 2 or values.shape[1] != REPRESENTATION_SIZE:
        raise ValueError(f"Expected N x {REPRESENTATION_SIZE} representations")
    median = np.median(values, axis=0)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    iqr[iqr < 1.0e-6] = 1.0
    return VectorState(median, iqr, tuple(sorted(set(map(str, units)))), domain)


def monotonic_progress(times: np.ndarray, soh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("Reference trajectory must contain three ordered cycles")
    smooth = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(times, soh)
    denominator = max(float(smooth[0]) - EOL_SOH, 1.0e-4)
    progress = np.clip((float(smooth[0]) - smooth) / denominator, 0.0, 1.0)
    progress = np.maximum.accumulate(progress)
    return np.asarray(smooth, np.float32), np.asarray(progress, np.float32)


def interpolate_progress(progress: np.ndarray, values: np.ndarray,
                         knots: np.ndarray) -> np.ndarray:
    order = np.argsort(progress, kind="stable")
    progress, values = progress[order], values[order]
    unique, inverse = np.unique(progress, return_inverse=True)
    collapsed = np.zeros((len(unique), values.shape[1]), float)
    counts = np.bincount(inverse).astype(float)
    for column in range(values.shape[1]):
        collapsed[:, column] = np.bincount(
            inverse, weights=values[:, column], minlength=len(unique),
        ) / counts
    if len(unique) == 1:
        return np.repeat(collapsed, len(knots), axis=0).astype(np.float32)
    return np.column_stack([
        np.interp(knots, unique, collapsed[:, column])
        for column in range(values.shape[1])
    ]).astype(np.float32)


def interpolate_progress_vector(progress: np.ndarray, values: np.ndarray,
                                knots: np.ndarray) -> np.ndarray:
    """Interpolate a scalar after safely collapsing repeated progress values."""
    return interpolate_progress(
        progress, np.asarray(values, float).reshape(-1, 1), knots,
    )[:, 0]


def flat_target(bundle: Any, base_prediction: pd.DataFrame) -> FlatTarget:
    rows: list[dict[str, Any]] = []
    representations: list[np.ndarray] = []
    prediction = base_prediction.set_index(["unit_id", "time"], verify_integrity=True)
    for index, unit_id in enumerate(bundle.unit_ids):
        valid_indices = np.flatnonzero(bundle.eval_mask[index])
        initial_soh = float(bundle.local_soh[index, 0])
        denominator = max(initial_soh - EOL_SOH, 0.02)
        for step in valid_indices:
            time = float(bundle.times[index, step])
            record = prediction.loc[(unit_id, time)]
            progress = float(np.clip(
                (initial_soh - float(record.predicted_soh)) / denominator, 0.0, 1.0,
            ))
            rows.append({
                "unit_id": unit_id, "time": time,
                "target_soh": float(bundle.target_soh[index, step]),
                "true_rul_cycles": float(bundle.true_rul[index, step]),
                "true_eol_cycle": float(bundle.true_eol[index, step]),
                "predicted_soh": float(record.predicted_soh),
                "base_rul": float(record.predicted_rul_raw),
                "progress": progress,
                # V1.1 supplies a row-level spectrum weight.  Keep it outside
                # the representation so lifetime-bin metadata can never leak
                # into inference inputs.  Legacy data has no such field and
                # therefore retains the original unit weight of one.
                "spectrum_weight": (
                    float(bundle.sample_weight[index, step])
                    if getattr(bundle, "sample_weight", None) is not None
                    else 1.0
                ),
            })
            representations.append(np.concatenate([
                bundle.local[index, step],
                np.asarray([bundle.local_soh[index, step]], np.float32),
                bundle.stats[index, step],
            ]))
    frame = pd.DataFrame(rows)
    representation = np.asarray(representations, np.float32)
    if representation.shape != (len(frame), REPRESENTATION_SIZE):
        raise RuntimeError("Flat target representation shape mismatch")
    return FlatTarget(frame, representation)


def _source_node_rows(unit_id: str, knots: np.ndarray, smooth: np.ndarray,
                      progress: np.ndarray, times: np.ndarray,
                      representations: np.ndarray, eol: float,
                      dynamics: dict[str, float | bool]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    nodes = interpolate_progress(progress, representations, knots)
    node_soh = interpolate_progress_vector(progress, smooth, knots)
    node_time = interpolate_progress_vector(progress, times, knots)
    # Progress may plateau after isotonic smoothing, so gradients on node_time
    # can contain zero denominators.  Differentiating in normalized degradation
    # progress and scaling by EOL keeps these audit fields finite and comparable.
    rate = -np.gradient(node_soh, knots, edge_order=1) / max(eol, 1.0)
    acceleration = np.gradient(rate, knots, edge_order=1) / max(eol, 1.0)
    knee_cycle = float(dynamics["knee_cycle"])
    has_knee = bool(dynamics["has_knee"])
    rows = [{
        "unit_id": unit_id, "progress": float(knot), "source_eol": eol,
        "smoothed_soh": float(node_soh[index]), "source_cycle": float(node_time[index]),
        "local_rate": float(rate[index]), "rate_acceleration": float(acceleration[index]),
        "has_knee": has_knee,
        "knee_state": bool(has_knee and node_time[index] >= knee_cycle),
        "knee_cycle": knee_cycle,
    } for index, knot in enumerate(knots)]
    return nodes, rows


def nasa_reference_bank(source: pd.DataFrame, features: list[str], model: DynamicsTCN,
                        config: Any, source_units: tuple[str, ...], knots_count: int,
                        device: torch.device) -> ReferenceBank:
    knots = np.linspace(0.0, 1.0, knots_count, dtype=np.float32)
    raw_nodes: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    eols: list[float] = []
    for unit_id in source_units:
        unit = source.loc[source.unit_id.astype(str).eq(unit_id)].sort_values("time").reset_index(drop=True)
        if len(unit) < 8:
            raise ValueError(f"NASA reference {unit_id} is too short")
        state = robust_fit(unit[["unit_id", "time", *features]], features, f"NASA:{unit_id}")
        raw = state.transform(unit[features].to_numpy(float))
        local, local_soh = encode_local(model, local_windows(raw, config.window), device)
        times = unit.time.to_numpy(float)
        eol = float(unit.true_eol_cycle.max())
        stats = causal_statistics(raw, local_soh, times, max(eol, 1.0))
        representation = np.concatenate([local, local_soh[:, None], stats], axis=1)
        smooth, progress = monotonic_progress(times, unit.target_soh.to_numpy(float))
        dynamics = fit_unit_dynamics(unit)
        nodes, rows = _source_node_rows(
            unit_id, knots, smooth, progress, times, representation, eol, dynamics,
        )
        raw_nodes.append(nodes)
        metadata.extend(rows)
        eols.append(eol)
    stacked = np.stack(raw_nodes).astype(np.float32)
    state = robust_vector_fit(
        stacked.reshape(-1, REPRESENTATION_SIZE), source_units, "NASA_reference_bank",
    )
    transformed = state.transform(stacked.reshape(-1, REPRESENTATION_SIZE)).reshape(stacked.shape)
    return ReferenceBank(list(source_units), knots, transformed, np.asarray(eols, np.float32),
                         pd.DataFrame(metadata), "nasa")


def choose_target_references(frame: pd.DataFrame, count: int = 5) -> list[str]:
    life = frame.groupby("unit_id", as_index=False).true_eol_cycle.max().sort_values(
        ["true_eol_cycle", "unit_id"], ignore_index=True,
    )
    indices = np.round(np.linspace(0, len(life) - 1, count)).astype(int)
    selected = list(dict.fromkeys(life.iloc[indices].unit_id.astype(str)))
    if len(selected) != count:
        raise RuntimeError("Target reference quantiles were not unique")
    return selected


def target_reference_bank(bundle: Any, flat: FlatTarget, state: VectorState,
                          reference_units: list[str], knots_count: int) -> ReferenceBank:
    knots = np.linspace(0.0, 1.0, knots_count, dtype=np.float32)
    nodes_all: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    eols: list[float] = []
    for unit_id in reference_units:
        row = bundle.unit_ids.index(unit_id)
        valid = bundle.mask[row]
        times = bundle.times[row, valid].astype(float)
        target_soh = bundle.target_soh[row, valid].astype(float)
        raw_reps = np.concatenate([
            bundle.local[row, valid], bundle.local_soh[row, valid, None],
            bundle.stats[row, valid],
        ], axis=1)
        reps = state.transform(raw_reps)
        smooth, progress = monotonic_progress(times, target_soh)
        eol = float(bundle.true_eol[row, valid].max())
        dynamics = fit_unit_dynamics(pd.DataFrame({
            "time": times, "target_soh": target_soh,
        }))
        nodes, rows = _source_node_rows(
            unit_id, knots, smooth, progress, times, reps, eol, dynamics,
        )
        nodes_all.append(nodes)
        metadata.extend(rows)
        eols.append(eol)
    return ReferenceBank(reference_units, knots, np.stack(nodes_all).astype(np.float32),
                         np.asarray(eols, np.float32), pd.DataFrame(metadata), "target_control")


def matched_references(progress: np.ndarray, bank: ReferenceBank) -> np.ndarray:
    position = np.clip(progress, 0.0, 1.0) * (len(bank.knots) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(bank.knots) - 1)
    fraction = (position - low).astype(np.float32)
    left = np.transpose(bank.representation[:, low, :], (1, 0, 2))
    right = np.transpose(bank.representation[:, high, :], (1, 0, 2))
    return (left * (1.0 - fraction[:, None, None])
            + right * fraction[:, None, None]).astype(np.float32)


def intercell_arrays(flat: FlatTarget, bank: ReferenceBank,
                     target_state: VectorState) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    frame = flat.frame
    target = target_state.transform(flat.representation)
    references = matched_references(frame.progress.to_numpy(float), bank)
    source_eol = np.broadcast_to(bank.eol[None, :], references.shape[:2]).astype(np.float32).copy()
    valid = np.ones(references.shape[:2], bool)
    if bank.domain == "target_control":
        units = frame.unit_id.astype(str).to_numpy()
        for column, reference_unit in enumerate(bank.unit_ids):
            valid[:, column] = units != reference_unit
    if np.any(valid.sum(axis=1) < 2):
        raise RuntimeError("Every target sample requires at least two valid references")
    time = frame.time.to_numpy(np.float32)
    eol = frame.true_eol_cycle.to_numpy(np.float32)
    rul = frame.true_rul_cycles.to_numpy(np.float32)
    progress = frame.progress.to_numpy(np.float32)
    difficult_weight = np.where(
        (time <= 15.0) | (rul > 60.0), 1.5, 1.0,
    ).astype(np.float32)
    spectrum_weight = frame.get(
        "spectrum_weight", pd.Series(1.0, index=frame.index),
    ).to_numpy(np.float32)
    if not np.isfinite(spectrum_weight).all() or np.any(spectrum_weight <= 0.0):
        raise ValueError("Inter-cell spectrum weights must be finite and positive")
    weight = (difficult_weight * spectrum_weight).astype(np.float32)
    arrays = (target, references, source_eol, valid, time, eol, rul, progress, weight)
    return arrays, valid


def intercell_loss(model: IntercellLifeHead, batch: tuple[torch.Tensor, ...],
                   maximum_lifetime: float, config: IntercellConfig) -> torch.Tensor:
    target, reference, source_eol, valid, time, true_eol, true_rul, progress, weight = batch
    delta = model(target, reference, progress)
    predicted_eol = torch.clamp(source_eol * torch.exp(delta), max=maximum_lifetime)
    predicted_eol = torch.maximum(predicted_eol, time[:, None])
    predicted_rul = torch.clamp(predicted_eol - time[:, None], min=0.0)
    valid_float = valid.float()
    count = torch.clamp(valid_float.sum(1), min=1.0)
    log_eol = F.smooth_l1_loss(
        torch.log(predicted_eol), torch.log(true_eol[:, None].expand_as(predicted_eol)),
        beta=0.05, reduction="none",
    )
    log_rul = F.smooth_l1_loss(
        torch.log1p(predicted_rul), torch.log1p(true_rul[:, None].expand_as(predicted_rul)),
        beta=0.05, reduction="none",
    )
    sign = torch.sign(torch.log(true_eol[:, None] / source_eol))
    ranking = F.softplus(-sign * delta)
    log_prediction = torch.log(predicted_eol)
    mean_log = (log_prediction * valid_float).sum(1) / count
    consistency = (((log_prediction - mean_log[:, None]) ** 2) * valid_float).sum(1) / count
    per_sample = (
        config.log_eol_weight * (log_eol * valid_float).sum(1) / count
        + config.log_rul_weight * (log_rul * valid_float).sum(1) / count
        + config.ranking_weight * (ranking * valid_float).sum(1) / count
        + config.consistency_weight * consistency
    )
    return (per_sample * weight).sum() / weight.sum()


def intercell_loader(arrays: tuple[np.ndarray, ...], config: IntercellConfig,
                     seed: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(value) for value in arrays]
    return DataLoader(
        TensorDataset(*tensors), batch_size=min(config.batch_size, len(tensors[0])),
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def predict_intercell(model: IntercellLifeHead, flat: FlatTarget, bank: ReferenceBank,
                      target_state: VectorState, maximum_lifetime: float,
                      config: IntercellConfig, device: torch.device) -> pd.DataFrame:
    arrays, valid = intercell_arrays(flat, bank, target_state)
    target, references, source_eol, _valid, time, *_ = arrays
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(target), config.batch_size):
            x = torch.from_numpy(target[start:start + config.batch_size]).to(device)
            ref = torch.from_numpy(references[start:start + config.batch_size]).to(device)
            p = torch.from_numpy(arrays[7][start:start + config.batch_size]).to(device)
            delta = model(x, ref, p).cpu().numpy()
            chunks.append(delta)
    delta = np.concatenate(chunks)
    predicted_eol = np.clip(source_eol * np.exp(delta), time[:, None], maximum_lifetime)
    predicted_eol[~valid] = np.nan
    per_reference_rul = np.maximum(predicted_eol - time[:, None], 0.0)
    result = flat.frame.copy()
    result["intercell_eol"] = np.nanmedian(predicted_eol, axis=1)
    result["intercell_rul"] = np.nanmedian(per_reference_rul, axis=1)
    result["reference_prediction_std"] = np.nanstd(per_reference_rul, axis=1)
    for column, unit_id in enumerate(bank.unit_ids):
        result[f"reference_rul__{unit_id}"] = per_reference_rul[:, column]
    return result


def choose_blend(base: np.ndarray, intercell: np.ndarray, truth: np.ndarray,
                 maximum: np.ndarray, minimum_improvement: float) -> tuple[float, float, float]:
    base_mae = float(np.mean(np.abs(base - truth)))
    candidates = []
    for weight in BLEND_GRID:
        prediction = np.clip((1.0 - weight) * base + weight * intercell, 0.0, maximum)
        candidates.append((float(np.mean(np.abs(prediction - truth))), float(weight)))
    best_mae, best_weight = min(candidates)
    if base_mae - best_mae < minimum_improvement:
        return 0.0, base_mae, base_mae
    return best_weight, best_mae, base_mae


def train_intercell_select(train: FlatTarget, validation: FlatTarget,
                           train_bank: ReferenceBank, validation_bank: ReferenceBank,
                           state: VectorState, maximum_lifetime: float, seed: int,
                           config: IntercellConfig, device: torch.device,
                           checkpoint: Path) -> tuple[IntercellLifeHead, int, float, float]:
    seed_all(seed)
    model = IntercellLifeHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_arrays, _ = intercell_arrays(train, train_bank, state)
    loader = intercell_loader(train_arrays, config, seed + 19, True)
    best_state, best_epoch, best_intercell, stale = snapshot(model), 0, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw_batch in loader:
            batch = tuple(value.to(device) for value in raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = intercell_loss(model, batch, maximum_lifetime, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_intercell(
            model, validation, validation_bank, state, maximum_lifetime, config, device,
        )
        mae = float(np.mean(np.abs(prediction.intercell_rul - prediction.true_rul_cycles)))
        if mae < best_intercell - 1.0e-6:
            best_state, best_epoch, best_intercell, stale = snapshot(model), epoch, mae, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    prediction = predict_intercell(
        model, validation, validation_bank, state, maximum_lifetime, config, device,
    )
    weight, blend_mae, base_mae = choose_blend(
        prediction.base_rul.to_numpy(float), prediction.intercell_rul.to_numpy(float),
        prediction.true_rul_cycles.to_numpy(float),
        maximum_lifetime - prediction.time.to_numpy(float),
        config.minimum_blend_improvement,
    )
    torch.save({
        "model_state": best_state, "best_epoch": best_epoch,
        "selected_blend_weight": weight, "intercell_validation_mae": best_intercell,
        "blended_validation_mae": blend_mae, "base_validation_mae": base_mae,
        "configuration": asdict(config), "reference_domain": train_bank.domain,
        "reference_units": train_bank.unit_ids,
    }, checkpoint)
    return model, best_epoch, weight, blend_mae


def train_intercell_fixed(train: FlatTarget, bank: ReferenceBank, state: VectorState,
                          maximum_lifetime: float, seed: int, epochs: int,
                          config: IntercellConfig, device: torch.device) -> IntercellLifeHead:
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("Invalid inter-cell refit epoch count")
    seed_all(seed)
    model = IntercellLifeHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    arrays, _ = intercell_arrays(train, bank, state)
    loader = intercell_loader(arrays, config, seed + 19, True)
    for _epoch in range(epochs):
        model.train()
        for raw_batch in loader:
            batch = tuple(value.to(device) for value in raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss = intercell_loss(model, batch, maximum_lifetime, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def apply_blend(prediction: pd.DataFrame, weight: float,
                maximum_lifetime: float) -> pd.DataFrame:
    output = prediction.copy()
    cap = np.maximum(maximum_lifetime - output.time.to_numpy(float), 0.0)
    output["predicted_rul_raw"] = np.clip(
        (1.0 - weight) * output.base_rul.to_numpy(float)
        + weight * output.intercell_rul.to_numpy(float), 0.0, cap,
    )
    output["blend_weight"] = weight
    return output


def target_ssl_state(target: pd.DataFrame, features: list[str], config: Any,
                     seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], Any]:
    unlabeled = unlabeled_view(target, features)
    state = robust_fit(unlabeled, features, "Basilisk:outer_train_unlabeled")
    values, _, _ = make_windows(unlabeled, features, state, config.window, False)
    model = ssl_fit(DynamicsTCN(len(features), config), [("target", values)],
                    seed, config, device)
    return snapshot(model), state


def save_base_checkpoint(path: Path, model: nn.Module, state: Any, units: list[str],
                         features: list[str], config: Any, seed: int, fold: int,
                         epochs: int, stage: str) -> None:
    torch.save({
        "model_state": snapshot(model), "seed": seed, "outer_fold": fold,
        "target_units": units, "features": features, "configuration": asdict(config),
        "target_scaler_median": state.median, "target_scaler_iqr": state.iqr,
        "target_scaler_fit_units": list(map(str, state.fit_units)),
        "best_epoch": epochs, "training_stage": stage,
        "nasa_supervised_pretraining": False,
    }, path)


def fold_paths(output: Path, seed: int, fold: int) -> tuple[Path, Path]:
    return (output / f"fold_predictions_seed_{seed}_fold_{fold}.csv",
            output / f"fold_result_seed_{seed}_fold_{fold}.json")


def run_fold(seed: int, fold: int, source: pd.DataFrame, target: pd.DataFrame,
             target_knees: pd.DataFrame, features: list[str], reference: dict[str, Any],
             fold_map: dict[str, int], selection_seed: int,
             history_config: HistoryConfig, intercell_config: IntercellConfig,
             source_units: tuple[str, ...], smoke: bool, output: Path,
             device: torch.device) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_units = sorted(unit for unit, value in fold_map.items() if value == fold)
    train_units = sorted(set(fold_map) - set(test_units))
    inner_train_units, inner_validation_units = inner_split(
        train_units, target, selection_seed, fold, 0.20 if smoke else 0.10,
    )
    outer_train, outer_test = subset(target, train_units), subset(target, test_units)
    inner_train = subset(target, inner_train_units)
    inner_validation = subset(target, inner_validation_units)
    if set(train_units) & set(test_units):
        raise AssertionError("Outer split leakage")
    config = build_config(reference, 24, smoke)
    maximum_lifetime = maximum_lifetime_without_test(outer_train)
    run_seed = seed + 100 * fold
    ssl_state, target_state = target_ssl_state(
        outer_train, features, config, run_seed, device,
    )
    if set(map(str, target_state.fit_units)) != set(train_units):
        raise AssertionError("Target SSL/scaler did not use exactly outer-train devices")

    # Selection pass.
    selected_tcn, tcn_epoch, _, _ = train_configuration(
        "target_ssl", inner_train, inner_validation, features, target_state,
        ssl_state, maximum_lifetime, config, run_seed, device,
    )
    for parameter in selected_tcn.parameters():
        parameter.requires_grad = False
    selection_train_bundle = make_bundle(
        inner_train, target_knees, features, target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device, inner_train_units,
    )
    selection_validation_bundle = make_bundle(
        inner_validation, target_knees, features, target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device, inner_validation_units,
    )
    fit_statistics(selection_train_bundle, selection_validation_bundle)
    selection_head, head_epoch, base_inner_mae = train_variant(
        "B_stats", selection_train_bundle, selection_validation_bundle,
        maximum_lifetime, run_seed, history_config, device,
        output / f"checkpoint_selection_bstats_seed_{seed}_fold_{fold}.pt",
    )
    selection_train_prediction = predict_bundle(
        selection_head, selection_train_bundle, maximum_lifetime, device,
        history_config.batch_devices,
    )
    selection_validation_prediction = predict_bundle(
        selection_head, selection_validation_bundle, maximum_lifetime, device,
        history_config.batch_devices,
    )
    selection_train_flat = flat_target(selection_train_bundle, selection_train_prediction)
    selection_validation_flat = flat_target(
        selection_validation_bundle, selection_validation_prediction,
    )
    selection_vector_state = robust_vector_fit(
        selection_train_flat.representation,
        selection_train_flat.frame.unit_id.astype(str), "target_inner_train",
    )
    nasa_selection_bank = nasa_reference_bank(
        source, features, selected_tcn, config, source_units,
        intercell_config.progress_knots, device,
    )
    target_selection_units = choose_target_references(inner_train, 5)
    target_selection_bank = target_reference_bank(
        selection_train_bundle, selection_train_flat, selection_vector_state,
        target_selection_units, intercell_config.progress_knots,
    )
    selection_models: dict[str, tuple[int, float, float]] = {}
    for method, bank in (
        ("target_reference_control", target_selection_bank),
        ("nasa_intercell", nasa_selection_bank),
    ):
        _model, epoch, weight, validation_mae = train_intercell_select(
            selection_train_flat, selection_validation_flat, bank, bank,
            selection_vector_state, maximum_lifetime, run_seed,
            intercell_config, device,
            output / f"checkpoint_selection_{method}_seed_{seed}_fold_{fold}.pt",
        )
        selection_models[method] = (epoch, weight, validation_mae)

    # Fixed-epoch refit on all 256 outer-training devices.
    refit_tcn = fit_multitask_fixed_epochs(
        ssl_state, outer_train, features, target_state, maximum_lifetime,
        config, run_seed, tcn_epoch, device,
    )
    for parameter in refit_tcn.parameters():
        parameter.requires_grad = False
    train_bundle = make_bundle(
        outer_train, target_knees, features, target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device, train_units,
    )
    test_bundle = make_bundle(
        outer_test, target_knees, features, target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device, test_units,
    )
    stats_median, stats_iqr = fit_statistics(train_bundle, test_bundle)
    refit_head = fit_bstats_fixed_epochs(
        train_bundle, maximum_lifetime, run_seed, head_epoch,
        history_config, device,
    )
    assert_causal(refit_head, test_bundle, maximum_lifetime, device)
    train_prediction = predict_bundle(
        refit_head, train_bundle, maximum_lifetime, device,
        history_config.batch_devices,
    )
    test_prediction = predict_bundle(
        refit_head, test_bundle, maximum_lifetime, device,
        history_config.batch_devices,
    )
    train_flat, test_flat = flat_target(train_bundle, train_prediction), flat_target(test_bundle, test_prediction)
    vector_state = robust_vector_fit(
        train_flat.representation, train_flat.frame.unit_id.astype(str), "target_outer_train",
    )
    nasa_bank = nasa_reference_bank(
        source, features, refit_tcn, config, source_units,
        intercell_config.progress_knots, device,
    )
    target_reference_units = choose_target_references(outer_train, 5)
    target_bank = target_reference_bank(
        train_bundle, train_flat, vector_state, target_reference_units,
        intercell_config.progress_knots,
    )
    prediction_frames: list[pd.DataFrame] = []
    base = test_flat.frame.copy()
    base["predicted_rul_raw"] = base.pop("base_rul")
    base["method"] = "target_ssl_bstats"
    base["blend_weight"] = 0.0
    prediction_frames.append(base)
    refit_rows: dict[str, Any] = {}
    for method, bank in (("target_reference_control", target_bank), ("nasa_intercell", nasa_bank)):
        epoch, weight, validation_mae = selection_models[method]
        model = train_intercell_fixed(
            train_flat, bank, vector_state, maximum_lifetime, run_seed,
            epoch, intercell_config, device,
        )
        prediction = predict_intercell(
            model, test_flat, bank, vector_state, maximum_lifetime,
            intercell_config, device,
        )
        prediction = apply_blend(prediction, weight, maximum_lifetime)
        prediction["method"] = method
        prediction_frames.append(prediction)
        torch.save({
            "model_state": snapshot(model), "method": method, "seed": seed,
            "outer_fold": fold, "refit_units": train_units,
            "outer_test_units": test_units, "selected_epoch": epoch,
            "selected_blend_weight": weight, "selection_validation_mae": validation_mae,
            "reference_units": bank.unit_ids, "reference_domain": bank.domain,
            "configuration": asdict(intercell_config),
            "target_vector_median": vector_state.median,
            "target_vector_iqr": vector_state.iqr,
            "sealed_accessed": False,
        }, output / f"checkpoint_refit_{method}_seed_{seed}_fold_{fold}.pt")
        refit_rows[method] = {
            "selection_epoch": epoch, "blend_weight": weight,
            "selection_validation_mae": validation_mae,
            "outer_rul_mae": regression_metrics(prediction, "predicted_rul_raw")["rul_mae"],
        }
    save_base_checkpoint(
        output / f"checkpoint_refit_targetssl_tcn_seed_{seed}_fold_{fold}.pt",
        refit_tcn, target_state, train_units, features, config, seed, fold,
        tcn_epoch, "outer_train_fixed_epoch_refit",
    )
    torch.save({
        "model_state": snapshot(refit_head), "variant": "B_stats",
        "seed": seed, "outer_fold": fold, "refit_units": train_units,
        "outer_test_units": test_units, "selected_epoch": head_epoch,
        "history_configuration": asdict(history_config),
        "statistics_median": stats_median, "statistics_iqr": stats_iqr,
    }, output / f"checkpoint_refit_targetssl_bstats_seed_{seed}_fold_{fold}.pt")
    nasa_bank.metadata.to_csv(
        output / f"nasa_reference_nodes_seed_{seed}_fold_{fold}.csv", index=False,
    )
    result = {
        "seed": seed, "outer_fold": fold,
        "outer_train_devices": len(train_units), "outer_test_devices": len(test_units),
        "selection_inner_train_devices": len(inner_train_units),
        "selection_inner_validation_devices": len(inner_validation_units),
        "tcn_selected_epoch": tcn_epoch, "bstats_selected_epoch": head_epoch,
        "base_inner_validation_mae": base_inner_mae,
        "target_reference_units": target_reference_units,
        "source_units": list(source_units), "methods": refit_rows,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
    }
    combined = pd.concat(prediction_frames, ignore_index=True, sort=False)
    combined.insert(0, "outer_fold", fold)
    combined.insert(0, "seed", seed)
    return combined, result


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "unit_id", "time", "outer_fold"]
    label_columns = ["target_soh", "true_rul_cycles", "true_eol_cycle"]
    grouped = predictions.groupby(keys, sort=False)
    for column in label_columns:
        spread = grouped[column].agg(lambda values: float(values.max() - values.min()))
        if float(spread.max()) > 1.0e-8:
            raise ValueError(f"Inconsistent {column} across ensemble seeds")
    output = predictions.groupby(keys, as_index=False).agg(
        target_soh=("target_soh", "first"),
        true_rul_cycles=("true_rul_cycles", "first"),
        true_eol_cycle=("true_eol_cycle", "first"),
        predicted_soh=("predicted_soh", "mean"),
        predicted_rul_raw=("predicted_rul_raw", "mean"),
        intercell_rul=("intercell_rul", "mean"),
        reference_prediction_std=("reference_prediction_std", "mean"),
        blend_weight=("blend_weight", "mean"),
        seed_count=("seed", "nunique"),
    )
    return output


def per_device_results(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, unit), group in ensemble.groupby(["method", "unit_id"], sort=True):
        rows.append({"method": method, "unit_id": unit, "samples": len(group),
                     "true_eol_cycle": float(group.true_eol_cycle.max()),
                     **regression_metrics(group, "predicted_rul_raw"), **soh_metrics(group)})
    return pd.DataFrame(rows)


def metric_rows(predictions: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    group_columns = ["method"] if level == "ensemble" else ["method", "seed"]
    for key, group in predictions.groupby(group_columns, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_columns, values)}
        row.update({"samples": len(group), "devices": int(group.unit_id.nunique()),
                    **regression_metrics(group, "predicted_rul_raw"), **soh_metrics(group)})
        rows.append(row)
    return pd.DataFrame(rows)


def source_reference_ablation(predictions: pd.DataFrame,
                              source_units: tuple[str, ...]) -> pd.DataFrame:
    """Score each NASA reference and each leave-one-source-out median.

    This is an inference ablation of the frozen five-reference model, not a
    retraining ablation.  The fold-selected blend weight is retained so the
    comparison changes only which synchronized NASA predictions are pooled.
    """
    nasa = predictions.loc[predictions.method.eq("nasa_intercell")].copy()
    reference_columns = [f"reference_rul__{unit}" for unit in source_units]
    missing = sorted(set(reference_columns) - set(nasa.columns))
    if missing:
        raise ValueError(f"Missing per-source NASA predictions: {missing}")
    candidates: list[pd.DataFrame] = []
    definitions: list[tuple[str, str, np.ndarray]] = []
    for unit, column in zip(source_units, reference_columns):
        definitions.append(("single_reference", unit, nasa[column].to_numpy(float)))
        remaining = [name for name in reference_columns if name != column]
        definitions.append((
            "leave_one_source_out", unit,
            np.nanmedian(nasa[remaining].to_numpy(float), axis=1),
        ))
    for mode, unit, intercell_rul in definitions:
        frame = nasa[[
            "seed", "unit_id", "time", "target_soh", "true_rul_cycles",
            "true_eol_cycle", "outer_fold", "predicted_soh",
        ]].copy()
        frame["predicted_rul_raw"] = (
            (1.0 - nasa.blend_weight.to_numpy(float)) * nasa.base_rul.to_numpy(float)
            + nasa.blend_weight.to_numpy(float) * intercell_rul
        )
        frame["ablation_mode"] = mode
        frame["source_unit"] = unit
        candidates.append(frame)
    by_seed = pd.concat(candidates, ignore_index=True)
    keys = [
        "ablation_mode", "source_unit", "unit_id", "time", "target_soh",
        "true_rul_cycles", "true_eol_cycle", "outer_fold",
    ]
    ensemble = by_seed.groupby(keys, as_index=False).agg(
        predicted_soh=("predicted_soh", "mean"),
        predicted_rul_raw=("predicted_rul_raw", "mean"),
        seed_count=("seed", "nunique"),
    )
    rows: list[dict[str, Any]] = []
    for (mode, unit), group in ensemble.groupby(
        ["ablation_mode", "source_unit"], sort=True,
    ):
        rows.append({
            "ablation_mode": mode, "source_unit": unit,
            "samples": len(group), "devices": int(group.unit_id.nunique()),
            **regression_metrics(group, "predicted_rul_raw"),
        })
    return pd.DataFrame(rows)


def paired_device_bootstrap(ensemble: pd.DataFrame, candidate: str, baseline: str,
                            draws: int = 10000, seed: int = 42) -> dict[str, float]:
    keys = ["unit_id", "time"]
    left = ensemble.loc[ensemble.method.eq(candidate), [*keys, "true_rul_cycles", "predicted_rul_raw"]].rename(
        columns={"predicted_rul_raw": "candidate"},
    )
    right = ensemble.loc[ensemble.method.eq(baseline), [*keys, "predicted_rul_raw"]].rename(
        columns={"predicted_rul_raw": "baseline"},
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["candidate_error"] = np.abs(paired.candidate - paired.true_rul_cycles)
    paired["baseline_error"] = np.abs(paired.baseline - paired.true_rul_cycles)
    device = paired.groupby("unit_id").agg(
        candidate_sum=("candidate_error", "sum"), baseline_sum=("baseline_error", "sum"),
        count=("candidate_error", "size"),
    )
    generator = np.random.default_rng(seed)
    indices = np.arange(len(device))
    differences = np.empty(draws, float)
    for draw in range(draws):
        chosen = generator.choice(indices, len(indices), replace=True)
        count = device["count"].to_numpy(float)[chosen].sum()
        differences[draw] = (
            device.candidate_sum.to_numpy(float)[chosen].sum()
            - device.baseline_sum.to_numpy(float)[chosen].sum()
        ) / count
    return {
        "paired_bootstrap_draws": draws,
        "mae_difference_candidate_minus_baseline": float(
            paired.candidate_error.mean() - paired.baseline_error.mean()
        ),
        "mae_difference_p025": float(np.quantile(differences, 0.025)),
        "mae_difference_p50": float(np.quantile(differences, 0.50)),
        "mae_difference_p975": float(np.quantile(differences, 0.975)),
    }


def difficult_metrics(frame: pd.DataFrame) -> dict[str, float]:
    early = frame.loc[frame.time.le(15.0)]
    far = frame.loc[frame.true_rul_cycles.gt(60.0)]
    life = frame.groupby("unit_id").true_eol_cycle.max().sort_values()
    cutoff = float(life.quantile(0.75))
    long = frame.loc[frame.unit_id.isin(life.loc[life.ge(cutoff)].index)]
    return {
        "early_7_15_mae": regression_metrics(early, "predicted_rul_raw")["rul_mae"],
        "far_rul_gt60_mae": regression_metrics(far, "predicted_rul_raw")["rul_mae"],
        "long_life_bias": regression_metrics(long, "predicted_rul_raw")["rul_bias"],
    }


def acceptance_report(predictions: pd.DataFrame, ensemble: pd.DataFrame,
                      fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_name, candidate_name = "target_ssl_bstats", "nasa_intercell"
    metrics = metric_rows(ensemble, "ensemble").set_index("method")
    baseline, candidate = metrics.loc[baseline_name], metrics.loc[candidate_name]
    gain = float((baseline.rul_mae - candidate.rul_mae) / baseline.rul_mae)
    bootstrap = paired_device_bootstrap(ensemble, candidate_name, baseline_name)
    fold_positive = 0
    for fold in sorted(ensemble.outer_fold.unique()):
        group = ensemble.loc[ensemble.outer_fold.eq(fold)]
        values = metric_rows(group, "ensemble").set_index("method")
        fold_positive += int(values.loc[candidate_name].rul_mae < values.loc[baseline_name].rul_mae)
    seed_metrics = metric_rows(predictions, "seed")
    pivot = seed_metrics.pivot(index="seed", columns="method", values="rul_mae")
    seed_positive = int((pivot[candidate_name] < pivot[baseline_name]).sum())
    baseline_difficult = difficult_metrics(ensemble.loc[ensemble.method.eq(baseline_name)])
    candidate_difficult = difficult_metrics(ensemble.loc[ensemble.method.eq(candidate_name)])
    early_gain = ((baseline_difficult["early_7_15_mae"] - candidate_difficult["early_7_15_mae"])
                  / baseline_difficult["early_7_15_mae"])
    far_gain = ((baseline_difficult["far_rul_gt60_mae"] - candidate_difficult["far_rul_gt60_mae"])
                / baseline_difficult["far_rul_gt60_mae"])
    long_bias_gain = ((abs(baseline_difficult["long_life_bias"])
                       - abs(candidate_difficult["long_life_bias"]))
                      / max(abs(baseline_difficult["long_life_bias"]), 1.0e-9))
    control_mae = float(metrics.loc["target_reference_control"].rul_mae)
    blend_weights = [
        float(row["methods"][candidate_name]["blend_weight"]) for row in fold_results
    ]
    conditions = {
        "rul_gain_at_least_5_percent": gain >= 0.05,
        "paired_bootstrap_upper_below_zero": bootstrap["mae_difference_p975"] < 0.0,
        "at_least_4_of_5_folds_positive": fold_positive >= 4,
        "at_least_2_of_3_seeds_positive": seed_positive >= 2,
        "early_mae_gain_at_least_8_percent": early_gain >= 0.08,
        "far_rul_mae_gain_at_least_8_percent": far_gain >= 0.08,
        "long_life_bias_gain_at_least_20_percent": long_bias_gain >= 0.20,
        "worst_device_not_worse_than_10_percent": candidate.worst_device_mae <= 1.10 * baseline.worst_device_mae,
        "soh_mae_not_worse_than_1_percent": candidate.soh_mae <= 1.01 * baseline.soh_mae,
        "nasa_beats_same_parameter_target_control": candidate.rul_mae < control_mae,
        "all_nasa_blend_weights_positive": all(weight > 0.0 for weight in blend_weights),
    }
    conditions = {name: bool(value) for name, value in conditions.items()}
    return {
        "stable_significant_positive_transfer": bool(all(conditions.values())),
        "baseline_rul_mae": float(baseline.rul_mae),
        "candidate_rul_mae": float(candidate.rul_mae),
        "relative_rul_gain": gain, "positive_fold_count": fold_positive,
        "positive_seed_count": seed_positive, "early_relative_gain": early_gain,
        "far_rul_relative_gain": far_gain, "long_life_bias_relative_gain": long_bias_gain,
        "target_control_rul_mae": control_mae, "nasa_blend_weights": blend_weights,
        "conditions": conditions, **bootstrap,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if requested_methods != METHODS:
        raise ValueError(
            "This frozen comparison must run all methods in order: " + ",".join(METHODS)
        )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    source, target, _validation, features, report, target_knees, _ = prepare_data(args.data_root)
    del report
    if len(features) != 16 or any(
        any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS) for name in features
    ):
        raise ValueError("Inter-cell transfer requires the frozen leak-free 16-feature contract")
    source_units = tuple(value.strip() for value in args.source_units.split(",") if value.strip())
    if source_units != SOURCE_UNITS:
        raise ValueError(f"Formal source bank must be exactly {SOURCE_UNITS}")
    seeds = tuple(int(value) for value in args.formal_seeds.split(","))
    if args.smoke:
        keep = sorted(target.unit_id.astype(str).unique())[:args.smoke_units]
        target = subset(target, keep)
        target_knees = target_knees.loc[target_knees.unit_id.astype(str).isin(keep)].copy()
        folds, seeds = 2, (42,)
    else:
        folds = args.outer_folds
        if folds != 5 or seeds != FORMAL_SEEDS:
            raise ValueError("Formal experiment requires five folds and seeds 42,43,44")
    fold_map = stratified_outer_folds(target, folds)
    history_config = HistoryConfig(
        epochs=3 if args.smoke else args.history_epochs,
        patience=2 if args.smoke else args.patience,
        batch_devices=args.batch_devices,
    )
    intercell_config = IntercellConfig(
        progress_knots=args.progress_knots,
        epochs=3 if args.smoke else args.intercell_epochs,
        patience=2 if args.smoke else args.patience,
        batch_size=args.intercell_batch_size,
    )
    prediction_frames: list[pd.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in range(folds):
            prediction_path, result_path = fold_paths(args.output_dir, seed, fold)
            if args.resume and prediction_path.exists() and result_path.exists():
                prediction = pd.read_csv(prediction_path)
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                prediction, result = run_fold(
                    seed, fold, source, target, target_knees, features, reference,
                    fold_map, args.selection_seed, history_config, intercell_config,
                    source_units, args.smoke, args.output_dir, device,
                )
                prediction.to_csv(prediction_path, index=False)
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            prediction_frames.append(prediction)
            fold_results.append(result)
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                args.output_dir / "oof_predictions_by_seed.csv", index=False,
            )
            print(json.dumps({"seed": seed, "outer_fold": fold,
                              "methods": result["methods"]}), flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not args.smoke:
        sequential = pd.read_csv(args.sequential_run / "oof_predictions_by_seed.csv")
        sequential["method"] = "sequential_nasa_bstats"
        sequential["blend_weight"] = 0.0
        sequential["intercell_rul"] = np.nan
        sequential["reference_prediction_std"] = np.nan
        predictions = pd.concat([predictions, sequential], ignore_index=True, sort=False)
    ensemble = ensemble_predictions(predictions)
    metrics = metric_rows(ensemble, "ensemble")
    seed_metrics = metric_rows(predictions, "seed")
    device_metrics = per_device_results(ensemble)
    source_ablation = source_reference_ablation(predictions, source_units)
    predictions.to_csv(args.output_dir / "oof_predictions_by_seed.csv", index=False)
    ensemble.to_csv(args.output_dir / "oof_ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "oof_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "oof_metrics_by_seed.csv", index=False)
    device_metrics.to_csv(args.output_dir / "oof_results_per_device.csv", index=False)
    source_ablation.to_csv(
        args.output_dir / "source_reference_ablation_metrics.csv", index=False,
    )
    pd.DataFrame([{
        "seed": row["seed"], "outer_fold": row["outer_fold"],
        "outer_train_devices": row["outer_train_devices"],
        "outer_test_devices": row["outer_test_devices"],
        "tcn_selected_epoch": row["tcn_selected_epoch"],
        "bstats_selected_epoch": row["bstats_selected_epoch"],
        "target_control_epoch": row["methods"]["target_reference_control"]["selection_epoch"],
        "target_control_blend": row["methods"]["target_reference_control"]["blend_weight"],
        "nasa_intercell_epoch": row["methods"]["nasa_intercell"]["selection_epoch"],
        "nasa_intercell_blend": row["methods"]["nasa_intercell"]["blend_weight"],
    } for row in fold_results]).to_csv(args.output_dir / "selection_and_refit_settings.csv", index=False)
    if args.smoke:
        audit = {"passed": True, "mode": "smoke", "devices": int(ensemble.unit_id.nunique()),
                 "sealed_features_accessed": False, "sealed_labels_accessed": False}
    else:
        audit = acceptance_report(predictions, ensemble, fold_results)
        expected_samples = len(target.loc[target.time.ge(7.0)])
        for method in ("target_ssl_bstats", "target_reference_control", "nasa_intercell"):
            group = predictions.loc[predictions.method.eq(method)]
            if len(group) != expected_samples * len(seeds):
                raise AssertionError(f"Incomplete OOF coverage for {method}")
            if group.groupby(["seed", "unit_id", "time"]).size().ne(1).any():
                raise AssertionError(f"Duplicate OOF keys for {method}")
        for row in fold_results:
            if row["outer_train_devices"] != 256 or row["outer_test_devices"] != 64:
                raise AssertionError("Formal device counts changed")
            if set(row["target_scaler_fit_units"]) & set(
                unit for unit, fold in fold_map.items() if fold == row["outer_fold"]
            ):
                raise AssertionError("Outer-test device entered target scaler")
        audit.update({
            "passed_structural_audit": True, "outer_folds": folds,
            "formal_seeds": list(seeds), "devices": 320,
            "samples_per_method": expected_samples,
            "source_units": list(source_units),
            "target_ssl_uses_outer_train_unlabeled_only": True,
            "future_target_labels_are_inputs": False,
            "sealed_features_accessed": False, "sealed_labels_accessed": False,
        })
    (args.output_dir / "intercell_transfer_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8",
    )
    conclusion = (
        "stable significant positive transfer"
        if audit.get("stable_significant_positive_transfer", False)
        else "existing five NASA batteries do not provide stable significant full-label transfer"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "V1.0 synchronized explicit inter-cell RUL transfer",
        "mode": "smoke" if args.smoke else "formal",
        "data_version": "Basilisk V1.0 unchanged",
        "features": features, "feature_count": len(features),
        "local_tcn_window": 24, "history_statistics": 38,
        "outer_folds": folds, "seeds": list(seeds),
        "source_units": list(source_units),
        "source_progress_definition": "(SOH0-SOHt)/(SOH0-0.80)",
        "reference_aggregation": "median over five synchronized references",
        "blend_grid": list(BLEND_GRID),
        "history_configuration": asdict(history_config),
        "intercell_configuration": asdict(intercell_config),
        "conclusion": conclusion, "audit": audit,
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
    parser.add_argument("--sequential-run", type=Path, default=DEFAULT_SEQUENTIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="Frozen documented interface; all four methods are always reported.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--formal-seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--selection-seed", type=int, default=41)
    parser.add_argument("--source-units", default=",".join(SOURCE_UNITS))
    parser.add_argument("--progress-knots", type=int, default=32)
    parser.add_argument("--history-epochs", type=int, default=45)
    parser.add_argument("--intercell-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-devices", type=int, default=32)
    parser.add_argument("--intercell-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-units", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
