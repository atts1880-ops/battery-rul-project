"""NASA multi-horizon degradation-dynamics pretraining for Basilisk V1.0.

Only source-domain labels are used here.  Inputs are causal TCN windows ending
at the current source cycle; future source SOH is used solely to construct the
training targets ΔSOH@1/4/8 EFC and a local degradation-rate target.  The
auxiliary heads are discarded after pretraining and only the existing
DynamicsTCN state is transferred to target adaptation.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_positive_transfer import RobustState, seed_all, snapshot
from train_bhump_v10_rul_multitask import DynamicsTCN


@dataclass(frozen=True)
class SourceDynamicsConfig:
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    soh_weight: float = 1.0
    delta_weight: float = 0.5
    rate_weight: float = 0.2
    consistency_weight: float = 0.1
    acceleration_weight: float = 0.1
    monotonic_weight: float = 0.1
    huber_beta_scaled: float = 0.05


@dataclass
class SourceDynamicsArrays:
    x: np.ndarray
    current_soh: np.ndarray
    deltas: np.ndarray
    delta_mask: np.ndarray
    rate: np.ndarray
    rate_mask: np.ndarray
    acceleration: np.ndarray
    acceleration_mask: np.ndarray
    weights: np.ndarray
    metadata: pd.DataFrame


class SourceDynamicsTrainer(nn.Module):
    """Training-only heads around the unchanged deployable DynamicsTCN."""

    def __init__(self, base: DynamicsTCN, horizons: Iterable[int]) -> None:
        super().__init__()
        self.base = base
        projection = base.soh_head.in_features
        self.horizons = tuple(int(value) for value in horizons)
        self.delta_heads = nn.ModuleList(nn.Linear(projection, 1) for _ in self.horizons)
        self.rate_head = nn.Linear(projection, 1)
        self.acceleration_head = nn.Linear(projection, 1)
        for head in (*self.delta_heads, self.rate_head, self.acceleration_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        soh, shared, _private = self.base(values, "source")
        # Capacity recovery/noise can make a short-horizon delta slightly negative.
        deltas = torch.stack(
            [0.10 * torch.tanh(head(shared).squeeze(-1)) for head in self.delta_heads],
            dim=1,
        )
        # Predict rate in log space so the same head can cover short- and
        # long-life cells without collapsing small positive rates to zero.
        log_rate = torch.clamp(self.rate_head(shared).squeeze(-1), -12.0, -2.0)
        rate = torch.exp(log_rate)
        acceleration = 0.01 * torch.tanh(
            self.acceleration_head(shared).squeeze(-1)
        )
        return {
            "soh": soh, "deltas": deltas, "rate": rate,
            "log_rate": log_rate, "acceleration": acceleration,
        }


def source_robust_fit(frame: pd.DataFrame, features: list[str]) -> RobustState:
    leaked = [
        name for name in features
        if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)
    ]
    if leaked:
        raise ValueError(f"Forbidden NASA dynamics inputs: {leaked}")
    values = frame[features].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite NASA dynamics input")
    median = np.median(values, axis=0)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    iqr[iqr < 1.0e-8] = 1.0
    return RobustState(
        median.astype(float), iqr.astype(float), "NASA5:dynamics_pretrain",
        tuple(sorted(frame.unit_id.astype(str).unique())),
    )


def _future_index(times: np.ndarray, current_index: int, horizon: float) -> int | None:
    target = float(times[current_index] + horizon)
    position = int(np.searchsorted(times, target, side="left"))
    if position < len(times) and abs(float(times[position]) - target) <= 1.0e-6:
        return position
    return None


def build_source_dynamics_arrays(
    source: pd.DataFrame, features: list[str], state: RobustState,
    window: int, horizons: tuple[int, ...] = (1, 4, 8, 16),
) -> SourceDynamicsArrays:
    """Build causal inputs and source-only future-dynamics supervision."""
    if tuple(sorted(horizons)) != tuple(horizons) or any(value <= 0 for value in horizons):
        raise ValueError("Horizons must be unique positive increasing integers")
    required = {"unit_id", "time", "target_soh", *features}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"NASA dynamics source is missing {sorted(missing)}")
    xs: list[np.ndarray] = []
    current_soh: list[float] = []
    deltas: list[list[float]] = []
    masks: list[list[float]] = []
    rates: list[float] = []
    rate_masks: list[float] = []
    accelerations: list[float] = []
    acceleration_masks: list[float] = []
    metadata: list[dict[str, Any]] = []

    ordered_source = source.sort_values(["unit_id", "time"])
    for unit_id, unit in ordered_source.groupby("unit_id", sort=True):
        unit = unit.reset_index(drop=True)
        times = unit.time.to_numpy(float)
        soh = unit.target_soh.to_numpy(float)
        first_end = min(7, window - 1)
        if len(unit) < first_end + 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError(f"NASA unit {unit_id} lacks ordered dynamics support")
        values = state.transform(unit[features].to_numpy(float))
        for end in range(first_end, len(unit) - 1):
            row_delta: list[float] = []
            row_mask: list[float] = []
            future_indices: list[int] = []
            for horizon in horizons:
                future = _future_index(times, end, horizon)
                if future is None:
                    row_delta.append(0.0)
                    row_mask.append(0.0)
                else:
                    row_delta.append(float(soh[end] - soh[future]))
                    row_mask.append(1.0)
                    future_indices.append(future)
            if not future_indices:
                continue
            furthest = max(future_indices)
            local_time = times[end:furthest + 1] - times[end]
            local_soh = soh[end:furthest + 1]
            if len(local_time) >= 2 and float(np.ptp(local_time)) > 0.0:
                slope = float(np.polyfit(local_time, local_soh, 1)[0])
                rate = max(-slope, 1.0e-8)
                rate_mask = 1.0
            else:
                rate, rate_mask = 0.0, 0.0
            # Difference between recent and forward rates is a causal-input,
            # future-label acceleration target. It is never part of x.
            recent_start = max(0, end - max(horizons))
            recent_time = times[recent_start:end + 1]
            recent_soh = soh[recent_start:end + 1]
            if (
                rate_mask and len(recent_time) >= 2
                and float(np.ptp(recent_time)) > 0.0
            ):
                recent_rate = max(
                    -float(np.polyfit(recent_time, recent_soh, 1)[0]), 1.0e-8,
                )
                acceleration = (rate - recent_rate) / max(
                    float(local_time[-1]), 1.0,
                )
                acceleration_mask = 1.0
            else:
                acceleration, acceleration_mask = 0.0, 0.0
            start = end - window + 1
            if start < 0:
                window_values = np.concatenate([
                    np.repeat(values[[0]], -start, axis=0), values[:end + 1],
                ], axis=0)
            else:
                window_values = values[start:end + 1]
            if len(window_values) != window:
                raise RuntimeError("NASA causal source window has unexpected length")
            xs.append(window_values)
            current_soh.append(float(soh[end]))
            deltas.append(row_delta)
            masks.append(row_mask)
            rates.append(rate)
            rate_masks.append(rate_mask)
            accelerations.append(acceleration)
            acceleration_masks.append(acceleration_mask)
            metadata.append({
                "unit_id": str(unit_id), "time": float(times[end]),
                "current_soh": float(soh[end]),
            })

    if not xs:
        raise ValueError("No NASA multi-horizon dynamics samples were built")
    meta = pd.DataFrame(metadata)
    counts = meta.unit_id.value_counts()
    weights = np.asarray([1.0 / counts[unit] for unit in meta.unit_id], np.float32)
    weights /= float(weights.mean())
    arrays = SourceDynamicsArrays(
        np.asarray(xs, np.float32), np.asarray(current_soh, np.float32),
        np.asarray(deltas, np.float32), np.asarray(masks, np.float32),
        np.asarray(rates, np.float32), np.asarray(rate_masks, np.float32),
        np.asarray(accelerations, np.float32),
        np.asarray(acceleration_masks, np.float32),
        weights, meta,
    )
    numeric = [arrays.x, arrays.current_soh, arrays.deltas, arrays.delta_mask,
               arrays.rate, arrays.rate_mask, arrays.acceleration,
               arrays.acceleration_mask, arrays.weights]
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("Non-finite NASA dynamics training arrays")
    return arrays


def _masked_huber(prediction: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor, beta: float) -> torch.Tensor:
    point = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return (point * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def dynamics_pretrain_loss(
    outputs: dict[str, torch.Tensor], current_soh: torch.Tensor,
    deltas: torch.Tensor, delta_mask: torch.Tensor, rate: torch.Tensor,
    rate_mask: torch.Tensor, acceleration: torch.Tensor,
    acceleration_mask: torch.Tensor, weights: torch.Tensor,
    config: SourceDynamicsConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    soh_point = F.smooth_l1_loss(
        outputs["soh"], current_soh, beta=0.01, reduction="none"
    )
    soh_loss = (soh_point * weights).mean()
    expanded_weight = weights[:, None]
    scaled_mask = delta_mask * expanded_weight
    delta_loss = _masked_huber(
        100.0 * outputs["deltas"], 100.0 * deltas,
        scaled_mask, config.huber_beta_scaled,
    )
    rate_loss = _masked_huber(
        outputs["log_rate"], torch.log(torch.clamp(rate, min=1.0e-8)),
        rate_mask * weights, config.huber_beta_scaled,
    )
    acceleration_loss = _masked_huber(
        1000.0 * outputs["acceleration"], 1000.0 * acceleration,
        acceleration_mask * weights, config.huber_beta_scaled,
    )
    horizons = torch.tensor(
        config.horizons, dtype=outputs["deltas"].dtype,
        device=outputs["deltas"].device,
    )[None, :]
    implied_rate = outputs["deltas"] / horizons
    consistency = _masked_huber(
        100.0 * implied_rate, 100.0 * outputs["rate"][:, None].expand_as(implied_rate),
        scaled_mask, config.huber_beta_scaled,
    )
    monotonic = torch.relu(
        outputs["deltas"][:, :-1] - outputs["deltas"][:, 1:]
    )
    pair_mask = scaled_mask[:, :-1] * scaled_mask[:, 1:]
    monotonic_loss = (monotonic * pair_mask).sum() / torch.clamp(
        pair_mask.sum(), min=1.0,
    )
    parts = {
        "soh": soh_loss, "delta": delta_loss,
        "rate": rate_loss, "acceleration": acceleration_loss,
        "consistency": consistency, "monotonic": monotonic_loss,
    }
    total = (
        config.soh_weight * soh_loss
        + config.delta_weight * delta_loss
        + config.rate_weight * rate_loss
        + config.acceleration_weight * acceleration_loss
        + config.consistency_weight * consistency
        + config.monotonic_weight * monotonic_loss
    )
    return total, parts


def fit_source_dynamics(
    initial_state: dict[str, torch.Tensor], arrays: SourceDynamicsArrays,
    input_size: int, model_config: Any, seed: int,
    config: SourceDynamicsConfig, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_all(seed)
    base = DynamicsTCN(input_size, model_config)
    base.load_state_dict(copy.deepcopy(initial_state))
    base.reset_soh_head()
    trainer = SourceDynamicsTrainer(base, config.horizons).to(device)
    tensors = (
        torch.from_numpy(arrays.x), torch.from_numpy(arrays.current_soh),
        torch.from_numpy(arrays.deltas), torch.from_numpy(arrays.delta_mask),
        torch.from_numpy(arrays.rate), torch.from_numpy(arrays.rate_mask),
        torch.from_numpy(arrays.acceleration),
        torch.from_numpy(arrays.acceleration_mask),
        torch.from_numpy(arrays.weights),
    )
    batches = DataLoader(
        TensorDataset(*tensors), batch_size=min(config.batch_size, len(arrays.x)),
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    parameters = [
        *trainer.base.encoder.parameters(), *trainer.base.shared_projection.parameters(),
        *trainer.base.soh_head.parameters(), *trainer.delta_heads.parameters(),
        *trainer.rate_head.parameters(), *trainer.acceleration_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    final_parts: dict[str, float] = {}
    for _epoch in range(config.epochs):
        trainer.train()
        for batch in batches:
            (values, soh, delta, mask, rate, rate_mask, acceleration,
             acceleration_mask, weight) = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = trainer(values)
            loss, parts = dynamics_pretrain_loss(
                outputs, soh, delta, mask, rate, rate_mask, acceleration,
                acceleration_mask, weight, config,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            final_parts = {name: float(value.detach().cpu()) for name, value in parts.items()}

    trainer.eval()
    predictions: dict[str, list[np.ndarray]] = {
        "soh": [], "deltas": [], "rate": [], "acceleration": [],
    }
    with torch.no_grad():
        for start in range(0, len(arrays.x), config.batch_size):
            values = torch.from_numpy(arrays.x[start:start + config.batch_size]).to(device)
            output = trainer(values)
            for name in predictions:
                predictions[name].append(output[name].cpu().numpy())
    predicted_soh = np.concatenate(predictions["soh"])
    predicted_deltas = np.concatenate(predictions["deltas"])
    predicted_rate = np.concatenate(predictions["rate"])
    predicted_acceleration = np.concatenate(predictions["acceleration"])
    delta_metrics = {}
    for index, horizon in enumerate(config.horizons):
        valid = arrays.delta_mask[:, index].astype(bool)
        delta_metrics[f"delta_{horizon}_mae"] = float(np.mean(np.abs(
            predicted_deltas[valid, index] - arrays.deltas[valid, index]
        )))
    valid_rate = arrays.rate_mask.astype(bool)
    valid_acceleration = arrays.acceleration_mask.astype(bool)
    audit = {
        "source_samples": int(len(arrays.x)),
        "source_units": sorted(arrays.metadata.unit_id.unique().tolist()),
        "horizons": list(config.horizons),
        "soh_mae": float(np.mean(np.abs(predicted_soh - arrays.current_soh))),
        "rate_mae": float(np.mean(np.abs(predicted_rate[valid_rate] - arrays.rate[valid_rate]))),
        "acceleration_mae": float(np.mean(np.abs(
            predicted_acceleration[valid_acceleration]
            - arrays.acceleration[valid_acceleration]
        ))),
        "delta_metrics": delta_metrics,
        "final_loss_parts": final_parts,
        "training_configuration": asdict(config),
        "auxiliary_heads_discarded": True,
        "retained_head_state": {
            name: value.detach().cpu().tolist()
            for name, value in {
                **{f"delta_heads.{name}": value for name, value in trainer.delta_heads.state_dict().items()},
                **{f"rate_head.{name}": value for name, value in trainer.rate_head.state_dict().items()},
                **{f"acceleration_head.{name}": value for name, value in trainer.acceleration_head.state_dict().items()},
            }.items()
        },
        "target_labels_used": False,
        "sealed_accessed": False,
    }
    return snapshot(trainer.base), audit


def nasa_dynamics_initial_state(
    source: pd.DataFrame, target_train: pd.DataFrame, features: list[str],
    ssl_state: dict[str, torch.Tensor], model_config: Any, seed: int,
    device: torch.device, audit_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Return a DynamicsTCN-compatible source dynamics initialization."""
    if set(target_train.unit_id.astype(str)) & set(source.unit_id.astype(str)):
        raise AssertionError("NASA and Basilisk device IDs overlap")
    source_state = source_robust_fit(source, features)
    training = SourceDynamicsConfig(
        epochs=2 if model_config.source_epochs <= 2 else model_config.source_epochs,
        batch_size=model_config.batch_size,
        learning_rate=model_config.source_learning_rate,
        weight_decay=model_config.weight_decay,
    )
    arrays = build_source_dynamics_arrays(
        source, features, source_state, model_config.window, training.horizons,
    )
    state, audit = fit_source_dynamics(
        ssl_state, arrays, len(features), model_config, seed + 1,
        training, device,
    )
    audit.update({
        "source_scaler_fit_units": list(source_state.fit_units),
        "target_outer_train_device_count": int(target_train.unit_id.nunique()),
        "input_feature_count": len(features),
        "input_features": features,
    })
    if audit_path is not None:
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return state
