"""Literature-guided positive-transfer experiment for NASA -> Basilisk SOH/RUL.

The implementation keeps the deployed compact TCN small while adding
training-only shared/private projections, masked reconstruction, temporal
order prediction, source selection, residual calibration and conditional MMD.
Target labels outside the selected few-shot devices are dropped before any
normalization or self-supervised training.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy.stats import ks_2samp, rankdata, theilslopes
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bhump_common import FORBIDDEN_INPUT_TOKENS, assert_feature_contract
from train import TcnEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_data"
DEFAULT_OUTPUT = ROOT / "bhump_positive_transfer_runs"
METHODS = (
    "target_only", "target_ssl", "selected_finetune", "selected_residual",
    "dr_tcn_conditional_mmd",
)
SOURCE_UNITS = ("B0005", "B0018", "B0033", "B0043", "B0044")
EXCLUDED_SOURCE_UNITS = ("B0030", "B0042", "B0038", "B0039")
LABEL_COLUMNS = ("target_soh", "true_rul_cycles", "true_eol_cycle")


@dataclass(frozen=True)
class Config:
    window: int = 8
    channels: tuple[int, ...] = (16, 24)
    kernel: int = 3
    dropout: float = 0.10
    projection: int = 16
    batch_size: int = 256
    ssl_epochs: int = 16
    source_epochs: int = 30
    adapt_epochs: int = 45
    patience: int = 7
    head_learning_rate: float = 5e-4
    encoder_learning_rate: float = 5e-5
    source_learning_rate: float = 3e-4
    ssl_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    huber_beta: float = 0.01
    mask_fraction: float = 0.15
    reconstruction_weight: float = 0.30
    order_weight: float = 0.10
    orthogonality_weight: float = 0.03
    alignment_weight: float = 0.003
    head_only_epochs: int = 5
    last_block_epochs: int = 10


@dataclass(frozen=True)
class RobustState:
    median: np.ndarray
    iqr: np.ndarray
    fit_domain: str
    fit_units: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.median) / self.iqr).astype(np.float32)


class PositiveTransferTCN(nn.Module):
    def __init__(self, input_size: int, config: Config) -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder = TcnEncoder(input_size, config.channels, config.kernel, config.dropout)
        self.shared_projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.projection), nn.SiLU(), nn.Dropout(config.dropout),
        )
        self.private_source_projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.projection), nn.SiLU(),
        )
        self.private_target_projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.projection), nn.SiLU(),
        )
        self.soh_head = nn.Linear(config.projection, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(config.projection, 8), nn.SiLU(), nn.Linear(8, 1),
        )
        self.reconstruction_head = nn.Linear(config.projection * 2, input_size)
        self.order_head = nn.Linear(config.projection, 1)
        self.reset_soh_head()
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode(self, values: torch.Tensor, domain: str = "target") -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)[:, -1]
        shared = self.shared_projection(encoded)
        private = (
            self.private_source_projection(encoded)
            if domain == "source" else self.private_target_projection(encoded)
        )
        return shared, private

    def forward(self, values: torch.Tensor, domain: str = "target") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared, private = self.encode(values, domain)
        prediction = 1.10 * torch.sigmoid(self.soh_head(shared).squeeze(-1))
        return prediction, shared, private

    def residual_prediction(self, values: torch.Tensor) -> torch.Tensor:
        base, shared, _ = self(values, "target")
        return torch.clamp(base + self.residual_head(shared).squeeze(-1), 0.0, 1.10)

    def reset_soh_head(self) -> None:
        nn.init.normal_(self.soh_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.soh_head.bias, 2.0)

    def deployment_parameters(self) -> Iterable[nn.Parameter]:
        return itertools.chain(self.encoder.parameters(), self.shared_projection.parameters(), self.soh_head.parameters())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def robust_fit(frame: pd.DataFrame, features: list[str], domain: str) -> RobustState:
    values = frame[features].to_numpy(float)
    median = np.median(values, axis=0)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    iqr[iqr < 1e-8] = 1.0
    return RobustState(median, iqr, domain, tuple(sorted(map(str, frame.unit_id.unique()))))


def unlabeled_view(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = frame[["unit_id", "time", *features]].copy()
    forbidden = [column for column in output if any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        raise ValueError(f"Unlabeled view retained forbidden columns: {forbidden}")
    return output


def make_windows(frame: pd.DataFrame, features: list[str], state: RobustState,
                 window: int, with_labels: bool) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[dict[str, Any]] = []
    metadata = ["unit_id", "time"] + ([*LABEL_COLUMNS] if with_labels else [])
    for _, unit in frame.sort_values(["unit_id", "time"]).groupby("unit_id", sort=True):
        unit = unit.reset_index(drop=True)
        values = state.transform(unit[features].to_numpy(float))
        for end in range(window - 1, len(unit)):
            xs.append(values[end - window + 1:end + 1])
            if with_labels:
                ys.append(float(unit.target_soh.iloc[end]))
            meta.append(unit.loc[end, metadata].to_dict())
    if not xs:
        raise ValueError("No causal windows could be built")
    labels = np.asarray(ys, np.float32) if with_labels else None
    return np.asarray(xs, np.float32), labels, pd.DataFrame(meta)


def loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, seed: int,
           weights: np.ndarray | None = None, shuffle: bool = True) -> DataLoader:
    tensors: list[torch.Tensor] = [torch.from_numpy(x)]
    if y is not None:
        tensors.append(torch.from_numpy(y))
    if weights is not None:
        tensors.append(torch.from_numpy(weights.astype(np.float32)))
    return DataLoader(
        TensorDataset(*tensors), batch_size=min(batch_size, len(x)), shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "soh_mae": float(np.mean(np.abs(error))),
        "soh_rmse": float(np.sqrt(np.mean(error**2))),
        "soh_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 1e-12 else 0.0,
        "soh_bias": float(np.mean(error)),
    }


def predict(model: PositiveTransferTCN, x: np.ndarray, config: Config, device: torch.device,
            residual: bool = False) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), config.batch_size):
            batch = torch.from_numpy(x[start:start + config.batch_size]).to(device)
            estimate = model.residual_prediction(batch) if residual else model(batch, "target")[0]
            values.append(estimate.cpu().numpy())
    return np.clip(np.concatenate(values), 0.0, 1.10)


def orthogonality_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    shared = shared - shared.mean(0, keepdim=True)
    private = private - private.mean(0, keepdim=True)
    shared = shared / (shared.norm(dim=0, keepdim=True) + 1e-6)
    private = private / (private.norm(dim=0, keepdim=True) + 1e-6)
    return (shared.T @ private).pow(2).mean()


def ssl_batch_loss(model: PositiveTransferTCN, values: torch.Tensor, domain: str,
                   config: Config, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    clean_last = values[:, -1].clone()
    mask = torch.rand(values.shape, device=values.device, generator=generator) < config.mask_fraction
    masked = values.masked_fill(mask, 0.0)
    shared, private = model.encode(masked, domain)
    reconstruction = model.reconstruction_head(torch.cat([shared, private], dim=1))
    reconstruction_loss = nn.functional.smooth_l1_loss(reconstruction, clean_last, beta=0.05)

    half = values.shape[1] // 2
    swapped = torch.cat([values[:, half:], values[:, :half]], dim=1)
    order_labels = torch.randint(0, 2, (len(values),), device=values.device, generator=generator)
    order_values = torch.where(order_labels[:, None, None].bool(), values, swapped)
    order_shared, _ = model.encode(order_values, domain)
    order_loss = nn.functional.binary_cross_entropy_with_logits(
        model.order_head(order_shared).squeeze(-1), order_labels.float(),
    )
    orthogonal = orthogonality_loss(shared, private)
    total = (
        config.reconstruction_weight * reconstruction_loss
        + config.order_weight * order_loss
        + config.orthogonality_weight * orthogonal
    )
    return total, {
        "reconstruction": float(reconstruction_loss.detach().cpu()),
        "order": float(order_loss.detach().cpu()),
        "orthogonality": float(orthogonal.detach().cpu()),
    }


def ssl_fit(model: PositiveTransferTCN, domains: list[tuple[str, np.ndarray]], seed: int,
            config: Config, device: torch.device) -> PositiveTransferTCN:
    seed_all(seed)
    model = model.to(device)
    parameters = itertools.chain(
        model.encoder.parameters(), model.shared_projection.parameters(),
        model.private_source_projection.parameters(), model.private_target_projection.parameters(),
        model.reconstruction_head.parameters(), model.order_head.parameters(),
    )
    optimizer = torch.optim.AdamW(parameters, lr=config.ssl_learning_rate, weight_decay=config.weight_decay)
    loaders = [loader(x, None, config.batch_size, seed + index * 31) for index, (_, x) in enumerate(domains)]
    steps = max(len(item) for item in loaders)
    generator = torch.Generator(device=device).manual_seed(seed + 901)
    for _ in range(config.ssl_epochs):
        model.train()
        iterators = [itertools.cycle(item) for item in loaders]
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for (domain, _), iterator in zip(domains, iterators):
                batch = next(iterator)[0].to(device)
                value, _ = ssl_batch_loss(model, batch, domain, config, generator)
                losses.append(value)
            torch.stack(losses).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def source_window_weights(metadata: pd.DataFrame, source_scores: dict[str, float]) -> np.ndarray:
    units = metadata.unit_id.astype(str)
    counts = units.value_counts()
    raw = np.asarray([source_scores[str(unit)] / counts[str(unit)] for unit in units], dtype=np.float32)
    return raw / raw.mean()


def supervised_source_fit(model: PositiveTransferTCN, x: np.ndarray, y: np.ndarray,
                          weights: np.ndarray, seed: int, config: Config,
                          device: torch.device) -> PositiveTransferTCN:
    seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.shared_projection.parameters()) + list(model.soh_head.parameters()),
        lr=config.source_learning_rate, weight_decay=config.weight_decay,
    )
    train_loader = loader(x, y, config.batch_size, seed, weights=weights)
    for _ in range(config.source_epochs):
        model.train()
        for values, labels, sample_weight in train_loader:
            values, labels, sample_weight = values.to(device), labels.to(device), sample_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(values, "source")[0]
            point = nn.functional.smooth_l1_loss(estimate, labels, beta=config.huber_beta, reduction="none")
            loss = (point * sample_weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def set_adaptation_trainable(model: PositiveTransferTCN, epoch: int, config: Config,
                             residual: bool = False) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if residual:
        for parameter in model.residual_head.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.soh_head.parameters():
        parameter.requires_grad = True
    if epoch > config.head_only_epochs:
        for parameter in model.shared_projection.parameters():
            parameter.requires_grad = True
        for parameter in model.encoder.network[-1].parameters():
            parameter.requires_grad = True
    if epoch > config.head_only_epochs + config.last_block_epochs:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True


def covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(0, keepdim=True)
    return centered.T @ centered / max(len(values) - 1, 1)


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([source, target], dim=0)
    distance = torch.cdist(combined, combined).pow(2)
    positive = distance.detach()[distance.detach() > 0]
    bandwidth = torch.median(positive) if len(positive) else torch.tensor(1.0, device=source.device)
    bandwidth = torch.clamp(bandwidth, min=1e-4)
    xx = torch.exp(-torch.cdist(source, source).pow(2) / bandwidth).mean()
    yy = torch.exp(-torch.cdist(target, target).pow(2) / bandwidth).mean()
    xy = torch.exp(-torch.cdist(source, target).pow(2) / bandwidth).mean()
    return xx + yy - 2.0 * xy


def conditional_mmd(source_z: torch.Tensor, source_y: torch.Tensor,
                    target_z: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    edges = (0.0, 0.85, 0.90, 0.95, 1.00, 1.10 + 1e-6)
    losses = []
    for low, high in zip(edges[:-1], edges[1:]):
        source_mask = (source_y >= low) & (source_y < high)
        target_mask = (target_y >= low) & (target_y < high)
        if int(source_mask.sum()) >= 2 and int(target_mask.sum()) >= 2:
            losses.append(mmd_loss(source_z[source_mask], target_z[target_mask]))
    return torch.stack(losses).mean() if losses else source_z.sum() * 0.0


def adapt_fit(model: PositiveTransferTCN, target_x: np.ndarray, target_y: np.ndarray,
              validation_x: np.ndarray, validation_y: np.ndarray, seed: int, config: Config,
              device: torch.device, source_arrays: tuple[np.ndarray, np.ndarray] | None = None,
              residual: bool = False) -> tuple[PositiveTransferTCN, int]:
    seed_all(seed)
    model = model.to(device)
    if not residual:
        model.reset_soh_head()
    target_loader = loader(target_x, target_y, config.batch_size, seed + 101)
    source_loader = None if source_arrays is None else loader(
        source_arrays[0], source_arrays[1], config.batch_size, seed + 202,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.shared_projection.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.soh_head.parameters(), "lr": config.head_learning_rate},
        {"params": model.residual_head.parameters(), "lr": config.head_learning_rate},
    ], weight_decay=config.weight_decay)
    best_state, best_mae, best_epoch, stale = snapshot(model), float("inf"), 0, 0
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    for epoch in range(1, config.adapt_epochs + 1):
        set_adaptation_trainable(model, epoch, config, residual=residual)
        model.train()
        source_iterator = itertools.cycle(source_loader) if source_loader is not None else None
        for target_values, target_labels in target_loader:
            target_values, target_labels = target_values.to(device), target_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model.residual_prediction(target_values) if residual else model(target_values, "target")[0]
            loss = loss_fn(estimate, target_labels)
            if source_iterator is not None and epoch <= math.ceil(config.adapt_epochs / 2):
                source_values, source_labels = next(source_iterator)
                source_values, source_labels = source_values.to(device), source_labels.to(device)
                with torch.no_grad():
                    source_z, _ = model.encode(source_values, "source")
                target_z, _ = model.encode(target_values, "target")
                ramp = min(1.0, epoch / max(config.adapt_epochs * 0.25, 1.0))
                loss = loss + ramp * config.alignment_weight * conditional_mmd(
                    source_z.detach(), source_labels, target_z, target_labels,
                )
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        estimate = predict(model, validation_x, config, device, residual=residual)
        mae = float(np.mean(np.abs(estimate - validation_y)))
        if mae < best_mae - 1e-7:
            best_state, best_mae, best_epoch, stale = snapshot(model), mae, epoch, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def rank_auc(left: np.ndarray, right: np.ndarray) -> float:
    ranks = rankdata(np.r_[left, right], method="average")
    n_left, n_right = len(left), len(right)
    auc = (float(ranks[n_left:].sum()) - n_right * (n_right + 1) / 2.0) / (n_left * n_right)
    return max(auc, 1.0 - auc)


def source_discrepancy(source: pd.DataFrame, target: pd.DataFrame,
                       features: list[str]) -> dict[str, float]:
    smd, ks, auc = [], [], []
    count = min(len(source), len(target), 5000)
    for feature in features:
        left = source[feature].to_numpy(float)
        right = target[feature].to_numpy(float)
        pooled = math.sqrt((left.var() + right.var()) / 2.0)
        smd.append(abs(float((right.mean() - left.mean()) / pooled)) if pooled > 1e-12 else 0.0)
        ks.append(float(ks_2samp(left, right).statistic))
        auc.append(rank_auc(
            source[feature].sample(count, random_state=41).to_numpy(float),
            target[feature].sample(count, random_state=42).to_numpy(float),
        ))
    return {"mean_smd": float(np.mean(smd)), "mean_ks": float(np.mean(ks)), "mean_univariate_auc": float(np.mean(auc))}


def monotonic_direction_rate(metadata: pd.DataFrame, prediction: np.ndarray) -> float:
    frame = metadata[["unit_id", "time"]].copy()
    frame["prediction"] = prediction
    directions = []
    for _, group in frame.sort_values(["unit_id", "time"]).groupby("unit_id"):
        directions.extend(np.diff(group.prediction.to_numpy(float)) <= 0.0)
    return float(np.mean(directions)) if directions else 0.0


def source_audit(source: pd.DataFrame, target_unlabeled: pd.DataFrame, validation: pd.DataFrame,
                 features: list[str], target_state: RobustState, config: Config,
                 seed: int, device: torch.device) -> pd.DataFrame:
    rows = []
    validation_x, validation_y, validation_meta = make_windows(
        validation, features, target_state, config.window, True,
    )
    assert validation_y is not None
    for unit_id in SOURCE_UNITS:
        unit = source.loc[source.unit_id.eq(unit_id)].copy()
        discrepancy = source_discrepancy(unit, target_unlabeled, features)
        source_state = robust_fit(unit, features, f"NASA:{unit_id}")
        source_x, source_y, source_meta = make_windows(unit, features, source_state, config.window, True)
        assert source_y is not None
        quick = replace(config, source_epochs=min(12, config.source_epochs))
        model = PositiveTransferTCN(len(features), quick)
        model = supervised_source_fit(
            model, source_x, source_y, np.ones(len(source_x), np.float32), seed, quick, device,
        )
        prediction = predict(model, validation_x, quick, device)
        rows.append({
            "unit_id": unit_id, **discrepancy,
            "source_only_validation_mae": float(np.mean(np.abs(prediction - validation_y))),
            "target_prediction_nonincrease_rate": monotonic_direction_rate(validation_meta, prediction),
            "source_rows": len(unit), "source_windows": len(source_meta),
        })
    result = pd.DataFrame(rows)
    discrepancy = (
        result.mean_smd.rank(pct=True) + result.mean_ks.rank(pct=True)
        + (result.mean_univariate_auc - 0.5).rank(pct=True)
    ) / 3.0
    pilot = result.source_only_validation_mae.rank(pct=True)
    result["selection_cost"] = 0.70 * discrepancy + 0.30 * pilot
    result = result.sort_values(["selection_cost", "unit_id"]).reset_index(drop=True)
    similarity = np.exp(-(result.selection_cost - result.selection_cost.min()) / 0.25)
    result["similarity_weight"] = similarity / similarity.sum()
    result["similarity_rank"] = np.arange(1, len(result) + 1)
    return result


def choose_nested_units(train: pd.DataFrame, sizes: list[int], seed: int) -> dict[int, list[str]]:
    units = np.asarray(sorted(map(str, train.unit_id.unique())))
    order = np.random.default_rng(seed).permutation(units)
    return {size: sorted(order[:size].tolist()) for size in sizes}


def per_unit_metrics(metadata: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    frame = metadata[["unit_id", "time", "target_soh"]].copy()
    frame["predicted_soh"] = prediction
    rows = []
    for unit_id, group in frame.groupby("unit_id"):
        rows.append({"unit_id": unit_id, **regression_metrics(
            group.target_soh.to_numpy(float), group.predicted_soh.to_numpy(float),
        )})
    return pd.DataFrame(rows)


def selected_source_arrays(source: pd.DataFrame, selected: list[str], features: list[str],
                           config: Config, score_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, RobustState]:
    chosen = source.loc[source.unit_id.isin(selected)].copy()
    state = robust_fit(chosen, features, "NASA:selected")
    x, y, metadata = make_windows(chosen, features, state, config.window, True)
    assert y is not None
    raw = score_table.set_index("unit_id").similarity_weight.to_dict()
    normalized = {unit: raw[unit] / sum(raw[name] for name in selected) for unit in selected}
    return x, y, source_window_weights(metadata, normalized), state


def pretrain_states(source_x: np.ndarray, source_y: np.ndarray, source_weights: np.ndarray,
                    target_unlabeled_x: np.ndarray, input_size: int, seed: int,
                    config: Config, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target_model = PositiveTransferTCN(input_size, config)
    target_model = ssl_fit(target_model, [("target", target_unlabeled_x)], seed, config, device)
    transfer_model = PositiveTransferTCN(input_size, config)
    transfer_model = ssl_fit(
        transfer_model, [("source", source_x), ("target", target_unlabeled_x)], seed + 1, config, device,
    )
    transfer_model = supervised_source_fit(
        transfer_model, source_x, source_y, source_weights, seed + 2, config, device,
    )
    return snapshot(target_model), snapshot(transfer_model)


def fit_method(method: str, target_x: np.ndarray, target_y: np.ndarray,
               validation_x: np.ndarray, validation_y: np.ndarray,
               source_arrays: tuple[np.ndarray, np.ndarray], target_ssl_state: dict[str, torch.Tensor],
               transfer_state: dict[str, torch.Tensor], input_size: int, seed: int,
               config: Config, device: torch.device) -> tuple[PositiveTransferTCN, int, bool]:
    model = PositiveTransferTCN(input_size, config)
    residual = method == "selected_residual"
    aligned = method == "dr_tcn_conditional_mmd"
    if method == "target_only":
        pass
    elif method == "target_ssl":
        model.load_state_dict(copy.deepcopy(target_ssl_state))
    else:
        model.load_state_dict(copy.deepcopy(transfer_state))
    model, epoch = adapt_fit(
        model, target_x, target_y, validation_x, validation_y, seed, config, device,
        source_arrays=source_arrays if aligned else None, residual=residual,
    )
    return model, epoch, residual


def tune_configuration(source: pd.DataFrame, target: pd.DataFrame, validation: pd.DataFrame,
                       target_unlabeled: pd.DataFrame, features: list[str], source_scores: pd.DataFrame,
                       target_state: RobustState, base: Config, seed: int,
                       device: torch.device) -> tuple[Config, list[str], pd.DataFrame]:
    units = choose_nested_units(target, [10], seed)[10]
    target_subset = target.loc[target.unit_id.astype(str).isin(units)].copy()
    target_x, target_y, _ = make_windows(target_subset, features, target_state, base.window, True)
    validation_x, validation_y, _ = make_windows(validation, features, target_state, base.window, True)
    unlabeled_x, _, _ = make_windows(target_unlabeled, features, target_state, base.window, False)
    assert target_y is not None and validation_y is not None
    records = []

    ssl_candidates = ((0.10, 0.05), (0.30, 0.05), (0.30, 0.10))
    ssl_best: tuple[float, float] = ssl_candidates[0]
    best_mae = float("inf")
    for reconstruction_weight, order_weight in ssl_candidates:
        config = replace(base, reconstruction_weight=reconstruction_weight, order_weight=order_weight)
        model = PositiveTransferTCN(len(features), config)
        model = ssl_fit(model, [("target", unlabeled_x)], seed, config, device)
        model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, config, device)
        prediction = predict(model, validation_x, config, device)
        mae = float(np.mean(np.abs(prediction - validation_y)))
        records.append({"stage": "ssl_weights", "candidate": f"rec={reconstruction_weight},order={order_weight}", "validation_mae": mae, "best_epoch": epoch})
        if mae < best_mae:
            best_mae, ssl_best = mae, (reconstruction_weight, order_weight)
    tuned = replace(base, reconstruction_weight=ssl_best[0], order_weight=ssl_best[1])

    ranked = source_scores.unit_id.tolist()
    source_best = ranked[:1]
    best_mae = float("inf")
    cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for count in (1, 3, 5):
        selected = ranked[:count]
        sx, sy, sw, _ = selected_source_arrays(source, selected, features, tuned, source_scores)
        cached[count] = (sx, sy, sw)
        _, transfer_state = pretrain_states(sx, sy, sw, unlabeled_x, len(features), seed, tuned, device)
        model = PositiveTransferTCN(len(features), tuned)
        model.load_state_dict(transfer_state)
        model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, tuned, device)
        prediction = predict(model, validation_x, tuned, device)
        mae = float(np.mean(np.abs(prediction - validation_y)))
        records.append({"stage": "source_count", "candidate": f"top_{count}", "validation_mae": mae, "best_epoch": epoch})
        if mae < best_mae:
            best_mae, source_best = mae, selected

    sx, sy, sw, _ = selected_source_arrays(source, source_best, features, tuned, source_scores)
    _, transfer_state = pretrain_states(sx, sy, sw, unlabeled_x, len(features), seed, tuned, device)
    alignment_best, best_mae = 0.003, float("inf")
    for weight in (0.001, 0.003, 0.01):
        config = replace(tuned, alignment_weight=weight)
        model = PositiveTransferTCN(len(features), config)
        model.load_state_dict(transfer_state)
        model, epoch = adapt_fit(
            model, target_x, target_y, validation_x, validation_y, seed, config, device,
            source_arrays=(sx, sy),
        )
        prediction = predict(model, validation_x, config, device)
        mae = float(np.mean(np.abs(prediction - validation_y)))
        records.append({"stage": "alignment_weight", "candidate": str(weight), "validation_mae": mae, "best_epoch": epoch})
        if mae < best_mae:
            best_mae, alignment_best = mae, weight
    tuned = replace(tuned, alignment_weight=alignment_best)

    schedule_candidates = ((5e-4, 5e-5, 7), (3e-4, 3e-5, 7), (5e-4, 5e-5, 10))
    schedule_best, best_mae = schedule_candidates[0], float("inf")
    for head_lr, encoder_lr, patience in schedule_candidates:
        config = replace(tuned, head_learning_rate=head_lr, encoder_learning_rate=encoder_lr, patience=patience)
        model = PositiveTransferTCN(len(features), config)
        model.load_state_dict(transfer_state)
        model, epoch = adapt_fit(model, target_x, target_y, validation_x, validation_y, seed, config, device)
        prediction = predict(model, validation_x, config, device)
        mae = float(np.mean(np.abs(prediction - validation_y)))
        records.append({"stage": "schedule", "candidate": f"head={head_lr},encoder={encoder_lr},patience={patience}", "validation_mae": mae, "best_epoch": epoch})
        if mae < best_mae:
            best_mae, schedule_best = mae, (head_lr, encoder_lr, patience)
    tuned = replace(
        tuned, head_learning_rate=schedule_best[0], encoder_learning_rate=schedule_best[1],
        patience=schedule_best[2],
    )
    return tuned, source_best, pd.DataFrame(records)


def causal_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty(len(values), dtype=float)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = min(output[index - 1], alpha * values[index] + (1.0 - alpha) * output[index - 1])
    return output


def slope_bounds(train: pd.DataFrame) -> tuple[float, float, float, float]:
    slopes, lifetimes = [], []
    for _, group in train.sort_values(["unit_id", "time"]).groupby("unit_id"):
        if len(group) >= 3:
            slope = float(theilslopes(group.target_soh.to_numpy(float), group.time.to_numpy(float)).slope)
            if np.isfinite(slope) and slope < -1e-7:
                slopes.append(slope)
        lifetimes.extend(group.true_eol_cycle.to_numpy(float))
    if not slopes:
        slopes = [-0.005]
    lower, upper = np.quantile(slopes, [0.05, 0.95])
    return float(lower), float(upper), float(np.median(slopes)), float(np.max(lifetimes))


def estimate_slope(times: np.ndarray, values: np.ndarray,
                   bounds: tuple[float, float, float, float]) -> float:
    lower, upper, prior, _ = bounds
    if len(values) < 3:
        return prior
    window = min(12, len(values))
    slope = float(theilslopes(values[-window:], times[-window:]).slope)
    if not np.isfinite(slope) or slope >= -1e-7:
        return prior
    return float(np.clip(slope, lower, upper))


def rul_evaluation(predictions: pd.DataFrame, train: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    bounds = slope_bounds(train)
    checkpoints: list[tuple[str, float | int]] = [
        ("fraction_0.2", 0.2), ("fraction_0.4", 0.4), ("fraction_0.6", 0.6),
        ("fraction_0.8", 0.8), ("fixed_30", 30),
    ]
    rows = []
    for unit_id, group in predictions.sort_values(["unit_id", "time"]).groupby("unit_id"):
        group = group.reset_index(drop=True)
        times = group.time.to_numpy(float)
        smooth = causal_smooth(group.predicted_soh.to_numpy(float), alpha)
        checkpoint_indices: dict[int, list[str]] = {}
        for name, spec in checkpoints:
            index = min(len(group) - 1, max(0, math.ceil(len(group) * spec) - 1)) if isinstance(spec, float) else min(len(group) - 1, int(spec) - 1)
            checkpoint_indices.setdefault(index, []).append(name)
        for index in range(len(group)):
            slope = estimate_slope(times[:index + 1], smooth[:index + 1], bounds)
            raw_rul = max((smooth[index] - 0.80) / (-slope), 0.0)
            capped_rul = min(raw_rul, max(bounds[3] - times[index], 0.0))
            rows.append({
                "unit_id": unit_id, "time": times[index], "smoothed_soh": smooth[index],
                "estimated_slope": slope, "predicted_rul_cycles": capped_rul,
                "true_rul_cycles": float(group.true_rul_cycles.iloc[index]),
                "predicted_eol_cycle": times[index] + capped_rul,
                "true_eol_cycle": float(group.true_eol_cycle.iloc[index]),
                "checkpoints": ";".join(checkpoint_indices.get(index, [])),
            })
    detail = pd.DataFrame(rows)
    summaries = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_points", detail)]
    for name, _ in checkpoints:
        groups.append((name, detail.loc[detail.checkpoints.str.split(";").map(lambda x: name in x)]))
    for name, group in groups:
        rul_error = group.predicted_rul_cycles - group.true_rul_cycles
        eol_error = group.predicted_eol_cycle - group.true_eol_cycle
        summaries.append({
            "checkpoint": name, "samples": len(group), "devices": group.unit_id.nunique(),
            "rul_mae_cycles": float(np.mean(np.abs(rul_error))),
            "rul_rmse_cycles": float(np.sqrt(np.mean(rul_error**2))),
            "rul_bias_cycles": float(np.mean(rul_error)),
            "eol_mae_cycles": float(np.mean(np.abs(eol_error))),
            "maximum_absolute_rul_error": float(np.max(np.abs(rul_error))),
        })
    return detail, pd.DataFrame(summaries)


def load_data(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    contracts = json.loads((data_root / "feature_contracts.json").read_text(encoding="utf-8"))
    if "bhump_compact_v2" not in contracts:
        raise ValueError("Run build_bhump_compact_v2_contract.py first")
    features = list(contracts["bhump_compact_v2"])
    assert_feature_contract(features)
    if not 24 <= len(features) <= 32:
        raise ValueError(f"compact-v2 feature count must be 24..32, got {len(features)}")
    source = pd.read_csv(data_root / "nasa_source_rich.csv")
    target = pd.read_csv(data_root / "basilisk_train_rich.csv")
    validation = pd.read_csv(data_root / "basilisk_validation_rich.csv")
    if set(map(str, source.unit_id)) != set(SOURCE_UNITS):
        raise ValueError("Unexpected NASA source units")
    if set(source.unit_id.astype(str)) & set(EXCLUDED_SOURCE_UNITS):
        raise ValueError("Excluded NASA unit was read")
    if set(target.unit_id.astype(str)) & set(validation.unit_id.astype(str)):
        raise ValueError("Target train/validation device leakage")
    for frame in (source, target, validation):
        if not np.isfinite(frame[features].to_numpy(float)).all():
            raise ValueError("Non-finite model input")
    return source, target, validation, features


def save_checkpoint(model: PositiveTransferTCN, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": snapshot(model), **metadata}, path)


def run(args: argparse.Namespace) -> None:
    base = Config(
        ssl_epochs=args.ssl_epochs, source_epochs=args.source_epochs,
        adapt_epochs=args.adapt_epochs, patience=args.patience,
    )
    methods = args.methods.split(",")
    if unknown := set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    subset_sizes = [int(item) for item in args.subset_sizes.split(",")]
    formal_seeds = [int(item) for item in args.formal_seeds.split(",")]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    source, target, validation, features = load_data(args.data_root)
    if args.smoke:
        base = replace(base, ssl_epochs=1, source_epochs=2, adapt_epochs=3, patience=2, batch_size=64)
        subset_sizes, formal_seeds = [min(5, target.unit_id.nunique())], [int(args.formal_seeds.split(",")[0])]
        keep_target = sorted(target.unit_id.astype(str).unique())[:12]
        keep_validation = sorted(validation.unit_id.astype(str).unique())[:4]
        target = target.loc[target.unit_id.astype(str).isin(keep_target)].copy()
        validation = validation.loc[validation.unit_id.astype(str).isin(keep_validation)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_unlabeled = unlabeled_view(target, features)
    target_state = robust_fit(target_unlabeled, features, "Basilisk:all_unlabeled_train")
    source_scores = source_audit(
        source, target_unlabeled, validation, features, target_state, base,
        args.tuning_seed, device,
    )
    source_scores.to_csv(args.output_dir / "source_domain_selection.csv", index=False)

    if args.skip_tuning or args.smoke:
        config = base
        selected_sources = source_scores.head(min(3, len(source_scores))).unit_id.tolist()
        tuning = pd.DataFrame([{"stage": "skipped", "candidate": "defaults", "validation_mae": np.nan, "best_epoch": 0}])
    else:
        config, selected_sources, tuning = tune_configuration(
            source, target, validation, target_unlabeled, features, source_scores,
            target_state, base, args.tuning_seed, device,
        )
    tuning.to_csv(args.output_dir / "tuning_results.csv", index=False)
    (args.output_dir / "frozen_tuning_config.json").write_text(json.dumps({
        "tuning_seed": args.tuning_seed, "selected_sources": selected_sources,
        "configuration": asdict(config), "selection_dataset": "Basilisk validation",
    }, indent=2), encoding="utf-8")

    source_x, source_y, source_weights, source_state = selected_source_arrays(
        source, selected_sources, features, config, source_scores,
    )
    target_unlabeled_x, _, _ = make_windows(
        target_unlabeled, features, target_state, config.window, False,
    )
    validation_x, validation_y, validation_meta = make_windows(
        validation, features, target_state, config.window, True,
    )
    assert validation_y is not None

    result_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    model_index: dict[tuple[str, int, int], tuple[PositiveTransferTCN, bool, list[str]]] = {}
    for seed in formal_seeds:
        seed_all(seed)
        nested = choose_nested_units(target, subset_sizes, seed)
        target_ssl_state, transfer_state = pretrain_states(
            source_x, source_y, source_weights, target_unlabeled_x,
            len(features), seed, config, device,
        )
        for subset_size in subset_sizes:
            selected_units = nested[subset_size]
            selected_rows.extend({"seed": seed, "subset_size": subset_size, "unit_id": unit} for unit in selected_units)
            target_subset = target.loc[target.unit_id.astype(str).isin(selected_units)].copy()
            target_x, target_y, _ = make_windows(target_subset, features, target_state, config.window, True)
            assert target_y is not None
            for method in methods:
                model, best_epoch, residual = fit_method(
                    method, target_x, target_y, validation_x, validation_y,
                    (source_x, source_y), target_ssl_state, transfer_state,
                    len(features), seed, config, device,
                )
                estimate = predict(model, validation_x, config, device, residual=residual)
                aggregate = regression_metrics(validation_y, estimate)
                device_metrics = per_unit_metrics(validation_meta, estimate)
                record = {
                    "method": method, "seed": seed, "subset_size": subset_size,
                    "target_labeled_units": len(selected_units), "target_labeled_windows": len(target_x),
                    "target_unlabeled_units": target.unit_id.nunique(), "best_epoch": best_epoch,
                    **aggregate, "device_mae_std": float(device_metrics.soh_mae.std(ddof=0)),
                    "worst_device_mae": float(device_metrics.soh_mae.max()),
                }
                result_rows.append(record)
                unit_rows.extend({"method": method, "seed": seed, "subset_size": subset_size, **row} for row in device_metrics.to_dict("records"))
                for metadata, value in zip(validation_meta.to_dict("records"), estimate):
                    prediction_rows.append({"method": method, "seed": seed, "subset_size": subset_size, **metadata, "predicted_soh": float(value)})
                save_checkpoint(model, args.output_dir / "checkpoints" / f"{method}_units_{subset_size:03d}_seed_{seed}.pt", {
                    "method": method, "seed": seed, "subset_size": subset_size,
                    "residual": residual, "features": features, "selected_sources": selected_sources,
                    "target_units": selected_units, "configuration": asdict(config),
                })
                model_index[(method, subset_size, seed)] = (copy.deepcopy(model).cpu(), residual, selected_units)
                print(json.dumps(record), flush=True)

    results = pd.DataFrame(result_rows)
    units = pd.DataFrame(unit_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected_frame = pd.DataFrame(selected_rows).drop_duplicates()
    results.to_csv(args.output_dir / "results_by_seed.csv", index=False)
    units.to_csv(args.output_dir / "results_per_unit.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    selected_frame.to_csv(args.output_dir / "selected_target_units.csv", index=False)

    baseline = results.loc[results.method.isin(["target_only", "target_ssl"])].sort_values("soh_mae").groupby(
        ["seed", "subset_size"], as_index=False,
    ).first()[["seed", "subset_size", "method", "soh_mae", "worst_device_mae"]].rename(columns={
        "method": "stronger_target_baseline", "soh_mae": "baseline_mae",
        "worst_device_mae": "baseline_worst_device_mae",
    })
    gains = results.merge(baseline, on=["seed", "subset_size"], how="left")
    gains["relative_gain"] = (gains.baseline_mae - gains.soh_mae) / gains.baseline_mae
    gains["worst_device_change"] = (
        gains.worst_device_mae - gains.baseline_worst_device_mae
    ) / gains.baseline_worst_device_mae
    gains["seed_positive"] = gains.relative_gain.gt(0.0)
    gains.to_csv(args.output_dir / "transfer_gains_by_seed.csv", index=False)

    summary = gains.groupby(["method", "subset_size"], as_index=False).agg(
        soh_mae_mean=("soh_mae", "mean"), soh_mae_std=("soh_mae", "std"),
        relative_gain_mean=("relative_gain", "mean"), positive_seed_count=("seed_positive", "sum"),
        worst_device_mae_mean=("worst_device_mae", "mean"),
        baseline_worst_device_mae_mean=("baseline_worst_device_mae", "mean"),
        device_mae_std_mean=("device_mae_std", "mean"),
    )
    summary["worst_device_change"] = (
        summary.worst_device_mae_mean - summary.baseline_worst_device_mae_mean
    ) / summary.baseline_worst_device_mae_mean
    summary["subset_success"] = (
        summary.relative_gain_mean.ge(0.05) & summary.positive_seed_count.ge(2)
        & summary.worst_device_change.le(0.10)
    )
    method_stability = summary.loc[summary.method.isin(METHODS[2:])].groupby("method", as_index=False).agg(
        successful_subset_count=("subset_success", "sum"),
        mean_relative_gain=("relative_gain_mean", "mean"),
        maximum_worst_device_change=("worst_device_change", "max"),
    )
    method_stability["stable_positive_transfer"] = method_stability.successful_subset_count.ge(2)
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    method_stability.to_csv(args.output_dir / "transfer_stability.csv", index=False)

    stable_methods = method_stability.loc[method_stability.stable_positive_transfer, "method"].tolist()
    rul_manifest: dict[str, Any] = {"executed": False, "reason": "no stable positive transfer evidence"}
    best_identity: dict[str, Any] | None = None
    if stable_methods:
        eligible = summary.loc[summary.method.isin(stable_methods) & summary.subset_success].sort_values(
            ["soh_mae_mean", "device_mae_std_mean", "subset_size"], ascending=[True, True, False],
        )
        chosen = eligible.iloc[0]
        method, subset_size = str(chosen.method), int(chosen.subset_size)
        seed_rows = results.loc[results.method.eq(method) & results.subset_size.eq(subset_size)].copy()
        median_mae = float(seed_rows.soh_mae.median())
        seed_rows["distance_to_median"] = (seed_rows.soh_mae - median_mae).abs()
        chosen_seed = int(seed_rows.sort_values(["distance_to_median", "seed"]).iloc[0].seed)
        model, residual, selected_units = model_index[(method, subset_size, chosen_seed)]
        chosen_predictions = predictions.loc[
            predictions.method.eq(method) & predictions.subset_size.eq(subset_size) & predictions.seed.eq(chosen_seed)
        ].copy()
        alpha_rows = []
        for alpha in (0.1, 0.2, 0.3, 0.5):
            smooth = np.concatenate([
                causal_smooth(group.predicted_soh.to_numpy(float), alpha)
                for _, group in chosen_predictions.sort_values(["unit_id", "time"]).groupby("unit_id")
            ])
            ordered_truth = np.concatenate([
                group.target_soh.to_numpy(float)
                for _, group in chosen_predictions.sort_values(["unit_id", "time"]).groupby("unit_id")
            ])
            alpha_rows.append({"alpha": alpha, "validation_soh_mae": float(np.mean(np.abs(smooth - ordered_truth)))})
        alpha_table = pd.DataFrame(alpha_rows).sort_values(["validation_soh_mae", "alpha"])
        alpha_table.to_csv(args.output_dir / "smoothing_selection.csv", index=False)
        alpha = float(alpha_table.iloc[0].alpha)
        slope_train = target.loc[target.unit_id.astype(str).isin(selected_units)].copy()
        rul_detail, rul_summary = rul_evaluation(chosen_predictions, slope_train, alpha)
        rul_detail.to_csv(args.output_dir / "validation_rul_predictions.csv", index=False)
        rul_summary.to_csv(args.output_dir / "rul_metrics.csv", index=False)
        best_identity = {"method": method, "subset_size": subset_size, "seed": chosen_seed}
        rul_manifest = {"executed": True, **best_identity, "smoothing_alpha": alpha, "slope_bounds": slope_bounds(slope_train)}

    deployment_count = sum(parameter.numel() for parameter in PositiveTransferTCN(len(features), config).deployment_parameters())
    training_count = sum(parameter.numel() for parameter in PositiveTransferTCN(len(features), config).parameters())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "direction": "NASA public source -> Basilisk V0.9 target -> SOH -> EOL/RUL",
        "feature_contract": "bhump_compact_v2", "features": features,
        "selected_sources": selected_sources, "excluded_nasa_units": list(EXCLUDED_SOURCE_UNITS),
        "formal_seeds": formal_seeds, "subset_sizes": subset_sizes,
        "configuration": asdict(config), "methods": methods,
        "deployment_parameter_count": deployment_count, "training_parameter_count": training_count,
        "stable_positive_transfer_found": bool(stable_methods),
        "stable_methods": stable_methods, "best_frozen_model": best_identity,
        "rul_evaluation": rul_manifest,
        "leakage_audit": {
            "unlabeled_target_labels_dropped": True,
            "target_scaler_fit_units": len(target_state.fit_units),
            "target_validation_used_for_scaler": False,
            "sealed_features_accessed": False, "sealed_labels_accessed": False,
            "nasa_external_or_sealed_accessed": False,
        },
        "conclusion": "stable positive transfer found" if stable_methods else "no stable positive transfer evidence",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset-sizes", default="5,10,20")
    parser.add_argument("--unlabeled-target", choices=("all",), default="all")
    parser.add_argument("--tuning-seed", type=int, default=41)
    parser.add_argument("--formal-seeds", default="42,43,44")
    parser.add_argument("--methods", default=",".join(METHODS))
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
