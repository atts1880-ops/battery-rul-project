"""Strict dual-domain OOF comparison for V1.0 and V1.5 battery RUL.

The registered comparison keeps NASA5, compact16, TCN24 and B_stats38 fixed.
Dataset identity is retained only for balancing and reporting and is never a
model input.  Public validation and every sealed path remain closed during
OOF model selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from train_bhump_positive_transfer import (
    RobustState, make_windows, robust_fit, seed_all, snapshot, ssl_fit, unlabeled_view,
)
from train_bhump_v10_bstats_oof import inner_split, subset
from train_bhump_v10_bstats_refit_oof import fit_multitask_fixed_epochs
from train_bhump_v10_history_ablation import (
    HistoryConfig, SequenceBundle, fit_statistics, make_bundle,
)
from train_bhump_v10_intercell_transfer import target_ssl_state
from train_bhump_v10_nasa_dynamics_pretrain import nasa_dynamics_initial_state
from train_bhump_v10_rul_multitask import (
    DynamicsTCN, attach_dynamics_labels, build_config, dynamics_loader,
    make_multitask_windows, multitask_loss, predict_dynamics,
    set_dynamics_trainable,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    ROOT / "bhump_transfer_v10_data",
    ROOT / "bhump_transfer_v15_nasa5_ecm_data",
)
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_v15_generalization_runs"
NASA5 = ("B0005", "B0018", "B0033", "B0043", "B0044")
DOMAINS = ("v10", "v15")
MAIN_METHODS = ("v10_only", "v15_only", "pooled", "multi_expert")
CONTROL_METHODS = ("pooled_wide_control", "pooled_target_only", "multi_expert_target_only")
ALL_METHODS = (*MAIN_METHODS, *CONTROL_METHODS)
MODE_NAMES = ("long_no_knee", "regular_no_knee", "early_knee", "late_knee")
EOL_SOH = 0.80


@dataclass(frozen=True)
class HeadConfig:
    epochs: int = 45
    patience: int = 7
    batch_devices: int = 32
    learning_rate: float = 7.5e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.10


class GeneralizationHead(nn.Module):
    """Single, parameter-matched wide, or four-expert causal RUL head."""

    def __init__(self, kind: str, config: HeadConfig) -> None:
        super().__init__()
        if kind not in {"single", "wide", "multi_expert"}:
            raise ValueError(f"Unknown head kind: {kind}")
        self.kind = kind
        # Keep the proven B_stats fusion path intact for SOH and the ordinary
        # pooled/specialist RUL head.  The expert router still consumes the
        # registered 55-D representation exactly as specified.
        self.fusion = nn.Sequential(
            nn.Linear(55, 48), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(48, 48), nn.SiLU(),
        )
        self.soh_residual_head = nn.Linear(48, 1)
        self.rate_head = nn.Linear(55, 1)
        if kind == "multi_expert":
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(55, 32), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(32, 1))
                for _ in MODE_NAMES
            ])
            self.gate = nn.Sequential(
                nn.Linear(55, 24), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(24, 4),
            )
        elif kind == "wide":
            width = 153
            self.rul_head = nn.Sequential(
                nn.Linear(55, width), nn.SiLU(), nn.Dropout(config.dropout), nn.Linear(width, 1),
            )
        else:
            self.rul_head = nn.Linear(48, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.soh_residual_head.weight)
        nn.init.zeros_(self.soh_residual_head.bias)
        if self.kind == "single":
            nn.init.zeros_(self.rul_head.weight)
            nn.init.zeros_(self.rul_head.bias)
        nn.init.constant_(self.rate_head.bias, -6.0)

    def forward(
        self, local: torch.Tensor, local_soh: torch.Tensor, stats: torch.Tensor,
        times: torch.Tensor, maximum_lifetime: float,
    ) -> dict[str, torch.Tensor]:
        representation = torch.cat([local, local_soh.unsqueeze(-1), stats], dim=-1)
        hidden = self.fusion(representation)
        soh = torch.clamp(
            local_soh + 0.05 * torch.tanh(self.soh_residual_head(hidden).squeeze(-1)),
            0.0, 1.10,
        )
        rate = F.softplus(self.rate_head(representation).squeeze(-1)) + 1.0e-7
        cap = torch.clamp(
            torch.as_tensor(maximum_lifetime, device=times.device) - times, min=0.0,
        )
        result = {"soh": soh, "rate": rate, "representation": representation}
        if self.kind == "multi_expert":
            logits = torch.stack([expert(representation).squeeze(-1) for expert in self.experts], -1)
            expert_rul = cap.unsqueeze(-1) * torch.sigmoid(logits)
            gate_logits = self.gate(representation)
            gate_probability = torch.softmax(gate_logits, dim=-1)
            result.update({
                "rul": torch.sum(gate_probability * expert_rul, dim=-1),
                "expert_rul": expert_rul, "gate_logits": gate_logits,
                "gate_probability": gate_probability,
            })
        elif self.kind == "wide":
            result["rul"] = cap * torch.sigmoid(self.rul_head(representation).squeeze(-1))
        else:
            result["rul"] = cap * torch.sigmoid(self.rul_head(hidden).squeeze(-1))
        return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_domain(root: Path, domain: str, validation: bool = False) -> tuple[pd.DataFrame, list[str]]:
    contract = json.loads((root / "feature_contracts.json").read_text(encoding="utf-8"))
    features = list(contract["bhump_degradation_invariant"])
    name = "basilisk_validation_rich.csv" if validation else "basilisk_train_rich.csv"
    frame = pd.read_csv(root / name)
    required = {"unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle", *features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{domain} misses required columns: {sorted(missing)}")
    original = frame.unit_id.astype(str)
    frame["unit_id"] = domain + "::" + original
    frame["evaluation_domain"] = domain
    if frame.unit_id.nunique() != (80 if validation else 320):
        raise ValueError(f"Unexpected {domain} device count")
    if not np.isfinite(frame[[*features, "target_soh", "true_rul_cycles"]].to_numpy(float)).all():
        raise ValueError(f"Non-finite {domain} values")
    frame, _ = attach_dynamics_labels(frame)
    return frame, features


def load_training_data(roots: tuple[Path, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    v10, features10 = read_domain(roots[0], "v10")
    v15, features15 = read_domain(roots[1], "v15")
    if features10 != features15 or len(features10) != 16:
        raise ValueError("V1.0/V1.5 frozen compact16 contracts differ")
    leaked = [f for f in features10 if any(token in f.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if leaked:
        raise ValueError(f"Forbidden features: {leaked}")
    source = pd.read_csv(roots[0] / "nasa_source_rich.csv")
    if tuple(sorted(source.unit_id.astype(str).unique())) != NASA5:
        raise ValueError("Source is not exact NASA5")
    return source, v10, v15, features10


def dynamics_report(frame: pd.DataFrame) -> pd.DataFrame:
    _, report = attach_dynamics_labels(frame)
    rows = []
    for item in report.itertuples(index=False):
        unit = frame.loc[frame.unit_id.eq(item.unit_id)].sort_values("time")
        progress = math.nan
        if bool(item.has_knee):
            index = int(np.argmin(np.abs(unit.time.to_numpy(float) - float(item.knee_cycle))))
            initial = float(unit.target_soh.iloc[0])
            progress = float((initial - float(unit.target_soh.iloc[index])) / max(initial - EOL_SOH, 1e-6))
        rows.append({**item._asdict(), "knee_progress": progress,
                     "evaluation_domain": str(unit.evaluation_domain.iloc[0])})
    return pd.DataFrame(rows)


def mode_thresholds(train: pd.DataFrame) -> dict[str, float]:
    return {
        domain: float(group.groupby("unit_id").true_eol_cycle.first().quantile(0.75))
        for domain, group in train.groupby("evaluation_domain")
    }


def mode_and_rate_arrays(
    bundle: SequenceBundle, report: pd.DataFrame, thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = report.set_index("unit_id").to_dict("index")
    modes = np.zeros(bundle.mask.shape, np.int64)
    rates = np.zeros(bundle.mask.shape, np.float32)
    fallback = float(np.median(list(thresholds.values())))
    for row, unit_id in enumerate(bundle.unit_ids):
        item = lookup[unit_id]
        domain = str(item["evaluation_domain"])
        if bool(item["has_knee"]):
            mode = 2 if float(item["knee_progress"]) < 0.5 else 3
        else:
            eol = float(bundle.true_eol[row, 0])
            mode = 0 if eol >= thresholds.get(domain, fallback) else 1
        modes[row, bundle.mask[row]] = mode
        times = bundle.times[row]
        before = times < float(item["knee_cycle"])
        rates[row] = np.where(before, float(item["pre_rate"]), float(item["post_rate"])).astype(np.float32)
    return modes, rates


def balanced_weights(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    domains = output.evaluation_domain.nunique()
    unit_counts = output.groupby("evaluation_domain").unit_id.nunique().to_dict()
    cycle_counts = output.groupby("unit_id").size().to_dict()
    raw = np.asarray([
        1.0 / (domains * unit_counts[d] * cycle_counts[u])
        for d, u in zip(output.evaluation_domain, output.unit_id)
    ], dtype=float)
    output["sample_weight"] = raw / raw.mean()
    output["unit_weight"] = 1.0
    masses = output.groupby("evaluation_domain").sample_weight.sum()
    if domains > 1 and float(masses.max() / masses.min()) > 1.000001:
        raise AssertionError("Domain sample masses are not equal")
    return output


def stratified_folds(frame: pd.DataFrame, report: pd.DataFrame, folds: int, seed: int) -> dict[str, int]:
    life = frame.groupby("unit_id", as_index=False).agg(eol=("true_eol_cycle", "first"))
    annotations = report[["unit_id", "has_knee", "knee_progress"]]
    life = life.merge(annotations, on="unit_id", validate="one_to_one")
    life["life_q"] = pd.qcut(life.eol.rank(method="first"), 4, labels=False)
    life["knee_group"] = np.where(
        ~life.has_knee.astype(bool), "none", np.where(life.knee_progress < 0.5, "early", "late"),
    )
    assignment: dict[str, int] = {}
    offset = 0
    for _, group in life.groupby(["life_q", "knee_group"], sort=True):
        ordered = sorted(group.unit_id.astype(str), key=lambda unit: int.from_bytes(
            hashlib.sha256(f"{seed}:{unit}".encode()).digest()[:8], "big",
        ))
        for index, unit in enumerate(ordered):
            assignment[unit] = (offset + index) % folds
        offset = (offset + len(ordered)) % folds
    counts = pd.Series(assignment).value_counts().sort_index()
    if len(counts) != folds or counts.max() - counts.min() > 1:
        raise RuntimeError(f"Invalid stratified folds: {counts.to_dict()}")
    return assignment


def pooled_ssl_state(
    train: pd.DataFrame, features: list[str], config: Any, seed: int, device: torch.device,
) -> tuple[dict[str, torch.Tensor], RobustState]:
    unlabeled = unlabeled_view(train, features)
    state = robust_fit(unlabeled, features, "Basilisk:v10+v15_outer_train_unlabeled")
    arrays = []
    for domain, group in train.groupby("evaluation_domain", sort=True):
        values, _, _ = make_windows(unlabeled.loc[unlabeled.unit_id.isin(group.unit_id)], features, state, config.window, False)
        arrays.append(("target", values))
    model = ssl_fit(DynamicsTCN(len(features), config), arrays, seed, config, device)
    return snapshot(model), state


def macro_device_mae(frame: pd.DataFrame) -> float:
    per_device = frame.assign(ae=np.abs(frame.predicted_rul_raw - frame.true_rul_cycles)).groupby(
        ["evaluation_domain", "unit_id"], as_index=False
    ).ae.mean()
    return float(per_device.groupby("evaluation_domain").ae.mean().mean())


def fit_tcn_select(
    initial: dict[str, torch.Tensor], train: pd.DataFrame, validation: pd.DataFrame,
    features: list[str], state: RobustState, maximum_lifetime: float,
    config: Any, seed: int, device: torch.device,
) -> tuple[DynamicsTCN, int, float]:
    train_x, train_y, _ = make_multitask_windows(train, features, state, config)
    val_x, val_y, val_meta = make_multitask_windows(validation, features, state, config)
    val_meta["evaluation_domain"] = val_meta.unit_id.str.split("::").str[0]
    seed_all(seed)
    model = DynamicsTCN(len(features), config)
    model.load_state_dict(copy.deepcopy(initial))
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
    loader = dynamics_loader(train_x, train_y, config.batch_size, seed + 101)
    best, best_epoch, best_mae, stale = snapshot(model), 1, float("inf"), 0
    for epoch in range(1, config.adapt_epochs + 1):
        set_dynamics_trainable(model, epoch, config)
        model.train()
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.dynamics(values)
            cap = torch.clamp(maximum_lifetime - labels[:, 6], min=0.0)
            loss, _ = multitask_loss(outputs, labels, cap)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        prediction = predict_dynamics(model, val_x, val_meta, maximum_lifetime, config, device)
        prediction["predicted_rul_raw"] = prediction.physical_rul
        mae = macro_device_mae(prediction)
        if mae < best_mae - 1e-7:
            best, best_epoch, best_mae, stale = snapshot(model), epoch, mae, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_mae


def head_dataset(bundle: SequenceBundle, modes: np.ndarray, rates: np.ndarray) -> TensorDataset:
    arrays = (
        bundle.local, bundle.local_soh, bundle.stats, bundle.times,
        bundle.target_soh, bundle.true_rul, bundle.eval_mask,
        bundle.unit_weight, modes, rates,
    )
    return TensorDataset(*(torch.from_numpy(item) for item in arrays))


def head_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], kind: str) -> torch.Tensor:
    _local, _local_soh, _stats, _times, true_soh, true_rul, eval_mask, unit_weight, modes, rates = batch
    valid = eval_mask.bool()

    def device_mean(point: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per = (point * mask).sum(1) / mask.sum(1).clamp_min(1)
        weight = unit_weight / unit_weight.sum().clamp_min(1e-8)
        return torch.sum(per * weight)

    soh = device_mean(F.smooth_l1_loss(outputs["soh"], true_soh, beta=0.01, reduction="none"), valid)
    rul = device_mean(F.smooth_l1_loss(
        torch.log1p(outputs["rul"]), torch.log1p(true_rul), beta=0.10, reduction="none",
    ), valid)
    dynamics = device_mean(F.smooth_l1_loss(
        100.0 * outputs["rate"], 100.0 * rates, beta=0.01, reduction="none",
    ), valid)
    total = soh + 0.5 * rul + 0.2 * dynamics
    if kind == "multi_expert":
        gate = F.cross_entropy(outputs["gate_logits"][valid], modes[valid])
        selected = torch.gather(outputs["expert_rul"], -1, modes.unsqueeze(-1)).squeeze(-1)
        expert = device_mean(F.smooth_l1_loss(
            torch.log1p(selected), torch.log1p(true_rul), beta=0.10, reduction="none",
        ), valid)
        total = total + 0.1 * gate + 0.1 * expert
    return total


def predict_head(
    model: GeneralizationHead, bundle: SequenceBundle, modes: np.ndarray,
    maximum_lifetime: float, device: torch.device, batch_devices: int,
) -> pd.DataFrame:
    model.eval()
    rows, offset = [], 0
    loader = DataLoader(head_dataset(bundle, modes, np.zeros_like(bundle.times)), batch_size=batch_devices)
    with torch.no_grad():
        for raw in loader:
            local, local_soh, stats, times = (value.to(device) for value in raw[:4])
            outputs = model(local, local_soh, stats, times, maximum_lifetime)
            for index in range(len(local)):
                source = offset + index
                valid = bundle.eval_mask[source]
                data: dict[str, Any] = {
                    "unit_id": bundle.unit_ids[source],
                    "evaluation_domain": bundle.unit_ids[source].split("::", 1)[0],
                    "time": bundle.times[source, valid],
                    "target_soh": bundle.target_soh[source, valid],
                    "true_rul_cycles": bundle.true_rul[source, valid],
                    "true_eol_cycle": bundle.true_eol[source, valid],
                    "predicted_soh": outputs["soh"][index].cpu().numpy()[valid],
                    "predicted_rul_raw": outputs["rul"][index].cpu().numpy()[valid],
                    "true_mode": modes[source, valid],
                }
                if model.kind == "multi_expert":
                    probabilities = outputs["gate_probability"][index].cpu().numpy()[valid]
                    experts = outputs["expert_rul"][index].cpu().numpy()[valid]
                    for column, name in enumerate(MODE_NAMES):
                        data[f"gate_{name}"] = probabilities[:, column]
                        data[f"expert_rul_{name}"] = experts[:, column]
                rows.append(pd.DataFrame(data))
            offset += len(local)
    return pd.concat(rows, ignore_index=True)


def train_head_select(
    kind: str, train: SequenceBundle, validation: SequenceBundle,
    train_modes: np.ndarray, train_rates: np.ndarray, val_modes: np.ndarray,
    val_rates: np.ndarray, maximum_lifetime: float, seed: int,
    config: HeadConfig, device: torch.device,
) -> tuple[GeneralizationHead, int, float]:
    seed_all(seed + {"single": 101, "wide": 202, "multi_expert": 303}[kind])
    model = GeneralizationHead(kind, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        head_dataset(train, train_modes, train_rates), batch_size=config.batch_devices,
        shuffle=True, generator=torch.Generator().manual_seed(seed + 71),
    )
    best, best_epoch, best_mae, stale = copy.deepcopy(model.state_dict()), 1, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw in loader:
            batch = tuple(value.to(device) for value in raw)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(*batch[:4], maximum_lifetime)
            loss = head_loss(outputs, batch, kind)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_head(model, validation, val_modes, maximum_lifetime, device, config.batch_devices)
        mae = macro_device_mae(prediction)
        if mae < best_mae - 1e-7:
            best, best_epoch, best_mae, stale = copy.deepcopy(model.state_dict()), epoch, mae, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best)
    return model, best_epoch, best_mae


def fit_head_fixed(
    kind: str, train: SequenceBundle, modes: np.ndarray, rates: np.ndarray,
    maximum_lifetime: float, seed: int, epochs: int, config: HeadConfig,
    device: torch.device,
) -> GeneralizationHead:
    seed_all(seed + {"single": 101, "wide": 202, "multi_expert": 303}[kind])
    model = GeneralizationHead(kind, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        head_dataset(train, modes, rates), batch_size=config.batch_devices, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 71),
    )
    for _ in range(epochs):
        model.train()
        for raw in loader:
            batch = tuple(value.to(device) for value in raw)
            optimizer.zero_grad(set_to_none=True)
            loss = head_loss(model(*batch[:4], maximum_lifetime), batch, kind)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def method_spec(method: str) -> tuple[tuple[str, ...], str, str]:
    mapping = {
        "v10_only": (("v10",), "nasa", "single"),
        "v15_only": (("v15",), "nasa", "single"),
        "pooled": (DOMAINS, "nasa", "single"),
        "pooled_wide_control": (DOMAINS, "nasa", "wide"),
        "multi_expert": (DOMAINS, "nasa", "multi_expert"),
        "pooled_target_only": (DOMAINS, "target", "single"),
        "multi_expert_target_only": (DOMAINS, "target", "multi_expert"),
    }
    return mapping[method]


def prepare_scope(
    scope: tuple[str, ...], init_kind: str, seed: int, fold: int,
    combined: pd.DataFrame, reports: pd.DataFrame, features: list[str], source: pd.DataFrame,
    fold_maps: dict[str, dict[str, int]], selection_seed: int, reference: dict[str, Any],
    smoke: bool, output: Path, device: torch.device,
) -> dict[str, Any]:
    train_units, test_units, inner_train_units, inner_val_units = [], [], [], []
    for domain in scope:
        mapping = fold_maps[domain]
        domain_test = sorted(unit for unit, value in mapping.items() if value == fold)
        domain_train = sorted(set(mapping) - set(domain_test))
        selection_train, selection_val = inner_split(
            domain_train, combined.loc[combined.evaluation_domain.eq(domain)],
            selection_seed, fold, 0.20 if smoke else 0.10,
        )
        train_units.extend(domain_train); test_units.extend(domain_test)
        inner_train_units.extend(selection_train); inner_val_units.extend(selection_val)
    outer_train = balanced_weights(subset(combined, train_units))
    inner_train_frame = balanced_weights(subset(combined, inner_train_units))
    inner_val_frame = balanced_weights(subset(combined, inner_val_units))
    all_test_units = []
    for domain in DOMAINS:
        all_test_units.extend(sorted(unit for unit, value in fold_maps[domain].items() if value == fold))
    # Some prepared feature tables retain historical training-weight columns.
    # Test/validation bundles must never consume them: they are neither model
    # inputs nor meaningful outside the current fold, and stale NaNs are
    # rejected by the shared bundle constructor.
    outer_test = subset(combined, all_test_units).drop(
        columns=["sample_weight", "unit_weight"], errors="ignore",
    )
    config = build_config(reference, 24, smoke)
    head_config = HeadConfig(epochs=3 if smoke else 45, patience=2 if smoke else 7)
    maximum_lifetime = float(outer_train.true_eol_cycle.max())
    run_seed = seed + 100 * fold + (0 if scope == DOMAINS else 10000)
    if scope == DOMAINS:
        ssl_state, target_state = pooled_ssl_state(outer_train, features, config, run_seed, device)
    else:
        ssl_state, target_state = target_ssl_state(outer_train, features, config, run_seed, device)
    if set(target_state.fit_units) != set(train_units):
        raise AssertionError("Scaler/SSL fit units differ from outer training scope")
    initial = ssl_state
    if init_kind == "nasa":
        audit_path = output / f"nasa_pretrain_{'_'.join(scope)}_seed_{seed}_fold_{fold}.json"
        initial = nasa_dynamics_initial_state(
            source, outer_train, features, ssl_state, config, run_seed, device, audit_path,
        )
    selected_tcn, tcn_epoch, tcn_mae = fit_tcn_select(
        initial, inner_train_frame, inner_val_frame, features, target_state,
        maximum_lifetime, config, run_seed, device,
    )
    for parameter in selected_tcn.parameters():
        parameter.requires_grad = False
    selection_train = make_bundle(
        inner_train_frame, reports, features, target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device,
    )
    selection_val = make_bundle(
        inner_val_frame, reports, features, target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device,
    )
    fit_statistics(selection_train, selection_val)
    refit_tcn = fit_multitask_fixed_epochs(
        initial, outer_train, features, target_state, maximum_lifetime,
        config, run_seed, tcn_epoch, device,
    )
    for parameter in refit_tcn.parameters():
        parameter.requires_grad = False
    refit_train = make_bundle(
        outer_train, reports, features, target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device,
    )
    refit_test = make_bundle(
        outer_test, reports, features, target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device,
    )
    statistics_median, statistics_iqr = fit_statistics(refit_train, refit_test)
    thresholds = mode_thresholds(outer_train)
    return {
        "scope": scope, "init_kind": init_kind, "outer_train": outer_train,
        "outer_test": outer_test, "selection_train": selection_train,
        "selection_val": selection_val, "refit_train": refit_train,
        "refit_test": refit_test, "target_state": target_state,
        "maximum_lifetime": maximum_lifetime, "config": config,
        "head_config": head_config, "tcn_epoch": tcn_epoch,
        "tcn_inner_macro_mae": tcn_mae, "thresholds": thresholds,
        "train_units": train_units, "test_units": all_test_units,
        "inner_train_units": inner_train_units, "inner_val_units": inner_val_units,
        "refit_tcn": refit_tcn, "statistics_median": statistics_median,
        "statistics_iqr": statistics_iqr,
    }


def run_group(
    seed: int, fold: int, methods: tuple[str, ...], source: pd.DataFrame,
    combined: pd.DataFrame, reports: pd.DataFrame, features: list[str],
    fold_maps: dict[str, dict[str, int]], selection_seed: int,
    reference: dict[str, Any], smoke: bool, output: Path, device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    predictions, method_rows = [], {}
    for method in methods:
        scope, init_kind, kind = method_spec(method)
        key = (scope, init_kind)
        if key not in cache:
            cache[key] = prepare_scope(
                scope, init_kind, seed, fold, combined, reports, features, source,
                fold_maps, selection_seed, reference, smoke, output, device,
            )
        state = cache[key]
        selection_modes, selection_rates = mode_and_rate_arrays(
            state["selection_train"], reports, state["thresholds"],
        )
        val_modes, val_rates = mode_and_rate_arrays(
            state["selection_val"], reports, state["thresholds"],
        )
        _, head_epoch, head_mae = train_head_select(
            kind, state["selection_train"], state["selection_val"],
            selection_modes, selection_rates, val_modes, val_rates,
            state["maximum_lifetime"], seed + 100 * fold,
            state["head_config"], device,
        )
        train_modes, train_rates = mode_and_rate_arrays(
            state["refit_train"], reports, state["thresholds"],
        )
        test_modes, _ = mode_and_rate_arrays(
            state["refit_test"], reports, state["thresholds"],
        )
        head = fit_head_fixed(
            kind, state["refit_train"], train_modes, train_rates,
            state["maximum_lifetime"], seed + 100 * fold, head_epoch,
            state["head_config"], device,
        )
        prediction = predict_head(
            head, state["refit_test"], test_modes, state["maximum_lifetime"],
            device, state["head_config"].batch_devices,
        )
        prediction["method"] = method
        prediction["seed"] = seed
        prediction["outer_fold"] = fold
        predictions.append(prediction)
        parameter_count = sum(parameter.numel() for parameter in head.parameters())
        method_rows[method] = {
            "scope": list(scope), "initialization": init_kind, "head_kind": kind,
            "tcn_epoch": int(state["tcn_epoch"]), "head_epoch": int(head_epoch),
            "inner_head_macro_device_mae": float(head_mae),
            "outer_macro_device_mae": macro_device_mae(prediction),
            "head_parameter_count": int(parameter_count),
            "outer_train_units": state["train_units"],
            "outer_test_units": state["test_units"],
            "inner_train_units": state["inner_train_units"],
            "inner_validation_units": state["inner_val_units"],
        }
        torch.save({
            "model_state": snapshot(head), "tcn_state": snapshot(state["refit_tcn"]),
            "method": method, "seed": seed,
            "outer_fold": fold, "head_kind": kind, "head_epoch": head_epoch,
            "head_configuration": asdict(state["head_config"]),
            "features": features, "maximum_lifetime": state["maximum_lifetime"],
            "statistics_median": state["statistics_median"],
            "statistics_iqr": state["statistics_iqr"],
            "feature_median": state["target_state"].median,
            "feature_iqr": state["target_state"].iqr,
            "outer_train_units": state["train_units"],
            "outer_test_units": state["test_units"],
            "domain_identity_input": False, "sealed_accessed": False,
        }, output / f"checkpoint_head_{method}_seed_{seed}_fold_{fold}.pt")
    if "multi_expert" in method_rows and "pooled_wide_control" in method_rows:
        multi = method_rows["multi_expert"]["head_parameter_count"]
        wide = method_rows["pooled_wide_control"]["head_parameter_count"]
        if abs(multi - wide) / multi > 0.05:
            raise AssertionError("Wide control is not parameter matched within 5%")
    return pd.concat(predictions, ignore_index=True), {
        "seed": seed, "outer_fold": fold, "methods": method_rows,
        "validation_accessed": False, "sealed_accessed": False,
    }


def ensemble_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "evaluation_domain", "unit_id", "time", "true_rul_cycles", "target_soh", "true_eol_cycle", "true_mode"]
    values = [column for column in frame if column.startswith("gate_") or column.startswith("expert_rul_")]
    return frame.groupby(keys, as_index=False).agg({
        "predicted_rul_raw": "mean", "predicted_soh": "mean", **{column: "mean" for column in values},
    })


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_frame in frame.groupby("method"):
        domain_device = method_frame.assign(
            absolute_error=np.abs(method_frame.predicted_rul_raw - method_frame.true_rul_cycles)
        ).groupby(["evaluation_domain", "unit_id"], as_index=False).absolute_error.mean()
        domain_mae = domain_device.groupby("evaluation_domain").absolute_error.mean().to_dict()
        for domain, group in method_frame.groupby("evaluation_domain"):
            error = group.predicted_rul_raw.to_numpy(float) - group.true_rul_cycles.to_numpy(float)
            device = domain_device.loc[domain_device.evaluation_domain.eq(domain)].absolute_error
            rows.append({
                "method": method, "domain": domain,
                "device_macro_mae": float(domain_mae[domain]),
                "point_mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error)),
                "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
                "worst_device_mae": float(device.max()),
                "soh_mae": float(np.mean(np.abs(group.predicted_soh - group.target_soh))),
            })
        rows.append({
            "method": method, "domain": "macro",
            "device_macro_mae": float(np.mean(list(domain_mae.values()))),
            "point_mae": float(np.mean(np.abs(method_frame.predicted_rul_raw - method_frame.true_rul_cycles))),
        })
    return pd.DataFrame(rows)


def screening_decision(by_seed: pd.DataFrame, ensemble_metrics: pd.DataFrame) -> dict[str, Any]:
    table = ensemble_metrics.pivot(index="method", columns="domain", values="device_macro_mae")
    required = {"v10_only", "pooled", "pooled_wide_control", "multi_expert"}
    if not required <= set(table.index):
        return {"passed": False, "reason": "required_methods_missing"}
    seed_table = by_seed.loc[by_seed.domain.eq("macro")].pivot(
        index="seed", columns="method", values="device_macro_mae",
    )
    wins = int((seed_table.multi_expert < seed_table.pooled).sum())
    conditions = {
        "macro_improvement_at_least_2_percent": float(table.loc["multi_expert", "macro"])
        <= 0.98 * float(table.loc["pooled", "macro"]),
        "at_least_two_of_three_seeds": wins >= 2,
        "v10_within_5_percent": float(table.loc["multi_expert", "v10"])
        <= 1.05 * float(table.loc["v10_only", "v10"]),
        "v15_not_worse_than_pooled": float(table.loc["multi_expert", "v15"])
        <= float(table.loc["pooled", "v15"]),
        "worst_domain_not_worse": float(table.loc["multi_expert", ["v10", "v15"]].max())
        <= float(table.loc["pooled", ["v10", "v15"]].max()),
        "beats_parameter_control_by_1_percent": float(table.loc["multi_expert", "macro"])
        <= 0.99 * float(table.loc["pooled_wide_control", "macro"]),
    }
    return {"passed": bool(all(conditions.values())), "conditions": conditions, "seed_wins": wins}


def paired_bootstrap(frame: pd.DataFrame, candidate: str, control: str, draws: int = 10000) -> dict[str, float]:
    selected = frame.loc[frame.method.isin([candidate, control])].copy()
    selected["ae"] = np.abs(selected.predicted_rul_raw - selected.true_rul_cycles)
    device = selected.groupby(["method", "evaluation_domain", "unit_id"], as_index=False).ae.mean()
    pivot = device.pivot(index=["evaluation_domain", "unit_id"], columns="method", values="ae").dropna()
    difference = pivot[candidate].to_numpy() - pivot[control].to_numpy()
    rng = np.random.default_rng(202608)
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = float(np.mean(rng.choice(difference, len(difference), replace=True)))
    return {
        "candidate_minus_control": float(np.mean(difference)),
        "p025": float(np.quantile(samples, 0.025)),
        "p50": float(np.quantile(samples, 0.50)),
        "p975": float(np.quantile(samples, 0.975)),
        "draws": draws,
    }


def detailed_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Report causal operating regions without using them for model selection."""
    working = frame.copy()
    unit_life = working[["evaluation_domain", "unit_id", "true_eol_cycle"]].drop_duplicates()
    quartile_parts = []
    for _domain, group in unit_life.groupby("evaluation_domain"):
        ranked = group.true_eol_cycle.rank(method="first")
        group = group.copy()
        group["lifetime_quartile"] = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        quartile_parts.append(group)
    working = working.merge(
        pd.concat(quartile_parts)[["evaluation_domain", "unit_id", "lifetime_quartile"]],
        on=["evaluation_domain", "unit_id"], how="left", validate="many_to_one",
    )
    slices: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", pd.Series(True, index=working.index)),
        ("cycle", "7-15", working.time.between(7, 15)),
        ("rul", ">60", working.true_rul_cycles > 60),
    ]
    slices.extend(
        ("mode", name, working.true_mode.eq(index)) for index, name in enumerate(MODE_NAMES)
    )
    slices.extend(
        ("lifetime_quartile", name, working.lifetime_quartile.astype(str).eq(name))
        for name in ("Q1", "Q2", "Q3", "Q4")
    )
    rows = []
    for method in sorted(working.method.unique()):
        for domain in DOMAINS:
            base = working.method.eq(method) & working.evaluation_domain.eq(domain)
            for dimension, level, condition in slices:
                group = working.loc[base & condition]
                if group.empty:
                    continue
                group = group.assign(ae=np.abs(group.predicted_rul_raw - group.true_rul_cycles))
                device = group.groupby("unit_id").ae.mean()
                error = group.predicted_rul_raw.to_numpy(float) - group.true_rul_cycles.to_numpy(float)
                rows.append({
                    "method": method, "evaluation_domain": domain,
                    "dimension": dimension, "level": level,
                    "samples": int(len(group)), "devices": int(group.unit_id.nunique()),
                    "device_macro_mae": float(device.mean()), "point_mae": float(group.ae.mean()),
                    "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error)),
                    "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
                    "worst_device_mae": float(device.max()),
                })
    return pd.DataFrame(rows)


def expert_usage_table(frame: pd.DataFrame) -> pd.DataFrame:
    gate_columns = [f"gate_{name}" for name in MODE_NAMES if f"gate_{name}" in frame]
    if not gate_columns:
        return pd.DataFrame()
    selected = frame.loc[frame[gate_columns].notna().any(axis=1)].copy()
    rows = []
    for (method, domain), group in selected.groupby(["method", "evaluation_domain"]):
        probability = group[gate_columns].mean()
        hard = group[gate_columns].idxmax(axis=1).value_counts(normalize=True)
        row: dict[str, Any] = {"method": method, "evaluation_domain": domain}
        for column in gate_columns:
            row[f"mean_{column}"] = float(probability[column])
            row[f"hard_usage_{column}"] = float(hard.get(column, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def formal_acceptance(
    frame: pd.DataFrame, metrics: pd.DataFrame, bootstrap: dict[str, Any],
) -> dict[str, Any]:
    if frame.seed.nunique() != 5:
        return {"passed": False, "reason": "five_formal_seeds_not_available"}
    table = metrics.pivot(index="method", columns="domain", values="device_macro_mae")
    required = {"multi_expert", "multi_expert_target_only", "v10_only"}
    if not required <= set(table.index):
        return {"passed": False, "reason": "required_methods_missing"}
    seed_rows, fold_rows = [], []
    for seed, group in frame.groupby("seed"):
        values = metric_table(group)
        seed_rows.append(values.loc[values.domain.eq("macro")].set_index("method").device_macro_mae.rename(seed))
    for fold, group in frame.groupby("outer_fold"):
        values = metric_table(ensemble_predictions(group))
        fold_rows.append(values.loc[values.domain.eq("macro")].set_index("method").device_macro_mae.rename(fold))
    seed_table = pd.DataFrame(seed_rows)
    fold_table = pd.DataFrame(fold_rows)
    comparison = bootstrap.get("multi_expert_vs_target_only", {})
    conditions = {
        "nasa_gain_at_least_2_percent": float(table.loc["multi_expert", "macro"])
        <= 0.98 * float(table.loc["multi_expert_target_only", "macro"]),
        "bootstrap_upper_below_zero": float(comparison.get("p975", math.inf)) < 0.0,
        "at_least_four_of_five_seeds": int((seed_table.multi_expert < seed_table.multi_expert_target_only).sum()) >= 4,
        "at_least_four_of_five_folds": int((fold_table.multi_expert < fold_table.multi_expert_target_only).sum()) >= 4,
        "v10_within_5_percent": float(table.loc["multi_expert", "v10"])
        <= 1.05 * float(table.loc["v10_only", "v10"]),
    }
    return {
        "passed": bool(all(conditions.values())), "conditions": conditions,
        "seed_wins": int((seed_table.multi_expert < seed_table.multi_expert_target_only).sum()),
        "fold_wins": int((fold_table.multi_expert < fold_table.multi_expert_target_only).sum()),
    }


def leakage_audit(
    combined: pd.DataFrame, features: list[str], fold_maps: dict[str, dict[str, int]],
    results: list[dict[str, Any]], methods: tuple[str, ...], folds: int,
) -> dict[str, Any]:
    violations = []
    forbidden = [name for name in features if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if forbidden:
        violations.append(f"forbidden_features:{forbidden}")
    for domain in DOMAINS:
        mapping = fold_maps[domain]
        expected = set(combined.loc[combined.evaluation_domain.eq(domain), "unit_id"].unique())
        if set(mapping) != expected:
            violations.append(f"incomplete_fold_map:{domain}")
        counts = pd.Series(mapping).value_counts()
        if set(counts.index) != set(range(folds)) or int(counts.max() - counts.min()) > 1:
            violations.append(f"unbalanced_fold_map:{domain}")
    for result in results:
        for method, row in result["methods"].items():
            train, test = set(row["outer_train_units"]), set(row["outer_test_units"])
            inner_train, inner_val = set(row["inner_train_units"]), set(row["inner_validation_units"])
            if train & test or inner_train & inner_val or not (inner_train | inner_val) <= train:
                violations.append(f"device_overlap:{method}:seed{result['seed']}:fold{result['outer_fold']}")
    return {
        "passed": not violations, "violations": violations,
        "feature_count": len(features), "domain_identity_input": False,
        "future_information_input": False, "validation_accessed_during_selection": False,
        "sealed_accessed": False,
    }


def selected_epochs(results: list[dict[str, Any]], method: str, seed: int) -> tuple[int, int]:
    rows = [
        result["methods"][method] for result in results
        if result["seed"] == seed and method in result["methods"]
    ]
    if len(rows) != 5:
        raise ValueError(f"Expected five OOF epoch selections for {method}, seed {seed}")
    return (
        max(1, int(round(float(np.median([row["tcn_epoch"] for row in rows]))))),
        max(1, int(round(float(np.median([row["head_epoch"] for row in rows]))))),
    )


def evaluate_frozen_validation(
    roots: tuple[Path, Path], methods: tuple[str, ...], seeds: tuple[int, ...],
    source: pd.DataFrame, combined: pd.DataFrame, reports: pd.DataFrame,
    features: list[str], reference: dict[str, Any], results: list[dict[str, Any]],
    output: Path, device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the OOF-frozen recipes on all 320+320 devices, then open validation once."""
    validation_parts = [read_domain(root, domain, validation=True)[0] for root, domain in zip(roots, DOMAINS)]
    validation = pd.concat(validation_parts, ignore_index=True).drop(
        columns=["sample_weight", "unit_weight"], errors="ignore",
    )
    validation_reports = dynamics_report(validation)
    all_reports = pd.concat([reports, validation_reports], ignore_index=True)
    predictions = []
    for seed in seeds:
        cache: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
        for method in methods:
            scope, init_kind, kind = method_spec(method)
            key = (scope, init_kind)
            if key not in cache:
                train_units = combined.loc[combined.evaluation_domain.isin(scope), "unit_id"].unique()
                full_train = balanced_weights(subset(combined, train_units))
                config = build_config(reference, 24, False)
                maximum_lifetime = float(full_train.true_eol_cycle.max())
                run_seed = seed + (0 if scope == DOMAINS else 10000)
                if scope == DOMAINS:
                    ssl_state, target_state = pooled_ssl_state(full_train, features, config, run_seed, device)
                else:
                    ssl_state, target_state = target_ssl_state(full_train, features, config, run_seed, device)
                initial = ssl_state
                if init_kind == "nasa":
                    initial = nasa_dynamics_initial_state(
                        source, full_train, features, ssl_state, config, run_seed, device,
                        output / f"final_nasa_pretrain_{'_'.join(scope)}_seed_{seed}.json",
                    )
                tcn_epoch, _ = selected_epochs(results, method, seed)
                tcn = fit_multitask_fixed_epochs(
                    initial, full_train, features, target_state, maximum_lifetime,
                    config, run_seed, tcn_epoch, device,
                )
                for parameter in tcn.parameters():
                    parameter.requires_grad = False
                train_bundle = make_bundle(
                    full_train, reports, features, target_state.median, target_state.iqr,
                    tcn, config, maximum_lifetime, device,
                )
                validation_bundle = make_bundle(
                    validation, all_reports, features, target_state.median, target_state.iqr,
                    tcn, config, maximum_lifetime, device,
                )
                statistics_median, statistics_iqr = fit_statistics(train_bundle, validation_bundle)
                cache[key] = {
                    "train": full_train, "train_bundle": train_bundle,
                    "validation_bundle": validation_bundle, "maximum_lifetime": maximum_lifetime,
                    "thresholds": mode_thresholds(full_train), "target_state": target_state,
                    "tcn": tcn, "tcn_epoch": tcn_epoch,
                    "statistics_median": statistics_median, "statistics_iqr": statistics_iqr,
                }
            state = cache[key]
            _, head_epoch = selected_epochs(results, method, seed)
            train_modes, train_rates = mode_and_rate_arrays(
                state["train_bundle"], reports, state["thresholds"],
            )
            validation_modes, _ = mode_and_rate_arrays(
                state["validation_bundle"], all_reports, state["thresholds"],
            )
            head_config = HeadConfig()
            head = fit_head_fixed(
                kind, state["train_bundle"], train_modes, train_rates,
                state["maximum_lifetime"], seed, head_epoch, head_config, device,
            )
            prediction = predict_head(
                head, state["validation_bundle"], validation_modes,
                state["maximum_lifetime"], device, head_config.batch_devices,
            )
            prediction["method"], prediction["seed"] = method, seed
            predictions.append(prediction)
            torch.save({
                "model_state": snapshot(head), "tcn_state": snapshot(state["tcn"]),
                "method": method, "seed": seed, "head_kind": kind,
                "tcn_epoch": state["tcn_epoch"], "head_epoch": head_epoch,
                "features": features, "maximum_lifetime": state["maximum_lifetime"],
                "feature_median": state["target_state"].median,
                "feature_iqr": state["target_state"].iqr,
                "statistics_median": state["statistics_median"],
                "statistics_iqr": state["statistics_iqr"],
                "training_units": sorted(state["train"].unit_id.unique()),
                "validation_accessed_after_freeze": True, "sealed_accessed": False,
            }, output / f"final_checkpoint_{method}_seed_{seed}.pt")
    by_seed = pd.concat(predictions, ignore_index=True)
    ensemble = ensemble_predictions(by_seed)
    return by_seed, ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", default=",".join(map(str, DEFAULT_DATA)))
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=202608)
    parser.add_argument("--tuning-seed", type=int, default=51)
    parser.add_argument("--screen-seeds", default="52,53,54")
    parser.add_argument("--additional-seeds", default="55,56")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = tuple(Path(value).resolve() for value in args.data_roots.split(","))
    if len(roots) != 2:
        raise ValueError("Exactly V1.0 and V1.5 data roots are required")
    methods = tuple(args.methods.split(","))
    if not set(methods) <= set(ALL_METHODS):
        raise ValueError("Unknown method requested")
    if args.outer_folds != 5 and not args.smoke:
        raise ValueError("Formal experiment requires five folds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, v10, v15, features = load_training_data(roots)
    if args.smoke:
        keep10 = sorted(v10.unit_id.unique())[:40]
        keep15 = sorted(v15.unit_id.unique())[:40]
        v10, v15 = subset(v10, keep10), subset(v15, keep15)
    combined = pd.concat([v10, v15], ignore_index=True)
    reports = dynamics_report(combined)
    folds = 2 if args.smoke else args.outer_folds
    fold_maps = {
        domain: stratified_folds(
            combined.loc[combined.evaluation_domain.eq(domain)],
            reports.loc[reports.evaluation_domain.eq(domain)], folds, args.split_seed,
        ) for domain in DOMAINS
    }
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    screen_seeds = (52,) if args.smoke else tuple(map(int, args.screen_seeds.split(",")))
    additional = () if args.smoke else tuple(map(int, args.additional_seeds.split(",")))
    frames, results = [], []

    def run_seeds(seeds: Iterable[int]) -> None:
        for seed in seeds:
            for fold in range(folds):
                prediction_path = args.output_dir / f"fold_predictions_seed_{seed}_fold_{fold}.csv"
                result_path = args.output_dir / f"fold_result_seed_{seed}_fold_{fold}.json"
                if args.resume and prediction_path.is_file() and result_path.is_file():
                    prediction = pd.read_csv(prediction_path)
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    prediction, result = run_group(
                        seed, fold, methods, source, combined, reports, features,
                        fold_maps, args.tuning_seed, reference, args.smoke,
                        args.output_dir, device,
                    )
                    prediction.to_csv(prediction_path, index=False)
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                frames.append(prediction); results.append(result)
                current = pd.concat(frames, ignore_index=True)
                current.to_csv(args.output_dir / "oof_predictions_by_seed.csv", index=False)
                print(json.dumps({"seed": seed, "fold": fold, "groups_complete": len(results)}), flush=True)

    run_seeds(screen_seeds)
    screen_frame = pd.concat(frames, ignore_index=True)
    screen_ensemble = ensemble_predictions(screen_frame)
    screen_metrics = metric_table(screen_ensemble)
    seed_metrics = pd.concat([
        metric_table(group).assign(seed=seed)
        for seed, group in screen_frame.groupby("seed")
    ], ignore_index=True)
    decision = screening_decision(seed_metrics, screen_metrics)
    (args.output_dir / "screening_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if decision.get("passed") and additional:
        run_seeds(additional)
    final_frame = pd.concat(frames, ignore_index=True)
    ensemble = ensemble_predictions(final_frame)
    metrics = metric_table(ensemble)
    ensemble.to_csv(args.output_dir / "oof_ensemble_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "model_summary.csv", index=False)
    detailed_metric_table(ensemble).to_csv(
        args.output_dir / "domain_mechanism_metrics.csv", index=False,
    )
    expert_usage_table(ensemble).to_csv(args.output_dir / "expert_usage.csv", index=False)
    bootstrap = {}
    if final_frame.seed.nunique() == 5 and {"multi_expert", "multi_expert_target_only"} <= set(methods):
        bootstrap["multi_expert_vs_target_only"] = paired_bootstrap(
            ensemble, "multi_expert", "multi_expert_target_only",
        )
    (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    acceptance = formal_acceptance(final_frame, metrics, bootstrap)
    (args.output_dir / "formal_acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8",
    )
    audit = leakage_audit(combined, features, fold_maps, results, methods, folds)
    (args.output_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["passed"]:
        raise AssertionError(f"Leakage audit failed: {audit['violations']}")
    validation_accessed = False
    validation_hashes: dict[str, str] = {}
    if acceptance.get("passed") and not args.smoke and not args.skip_validation:
        validation_by_seed, validation_ensemble = evaluate_frozen_validation(
            roots, methods, tuple(sorted(final_frame.seed.unique())), source, combined,
            reports, features, reference, results, args.output_dir, device,
        )
        validation_by_seed.to_csv(args.output_dir / "validation_predictions_by_seed.csv", index=False)
        validation_ensemble.to_csv(args.output_dir / "validation_ensemble_predictions.csv", index=False)
        metric_table(validation_ensemble).to_csv(args.output_dir / "validation_model_summary.csv", index=False)
        detailed_metric_table(validation_ensemble).to_csv(
            args.output_dir / "validation_domain_mechanism_metrics.csv", index=False,
        )
        validation_accessed = True
        validation_hashes = {
            str(root): file_sha256(root / "basilisk_validation_rich.csv") for root in roots
        }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "bhump-v10-v15-generalization-v1",
        "methods": methods, "screen_seeds": screen_seeds,
        "additional_seeds_run": sorted(set(final_frame.seed) - set(screen_seeds)),
        "features": features, "feature_count": len(features), "tcn_window": 24,
        "history_statistics": 38, "source_units": list(NASA5),
        "device": str(device), "screening_decision": decision,
        "formal_acceptance": acceptance,
        "data_hashes": {
            str(root): file_sha256(root / "basilisk_train_rich.csv") for root in roots
        },
        "domain_identity_is_model_input": False,
        "future_information_is_model_input": False,
        "validation_hashes": validation_hashes,
        "validation_accessed": validation_accessed, "sealed_accessed": False,
        "note": "Validation is opened only after the five-seed OOF acceptance gate passes.",
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
