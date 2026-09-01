"""V1.0-anchored continual transfer over small V1.5 mechanism domains.

The script inherits the registered full320 TCN24+B_stats38 checkpoints, keeps
the V1.0 compact16/scalers frozen, and adds V1.5 domains sequentially with
cumulative replay.  Dataset identity is used only by the sampler and loss; it
is never passed to the model.  Public validation/pseudo-blind tables are read
only after the winning structure has been frozen.
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
from torch.func import functional_call

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_positive_transfer import Config, seed_all
from train_bhump_v10_history_ablation import (
    HistoryAblationModel, HistoryConfig, causal_statistics, encode_local, local_windows,
)
from train_bhump_v10_rul_multitask import DynamicsTCN


ROOT = Path(__file__).resolve().parent
DEFAULT_V10 = ROOT / "bhump_transfer_v10_data"
DEFAULT_V15 = ROOT / "bhump_transfer_v15_nasa5_ecm_data"
DEFAULT_MICRO = ROOT / "bhump_transfer_v15_microdomains_data" / "formal"
DEFAULT_NASA_PARENT = ROOT / "bhump_v10_nasa_dynamics_full320_runs"
DEFAULT_TARGET_PARENT = ROOT / "bhump_v10_full320_target_control_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_v15_continual_runs"
MICRODOMAINS = ("knee_spectrum", "thermal_load", "decoupled_aging", "path_nonstationary")
METHODS = ("inherited_ft", "inherited_mldg", "inherited_mldg_groupdro")
EOL_SOH = 0.80


@dataclass(frozen=True)
class TrainConfig:
    head_only_epochs: int = 3
    last_block_epochs: int = 5
    maximum_epochs: int = 20
    patience: int = 5
    batch_size: int = 96
    steps_per_epoch: int = 12
    head_learning_rate: float = 5.0e-4
    encoder_learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-4
    l2sp_weight: float = 1.0e-4
    mldg_beta: float = 0.5
    mldg_inner_rate: float = 2.5e-4
    groupdro_eta: float = 0.05
    maximum_lifetime: float = 130.0


@dataclass
class DomainSamples:
    domain: str
    unit_ids: np.ndarray
    windows: np.ndarray
    stats: np.ndarray
    times: np.ndarray
    soh: np.ndarray
    rul: np.ndarray
    unit_to_rows: dict[str, np.ndarray]

    def sample(self, count: int, rng: np.random.Generator, device: torch.device) -> tuple[torch.Tensor, ...]:
        units = np.asarray(sorted(self.unit_to_rows), dtype=object)
        chosen = rng.choice(units, size=count, replace=len(units) < count)
        rows = np.asarray([rng.choice(self.unit_to_rows[str(unit)]) for unit in chosen], dtype=int)
        return tuple(torch.from_numpy(value[rows]).to(device) for value in (
            self.windows, self.stats, self.times, self.soh, self.rul,
        ))


class ContinualRULModel(nn.Module):
    """Exact parent TCN and B_stats modules joined for end-to-end adaptation."""

    def __init__(self, tcn: DynamicsTCN, head: HistoryAblationModel) -> None:
        super().__init__()
        self.tcn = tcn
        self.head = head

    def forward(self, windows: torch.Tensor, stats: torch.Tensor,
                times: torch.Tensor, maximum_lifetime: float) -> dict[str, torch.Tensor]:
        local_soh, local, _private = self.tcn(windows, "target")
        raw = windows[:, -1].unsqueeze(1)
        outputs = self.head(
            raw, local.unsqueeze(1), local_soh.unsqueeze(1), stats.unsqueeze(1),
            times.unsqueeze(1), maximum_lifetime,
        )
        return {name: value.squeeze(1) for name, value in outputs.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_config(payload: dict[str, Any]) -> Config:
    values = dict(payload)
    values["channels"] = tuple(values["channels"])
    return Config(**values)


def parent_path(directory: Path, seed: int, lineage: str) -> Path:
    if lineage == "nasa":
        name = f"checkpoint_full320_nasa_dynamics_adaptive_seed_{seed}.pt"
    else:
        name = f"checkpoint_full320_target_controls_seed_{seed}.pt"
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_parent(path: Path, expected_seed: int, device: torch.device) -> tuple[
    ContinualRULModel, dict[str, Any], Config, HistoryConfig,
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["seed"]) != expected_seed:
        raise AssertionError(f"Seed-chain violation: requested {expected_seed}, checkpoint has {payload['seed']}")
    if len(payload["features"]) != 16:
        raise ValueError("Parent does not use compact16")
    config = load_config(payload["tcn_configuration"])
    history_config = HistoryConfig(**payload["history_configuration"])
    tcn = DynamicsTCN(16, config)
    head = HistoryAblationModel("B_stats", history_config)
    tcn.load_state_dict(payload["model_state_tcn"], strict=True)
    head.load_state_dict(payload["model_state_bstats"], strict=True)
    model = ContinualRULModel(tcn, head).to(device)
    expected = state_sha256({
        **{f"tcn.{k}": v for k, v in payload["model_state_tcn"].items()},
        **{f"head.{k}": v for k, v in payload["model_state_bstats"].items()},
    })
    actual = state_sha256(model_state(model))
    if expected != actual:
        raise AssertionError("Parent parameters changed while constructing continual model")
    return model, payload, config, history_config


def read_contract(root: Path) -> list[str]:
    payload = json.loads((root / "feature_contracts.json").read_text(encoding="utf-8"))
    features = list(payload["bhump_degradation_invariant"])
    if len(features) != 16:
        raise ValueError(f"Expected compact16 at {root}, received {len(features)}")
    leaked = [name for name in features if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)]
    if leaked:
        raise ValueError(f"Forbidden features: {leaked}")
    return features


def read_table(root: Path, domain: str, split: str, expected_features: list[str]) -> pd.DataFrame:
    features = read_contract(root)
    if features != expected_features:
        raise ValueError(f"Frozen compact16 mismatch for {domain}")
    filename = "basilisk_train_rich.csv" if split == "train" else "basilisk_validation_rich.csv"
    frame = pd.read_csv(root / filename)
    required = {"unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle", *features}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{domain} missing columns: {sorted(missing)}")
    numeric = frame[[*features, "target_soh", "true_rul_cycles", "true_eol_cycle"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite values in {domain}/{split}")
    frame = frame.copy()
    frame["original_unit_id"] = frame.unit_id.astype(str)
    frame["unit_id"] = domain + "::" + frame.original_unit_id
    frame["evaluation_domain"] = domain
    return frame


def knee_label(unit: pd.DataFrame) -> str:
    ordered = unit.sort_values("time")
    values = ordered.target_soh.to_numpy(float)
    if len(values) < 20:
        return "unknown"
    middle = len(values) // 2
    pre = np.polyfit(np.arange(middle), values[:middle], 1)[0]
    post = np.polyfit(np.arange(len(values) - middle), values[middle:], 1)[0]
    if pre < 0 and post < 0 and abs(post) >= 1.25 * abs(pre):
        return "knee"
    return "smooth"


def unit_strata(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit_id, unit in frame.groupby("unit_id", sort=True):
        rows.append({
            "unit_id": str(unit_id),
            "eol": float(unit.true_eol_cycle.iloc[0]),
            "knee": knee_label(unit),
        })
    report = pd.DataFrame(rows)
    report["life"] = pd.qcut(report.eol.rank(method="first"), 4, labels=False).astype(int)
    report["stratum"] = report.life.astype(str) + "::" + report.knee
    return report


def stratified_units(frame: pd.DataFrame, count: int, seed: int) -> list[str]:
    report = unit_strata(frame)
    if count > len(report):
        raise ValueError(f"Cannot sample {count} from {len(report)} devices")
    rng = np.random.default_rng(seed)
    buckets = {
        str(name): list(rng.permutation(group.unit_id.astype(str).to_numpy()))
        for name, group in report.groupby("stratum", sort=True)
    }
    chosen: list[str] = []
    names = sorted(buckets)
    while len(chosen) < count:
        progressed = False
        for name in names:
            if buckets[name] and len(chosen) < count:
                chosen.append(str(buckets[name].pop()))
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise AssertionError("Stratified selection did not reach requested size")
    return sorted(chosen)


def subset(frame: pd.DataFrame, units: Iterable[str]) -> pd.DataFrame:
    allowed = set(map(str, units))
    return frame.loc[frame.unit_id.astype(str).isin(allowed)].copy()


def prepare_domains(args: argparse.Namespace) -> tuple[
    dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str], dict[str, Any],
]:
    features = read_contract(args.v10_root)
    v10 = read_table(args.v10_root, "v10", "train", features)
    v15 = read_table(args.v15_root, "v15", "train", features)
    bridge_units = stratified_units(v15, args.v15_train_units, args.split_seed)
    domains = {"v10": v10, "v15_bridge": subset(v15, bridge_units)}
    locked: dict[str, pd.DataFrame] = {"v15_blind": subset(v15, set(v15.unit_id) - set(bridge_units))}
    for domain in args.microdomains:
        root = args.micro_root / domain
        domains[domain] = read_table(root, domain, "train", features)
    split_manifest = {
        "v15_bridge_units": bridge_units,
        "v15_blind_units": sorted(set(v15.unit_id.astype(str)) - set(bridge_units)),
        "bridge_selection": "lifetime_quartile_and_offline_knee_stratified_before_model_training",
        "split_seed": args.split_seed,
    }
    return domains, locked, features, split_manifest


def split_inner_domains(domains: dict[str, pd.DataFrame], seed: int, smoke: bool) -> tuple[
    dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]],
]:
    train, monitor, manifest = {}, {}, {}
    for index, (domain, frame) in enumerate(domains.items()):
        count = frame.unit_id.nunique()
        desired = 2 if smoke else (32 if domain == "v10" else 8)
        held_count = min(desired, max(1, count // 5))
        held = stratified_units(frame, held_count, seed + 1009 * (index + 1))
        remaining = sorted(set(frame.unit_id.astype(str)) - set(held))
        train[domain], monitor[domain] = subset(frame, remaining), subset(frame, held)
        manifest[domain] = {"train": remaining, "monitor": held}
    return train, monitor, manifest


def build_samples(frame: pd.DataFrame, domain: str, features: list[str],
                  model: ContinualRULModel, parent: dict[str, Any], config: Config,
                  maximum_lifetime: float, device: torch.device,
                  smoke: bool = False) -> DomainSamples:
    median = np.asarray(parent["target_scaler_median"], dtype=float)
    iqr = np.asarray(parent["target_scaler_iqr"], dtype=float)
    stats_median = np.asarray(parent["bstats_median"], dtype=float)
    stats_iqr = np.asarray(parent["bstats_iqr"], dtype=float)
    windows, stats, times, soh, rul, unit_ids = [], [], [], [], [], []
    model.eval()
    working = frame
    if smoke:
        keep = sorted(frame.unit_id.astype(str).unique())[:4]
        working = subset(frame, keep)
    for unit_id, unit in working.groupby("unit_id", sort=True):
        unit = unit.sort_values("time").reset_index(drop=True)
        raw = ((unit[features].to_numpy(float) - median) / iqr).astype(np.float32)
        local_window = local_windows(raw, config.window)
        _local, local_soh = encode_local(model.tcn, local_window, device)
        unit_times = unit.time.to_numpy(float)
        causal = causal_statistics(raw, local_soh, unit_times, maximum_lifetime)
        causal = ((causal - stats_median) / stats_iqr).astype(np.float32)
        valid = unit_times >= 7.0
        if smoke:
            valid &= unit_times <= 16.0
        indices = np.flatnonzero(valid)
        windows.append(local_window[indices])
        stats.append(causal[indices])
        times.append(unit_times[indices].astype(np.float32))
        soh.append(unit.target_soh.to_numpy(np.float32)[indices])
        rul.append(unit.true_rul_cycles.to_numpy(np.float32)[indices])
        unit_ids.extend([str(unit_id)] * len(indices))
    if not windows:
        raise ValueError(f"No valid samples for {domain}")
    ids = np.asarray(unit_ids, dtype=object)
    mapping = {str(unit): np.flatnonzero(ids == unit) for unit in sorted(set(ids))}
    return DomainSamples(
        domain, ids, np.concatenate(windows), np.concatenate(stats), np.concatenate(times),
        np.concatenate(soh), np.concatenate(rul), mapping,
    )


def point_loss(outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    _window, _stats, _time, true_soh, true_rul = batch
    soh = F.smooth_l1_loss(outputs["soh"], true_soh, beta=0.01)
    rul = F.smooth_l1_loss(torch.log1p(outputs["rul"]), torch.log1p(true_rul), beta=0.10)
    return soh + 0.5 * rul


def domain_weights(domains: Iterable[str], anchor_weight: float,
                   dro: dict[str, float] | None = None) -> dict[str, float]:
    names = list(domains)
    if "v10" not in names:
        raise ValueError("V1.0 anchor missing")
    auxiliary = [name for name in names if name != "v10"]
    if not auxiliary:
        return {"v10": 1.0}
    if dro is None:
        other = {name: 1.0 / len(auxiliary) for name in auxiliary}
    else:
        total = sum(max(float(dro.get(name, 0.0)), 0.0) for name in auxiliary)
        other = {name: max(float(dro.get(name, 0.0)), 0.0) / max(total, 1e-12) for name in auxiliary}
    return {"v10": anchor_weight, **{name: (1.0 - anchor_weight) * other[name] for name in auxiliary}}


def l2sp_loss(model: nn.Module, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for name, parameter in model.named_parameters():
        if name in anchor:
            losses.append(torch.mean((parameter - anchor[name].to(parameter.device)) ** 2))
    return torch.stack(losses).mean() if losses else next(model.parameters()).new_zeros(())


def set_stage_trainable(model: ContinualRULModel, epoch: int, config: TrainConfig) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    for parameter in model.tcn.soh_head.parameters():
        parameter.requires_grad = True
    if epoch > config.head_only_epochs:
        for parameter in model.tcn.encoder.network[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.tcn.shared_projection.parameters():
            parameter.requires_grad = True
    if epoch > config.head_only_epochs + config.last_block_epochs:
        for parameter in model.tcn.parameters():
            parameter.requires_grad = True


def optimizer_for(model: ContinualRULModel, config: TrainConfig,
                  encoder_scale: float = 1.0) -> torch.optim.Optimizer:
    encoder_ids = {id(parameter) for parameter in model.tcn.encoder.parameters()}
    encoder = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    heads = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.AdamW([
        {"params": encoder, "lr": config.encoder_learning_rate * encoder_scale},
        {"params": heads, "lr": config.head_learning_rate},
    ], weight_decay=config.weight_decay)


def evaluate(model: ContinualRULModel, samples: DomainSamples, config: TrainConfig,
             device: torch.device, batch_size: int = 512) -> tuple[pd.DataFrame, float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(samples.times), batch_size):
            stop = min(start + batch_size, len(samples.times))
            batch = tuple(torch.from_numpy(value[start:stop]).to(device) for value in (
                samples.windows, samples.stats, samples.times, samples.soh, samples.rul,
            ))
            outputs = model(batch[0], batch[1], batch[2], config.maximum_lifetime)
            predictions.append((outputs["soh"].cpu().numpy(), outputs["rul"].cpu().numpy()))
    pred_soh = np.concatenate([item[0] for item in predictions])
    pred_rul = np.concatenate([item[1] for item in predictions])
    frame = pd.DataFrame({
        "evaluation_domain": samples.domain, "unit_id": samples.unit_ids,
        "time": samples.times, "target_soh": samples.soh,
        "true_rul_cycles": samples.rul, "predicted_soh": pred_soh,
        "predicted_rul": pred_rul,
    })
    device_mae = frame.assign(ae=np.abs(frame.predicted_rul - frame.true_rul_cycles)).groupby("unit_id").ae.mean()
    return frame, float(device_mae.mean())


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states to average")
    output = {}
    for name in states[0]:
        values = [state[name] for state in states]
        if torch.is_floating_point(values[0]):
            output[name] = torch.stack(values).mean(0)
        else:
            output[name] = values[-1].clone()
    return output


def train_stage(model: ContinualRULModel, anchor: dict[str, torch.Tensor],
                train: dict[str, DomainSamples], monitor: dict[str, DomainSamples],
                method: str, seed: int, stage: int, config: TrainConfig,
                device: torch.device, fixed_epochs: int | None = None,
                encoder_scale: float = 1.0, swad: bool = False,
                progress_path: Path | None = None, resume: bool = False) -> tuple[
                    ContinualRULModel, dict[str, Any], list[pd.DataFrame],
                ]:
    optimizer = optimizer_for(model, config, encoder_scale)
    rng = np.random.default_rng(seed + 10_007 * stage)
    dro = {name: 1.0 for name in train if name != "v10"}
    best_state, best_score, best_epoch, stale = model_state(model), float("inf"), 0, 0
    start_epoch, swad_states, history = 1, [], []
    maximum_epochs = fixed_epochs or config.maximum_epochs
    if resume and progress_path is not None and progress_path.is_file():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best_state, best_score = saved["best_state"], float(saved["best_score"])
        best_epoch, stale, start_epoch = int(saved["best_epoch"]), int(saved["stale"]), int(saved["epoch"]) + 1
        dro = dict(saved["groupdro"])
        swad_states = list(saved.get("swad_states", []))
        history = list(saved.get("history", []))
    domains = list(train)
    for epoch in range(start_epoch, maximum_epochs + 1):
        set_stage_trainable(model, epoch, config)
        model.train()
        epoch_losses = {name: [] for name in domains}
        for step in range(config.steps_per_epoch):
            per_domain_batch = {
                name: samples.sample(max(8, config.batch_size // len(domains)), rng, device)
                for name, samples in train.items()
            }
            per_domain_loss = {
                name: point_loss(model(batch[0], batch[1], batch[2], config.maximum_lifetime), batch)
                for name, batch in per_domain_batch.items()
            }
            for name, value in per_domain_loss.items():
                epoch_losses[name].append(float(value.detach().cpu()))
            if method == "inherited_mldg_groupdro" and dro:
                for name in dro:
                    dro[name] *= math.exp(config.groupdro_eta * float(per_domain_loss[name].detach().cpu()))
                normalizer = sum(dro.values())
                dro = {name: value / normalizer for name, value in dro.items()}
            weights = domain_weights(domains, 0.5, dro if "groupdro" in method else None)
            optimizer.zero_grad(set_to_none=True)
            if "mldg" in method and len(domains) > 1:
                meta_cycle = ["v10", *[name for name in domains if name != "v10"] * 3]
                meta = meta_cycle[(epoch * config.steps_per_epoch + step) % len(meta_cycle)]
                inner_names = [name for name in domains if name != meta]
                inner_weight = {name: weights[name] for name in inner_names}
                scale = sum(inner_weight.values())
                inner = sum(inner_weight[name] / scale * per_domain_loss[name] for name in inner_names)
                trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                gradients = torch.autograd.grad(inner, [item[1] for item in trainable], create_graph=True, allow_unused=True)
                updated = dict(model.named_parameters())
                for (name, parameter), gradient in zip(trainable, gradients):
                    if gradient is not None:
                        updated[name] = parameter - config.mldg_inner_rate * gradient
                meta_batch = per_domain_batch[meta]
                meta_outputs = functional_call(
                    model, updated, (meta_batch[0], meta_batch[1], meta_batch[2], config.maximum_lifetime),
                )
                loss = inner + config.mldg_beta * point_loss(meta_outputs, meta_batch)
            else:
                loss = sum(weights[name] * per_domain_loss[name] for name in domains)
            loss = loss + config.l2sp_weight * l2sp_loss(model, anchor)
            loss.backward()
            nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
        scores, epoch_predictions = {}, []
        for name, samples in monitor.items():
            prediction, score = evaluate(model, samples, config, device)
            scores[name], epoch_predictions = score, [*epoch_predictions, prediction]
        worst = max(scores.values())
        record = {
            "epoch": epoch, "worst_domain_mae": worst, "domain_mae": scores,
            "train_loss": {name: float(np.mean(values)) for name, values in epoch_losses.items()},
            "groupdro": dict(dro),
        }
        history.append(record)
        if worst < best_score - 1e-7:
            best_state, best_score, best_epoch, stale = model_state(model), worst, epoch, 0
        else:
            stale += 1
        if swad and worst <= best_score * 1.01:
            swad_states.append(model_state(model))
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state": model_state(model),
                "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                "groupdro": dro, "swad_states": swad_states[-8:], "history": history,
            }, progress_path)
        if fixed_epochs is None and epoch >= config.head_only_epochs + config.last_block_epochs and stale >= config.patience:
            break
    if fixed_epochs is not None:
        # Refit is deliberately trained for the selection-chosen number of
        # epochs.  It must not silently become another early-stopping pass.
        best_state, best_epoch = model_state(model), int(fixed_epochs)
        best_score = max(evaluate(model, samples, config, device)[1] for samples in monitor.values())
    if swad and swad_states:
        candidate = average_states(swad_states)
        model.load_state_dict(candidate)
        candidate_score = max(evaluate(model, samples, config, device)[1] for samples in monitor.values())
        if candidate_score <= best_score:
            best_state, best_score = candidate, candidate_score
    model.load_state_dict(best_state)
    predictions = [evaluate(model, samples, config, device)[0] for samples in monitor.values()]
    return model, {
        "best_epoch": best_epoch, "best_worst_domain_mae": best_score,
        "history": history, "groupdro": dro,
    }, predictions


def instantiate_from_state(parent_path_value: Path, seed: int, device: torch.device,
                           state: dict[str, torch.Tensor] | None = None) -> tuple[
                               ContinualRULModel, dict[str, Any], Config, HistoryConfig,
                           ]:
    model, payload, config, history_config = load_parent(parent_path_value, seed, device)
    if state is not None:
        model.load_state_dict(copy.deepcopy(state), strict=True)
    return model, payload, config, history_config


def stage_domains(all_domains: dict[str, pd.DataFrame], microdomains: tuple[str, ...]) -> list[tuple[str, ...]]:
    order = ["v15_bridge", *microdomains]
    return [tuple(["v10", *order[:index + 1]]) for index in range(len(order))]


def run_chain(args: argparse.Namespace, method: str, seed: int, lineage: str,
              domains: dict[str, pd.DataFrame], features: list[str],
              parent_file: Path, train_config: TrainConfig,
              device: torch.device, swad: bool = False) -> dict[str, Any]:
    selection_frames, monitor_frames, inner_manifest = split_inner_domains(
        domains, args.split_seed + seed, args.smoke,
    )
    initial_model, parent, tcn_config, _history_config = load_parent(parent_file, seed, device)
    anchor = model_state(initial_model)
    selection_model = initial_model
    refit_model, _payload, _config, _history = instantiate_from_state(parent_file, seed, device)
    parent_hash = state_sha256(anchor)
    stages = []
    for stage_index, active in enumerate(stage_domains(domains, args.microdomains), 1):
        before_selection = model_state(selection_model)
        train_samples = {
            name: build_samples(selection_frames[name], name, features, selection_model, parent, tcn_config,
                                train_config.maximum_lifetime, device, args.smoke)
            for name in active
        }
        monitor_samples = {
            name: build_samples(monitor_frames[name], name, features, selection_model, parent, tcn_config,
                                train_config.maximum_lifetime, device, args.smoke)
            for name in active
        }
        v10_before = evaluate(selection_model, monitor_samples["v10"], train_config, device)[1]
        progress = args.output_dir / "progress" / f"{lineage}_{method}_seed{seed}_stage{stage_index}.pt"
        selection_model, selected, predictions = train_stage(
            selection_model, anchor, train_samples, monitor_samples, method, seed, stage_index,
            train_config, device, swad=swad and stage_index == len(stage_domains(domains, args.microdomains)),
            progress_path=progress, resume=args.resume,
        )
        v10_after = evaluate(selection_model, monitor_samples["v10"], train_config, device)[1]
        retried, promoted = False, True
        if v10_after > 1.10 * max(v10_before, 1e-8):
            retried = True
            selection_model.load_state_dict(before_selection)
            retry_progress = progress.with_name(progress.stem + "_retry.pt")
            selection_model, selected, predictions = train_stage(
                selection_model, anchor, train_samples, monitor_samples, method, seed, stage_index,
                train_config, device, encoder_scale=0.5, swad=swad,
                progress_path=retry_progress, resume=args.resume,
            )
            v10_after = evaluate(selection_model, monitor_samples["v10"], train_config, device)[1]
            if v10_after > 1.10 * max(v10_before, 1e-8):
                selection_model.load_state_dict(before_selection)
                promoted = False
        if promoted:
            refit_samples = {
                name: build_samples(domains[name], name, features, refit_model, parent, tcn_config,
                                    train_config.maximum_lifetime, device, args.smoke)
                for name in active
            }
            # Refit monitoring is used only to keep the fixed-epoch routine observable.
            refit_monitor = {name: refit_samples[name] for name in active}
            refit_model, refit_info, _ = train_stage(
                refit_model, anchor, refit_samples, refit_monitor, method, seed, stage_index,
                train_config, device, fixed_epochs=max(1, int(selected["best_epoch"])),
                swad=swad and stage_index == len(stage_domains(domains, args.microdomains)),
            )
        else:
            refit_info = {"skipped": True}
        checkpoint = args.output_dir / "checkpoints" / f"{lineage}_{method}_seed{seed}_stage{stage_index}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1, "lineage": lineage, "method": method, "seed": seed,
            "stage": stage_index, "active_domains": list(active),
            "model_state": model_state(refit_model), "anchor_state_sha256": parent_hash,
            "parent_checkpoint": str(parent_file.resolve()), "parent_file_sha256": sha256(parent_file),
            "features": features, "target_scaler_median": parent["target_scaler_median"],
            "target_scaler_iqr": parent["target_scaler_iqr"],
            "bstats_median": parent["bstats_median"], "bstats_iqr": parent["bstats_iqr"],
            "selected_epochs": int(selected["best_epoch"]), "promoted": promoted,
            "optimizer_inherited_from_previous_stage": False,
            "model_parameters_inherited_from_previous_stage": True,
            "validation_accessed": False, "sealed_accessed": False,
        }, checkpoint)
        stage_prediction = pd.concat(predictions, ignore_index=True)
        stage_prediction["stage"] = stage_index
        stage_prediction["seed"] = seed
        stage_prediction["method"] = method
        stage_prediction["lineage"] = lineage
        stage_prediction.to_csv(
            args.output_dir / f"selection_predictions_{lineage}_{method}_seed{seed}_stage{stage_index}.csv",
            index=False,
        )
        stages.append({
            "stage": stage_index, "active_domains": list(active), "selection": selected,
            "refit": refit_info, "v10_before": v10_before, "v10_after": v10_after,
            "retried_half_encoder_lr": retried, "promoted": promoted,
            "checkpoint": str(checkpoint.resolve()),
        })
    final_score = float(stages[-1]["selection"]["best_worst_domain_mae"])
    return {
        "method": method, "seed": seed, "lineage": lineage, "selection_worst_domain_mae": final_score,
        "final_state": model_state(refit_model), "parent": parent,
        "tcn_config": asdict(tcn_config), "inner_manifest": inner_manifest, "stages": stages,
    }


def final_test_frames(args: argparse.Namespace, locked: dict[str, pd.DataFrame],
                      features: list[str]) -> dict[str, pd.DataFrame]:
    frames = {
        "v10_validation": read_table(args.v10_root, "v10_validation", "validation", features),
        "v15_blind": locked["v15_blind"],
    }
    for domain in args.microdomains:
        frames[domain + "_locked"] = read_table(args.micro_root / domain, domain + "_locked", "validation", features)
    if args.smoke:
        frames = {
            name: subset(frame, sorted(frame.unit_id.astype(str).unique())[:5])
            for name, frame in frames.items()
        }
    return frames


def evaluate_ensemble(results: list[dict[str, Any]], frames: dict[str, pd.DataFrame],
                      features: list[str], parent_file_for_seed: dict[int, Path],
                      train_config: TrainConfig, device: torch.device, output: Path,
                      label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = []
    for result in results:
        seed = int(result["seed"])
        model, parent, config, _ = load_parent(parent_file_for_seed[seed], seed, device)
        model.load_state_dict(result["final_state"], strict=True)
        for domain, frame in frames.items():
            samples = build_samples(frame, domain, features, model, parent, config,
                                    train_config.maximum_lifetime, device, False)
            prediction, _ = evaluate(model, samples, train_config, device)
            prediction["seed"] = seed
            prediction["lineage"] = label
            by_seed.append(prediction)
    predictions = pd.concat(by_seed, ignore_index=True)
    keys = ["evaluation_domain", "unit_id", "time", "target_soh", "true_rul_cycles"]
    ensemble = predictions.groupby(keys, as_index=False).agg(
        predicted_soh=("predicted_soh", "mean"), predicted_rul=("predicted_rul", "mean"),
        ensemble_std=("predicted_rul", "std"),
    )
    metrics = []
    for domain, group in ensemble.groupby("evaluation_domain"):
        error = group.predicted_rul - group.true_rul_cycles
        device = group.assign(ae=np.abs(error)).groupby("unit_id").ae.mean()
        metrics.append({
            "lineage": label, "evaluation_domain": domain,
            "device_macro_rul_mae": float(device.mean()),
            "point_rul_mae": float(np.mean(np.abs(error))),
            "rul_rmse": float(np.sqrt(np.mean(error ** 2))),
            "rul_bias": float(np.mean(error)), "worst_device_mae": float(device.max()),
            "soh_mae": float(np.mean(np.abs(group.predicted_soh - group.target_soh))),
        })
    metric_frame = pd.DataFrame(metrics)
    metric_frame = pd.concat([metric_frame, pd.DataFrame([{
        "lineage": label, "evaluation_domain": "worst_domain",
        "device_macro_rul_mae": float(metric_frame.device_macro_rul_mae.max()),
        "point_rul_mae": float("nan"), "rul_rmse": float("nan"),
        "rul_bias": float("nan"), "worst_device_mae": float(metric_frame.worst_device_mae.max()),
        "soh_mae": float(metric_frame.soh_mae.max()),
    }])], ignore_index=True)
    predictions.to_csv(output / f"final_predictions_by_seed_{label}.csv", index=False)
    ensemble.to_csv(output / f"final_ensemble_predictions_{label}.csv", index=False)
    return ensemble, metric_frame


def paired_bootstrap(candidate: pd.DataFrame, control: pd.DataFrame, draws: int = 10000) -> dict[str, float]:
    keys = ["evaluation_domain", "unit_id", "time"]
    merged = candidate.merge(control, on=keys, suffixes=("_candidate", "_control"))
    merged["ae_candidate"] = np.abs(merged.predicted_rul_candidate - merged.true_rul_cycles_candidate)
    merged["ae_control"] = np.abs(merged.predicted_rul_control - merged.true_rul_cycles_control)
    device = merged.groupby(["evaluation_domain", "unit_id"])[["ae_candidate", "ae_control"]].mean()
    difference = (device.ae_candidate - device.ae_control).to_numpy(float)
    rng = np.random.default_rng(202608)
    bootstrap = np.asarray([np.mean(rng.choice(difference, len(difference), replace=True)) for _ in range(draws)])
    return {
        "candidate_minus_control_device_mae": float(np.mean(difference)),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "draws": draws,
    }


def acceptance_report(metrics: pd.DataFrame, bootstrap: dict[str, float] | None,
                      selection: pd.DataFrame, winner: str) -> dict[str, Any]:
    nasa = metrics.loc[metrics.lineage.eq("nasa")]
    control = metrics.loc[metrics.lineage.eq("target_control")]
    v10 = nasa.loc[nasa.evaluation_domain.eq("v10_validation")]
    nasa_worst = nasa.loc[nasa.evaluation_domain.eq("worst_domain")]
    control_worst = control.loc[control.evaluation_domain.eq("worst_domain")]
    inherited = selection.loc[selection.method.eq("inherited_ft"), "selection_worst_domain_mae"]
    winning = selection.loc[selection.method.eq(winner), "selection_worst_domain_mae"]
    checks = {
        "v10_validation_mae_le_5": bool(len(v10) == 1 and float(v10.iloc[0].device_macro_rul_mae) <= 5.0),
        "selection_worst_domain_improves_5pct_over_inherited_ft": bool(
            len(inherited) == 1 and len(winning) == 1
            and float(winning.iloc[0]) <= 0.95 * float(inherited.iloc[0])
        ),
        "nasa_improves_2pct_over_target_control": bool(
            len(nasa_worst) == 1 and len(control_worst) == 1
            and float(nasa_worst.iloc[0].device_macro_rul_mae)
            <= 0.98 * float(control_worst.iloc[0].device_macro_rul_mae)
        ),
        "paired_bootstrap_ci95_upper_below_zero": bool(
            bootstrap is not None and float(bootstrap["ci95_upper"]) < 0.0
        ),
        "soh_mae_within_5pct_of_registered_0p01151": bool(
            len(v10) == 1 and float(v10.iloc[0].soh_mae) <= 1.05 * 0.01151
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "note": "Per-domain <=10% degradation requires registered domain-specific baselines and is reported separately.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-root", type=Path, default=DEFAULT_V10)
    parser.add_argument("--v15-root", type=Path, default=DEFAULT_V15)
    parser.add_argument("--micro-root", type=Path, default=DEFAULT_MICRO)
    parser.add_argument("--parent-nasa-dir", type=Path, default=DEFAULT_NASA_PARENT)
    parser.add_argument("--parent-target-dir", type=Path, default=DEFAULT_TARGET_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v15-train-units", type=int, default=40)
    parser.add_argument("--microdomains", default=",".join(MICRODOMAINS))
    parser.add_argument("--v10-anchor-weight", type=float, default=0.5)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="52,53,54")
    parser.add_argument("--split-seed", type=int, default=202608)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swad-best", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--skip-target-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.microdomains = tuple(item.strip() for item in args.microdomains.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if args.smoke:
        methods, seeds, args.microdomains = methods[:1], seeds[:1], args.microdomains[:1]
    if set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {sorted(set(methods) - set(METHODS))}")
    if set(args.microdomains) - set(MICRODOMAINS):
        raise ValueError("Unknown microdomain")
    if abs(args.v10_anchor_weight - 0.5) > 1e-12:
        raise ValueError("Registered experiment fixes V1.0 supervision mass at exactly 0.5")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for smoke")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains, locked, features, bridge_manifest = prepare_domains(args)
    if args.smoke:
        domains = {name: subset(frame, sorted(frame.unit_id.unique())[:5]) for name, frame in domains.items()
                   if name in {"v10", "v15_bridge", *args.microdomains}}
    train_config = TrainConfig(
        maximum_epochs=2 if args.smoke else 20,
        patience=1 if args.smoke else 5,
        steps_per_epoch=1 if args.smoke else 12,
        batch_size=16 if args.smoke else 96,
    )
    nasa_results = []
    for method in methods:
        for seed in seeds:
            print(json.dumps({"stage": "continual_chain", "lineage": "nasa", "method": method, "seed": seed}), flush=True)
            nasa_results.append(run_chain(
                args, method, seed, "nasa", domains, features,
                parent_path(args.parent_nasa_dir, seed, "nasa"), train_config, device,
            ))
    summary = pd.DataFrame([{
        "method": row["method"], "seed": row["seed"],
        "selection_worst_domain_mae": row["selection_worst_domain_mae"],
    } for row in nasa_results])
    method_summary = summary.groupby("method", as_index=False).selection_worst_domain_mae.mean().sort_values(
        ["selection_worst_domain_mae", "method"], kind="stable",
    )
    winner = str(method_summary.iloc[0].method)
    winning_results = [row for row in nasa_results if row["method"] == winner]
    if args.swad_best:
        winning_results = []
        for seed in seeds:
            winning_results.append(run_chain(
                args, winner, seed, "nasa_swad", domains, features,
                parent_path(args.parent_nasa_dir, seed, "nasa"), train_config, device, swad=True,
            ))
    control_results = []
    if not args.skip_target_control:
        for seed in seeds:
            control_results.append(run_chain(
                args, winner, seed, "target_control", domains, features,
                parent_path(args.parent_target_dir, seed, "target"), train_config, device,
                swad=args.swad_best,
            ))
    final_metrics, bootstrap, acceptance = pd.DataFrame(), None, None
    if not args.skip_final_evaluation:
        tests = final_test_frames(args, locked, features)
        nasa_parent = {seed: parent_path(args.parent_nasa_dir, seed, "nasa") for seed in seeds}
        nasa_prediction, nasa_metrics = evaluate_ensemble(
            winning_results, tests, features, nasa_parent, train_config, device, args.output_dir, "nasa",
        )
        final_metrics = nasa_metrics
        if control_results:
            target_parent = {seed: parent_path(args.parent_target_dir, seed, "target") for seed in seeds}
            target_prediction, target_metrics = evaluate_ensemble(
                control_results, tests, features, target_parent, train_config, device,
                args.output_dir, "target_control",
            )
            final_metrics = pd.concat([nasa_metrics, target_metrics], ignore_index=True)
            bootstrap = paired_bootstrap(nasa_prediction, target_prediction)
        final_metrics.to_csv(args.output_dir / "final_domain_metrics.csv", index=False)
        if bootstrap is not None:
            (args.output_dir / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
        if control_results:
            acceptance = acceptance_report(final_metrics, bootstrap, method_summary, winner)
            (args.output_dir / "acceptance_report.json").write_text(
                json.dumps(acceptance, indent=2), encoding="utf-8",
            )
    summary.to_csv(args.output_dir / "selection_results_by_seed.csv", index=False)
    method_summary.to_csv(args.output_dir / "selection_method_summary.csv", index=False)
    manifest = {
        "schema_version": "bhump-v10-v15-continual-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods_screened": list(methods), "winning_method": winner,
        "seeds": list(seeds), "microdomains": list(args.microdomains),
        "stage_order": ["v15_bridge", *args.microdomains],
        "v10_anchor_weight": 0.5, "train_config": asdict(train_config),
        "bridge_split": bridge_manifest,
        "feature_contract": features, "feature_count": len(features),
        "parent_nasa_checkpoints": {
            str(seed): {"path": str(parent_path(args.parent_nasa_dir, seed, "nasa").resolve()),
                        "sha256": sha256(parent_path(args.parent_nasa_dir, seed, "nasa"))}
            for seed in seeds
        },
        "parameter_inheritance": True, "optimizer_reset_between_stages": True,
        "optimizer_restored_only_for_same_stage_resume": True,
        "cumulative_replay": True, "l2sp_anchor": "original_v10_parent",
        "normalization": "frozen_original_v10_parent",
        "v15_blind_used_before_final_evaluation": False,
        "sealed_features_accessed": False, "sealed_labels_accessed": False,
        "final_evaluation_executed": not args.skip_final_evaluation,
        "paired_bootstrap": bootstrap,
        "acceptance": acceptance,
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "winner": winner,
        "selection": method_summary.to_dict("records"),
        "final_metrics": final_metrics.to_dict("records"),
        "bootstrap": bootstrap,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
