"""Standalone causal inference for the final V1.0 three-seed RUL ensemble."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import weight_norm

from bhump_common import add_causal_baseline_deltas, curve_features


EOL_SOH = 0.80
REPRESENTATION_SIZE = 55
RAW_CURVE_COLUMNS = {
    "unit_id", "time", "elapsed_s", "voltage_v", "current_a", "temperature_c",
}


@dataclass(frozen=True)
class Config:
    window: int = 24
    channels: tuple[int, ...] = (16, 24, 24)
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


class Chomp1d(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, :-self.amount] if self.amount else values


class TemporalBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(
                input_size, output_size, kernel,
                padding=padding, dilation=dilation,
            )),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(
                output_size, output_size, kernel,
                padding=padding, dilation=dilation,
            )),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = (
            nn.Conv1d(input_size, output_size, 1)
            if input_size != output_size else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(values) + self.skip(values))


class TcnEncoder(nn.Module):
    def __init__(self, input_size: int, channels: tuple[int, ...],
                 kernel: int, dropout: float) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_size
        for level, output in enumerate(channels):
            blocks.append(TemporalBlock(
                current, output, kernel, 2**level, dropout,
            ))
            current = output
        self.network = nn.Sequential(*blocks)
        self.output_size = current

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.transpose(1, 2)).transpose(1, 2)


class DynamicsTCN(nn.Module):
    def __init__(self, input_size: int, config: Config) -> None:
        super().__init__()
        self.encoder = TcnEncoder(
            input_size, config.channels, config.kernel, config.dropout,
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.projection),
            nn.SiLU(), nn.Dropout(config.dropout),
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
        self.pre_rate_head = nn.Linear(config.projection, 1)
        self.post_delta_head = nn.Linear(config.projection, 1)
        self.knee_time_head = nn.Linear(config.projection, 1)
        self.knee_probability_head = nn.Linear(config.projection, 1)

    def forward(self, values: torch.Tensor,
                domain: str = "target") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)[:, -1]
        shared = self.shared_projection(encoded)
        private = (
            self.private_source_projection(encoded)
            if domain == "source" else self.private_target_projection(encoded)
        )
        soh = 1.10 * torch.sigmoid(self.soh_head(shared).squeeze(-1))
        return soh, shared, private


class HistoryBStats(nn.Module):
    def __init__(self, config: HistoryConfig) -> None:
        super().__init__()
        input_size = config.local_projection + 1 + config.statistics
        self.fusion = nn.Sequential(
            nn.Linear(input_size, config.fusion_hidden), nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, config.fusion_hidden), nn.SiLU(),
        )
        self.soh_residual_head = nn.Linear(config.fusion_hidden, 1)
        self.rul_fraction_head = nn.Linear(config.fusion_hidden, 1)

    def forward(self, local: torch.Tensor, local_soh: torch.Tensor,
                stats: torch.Tensor, times: torch.Tensor,
                maximum_lifetime: float) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.fusion(torch.cat([
            local, local_soh.unsqueeze(-1), stats,
        ], dim=-1))
        soh = torch.clamp(
            local_soh
            + 0.05 * torch.tanh(self.soh_residual_head(hidden).squeeze(-1)),
            0.0, 1.10,
        )
        cap = torch.clamp(
            torch.as_tensor(maximum_lifetime, device=times.device) - times,
            min=0.0,
        )
        rul = cap * torch.sigmoid(self.rul_fraction_head(hidden).squeeze(-1))
        return soh, rul


class IntercellLifeHead(nn.Module):
    def __init__(self, config: IntercellConfig) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(REPRESENTATION_SIZE, config.projection), nn.SiLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(4 * config.projection + 1, config.hidden), nn.SiLU(),
            nn.Dropout(0.10), nn.Linear(config.hidden, 1),
        )

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


def local_windows(values: np.ndarray, window: int) -> np.ndarray:
    windows: list[np.ndarray] = []
    for end in range(len(values)):
        start = end - window + 1
        if start < 0:
            item = np.concatenate([
                np.repeat(values[[0]], -start, axis=0), values[:end + 1],
            ])
        else:
            item = values[start:end + 1]
        windows.append(item)
    return np.asarray(windows, np.float32)


def encode_local(model: DynamicsTCN, windows: np.ndarray,
                 device: torch.device, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            values = torch.from_numpy(windows[start:start + batch_size]).to(device)
            soh, shared, _ = model(values, "target")
            embeddings.append(shared.cpu().numpy())
            predictions.append(soh.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def safe_slope(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.zeros(values.shape[1:] or (), dtype=float)
    centered = times - times.mean()
    denominator = float(np.sum(centered**2))
    if denominator < 1.0e-12:
        return np.zeros(values.shape[1:] or (), dtype=float)
    shaped = centered.reshape((-1,) + (1,) * (values.ndim - 1))
    return np.sum(shaped * values, axis=0) / denominator


def causal_statistics(raw: np.ndarray, local_soh: np.ndarray,
                      times: np.ndarray, maximum_lifetime: float) -> np.ndarray:
    rows: list[np.ndarray] = []
    ewma = float(local_soh[0])
    for end in range(len(raw)):
        history_t = times[:end + 1]
        history_x = raw[:end + 1]
        history_soh = local_soh[:end + 1]
        ewma = (
            0.2 * float(local_soh[end]) + 0.8 * ewma
            if end else float(local_soh[0])
        )
        recent_start = max(0, end - 11)
        long_slope = float(safe_slope(history_t, history_soh))
        recent_slope = float(safe_slope(
            times[recent_start:end + 1], local_soh[recent_start:end + 1],
        ))
        rows.append(np.concatenate([
            raw[end] - raw[0],
            np.asarray(safe_slope(history_t, history_x), dtype=float),
            np.asarray([
                long_slope, recent_slope, recent_slope - long_slope,
                float(local_soh[end]) - ewma,
                float(times[end]) / maximum_lifetime,
                math.log1p(end + 1) / math.log1p(maximum_lifetime + 1.0),
            ]),
        ]))
    output = np.asarray(rows, np.float32)
    if output.shape[1] != 38:
        raise RuntimeError(f"Expected 38 B_stats values, received {output.shape[1]}")
    return output


def matched_references(progress: np.ndarray, bank: dict[str, Any]) -> np.ndarray:
    knots = np.asarray(bank["knots"], np.float32)
    representations = np.asarray(bank["representation"], np.float32)
    position = np.clip(progress, 0.0, 1.0) * (len(knots) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(knots) - 1)
    fraction = (position - low).astype(np.float32)
    left = np.transpose(representations[:, low, :], (1, 0, 2))
    right = np.transpose(representations[:, high, :], (1, 0, 2))
    return (
        left * (1.0 - fraction[:, None, None])
        + right * fraction[:, None, None]
    ).astype(np.float32)


class EnsembleMember:
    def __init__(self, payload: dict[str, Any], device: torch.device) -> None:
        self.payload = payload
        tcn_values = dict(payload["tcn_configuration"])
        tcn_values["channels"] = tuple(tcn_values["channels"])
        self.tcn_config = Config(**tcn_values)
        self.history_config = HistoryConfig(**payload["history_configuration"])
        self.intercell_config = IntercellConfig(**payload["intercell_configuration"])
        self.device = device
        self.tcn = DynamicsTCN(len(payload["features"]), self.tcn_config).to(device)
        self.bstats = HistoryBStats(self.history_config).to(device)
        self.intercell = IntercellLifeHead(self.intercell_config).to(device)
        self.tcn.load_state_dict(payload["model_state_tcn"], strict=True)
        self.bstats.load_state_dict(payload["model_state_bstats"], strict=True)
        self.intercell.load_state_dict(payload["model_state_intercell"], strict=True)
        self.tcn.eval()
        self.bstats.eval()
        self.intercell.eval()

    def predict_unit(self, unit: pd.DataFrame) -> pd.DataFrame:
        features = list(self.payload["features"])
        times = unit.time.to_numpy(np.float32)
        values = unit[features].to_numpy(np.float32)
        raw = (
            (values - np.asarray(self.payload["target_scaler_median"], np.float32))
            / np.asarray(self.payload["target_scaler_iqr"], np.float32)
        ).astype(np.float32)
        local, local_soh = encode_local(
            self.tcn, local_windows(raw, self.tcn_config.window), self.device,
        )
        maximum = float(self.payload["maximum_training_lifetime"])
        stats = causal_statistics(raw, local_soh, times, maximum)
        stats = (
            (stats - np.asarray(self.payload["bstats_median"], np.float32))
            / np.asarray(self.payload["bstats_iqr"], np.float32)
        ).astype(np.float32)
        with torch.inference_mode():
            predicted_soh, base_rul = self.bstats(
                torch.from_numpy(local[None]).to(self.device),
                torch.from_numpy(local_soh[None]).to(self.device),
                torch.from_numpy(stats[None]).to(self.device),
                torch.from_numpy(times[None]).to(self.device),
                maximum,
            )
        soh = predicted_soh[0].cpu().numpy()
        base = base_rul[0].cpu().numpy()
        representation = np.concatenate([
            local, local_soh[:, None], stats,
        ], axis=1).astype(np.float32)
        representation = (
            (representation - np.asarray(self.payload["target_vector_median"], np.float32))
            / np.asarray(self.payload["target_vector_iqr"], np.float32)
        ).astype(np.float32)
        denominator = max(float(local_soh[0]) - EOL_SOH, 0.02)
        progress = np.clip(
            (float(local_soh[0]) - soh) / denominator, 0.0, 1.0,
        ).astype(np.float32)
        bank = self.payload["nasa_reference_bank"]
        references = matched_references(progress, bank)
        deltas: list[np.ndarray] = []
        batch = self.intercell_config.batch_size
        with torch.inference_mode():
            for start in range(0, len(unit), batch):
                deltas.append(self.intercell(
                    torch.from_numpy(representation[start:start + batch]).to(self.device),
                    torch.from_numpy(references[start:start + batch]).to(self.device),
                    torch.from_numpy(progress[start:start + batch]).to(self.device),
                ).cpu().numpy())
        delta = np.concatenate(deltas)
        source_eol = np.asarray(bank["eol"], np.float32)[None, :]
        predicted_eol = np.minimum(
            np.maximum(source_eol * np.exp(delta), times[:, None]), maximum,
        )
        reference_rul = np.maximum(predicted_eol - times[:, None], 0.0)
        policy = self.payload["lifetime_policy"]
        mode = str(policy["mode"])
        blend = float(policy["blend_weight"])
        if mode in {"uniform_median", "uniform_fallback"}:
            intercell_rul = np.nanmedian(reference_rul, axis=1)
            effective = np.full(len(unit), blend, dtype=float)
        else:
            known_eol = np.asarray(bank["eol"], float)
            gamma = float(policy["gamma"])
            base_eol = np.maximum(base.astype(float) + times, times + 1.0e-3)
            log_gap = np.abs(np.log(
                (base_eol[:, None] + 1.0e-3) / (known_eol[None, :] + 1.0e-3)
            ))
            logits = -gamma * log_gap
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
            aggregate = np.exp(np.sum(
                weights * np.log(np.maximum(predicted_eol, times[:, None] + 1.0e-3)),
                axis=1,
            ))
            intercell_rul = np.maximum(aggregate - times, 0.0)
            lower = np.maximum(np.log(
                (float(np.min(known_eol)) + 1.0e-3) / (base_eol + 1.0e-3)
            ), 0.0)
            upper = np.maximum(np.log(
                (base_eol + 1.0e-3) / (float(np.max(known_eol)) + 1.0e-3)
            ), 0.0)
            effective = blend * np.exp(-gamma * (lower + upper))
        cap = np.maximum(maximum - times, 0.0)
        final = np.clip(
            (1.0 - effective) * base + effective * intercell_rul,
            0.0, cap,
        )
        return pd.DataFrame({
            "unit_id": unit.unit_id.astype(str).to_numpy(),
            "time": times.astype(float),
            "predicted_soh": soh.astype(float),
            "base_rul": base.astype(float),
            "intercell_rul": intercell_rul.astype(float),
            "predicted_rul": final.astype(float),
            "reference_prediction_std": np.std(reference_rul, axis=1),
            "effective_blend_weight": effective,
        })


class BatteryRulEnsemble:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.device = torch.device(resolved_device)
        bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        if bundle.get("format") != "bhump_v10_final_ensemble":
            raise ValueError("Unsupported ensemble model format")
        if len(bundle.get("members", [])) != 3:
            raise ValueError("Final ensemble must contain exactly three members")
        self.bundle = bundle
        self.features = list(bundle["features"])
        self.members = [
            EnsembleMember(payload, self.device)
            for payload in bundle["members"]
        ]

    def curve_table_to_features(self, curves: pd.DataFrame) -> pd.DataFrame:
        missing = RAW_CURVE_COLUMNS - set(curves.columns)
        if missing:
            raise ValueError(f"Raw curve input is missing columns: {sorted(missing)}")
        order = ["unit_id", "time"] + (
            ["sample_index"] if "sample_index" in curves.columns else ["elapsed_s"]
        )
        curves = curves.sort_values(order).copy()
        duplicate_key = ["unit_id", "time", "sample_index"] \
            if "sample_index" in curves.columns else ["unit_id", "time", "elapsed_s"]
        if curves.duplicated(duplicate_key).any():
            raise ValueError("Raw curve input contains duplicate samples")
        rows: list[dict[str, Any]] = []
        for (unit_id, cycle), group in curves.groupby(["unit_id", "time"], sort=True):
            elapsed = group.elapsed_s.to_numpy(float)
            if np.any(np.diff(elapsed) < 0.0):
                raise ValueError(f"Curve {unit_id}/{cycle} has decreasing elapsed_s")
            ambient = (
                float(group.ambient_temperature_c.iloc[0])
                if "ambient_temperature_c" in group.columns
                else float(group.temperature_c.iloc[0])
            )
            extracted = curve_features(
                group.voltage_v.to_numpy(float),
                group.current_a.to_numpy(float),
                group.temperature_c.to_numpy(float),
                elapsed,
                ambient,
            )
            rows.append({"unit_id": str(unit_id), "time": float(cycle), **extracted})
        features, _ = add_causal_baseline_deltas(
            pd.DataFrame(rows), {"unit_id", "time"},
        )
        return features

    def validate_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"unit_id", "time", *self.features}
        if not required.issubset(frame.columns) and RAW_CURVE_COLUMNS.issubset(frame.columns):
            frame = self.curve_table_to_features(frame)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        selected = frame[["unit_id", "time", *self.features]].copy()
        selected["unit_id"] = selected.unit_id.astype(str)
        if selected.duplicated(["unit_id", "time"]).any():
            raise ValueError("Input contains duplicate unit_id/time rows")
        numeric = selected[["time", *self.features]].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise ValueError("Input contains non-finite time or feature values")
        selected = selected.sort_values(["unit_id", "time"]).reset_index(drop=True)
        for unit_id, unit in selected.groupby("unit_id", sort=False):
            if len(unit) < 1 or np.any(np.diff(unit.time.to_numpy(float)) <= 0.0):
                raise ValueError(f"Unit {unit_id} does not have strictly increasing time")
        return selected

    def predict(self, frame: pd.DataFrame, diagnostics: bool = False) -> pd.DataFrame:
        selected = self.validate_input(frame)
        member_frames: list[pd.DataFrame] = []
        for member in self.members:
            predictions = [
                member.predict_unit(unit.reset_index(drop=True))
                for _, unit in selected.groupby("unit_id", sort=True)
            ]
            member_frames.append(pd.concat(predictions, ignore_index=True))
        keys = member_frames[0][["unit_id", "time"]].copy()
        for frame_item in member_frames[1:]:
            if not keys.equals(frame_item[["unit_id", "time"]]):
                raise RuntimeError("Ensemble members produced inconsistent row keys")
        soh = np.column_stack([
            item.predicted_soh.to_numpy(float) for item in member_frames
        ])
        rul = np.column_stack([
            item.predicted_rul.to_numpy(float) for item in member_frames
        ])
        output = keys.copy()
        output["predicted_soh"] = soh.mean(axis=1)
        output["predicted_rul_cycles"] = rul.mean(axis=1)
        output["rul_ensemble_std"] = rul.std(axis=1)
        output["model_count"] = len(self.members)
        if diagnostics:
            for member, item in zip(self.members, member_frames):
                seed = int(member.payload["seed"])
                output[f"predicted_soh_seed_{seed}"] = item.predicted_soh.to_numpy(float)
                output[f"predicted_rul_seed_{seed}"] = item.predicted_rul.to_numpy(float)
                output[f"base_rul_seed_{seed}"] = item.base_rul.to_numpy(float)
                output[f"intercell_rul_seed_{seed}"] = item.intercell_rul.to_numpy(float)
        if not np.isfinite(
            output[["predicted_soh", "predicted_rul_cycles", "rul_ensemble_std"]]
            .to_numpy(float)
        ).all():
            raise RuntimeError("Ensemble produced non-finite predictions")
        if (output.predicted_rul_cycles < 0.0).any():
            raise RuntimeError("Ensemble produced negative RUL")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parent / "battery_rul_ensemble_v10.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    model = BatteryRulEnsemble(args.model, args.device)
    prediction = model.predict(frame, diagnostics=args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(
        f"Predicted {len(prediction)} rows for "
        f"{prediction.unit_id.nunique()} devices on {model.device}."
    )


if __name__ == "__main__":
    main()
