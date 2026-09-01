"""Independent PyTorch implementation of battery SOH/RUL sequence models."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowDataset(Dataset):
    def __init__(self, series: list[np.ndarray], window: int) -> None:
        self.samples: list[tuple[np.ndarray, np.float32]] = []
        for values in series:
            for end in range(window, len(values)):
                self.samples.append((values[end - window : end], np.float32(values[end])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[index]
        return torch.from_numpy(x[:, None]).float(), torch.tensor([y]).float()


class Chomp1d(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.amount] if self.amount else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.skip(x))


class TcnEncoder(nn.Module):
    def __init__(self, input_size: int, channels: tuple[int, ...], kernel: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        current = input_size
        for level, output in enumerate(channels):
            blocks.append(TemporalBlock(current, output, kernel, 2**level, dropout))
            current = output
        self.network = nn.Sequential(*blocks)
        self.output_size = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class SequenceRegressor(nn.Module):
    """Common model interface returning prediction and transferable features."""

    def __init__(
        self,
        kind: str,
        input_size: int,
        hidden: int,
        channels: tuple[int, ...],
        kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.sequence = nn.LSTM(input_size, hidden, batch_first=True)
            feature_size = hidden
        elif kind == "tcn":
            self.sequence = TcnEncoder(input_size, channels, kernel, dropout)
            feature_size = self.sequence.output_size
        elif kind == "cnn_lstm":
            self.cnn = nn.Sequential(
                nn.Conv1d(input_size, channels[0], kernel, padding=kernel // 2),
                nn.ReLU(),
                nn.Conv1d(channels[0], channels[-1], kernel, padding=kernel // 2),
                nn.ReLU(),
            )
            self.sequence = nn.LSTM(channels[-1], hidden, batch_first=True)
            feature_size = hidden
        elif kind == "tcn_lstm":
            self.tcn = TcnEncoder(input_size, channels, kernel, dropout)
            self.sequence = nn.LSTM(self.tcn.output_size, hidden, batch_first=True)
            feature_size = hidden
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        self.head = nn.Sequential(nn.Linear(feature_size, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "lstm":
            features, _ = self.sequence(x)
            vector = features[:, -1]
        elif self.kind == "tcn":
            features = self.sequence(x)
            vector = features[:, -1]
        elif self.kind == "cnn_lstm":
            features = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            features, _ = self.sequence(features)
            vector = features[:, -1]
        else:
            features = self.tcn(x)
            features, _ = self.sequence(features)
            vector = features[:, -1]
        return self.head(vector), vector


@dataclass
class Config:
    model: str
    seed: int
    window: int
    hidden: int
    channels: tuple[int, ...]
    kernel: int
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    eol_soh: float
    max_forecast_cycles: int


def read_soh(data_root: Path, split: str, cell: str) -> np.ndarray:
    frame = pd.read_csv(data_root / split / f"{cell}.csv")
    if not np.allclose(frame["SOH"], frame["Capacity"] / 2.0, atol=1e-12):
        raise ValueError(f"SOH contract failed for {cell}")
    return frame["SOH"].to_numpy(dtype=np.float32)


def epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    values = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x)
            loss = loss_fn(prediction, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            values.append(float(loss.detach().cpu()) * len(x))
    return float(sum(values) / len(loader.dataset))


def forecast(model: nn.Module, observed: np.ndarray, history: int, maximum: int, device: torch.device) -> np.ndarray:
    generated = list(map(float, observed[:history]))
    model.eval()
    with torch.no_grad():
        while len(generated) < maximum:
            window = np.asarray(generated[-history:], dtype=np.float32)
            x = torch.from_numpy(window[None, :, None]).to(device)
            prediction, _ = model(x)
            generated.append(float(prediction.item()))
    return np.asarray(generated)


def first_crossing(values: np.ndarray, threshold: float) -> int | None:
    found = np.flatnonzero(values <= threshold)
    return int(found[0]) if len(found) else None


def evaluate(
    model: nn.Module,
    test: np.ndarray,
    config: Config,
    device: torch.device,
    output: Path,
) -> pd.DataFrame:
    true_eol = first_crossing(test, config.eol_soh)
    if true_eol is None:
        raise ValueError("Test cell does not reach configured EOL")
    histories = [x for x in (20, 30, 40, 50, 60, 70, 80, 90) if x >= config.window and x < len(test)]
    rows = []
    for history in histories:
        prediction = forecast(model, test, history, config.max_forecast_cycles, device)
        predicted_eol = first_crossing(prediction, config.eol_soh)
        overlap = min(len(test), len(prediction))
        soh_mae = float(np.mean(np.abs(prediction[:overlap] - test[:overlap])))
        true_rul = max(0, true_eol - (history - 1))
        predicted_rul = None if predicted_eol is None else max(0, predicted_eol - (history - 1))
        rows.append(
            {
                "history_length": history,
                "soh_mae_observed_range": soh_mae,
                "true_eol_index": true_eol,
                "predicted_eol_index": predicted_eol,
                "predicted_crossed_eol": predicted_eol is not None,
                "true_rul_cycles": true_rul,
                "predicted_rul_cycles": predicted_rul,
                "rul_absolute_error_cycles": None if predicted_rul is None else abs(predicted_rul - true_rul),
            }
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(len(test)), test, label="B0018 true SOH")
        ax.plot(np.arange(len(prediction)), prediction, label=f"{config.model}, history={history}")
        ax.axhline(config.eol_soh, color="red", linestyle="--", label=f"EOL={config.eol_soh}")
        ax.axvline(history - 1, color="grey", linestyle=":", label="forecast start")
        ax.set(xlabel="Discharge cycle index", ylabel="SOH", title="Autoregressive battery forecast")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"forecast_history_{history:03d}.png", dpi=160)
        plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_data = (
        Path(__file__).parents[1]
        / "battery_lstm_public_reproduction"
        / "upstream"
        / "02_data_driven_rul_prediction"
        / "data"
        / "preprocessed"
    )
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--model", choices=("lstm", "tcn", "cnn_lstm", "tcn_lstm"), default="tcn_lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--channels", default="32,32,64")
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eol-soh", type=float, default=0.70)
    parser.add_argument("--max-forecast-cycles", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    channels = tuple(int(x) for x in args.channels.split(","))
    config = Config(
        model=args.model,
        seed=args.seed,
        window=args.window,
        hidden=args.hidden,
        channels=channels,
        kernel=args.kernel,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        eol_soh=args.eol_soh,
        max_forecast_cycles=args.max_forecast_cycles,
    )
    output = args.output_root / args.model / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    train_values = [read_soh(args.data_root, "training", cell) for cell in ("B0006", "B0007")]
    validation_values = read_soh(args.data_root, "validation", "B0005")
    test_values = read_soh(args.data_root, "test", "B0018")
    train_dataset = WindowDataset(train_values, config.window)
    validation_dataset = WindowDataset([validation_values], config.window)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = SequenceRegressor(
        config.model, 1, config.hidden, config.channels, config.kernel, config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    loss_fn = nn.HuberLoss(delta=0.02)
    best = math.inf
    remaining_patience = config.patience
    records = []
    for epoch in range(1, config.epochs + 1):
        train_loss = epoch_loss(model, train_loader, loss_fn, device, optimizer)
        validation_loss = epoch_loss(model, validation_loader, loss_fn, device, None)
        scheduler.step(validation_loss)
        records.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best - 1e-8:
            best = validation_loss
            remaining_patience = config.patience
            torch.save(model.state_dict(), output / "best_model_state.pt")
        else:
            remaining_patience -= 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:03d} train={train_loss:.8f} validation={validation_loss:.8f}")
        if remaining_patience == 0:
            break

    pd.DataFrame(records).to_csv(output / "training_history.csv", index=False)
    model.load_state_dict(torch.load(output / "best_model_state.pt", map_location=device, weights_only=True))
    metrics = evaluate(model, test_values, config, device, output)
    metrics.to_csv(output / "test_metrics.csv", index=False)
    metadata = {
        **asdict(config),
        "channels": list(config.channels),
        "device": str(device),
        "torch_version": torch.__version__,
        "train_cells": ["B0006", "B0007"],
        "validation_cells": ["B0005"],
        "test_cells": ["B0018"],
        "best_validation_huber": best,
        "epochs_completed": len(records),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "feature_interface": "forward returns (prediction, transferable_feature_vector)",
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
