"""Train the independent long-life ALT26 -> Basilisk V1.4.1 model.

This entry intentionally uses only the 12 NASA ALT26 complete-EOL batteries as
the source domain. It compares the transfer candidates with target-only SSL +
B_stats and the same-capacity target-reference control under strict device OOF.
No training starts unless this script is executed explicitly.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

import train_bhump_v10_source_gated_confirmation as experiment


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v14_alt26_diverse_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v14_alt26_transfer_runs"
ALIASES = {
    "target_ssl_bstats": "target_ssl_bstats",
    "target_reference_control": "target_reference_control",
    "nasa_all5_uniform": "alt26_direct",
    "nasa_adaptive": "alt26_progress_intercell",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--formal-seeds", default="42,43,44")
    parser.add_argument("--outer-fold-seed", type=int, default=202614)
    parser.add_argument("--selection-seed", type=int, default=202615)
    parser.add_argument("--progress-knots", type=int, default=32)
    parser.add_argument("--history-epochs", type=int, default=45)
    parser.add_argument("--intercell-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-devices", type=int, default=32)
    parser.add_argument("--intercell-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-units", type=int, default=12)
    return parser.parse_args()


def run(args: argparse.Namespace):
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(data_root / "nasa17_source_compact16.csv")
    alt26 = source.loc[source.source_dataset.eq("NASA_ALT26")].copy()
    if alt26.unit_id.nunique() != 12:
        raise ValueError(f"Expected 12 ALT26 sources, found {alt26.unit_id.nunique()}")
    if float(alt26.true_eol_cycle.min()) < 202 or float(alt26.true_eol_cycle.max()) > 541:
        raise ValueError("ALT26 source EOL support is outside 202-541 EFC")
    source_dir = output / "source_pool"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "alt26_complete_eol_12.csv"
    alt26.to_csv(source_file, index=False)
    (output / "method_aliases.json").write_text(
        json.dumps(ALIASES, indent=2), encoding="utf-8",
    )
    return experiment.run(Namespace(
        data_root=data_root,
        source_file=source_file.resolve(),
        reference_run=args.reference_run.resolve(),
        output_dir=output,
        outer_folds=2 if args.smoke else args.outer_folds,
        expected_target_units=320,
        data_version="Basilisk V1.4.1 ALT26-matched mechanism-diverse",
        outer_fold_seed=args.outer_fold_seed,
        selection_seed=args.selection_seed,
        formal_seeds=args.formal_seeds,
        progress_knots=args.progress_knots,
        history_epochs=args.history_epochs,
        intercell_epochs=args.intercell_epochs,
        patience=args.patience,
        batch_devices=args.batch_devices,
        intercell_batch_size=args.intercell_batch_size,
        confirm_validation_once=False,
        device=args.device,
        resume=args.resume,
        smoke=args.smoke,
        smoke_units=args.smoke_units,
        include_nasa_finetune=False,
        nasa_pretrain_mode="soh_dynamics",
        adaptive_v2=False,
        minimal_v2=False,
    ))


if __name__ == "__main__":
    run(parse_args())
