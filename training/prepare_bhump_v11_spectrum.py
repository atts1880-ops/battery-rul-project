"""Extract V1.1 BHUMP features and attach the frozen 16-feature contract.

Development mode reads only public train/validation curves.  Sealed prefixes
are intentionally unsupported here so model selection cannot open them by
accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bhump_common import FORBIDDEN_INPUT_TOKENS, assert_feature_contract
from prepare_bhump_transfer_data import (
    domain_report, extract_basilisk_role, legacy_frame, rich_columns,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_V11 = (
    ROOT.parent / "battery_target_domain" / "output" / "v1.1_lifetime_spectrum"
    / "formal" / "battery_v11_public"
)
DEFAULT_CONTRACTS = ROOT / "bhump_transfer_v10_data" / "feature_contracts.json"
DEFAULT_NASA17 = (
    ROOT / "nasa_external_sources" / "expanded_source_pool" / "nasa_expanded_source_rich.csv"
)
DEFAULT_OUTPUT = ROOT / "bhump_transfer_v11_spectrum_data"
BIN_EDGES = (35, 61, 91, 131, 201, 351, 551)
BIN_NAMES = ("L035_060", "L061_090", "L091_130", "L131_200", "L201_350", "L351_550")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lifetime_assignments(frame: pd.DataFrame, split: str,
                         bin_edges: tuple[int, ...] = BIN_EDGES,
                         bin_names: tuple[str, ...] = BIN_NAMES) -> pd.DataFrame:
    devices = frame.groupby("unit_id", as_index=False).true_eol_cycle.first()
    if len(bin_edges) != len(bin_names) + 1:
        raise ValueError("Lifetime bin edges/names length mismatch")
    indices = np.digitize(devices.true_eol_cycle.to_numpy(float), bin_edges[1:-1], right=False)
    devices["lifetime_bin"] = [bin_names[int(index)] for index in indices]
    devices.insert(1, "split", split)
    return devices


def assert_no_input_leak(features: list[str]) -> None:
    forbidden = [
        feature for feature in features
        if any(token in feature.lower() for token in (*FORBIDDEN_INPUT_TOKENS, "lifetime_bin", "soc"))
    ]
    if forbidden:
        raise ValueError(f"Frozen feature contract contains forbidden inputs: {forbidden}")


def attach_balanced_weights(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Equal total mass per bin, per unit within bin, and per cycle within unit."""
    output = frame.merge(
        assignments[["unit_id", "lifetime_bin"]], on="unit_id", how="left", validate="many_to_one",
    )
    if output.lifetime_bin.isna().any():
        raise ValueError("Missing lifetime-bin assignment")
    bin_devices = assignments.groupby("lifetime_bin").unit_id.nunique().to_dict()
    cycle_counts = output.groupby("unit_id").size().to_dict()
    bin_count = assignments.lifetime_bin.nunique()
    output["unit_weight"] = output.apply(
        lambda row: len(assignments) / (bin_count * bin_devices[row.lifetime_bin]), axis=1,
    )
    output["sample_weight"] = output.apply(
        lambda row: 1.0 / (
            bin_count * bin_devices[row.lifetime_bin] * cycle_counts[row.unit_id]
        ), axis=1,
    )
    output["sample_weight"] *= len(output) / output.sample_weight.sum()
    return output


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if "sealed" in str(args.v11_root).lower():
        raise ValueError("V1.1 development preparation cannot use a sealed-specific path")
    contracts = json.loads(args.contracts.read_text(encoding="utf-8"))
    compact = list(contracts["bhump_degradation_invariant"])
    if len(compact) != 16:
        raise ValueError(f"Expected frozen 16-feature contract, received {len(compact)}")
    assert_feature_contract(compact)
    assert_no_input_leak(compact)

    soh_upper_bound = float(getattr(args, "soh_upper_bound", 1.10))
    train_rich, rejected_train = extract_basilisk_role(
        args.v11_root, "train", True, soh_upper_bound=soh_upper_bound
    )
    validation_rich, rejected_validation = extract_basilisk_role(
        args.v11_root, "validation", True, soh_upper_bound=soh_upper_bound
    )
    train_legacy = legacy_frame(args.v11_root, "train", True)
    validation_legacy = legacy_frame(args.v11_root, "validation", True)
    rich = rich_columns(train_rich)
    if rich != rich_columns(validation_rich):
        raise ValueError("V1.1 train/validation rich feature contracts differ")
    if len(rich) != 252:
        raise ValueError(f"Expected 252 rich features, received {len(rich)}")
    missing = [feature for feature in compact if feature not in rich]
    if missing:
        raise ValueError(f"V1.1 rich table misses frozen compact features: {missing}")

    source = pd.read_csv(args.nasa17)
    source_required = {
        "source_dataset", "unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle", *compact,
    }
    missing_source = source_required - set(source.columns)
    if missing_source:
        raise ValueError(f"NASA17 pool is missing {sorted(missing_source)}")
    source = source[[
        "source_dataset", "unit_id", "time", "target_soh", "true_rul_cycles", "true_eol_cycle", *compact,
    ]].copy()
    source.insert(0, "domain", "nasa_source")
    source.insert(4, "split", "source_train")
    if source.unit_id.nunique() != 17:
        raise ValueError(f"Expected 17 exact-EOL NASA sources, received {source.unit_id.nunique()}")
    numeric = source[["target_soh", "true_rul_cycles", "true_eol_cycle", *compact]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("NASA17 pool contains non-finite values")

    train_units = set(train_rich.unit_id.astype(str))
    validation_units = set(validation_rich.unit_id.astype(str))
    if train_units & validation_units:
        raise ValueError("V1.1 train/validation device leakage")
    bin_edges = tuple(getattr(args, "lifetime_bin_edges", BIN_EDGES))
    bin_names = tuple(getattr(args, "lifetime_bin_names", BIN_NAMES))
    assignments = pd.concat([
        lifetime_assignments(train_rich, "train", bin_edges, bin_names),
        lifetime_assignments(validation_rich, "validation", bin_edges, bin_names),
    ], ignore_index=True)
    counts = assignments.groupby(["split", "lifetime_bin"], as_index=False).size()
    spread = counts.groupby("split")["size"].agg(lambda values: int(values.max() - values.min()))
    if any(int(value) > 1 for value in spread):
        raise ValueError(f"Unbalanced V1.1 lifetime strata: {spread.to_dict()}")
    train_rich = attach_balanced_weights(
        train_rich, assignments.loc[assignments.split.eq("train")],
    )
    validation_rich = attach_balanced_weights(
        validation_rich, assignments.loc[assignments.split.eq("validation")],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "basilisk_train_rich.csv": train_rich,
        "basilisk_validation_rich.csv": validation_rich,
        "basilisk_train_legacy11.csv": train_legacy,
        "basilisk_validation_legacy11.csv": validation_legacy,
        "nasa17_source_compact16.csv": source,
        "lifetime_assignments.csv": assignments,
        "lifetime_spectrum_counts.csv": counts,
        "rejected_target_cycles.csv": pd.concat([rejected_train, rejected_validation], ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False)
    output_contracts = {
        "legacy11": list(contracts["legacy11"]),
        "bhump_rich": rich,
        "bhump_degradation_invariant": compact,
        "contract_source": str(args.contracts.resolve()),
        "contract_source_sha256": file_sha256(args.contracts),
        "v11_labels_used_for_feature_selection": False,
    }
    (args.output_dir / "feature_contracts.json").write_text(
        json.dumps(output_contracts, indent=2), encoding="utf-8",
    )
    (args.output_dir / "degradation_contract_report.json").write_text(
        json.dumps({
            "schema_version": "bhump-degradation-invariant-v11-frozen-from-v10",
            "features": compact,
            "feature_count": len(compact),
            "selection_dataset": "Basilisk V1.0/NASA source; frozen before V1.1",
            "v11_labels_used_for_selection": False,
        }, indent=2),
        encoding="utf-8",
    )
    shift_table, shift_summary = domain_report(
        source, train_rich, compact, "bhump_degradation_invariant_nasa17_vs_v11",
    )
    shift_table.to_csv(args.output_dir / "domain_shift_compact16.csv", index=False)
    (args.output_dir / "domain_shift_summary.json").write_text(
        json.dumps(shift_summary, indent=2), encoding="utf-8",
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "bhump-v11-spectrum-development-v1",
        "v11_root": str(args.v11_root.resolve()),
        "train_units": len(train_units), "validation_units": len(validation_units),
        "rich_feature_count": len(rich), "compact_feature_count": len(compact),
        "nasa_source_units": int(source.unit_id.nunique()),
        "nasa_legacy_units": int(source.loc[source.source_dataset.eq("NASA_ARC_legacy"), "unit_id"].nunique()),
        "nasa_alt26_units": int(source.loc[source.source_dataset.eq("NASA_ALT26"), "unit_id"].nunique()),
        "lifetime_bin_mapping_is_model_input": False,
        "v11_labels_used_for_contract_selection": False,
        "sealed_files_read": False, "sealed_labels_read": False,
        "rejected_cycles": int(len(tables["rejected_target_cycles.csv"])),
    }
    (args.output_dir / "development_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-root", type=Path, default=DEFAULT_V11)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--nasa17", type=Path, default=DEFAULT_NASA17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
