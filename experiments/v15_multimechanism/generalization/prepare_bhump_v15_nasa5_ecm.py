"""Prepare frozen-contract BHUMP tables for V1.5 NASA5-protocol ECM data.

Only public train and validation curves are accepted.  The frozen 16-feature
contract and the exact NASA5 rich table are copied from the registered V1.0
experiment; V1.5 labels never participate in feature selection.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

import prepare_bhump_v11_spectrum as shared


ROOT = Path(__file__).resolve().parent
DEFAULT_V15 = (
    ROOT.parent / "battery_target_domain" / "output"
    / "v1.5_nasa5_ecm_calibrated_run4" / "formal" / "battery_v15_public"
)
DEFAULT_CONTRACTS = ROOT / "bhump_transfer_v10_data" / "feature_contracts.json"
DEFAULT_NASA17 = (
    ROOT / "nasa_external_sources" / "expanded_source_pool"
    / "nasa_expanded_source_rich.csv"
)
DEFAULT_NASA5 = ROOT / "bhump_transfer_v10_data" / "nasa_source_rich.csv"
DEFAULT_OUTPUT = ROOT / "bhump_transfer_v15_nasa5_ecm_data"
NASA5 = ("B0005", "B0018", "B0033", "B0043", "B0044")


def rewrite(path: Path, updates: dict, remove: tuple[str, ...] = ()) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in remove:
        payload.pop(key, None)
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict:
    lowered = str(args.v15_root).lower()
    if any(token in lowered for token in ("sealed", "custodian", "private")):
        raise ValueError("V1.5 preparation accepts only the public dataset root")
    audit = shared.prepare(Namespace(
        v11_root=args.v15_root,
        contracts=args.contracts,
        nasa17=args.nasa17,
        output_dir=args.output_dir,
        soh_upper_bound=1.11,
        lifetime_bin_edges=(35, 49, 63, 77, 96),
        lifetime_bin_names=("L035_048", "L049_062", "L063_076", "L077_095"),
    ))
    nasa5 = pd.read_csv(args.nasa5)
    units = tuple(sorted(nasa5.unit_id.astype(str).unique()))
    if units != NASA5:
        raise ValueError(f"Expected exact NASA5 source order {NASA5}, received {units}")
    nasa5.to_csv(args.output_dir / "nasa_source_rich.csv", index=False)
    rewrite(args.output_dir / "feature_contracts.json", {
        "contract_target": "Basilisk V1.5 NASA5-protocol ECM",
        "v15_labels_used_for_feature_selection": False,
    }, ("v11_labels_used_for_feature_selection",))
    rewrite(args.output_dir / "degradation_contract_report.json", {
        "schema_version": "bhump-degradation-invariant-v15-frozen-from-v10",
        "selection_dataset": "Basilisk V1.0/NASA5; frozen before V1.5",
        "v15_labels_used_for_selection": False,
    }, ("v11_labels_used_for_selection",))
    rewrite(args.output_dir / "domain_shift_summary.json", {
        "comparison": "NASA17 audit pool versus Basilisk V1.5 train",
        "registered_transfer_source": list(NASA5),
    })
    audit_path = args.output_dir / "development_audit.json"
    rewrite(audit_path, {
        "schema_version": "bhump-v15-nasa5-ecm-development-v1",
        "v15_root": str(args.v15_root.resolve()),
        "registered_transfer_source_units": list(NASA5),
        "v15_labels_used_for_contract_selection": False,
        "frozen_feature_count": 16,
        "sealed_files_read": False,
        "sealed_labels_read": False,
        "custodian_files_read": False,
    }, ("v11_root", "v11_labels_used_for_contract_selection"))
    result = json.loads(audit_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15-root", type=Path, default=DEFAULT_V15)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--nasa17", type=Path, default=DEFAULT_NASA17)
    parser.add_argument("--nasa5", type=Path, default=DEFAULT_NASA5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
