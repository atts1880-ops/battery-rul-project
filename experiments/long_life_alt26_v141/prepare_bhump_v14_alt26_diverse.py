"""Prepare leak-free frozen-contract BHUMP tables for Basilisk V1.4.1.

The extractor and 16-feature contract are reused unchanged from V1.0. Only
public train/validation curves are read; sealed and custodian files are not.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import prepare_bhump_v11_spectrum as shared


ROOT = Path(__file__).resolve().parent
DEFAULT_V14 = (
    ROOT.parent / "battery_target_domain" / "output"
    / "v1.4_alt26_diverse_final" / "formal" / "battery_v14_public"
)
DEFAULT_CONTRACTS = ROOT / "bhump_transfer_v10_data" / "feature_contracts.json"
DEFAULT_NASA17 = (
    ROOT / "nasa_external_sources" / "expanded_source_pool"
    / "nasa_expanded_source_rich.csv"
)
DEFAULT_OUTPUT = ROOT / "bhump_transfer_v14_alt26_diverse_data"


def rewrite_json(path: Path, updates: dict[str, Any], remove: tuple[str, ...] = ()) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in remove:
        payload.pop(key, None)
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root_text = str(args.v14_root).lower()
    if "sealed" in root_text or "custodian" in root_text or "private" in root_text:
        raise ValueError("V1.4 preparation accepts only the public dataset root")
    audit = shared.prepare(Namespace(
        v11_root=args.v14_root,
        contracts=args.contracts,
        nasa17=args.nasa17,
        output_dir=args.output_dir,
        soh_upper_bound=1.11,
        lifetime_bin_edges=(202, 231, 261, 301, 361, 451, 542),
        lifetime_bin_names=(
            "L202_230", "L231_260", "L261_300",
            "L301_360", "L361_450", "L451_541",
        ),
    ))
    rewrite_json(
        args.output_dir / "feature_contracts.json",
        {
            "contract_target": "Basilisk V1.4.1 ALT26-matched diverse",
            "v14_labels_used_for_feature_selection": False,
        },
        ("v11_labels_used_for_feature_selection",),
    )
    rewrite_json(
        args.output_dir / "degradation_contract_report.json",
        {
            "schema_version": "bhump-degradation-invariant-v14-frozen-from-v10",
            "selection_dataset": "Basilisk V1.0/NASA source; frozen before V1.4",
            "v14_labels_used_for_selection": False,
        },
        ("v11_labels_used_for_selection",),
    )
    rewrite_json(
        args.output_dir / "domain_shift_summary.json",
        {"comparison": "NASA17 versus Basilisk V1.4.1 ALT26-matched diverse train"},
    )
    audit_path = args.output_dir / "development_audit.json"
    rewrite_json(
        audit_path,
        {
            "schema_version": "bhump-v14-alt26-diverse-development-v1",
            "v14_root": str(args.v14_root.resolve()),
            "v14_labels_used_for_contract_selection": False,
            "frozen_feature_count": 16,
            "supervision_soh_range": [0.0, 1.11],
            "sealed_files_read": False,
            "sealed_labels_read": False,
            "custodian_files_read": False,
        },
        ("v11_root", "v11_labels_used_for_contract_selection"),
    )
    final_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    print(json.dumps(final_audit, indent=2), flush=True)
    return final_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-root", type=Path, default=DEFAULT_V14)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--nasa17", type=Path, default=DEFAULT_NASA17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
