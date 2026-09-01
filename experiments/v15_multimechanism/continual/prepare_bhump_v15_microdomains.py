"""Extract frozen V1.0 compact16 BHUMP features for V1.5 microdomains."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import prepare_bhump_v11_spectrum as shared


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT.parent / "battery_target_domain" / "output" / "v1.5_incremental_microdomains"
DEFAULT_OUTPUT = ROOT / "bhump_transfer_v15_microdomains_data"
DEFAULT_CONTRACTS = ROOT / "bhump_transfer_v10_data" / "feature_contracts.json"
DEFAULT_NASA17 = ROOT / "nasa_external_sources" / "expanded_source_pool" / "nasa_expanded_source_rich.csv"
DEFAULT_NASA5 = ROOT / "bhump_transfer_v10_data" / "nasa_source_rich.csv"
DOMAINS = ("knee_spectrum", "thermal_load", "decoupled_aging", "path_nonstationary")
NASA5 = ("B0005", "B0018", "B0033", "B0043", "B0044")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--nasa17", type=Path, default=DEFAULT_NASA17)
    parser.add_argument("--nasa5", type=Path, default=DEFAULT_NASA5)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    unknown = sorted(set(selected) - set(DOMAINS))
    if unknown:
        raise ValueError(f"Unknown microdomains: {unknown}")
    nasa5 = pd.read_csv(args.nasa5)
    if tuple(sorted(nasa5.unit_id.astype(str).unique())) != NASA5:
        raise ValueError("Registered source must be exact NASA5")
    records = []
    for domain in selected:
        public = args.raw_root / domain / args.mode / "battery_v15_public"
        lowered = str(public).lower()
        if any(token in lowered for token in ("sealed", "private", "custodian")):
            raise ValueError(f"Unsafe public root: {public}")
        output = args.output_dir / args.mode / domain
        audit = shared.prepare(Namespace(
            v11_root=public,
            contracts=args.contracts,
            nasa17=args.nasa17,
            output_dir=output,
            soh_upper_bound=1.11,
            lifetime_bin_edges=(35, 56, 76, 101, 131),
            lifetime_bin_names=("L035_055", "L056_075", "L076_100", "L101_130"),
        ))
        nasa5.to_csv(output / "nasa_source_rich.csv", index=False)
        contract_path = output / "feature_contracts.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.update({
            "contract_target": f"Basilisk V1.5 microdomain {domain}",
            "contract_source_sha256": file_sha256(args.contracts),
            "microdomain_labels_used_for_feature_selection": False,
        })
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        records.append({
            "domain": domain,
            "raw_public_root": str(public.resolve()),
            "feature_root": str(output.resolve()),
            "train_units": int(audit["train_units"]),
            "locked_test_units": int(audit["validation_units"]),
            "compact_feature_count": int(audit["compact_feature_count"]),
        })
    manifest = {
        "schema_version": "bhump-v15-incremental-microdomains-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "domains": records,
        "feature_contract": str(args.contracts.resolve()),
        "feature_contract_sha256": file_sha256(args.contracts),
        "registered_source_units": list(NASA5),
        "labels_used_for_feature_selection": False,
        "sealed_files_read": False,
        "sealed_labels_read": False,
    }
    target = args.output_dir / args.mode / "feature_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
