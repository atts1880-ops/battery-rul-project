"""Create unified NASA/Basilisk BHUMP features without opening sealed data during development."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:  # Reproducible lightweight fallback for minimal runtimes.
    LogisticRegression = None

from bhump_common import LEGACY_FEATURES, add_causal_baseline_deltas, assert_feature_contract, curve_features


ROOT = Path(__file__).resolve().parent
DEFAULT_NASA = ROOT / "bhump_source_train_data" / "bhump_cycle_features.csv"
DEFAULT_V09 = ROOT.parent / "battery_target_domain" / "output" / "v0.9_bhump" / "formal" / "battery_v09_public"
DEFAULT_OUTPUT = ROOT / "bhump_transfer_data"
NASA_TRAIN_UNITS = ("B0005", "B0018", "B0033", "B0043", "B0044")
NASA_FORBIDDEN_UNITS = ("B0030", "B0042", "B0038", "B0039")
METADATA = {"domain", "unit_id", "time", "split", "target_soh", "true_rul_cycles", "true_eol_cycle"}
SEALED_ROLES = ("sealed_id", "sealed_ood_temperature", "sealed_ood_load")


def rich_columns(frame: pd.DataFrame) -> list[str]:
    features = [column for column in frame.columns if column not in METADATA and column != "cycle_position"]
    assert_feature_contract(features)
    return features


def extract_basilisk_role(v09_root: Path, role: str, with_labels: bool,
                           soh_upper_bound: float = 1.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_path = v09_root / f"battery_{role}_curves.csv.gz"
    legacy_path = v09_root / f"battery_{role}_legacy11.csv"
    curves = pd.read_csv(curve_path)
    legacy = pd.read_csv(legacy_path)
    expected_curve_columns = {
        "unit_id", "time", "sample_index", "elapsed_s", "voltage_v", "current_a", "temperature_c",
    }
    if set(curves.columns) != expected_curve_columns:
        raise ValueError(f"Unexpected V0.9 curve schema for {role}: {curves.columns.tolist()}")
    if curves.duplicated(["unit_id", "time", "sample_index"]).any():
        raise ValueError(f"Duplicate curve sample keys in {role}")
    ambient = legacy.set_index(["unit_id", "time"])["ambient_temperature_c"]
    rows, rejected = [], []
    for (unit_id, cycle), group in curves.groupby(["unit_id", "time"], sort=False):
        group = group.sort_values("sample_index")
        try:
            features = curve_features(
                group.voltage_v.to_numpy(float), group.current_a.to_numpy(float),
                group.temperature_c.to_numpy(float), group.elapsed_s.to_numpy(float),
                float(ambient.loc[(unit_id, cycle)]),
            )
            rows.append({
                "domain": "basilisk_target", "unit_id": unit_id, "time": int(cycle), "split": role,
                **features,
            })
        except (KeyError, ValueError) as exc:
            rejected.append({"domain": "basilisk_target", "unit_id": unit_id, "time": int(cycle), "reason": str(exc)})
    frame = pd.DataFrame(rows)
    if with_labels:
        labels = pd.read_csv(v09_root / f"battery_{role}_labels.csv").rename(columns={
            "soh": "target_soh", "rul_cycles": "true_rul_cycles", "eol_cycle": "true_eol_cycle",
        })
        frame = frame.merge(labels, on=["unit_id", "time"], how="inner", validate="one_to_one")
        if not frame.target_soh.between(0.0, float(soh_upper_bound)).all():
            raise ValueError(f"SOH outside [0, {soh_upper_bound:.3f}] in {role}")
    frame, _ = add_causal_baseline_deltas(frame, METADATA)
    rejected_frame = pd.DataFrame(
        rejected, columns=["domain", "unit_id", "time", "reason"]
    )
    return frame.sort_values(["unit_id", "time"]).reset_index(drop=True), rejected_frame


def legacy_frame(v09_root: Path, role: str, with_labels: bool) -> pd.DataFrame:
    frame = pd.read_csv(v09_root / f"battery_{role}_legacy11.csv")
    output = frame[["unit_id", "time", *LEGACY_FEATURES]].copy()
    output.insert(0, "domain", "basilisk_target")
    output.insert(3, "split", role)
    if with_labels:
        labels = pd.read_csv(v09_root / f"battery_{role}_labels.csv").rename(columns={
            "soh": "target_soh", "rul_cycles": "true_rul_cycles", "eol_cycle": "true_eol_cycle",
        })
        output = output.merge(labels, on=["unit_id", "time"], validate="one_to_one")
    assert_feature_contract(LEGACY_FEATURES)
    return output.sort_values(["unit_id", "time"]).reset_index(drop=True)


def nasa_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(path)
    accessed = set(source.unit_id.unique())
    forbidden = accessed & set(NASA_FORBIDDEN_UNITS)
    if forbidden:
        raise ValueError(f"NASA source file contains prohibited external/sealed units: {sorted(forbidden)}")
    source = source.loc[source.unit_id.isin(NASA_TRAIN_UNITS)].copy()
    if set(source.unit_id.unique()) != set(NASA_TRAIN_UNITS):
        raise ValueError("NASA source training units are incomplete")
    rich = source.drop(columns=["cycle_position"], errors="ignore")
    rich.insert(0, "domain", "nasa_source")
    rich["split"] = "source_train"
    rich["true_eol_cycle"] = rich.time + rich.true_rul_cycles
    legacy = rich[["domain", "unit_id", "time", "split", "target_soh", "true_rul_cycles", "true_eol_cycle"]].copy()
    # Reconstruct the exact 11-feature baseline from the source raw rows already audited by BHUMP.
    missing = [feature for feature in LEGACY_FEATURES if feature not in rich]
    if missing:
        legacy_root = ROOT.parent / "battery_dataset" / "word-1-2-3-4-csv" / "data" / "nasa_real_target_v1"
        source_legacy = pd.read_csv(legacy_root / "nasa_target_train_features.csv")
        source_legacy = source_legacy.loc[source_legacy.unit_id.isin(NASA_TRAIN_UNITS), ["unit_id", "time", *LEGACY_FEATURES]]
        legacy = legacy.merge(source_legacy, on=["unit_id", "time"], validate="one_to_one")
    else:
        legacy[list(LEGACY_FEATURES)] = rich[list(LEGACY_FEATURES)]
    assert_feature_contract(rich_columns(rich))
    assert_feature_contract(LEGACY_FEATURES)
    return rich.sort_values(["unit_id", "time"]).reset_index(drop=True), legacy


def domain_report(source: pd.DataFrame, target: pd.DataFrame, features: list[str], variant: str,
                  seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rows = []
    for feature in features:
        left = source[feature].to_numpy(float)
        right = target[feature].to_numpy(float)
        pooled = math.sqrt((left.var() + right.var()) / 2.0)
        p01, p99 = np.quantile(left, [0.01, 0.99])
        rows.append({
            "feature_variant": variant,
            "feature": feature,
            "source_mean": float(left.mean()),
            "target_mean": float(right.mean()),
            "absolute_smd": abs(float((right.mean() - left.mean()) / pooled)) if pooled > 1e-12 else 0.0,
            "ks_statistic": float(ks_2samp(left, right).statistic),
            "source_p01": float(p01),
            "source_p99": float(p99),
            "target_outside_source_p01_p99_fraction": float(np.mean((right < p01) | (right > p99))),
        })
    count = min(len(source), len(target))
    source_sample = source.sample(count, random_state=seed)[features].to_numpy(float)
    target_sample = target.sample(count, random_state=seed + 1)[features].to_numpy(float)
    values = np.vstack([source_sample, target_sample])
    labels = np.r_[np.zeros(count, dtype=int), np.ones(count, dtype=int)]
    if LogisticRegression is not None:
        train_x, test_x, train_y, test_y = train_test_split(
            values, labels, test_size=0.35, random_state=seed, stratify=labels,
        )
    else:
        rng = np.random.default_rng(seed)
        train_indices, test_indices = [], []
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            cut = max(1, int(round(0.65 * len(indices))))
            train_indices.extend(indices[:cut])
            test_indices.extend(indices[cut:])
        train_x, train_y = values[train_indices], labels[train_indices]
        test_x, test_y = values[test_indices], labels[test_indices]
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    normalized_train = (train_x - mean) / std
    normalized_test = (test_x - mean) / std
    if LogisticRegression is not None:
        classifier = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
        classifier.fit(normalized_train, train_y)
        scores = classifier.predict_proba(normalized_test)[:, 1]
        auc = roc_auc_score(test_y, scores)
        classifier_name = "sklearn_logistic_regression"
    else:
        design = np.column_stack([np.ones(len(normalized_train)), normalized_train])
        ridge = np.eye(design.shape[1]) * 10.0
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ train_y)
        scores = np.column_stack([np.ones(len(normalized_test)), normalized_test]) @ coefficients
        ranks = rankdata(scores)
        positive = test_y == 1
        negative = ~positive
        auc = (
            (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
            / (positive.sum() * negative.sum())
        )
        classifier_name = "ridge_linear_fallback"
    source_correlation = np.nan_to_num(source[features].corr().to_numpy(float))
    target_correlation = np.nan_to_num(target[features].corr().to_numpy(float))
    summary = {
        "feature_variant": variant,
        "feature_count": len(features),
        "source_rows": len(source),
        "target_rows": len(target),
        "domain_classifier_auc": float(auc),
        "domain_classifier_implementation": classifier_name,
        "correlation_matrix_frobenius_difference": float(np.linalg.norm(source_correlation - target_correlation)),
    }
    return pd.DataFrame(rows), summary


def prepare_development(args: argparse.Namespace) -> None:
    if "sealed" in str(args.v09_root).lower() and not args.v09_root.name.startswith("battery_v09_public"):
        raise ValueError("Development stage cannot use a sealed-specific root")
    source_rich, source_legacy = nasa_frames(args.nasa_features)
    target_train_rich, rejected_train = extract_basilisk_role(args.v09_root, "train", True)
    target_validation_rich, rejected_validation = extract_basilisk_role(args.v09_root, "validation", True)
    target_train_legacy = legacy_frame(args.v09_root, "train", True)
    target_validation_legacy = legacy_frame(args.v09_root, "validation", True)
    train_units = set(target_train_rich.unit_id)
    validation_units = set(target_validation_rich.unit_id)
    if train_units & validation_units:
        raise ValueError("Basilisk train/validation device leakage")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "nasa_source_rich.csv": source_rich,
        "nasa_source_legacy11.csv": source_legacy,
        "basilisk_train_rich.csv": target_train_rich,
        "basilisk_validation_rich.csv": target_validation_rich,
        "basilisk_train_legacy11.csv": target_train_legacy,
        "basilisk_validation_legacy11.csv": target_validation_legacy,
    }
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False)
    rejected = pd.concat([rejected_train, rejected_validation], ignore_index=True)
    rejected.to_csv(args.output_dir / "rejected_target_cycles.csv", index=False)
    rich = rich_columns(source_rich)
    if rich != rich_columns(target_train_rich):
        raise ValueError("NASA and Basilisk rich feature contracts differ")
    contracts = {"legacy11": list(LEGACY_FEATURES), "bhump_rich": rich}
    (args.output_dir / "feature_contracts.json").write_text(json.dumps(contracts, indent=2), encoding="utf-8")

    report_tables, report_summaries = [], []
    for variant, features, source, target in (
        ("legacy11", list(LEGACY_FEATURES), source_legacy, target_train_legacy),
        ("bhump_rich", rich, source_rich, target_train_rich),
    ):
        table, summary = domain_report(source, target, features, variant)
        report_tables.append(table)
        report_summaries.append(summary)
    pd.concat(report_tables, ignore_index=True).to_csv(args.output_dir / "domain_shift_features.csv", index=False)
    (args.output_dir / "domain_shift_summary.json").write_text(
        json.dumps(report_summaries, indent=2), encoding="utf-8"
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development",
        "nasa_train_units": list(NASA_TRAIN_UNITS),
        "nasa_units_explicitly_excluded": list(NASA_FORBIDDEN_UNITS),
        "basilisk_train_units": len(train_units),
        "basilisk_validation_units": len(validation_units),
        "sealed_files_read": False,
        "target_rejected_cycles": len(rejected),
        "feature_contracts": {key: len(value) for key, value in contracts.items()},
    }
    (args.output_dir / "development_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def prepare_sealed(args: argparse.Namespace) -> None:
    if args.sealed_role not in SEALED_ROLES:
        raise ValueError(f"Unknown sealed role: {args.sealed_role}")
    rich, rejected = extract_basilisk_role(args.v09_root, args.sealed_role, False)
    legacy = legacy_frame(args.v09_root, args.sealed_role, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rich.to_csv(args.output_dir / f"basilisk_{args.sealed_role}_rich.csv", index=False)
    legacy.to_csv(args.output_dir / f"basilisk_{args.sealed_role}_legacy11.csv", index=False)
    rejected.to_csv(args.output_dir / f"rejected_{args.sealed_role}_cycles.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "sealed"), default="development")
    parser.add_argument("--nasa-features", type=Path, default=DEFAULT_NASA)
    parser.add_argument("--v09-root", type=Path, default=DEFAULT_V09)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sealed-role", choices=SEALED_ROLES)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stage == "development":
        prepare_development(arguments)
    else:
        if not arguments.sealed_role:
            raise SystemExit("--sealed-role is required for --stage sealed")
        prepare_sealed(arguments)
