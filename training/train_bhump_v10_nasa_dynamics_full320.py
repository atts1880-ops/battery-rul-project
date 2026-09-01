"""Final full-320 refit for the frozen NASA5 dynamics-adaptive model.

This is a deployment refit, not another model-selection experiment.  Epoch
counts and the lifetime policy are aggregated deterministically from the
completed strict nested-OOF run.  All 320 Basilisk V1.0 training devices are
then used for target SSL, supervised target adaptation, B_stats training and
the NASA inter-cell head.  Validation and sealed files are never opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from train_bhump_positive_transfer import snapshot
from train_bhump_v10_bstats_oof import maximum_lifetime_without_test
from train_bhump_v10_bstats_refit_oof import (
    fit_bstats_fixed_epochs,
    fit_multitask_fixed_epochs,
)
from train_bhump_v10_history_ablation import (
    HistoryConfig,
    fit_statistics,
    make_bundle,
    predict_bundle,
    regression_metrics,
    soh_metrics,
)
from train_bhump_v10_intercell_transfer import (
    IntercellConfig,
    flat_target,
    nasa_reference_bank,
    predict_intercell,
    robust_vector_fit,
    target_ssl_state,
    train_intercell_fixed,
)
from train_bhump_v10_nasa_dynamics_pretrain import nasa_dynamics_initial_state
from train_bhump_v10_rul_multitask import build_config
from train_bhump_v10_source_gated_confirmation import (
    apply_lifetime_policy,
    prepare_oof_data,
    subset_bank,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_SELECTION = ROOT / "bhump_v10_nasa_dynamics_pretrain_strict_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_nasa_dynamics_full320_runs"
NASA5 = ("B0005", "B0018", "B0033", "B0043", "B0044")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="52,53,54")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-units", type=int, default=40)
    return parser.parse_args()


def _policy_key(policy: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(policy["mode"]),
        tuple(map(str, policy["source_units"])),
        float(policy["gamma"]),
        float(policy["blend_weight"]),
        bool(policy.get("support_gated", False)),
        bool(policy.get("fallback_to_uniform", False)),
    )


def frozen_settings(selection_run: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(selection_run.glob("fold_result_seed_*_fold_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row["seed"]) in seeds:
            rows.append(row)
    expected = 5 * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} frozen OOF selection rows, found {len(rows)}"
        )
    settings: dict[str, Any] = {"per_seed": {}}
    for seed in seeds:
        selected = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: int(row["outer_fold"]),
        )
        if len(selected) != 5:
            raise RuntimeError(f"Seed {seed} does not contain five OOF folds")
        settings["per_seed"][str(seed)] = {
            "tcn_epochs": int(median([
                int(row["methods"]["nasa_dynamics_finetune_bstats"]["selection_tcn_epoch"])
                for row in selected
            ])),
            "bstats_epochs": int(median([
                int(row["methods"]["nasa_dynamics_finetune_bstats"]["selection_bstats_epoch"])
                for row in selected
            ])),
            "intercell_epochs": int(median([
                int(row["methods"]["nasa_dynamics_adaptive"]["selection_epoch"])
                for row in selected
            ])),
        }
    counts = Counter(
        _policy_key(row["methods"]["nasa_dynamics_adaptive"]["policy"])
        for row in rows
    )
    key, count = counts.most_common(1)[0]
    if count < (len(rows) // 2 + 1):
        raise RuntimeError("No majority frozen NASA lifetime policy exists")
    mode, units, gamma, blend, support_gated, fallback = key
    settings["policy"] = {
        "mode": mode,
        "source_units": list(units),
        "gamma": gamma,
        "blend_weight": blend,
        "support_gated": support_gated,
        "fallback_to_uniform": fallback,
        "selection_provenance": "majority_of_strict_nested_oof_folds",
        "selection_frequency": int(count),
        "selection_total": int(len(rows)),
    }
    settings["selection_files"] = expected
    return settings


def _bank_payload(bank: Any) -> dict[str, Any]:
    return {
        "unit_ids": list(map(str, bank.unit_ids)),
        "knots": bank.knots,
        "representation": bank.representation,
        "eol": bank.eol,
        "metadata": bank.metadata.to_dict(orient="list"),
        "domain": str(bank.domain),
    }


def train_seed(
    seed: int,
    source: Any,
    target: Any,
    target_knees: Any,
    features: list[str],
    reference: dict[str, Any],
    settings: dict[str, Any],
    output: Path,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    units = sorted(target.unit_id.astype(str).unique())
    expected_units = len(units) if smoke else 320
    if len(units) != expected_units:
        raise AssertionError(f"Expected {expected_units} full-refit devices")
    source_units = tuple(sorted(source.unit_id.astype(str).unique()))
    if source_units != tuple(sorted(NASA5)):
        raise ValueError(f"Final refit requires exactly NASA5, received {source_units}")

    selected = settings["per_seed"][str(seed)]
    tcn_epochs = 2 if smoke else int(selected["tcn_epochs"])
    bstats_epochs = 2 if smoke else int(selected["bstats_epochs"])
    intercell_epochs = 2 if smoke else int(selected["intercell_epochs"])
    history_config = HistoryConfig(
        epochs=3 if smoke else 45,
        patience=2 if smoke else 7,
        batch_devices=32,
    )
    intercell_config = IntercellConfig(
        progress_knots=32,
        epochs=3 if smoke else 40,
        patience=2 if smoke else 7,
        batch_size=256,
    )
    config = build_config(reference, 24, smoke)
    maximum_lifetime = maximum_lifetime_without_test(target)

    print(json.dumps({
        "stage": "target_ssl", "seed": seed,
        "target_devices": len(units), "device": str(device),
    }), flush=True)
    ssl_state, target_state = target_ssl_state(
        target, features, config, seed, device,
    )
    if set(map(str, target_state.fit_units)) != set(units):
        raise AssertionError("Full320 scaler/SSL did not use exactly all train devices")

    dynamics_audit_path = output / f"nasa_dynamics_pretrain_full320_seed_{seed}.json"
    print(json.dumps({"stage": "nasa_dynamics_pretrain", "seed": seed}), flush=True)
    nasa_state = nasa_dynamics_initial_state(
        source, target, features, ssl_state, config, seed, device,
        dynamics_audit_path,
    )

    print(json.dumps({
        "stage": "target_supervised_refit", "seed": seed,
        "epochs": tcn_epochs,
    }), flush=True)
    tcn = fit_multitask_fixed_epochs(
        nasa_state, target, features, target_state, maximum_lifetime,
        config, seed, tcn_epochs, device,
    )
    for parameter in tcn.parameters():
        parameter.requires_grad = False

    print(json.dumps({"stage": "build_bstats", "seed": seed}), flush=True)
    train_bundle = make_bundle(
        target, target_knees, features,
        target_state.median, target_state.iqr,
        tcn, config, maximum_lifetime, device, units,
    )
    statistics_shadow = copy.deepcopy(train_bundle)
    stats_median, stats_iqr = fit_statistics(train_bundle, statistics_shadow)
    print(json.dumps({
        "stage": "bstats_refit", "seed": seed, "epochs": bstats_epochs,
    }), flush=True)
    head = fit_bstats_fixed_epochs(
        train_bundle, maximum_lifetime, seed,
        bstats_epochs, history_config, device,
    )
    base_prediction = predict_bundle(
        head, train_bundle, maximum_lifetime, device,
        history_config.batch_devices,
    )
    train_flat = flat_target(train_bundle, base_prediction)
    vector_state = robust_vector_fit(
        train_flat.representation,
        train_flat.frame.unit_id.astype(str),
        "target_full320_after_nasa_dynamics_pretrain",
    )

    full_bank = nasa_reference_bank(
        source, features, tcn, config, source_units,
        intercell_config.progress_knots, device,
    )
    policy = dict(settings["policy"])
    selected_bank = subset_bank(
        full_bank, tuple(map(str, policy["source_units"])),
    )
    print(json.dumps({
        "stage": "nasa_intercell_refit", "seed": seed,
        "epochs": intercell_epochs,
        "source_units": selected_bank.unit_ids,
    }), flush=True)
    intercell = train_intercell_fixed(
        train_flat, selected_bank, vector_state, maximum_lifetime,
        seed, intercell_epochs, intercell_config, device,
    )
    raw = predict_intercell(
        intercell, train_flat, selected_bank, vector_state,
        maximum_lifetime, intercell_config, device,
    )
    fitted = apply_lifetime_policy(raw, selected_bank, policy, maximum_lifetime)
    train_metrics = {
        **regression_metrics(fitted, "predicted_rul_raw"),
        **soh_metrics(fitted),
    }

    checkpoint = {
        "format_version": 1,
        "method": "nasa_dynamics_adaptive_full320",
        "seed": seed,
        "model_state_tcn": snapshot(tcn),
        "model_state_bstats": snapshot(head),
        "model_state_intercell": snapshot(intercell),
        "features": features,
        "tcn_configuration": asdict(config),
        "history_configuration": asdict(history_config),
        "intercell_configuration": asdict(intercell_config),
        "target_scaler_median": target_state.median,
        "target_scaler_iqr": target_state.iqr,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "bstats_median": stats_median,
        "bstats_iqr": stats_iqr,
        "target_vector_median": vector_state.median,
        "target_vector_iqr": vector_state.iqr,
        "target_vector_fit_units": list(vector_state.fit_units),
        "nasa_reference_bank": _bank_payload(selected_bank),
        "lifetime_policy": policy,
        "maximum_training_lifetime": maximum_lifetime,
        "target_units": units,
        "target_supervised_device_count": len(units),
        "source_units": list(source_units),
        "tcn_epochs": tcn_epochs,
        "bstats_epochs": bstats_epochs,
        "intercell_epochs": intercell_epochs,
        "training_resubstitution_metrics_not_for_model_claims": train_metrics,
        "validation_accessed": False,
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }
    checkpoint_path = output / f"checkpoint_full320_nasa_dynamics_adaptive_seed_{seed}.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "seed": seed,
        "checkpoint": checkpoint_path.name,
        "target_devices": len(units),
        "source_devices": len(source_units),
        "tcn_epochs": tcn_epochs,
        "bstats_epochs": bstats_epochs,
        "intercell_epochs": intercell_epochs,
        "policy": policy,
        "training_resubstitution_metrics_not_for_model_claims": train_metrics,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / f"full320_result_seed_{seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "stage": "seed_complete", "seed": seed,
        "checkpoint": checkpoint_path.name,
    }), flush=True)
    return result


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a non-empty unique list")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    settings = frozen_settings(args.selection_run, seeds)
    source, target, features, _report, target_knees = prepare_oof_data(
        args.data_root, args.source_file, expected_target_units=320,
    )
    if args.smoke:
        keep = sorted(target.unit_id.astype(str).unique())[: args.smoke_units]
        target = target.loc[target.unit_id.astype(str).isin(keep)].copy()
        target_knees = target_knees.loc[
            target_knees.unit_id.astype(str).isin(keep)
        ].copy()
        seeds = seeds[:1]
    reference = json.loads(
        (args.reference_run / "experiment_manifest.json").read_text(
            encoding="utf-8",
        )
    )
    results: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = (
            args.output_dir
            / f"checkpoint_full320_nasa_dynamics_adaptive_seed_{seed}.pt"
        )
        result_path = args.output_dir / f"full320_result_seed_{seed}.json"
        if args.resume and checkpoint.exists() and result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            print(json.dumps({
                "stage": "seed_resumed", "seed": seed,
                "checkpoint": checkpoint.name,
            }), flush=True)
            continue
        results.append(train_seed(
            seed, source, target, target_knees, features, reference,
            settings, args.output_dir, device, args.smoke,
        ))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full320_final_refit",
        "method": "NASA5 dynamics pretrain + TCN24 + B_stats38 + inter-cell adaptive",
        "source_units": list(NASA5),
        "target_device_count": int(target.unit_id.nunique()),
        "features": features,
        "feature_count": len(features),
        "seeds": list(seeds),
        "frozen_settings": settings,
        "results": results,
        "performance_claim_source": str(args.selection_run / "oof_metrics.csv"),
        "performance_claim_rul_mae_efc": 4.345603979546951,
        "validation_accessed": False,
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }
    (args.output_dir / "full320_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
