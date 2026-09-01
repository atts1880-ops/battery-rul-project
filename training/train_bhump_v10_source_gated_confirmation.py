"""Fresh-fold confirmation of NASA source screening and lifetime-support gating.

This experiment preserves Basilisk V1.0, the frozen 16-feature contract,
TCN24 and B_stats38.  Source subsets and lifetime-support strength are chosen
only on each outer fold's inner validation devices.  The 80-device development
validation split is accessed only if the complete fresh-fold OOF experiment
passes every pre-registered OOF condition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bhump_common import FORBIDDEN_INPUT_TOKENS
from train_bhump_degradation_transfer import calibrate_source_labels, l2sp_source_fit
from train_bhump_positive_transfer import make_windows, robust_fit, snapshot
from train_bhump_v10_bstats_oof import (
    device_lifetimes, inner_split, maximum_lifetime_without_test, subset,
)
from train_bhump_v10_bstats_refit_oof import (
    fit_bstats_fixed_epochs, fit_multitask_fixed_epochs,
)
from train_bhump_v10_history_ablation import (
    HistoryConfig, assert_causal, fit_statistics, make_bundle, predict_bundle,
    regression_metrics, soh_metrics, train_variant,
)
from train_bhump_v10_intercell_transfer import (
    BLEND_GRID, FlatTarget, IntercellConfig, ReferenceBank,
    apply_blend, choose_blend, choose_target_references, difficult_metrics,
    ensemble_predictions, flat_target, metric_rows, nasa_reference_bank,
    paired_device_bootstrap, per_device_results, predict_intercell,
    robust_vector_fit, save_base_checkpoint, target_reference_bank,
    target_ssl_state, train_intercell_fixed, train_intercell_select,
)
from train_bhump_v10_rul_multitask import (
    DynamicsTCN, attach_dynamics_labels, build_config, train_configuration,
)
from train_bhump_v10_nasa_dynamics_pretrain import nasa_dynamics_initial_state


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "bhump_transfer_v10_data"
DEFAULT_REFERENCE = ROOT / "bhump_v10_law_transfer_runs"
DEFAULT_OUTPUT = ROOT / "bhump_v10_source_gated_confirmation_strict_runs"
DEFAULT_DYNAMICS_OUTPUT = ROOT / "bhump_v10_nasa_dynamics_pretrain_strict_runs"
FORMAL_SEEDS = (52, 53, 54)
METHODS = (
    "target_ssl_bstats", "nasa_all5_uniform",
    "target_reference_control", "nasa_adaptive",
)
GAMMA_GRID = (0.0, 1.0, 2.0, 4.0)
MINIMUM_SUBSET_SIZE = 3
SUBSET_REQUIRED_IMPROVEMENT = 0.10
SUBSET_TIE_TOLERANCE = 0.02


def prepare_oof_data(
    data_root: Path, source_file: Path | None = None,
    expected_target_units: int | None = 320,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], pd.DataFrame]:
    """Load NASA and the 320 target-train devices without opening validation."""
    contracts = json.loads(
        (data_root / "feature_contracts.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (data_root / "degradation_contract_report.json").read_text(
            encoding="utf-8",
        )
    )
    features = list(contracts["bhump_degradation_invariant"])
    if features != list(report["features"]) or len(features) != 16:
        raise ValueError("Frozen degradation feature contract mismatch")
    source_path = source_file or Path("nasa_source_rich.csv")
    if not source_path.is_absolute():
        source_path = data_root / source_path
    source = pd.read_csv(source_path)
    target = pd.read_csv(data_root / "basilisk_train_rich.csv")
    source_units = tuple(sorted(source.unit_id.astype(str).unique()))
    required = {
        "unit_id", "time", "target_soh", "true_eol_cycle",
        "true_rul_cycles", *features,
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"NASA source file is missing {sorted(missing)}")
    if len(source_units) < MINIMUM_SUBSET_SIZE:
        raise ValueError(
            f"Need at least {MINIMUM_SUBSET_SIZE} exact-RUL source devices"
        )
    if source.groupby("unit_id").size().min() < 8:
        raise ValueError("Every NASA RUL reference must contain at least 8 curves")
    if not np.isfinite(
        source[["target_soh", "true_eol_cycle", "true_rul_cycles"]].to_numpy(float)
    ).all():
        raise ValueError("NASA RUL source contains censored or non-finite labels")
    if (
        expected_target_units is not None
        and target.unit_id.nunique() != int(expected_target_units)
    ):
        raise ValueError(
            f"Expected exactly {expected_target_units} target-training devices, "
            f"received {target.unit_id.nunique()}"
        )
    for frame in (source, target):
        if not np.isfinite(frame[features].to_numpy(float)).all():
            raise ValueError("Non-finite model input")
    target, target_knees = attach_dynamics_labels(target)
    return source, target, features, report, target_knees


def seeded_stratified_outer_folds(frame: pd.DataFrame, folds: int,
                                   assignment_seed: int) -> dict[str, int]:
    """Lifetime-stratified deterministic folds with a fresh hash-based order."""
    life = device_lifetimes(frame).copy()
    bin_count = min(20, max(folds, len(life) // 12))
    life["life_bin"] = pd.qcut(
        life.eol_cycle.rank(method="first"), bin_count,
        labels=False, duplicates="drop",
    )
    assignment: dict[str, int] = {}
    fold_offset = 0
    for _, group in life.groupby("life_bin", sort=True):
        ordered = sorted(
            group.unit_id.astype(str),
            key=lambda unit: int.from_bytes(
                hashlib.sha256(
                    f"{assignment_seed}:{unit}".encode("utf-8"),
                ).digest()[:8],
                "big",
            ),
        )
        for index, unit in enumerate(ordered):
            assignment[unit] = int((fold_offset + index) % folds)
        fold_offset = (fold_offset + len(ordered)) % folds
    counts = pd.Series(assignment).value_counts().sort_index()
    if (
        len(assignment) != frame.unit_id.nunique()
        or len(counts) != folds
        or counts.max() - counts.min() > 1
    ):
        raise RuntimeError(f"Invalid seeded outer folds: {counts.to_dict()}")
    return assignment


def subset_bank(bank: ReferenceBank, selected_units: tuple[str, ...]) -> ReferenceBank:
    indices = [bank.unit_ids.index(unit) for unit in selected_units]
    metadata = bank.metadata.loc[
        bank.metadata.unit_id.astype(str).isin(selected_units)
    ].copy()
    return ReferenceBank(
        list(selected_units), bank.knots.copy(),
        bank.representation[indices].copy(), bank.eol[indices].copy(),
        metadata, bank.domain,
    )


def remap_reference_policy(
    policy: dict[str, Any], selection_bank: ReferenceBank,
    refit_bank: ReferenceBank,
) -> dict[str, Any]:
    """Map target-reference choices by lifetime-quantile position, not unit ID."""
    if len(selection_bank.unit_ids) != len(refit_bank.unit_ids):
        raise ValueError("Selection and refit reference banks must have equal size")
    positions = [
        selection_bank.unit_ids.index(str(unit))
        for unit in policy["source_units"]
    ]
    output = dict(policy)
    output["selection_source_units"] = list(map(str, policy["source_units"]))
    output["source_positions"] = positions
    output["source_units"] = [refit_bank.unit_ids[index] for index in positions]
    return output


def nasa_supervised_initial_state(
    source: pd.DataFrame, target_train: pd.DataFrame, features: list[str],
    ssl_state: dict[str, torch.Tensor], config: Any, seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pretrain on every configured NASA source, then return encoder weights.

    This is the ordinary-pretraining control: no progress-reference lookup,
    source selection, support gate, CORAL or MMD is used.  Source curves are
    robust-scaled in the source domain and SOH amplitudes are mapped only to
    the median initial SOH of the current outer-training target devices.
    """
    initial = (
        target_train.sort_values(["unit_id", "time"])
        .groupby("unit_id", as_index=False).first().target_soh.median()
    )
    calibrated = calibrate_source_labels(source, float(initial))
    source_state = robust_fit(calibrated, features, "NASA17:supervised_pretrain")
    source_x, source_y, _ = make_windows(
        calibrated, features, source_state, config.window, True,
    )
    if source_y is None:
        raise AssertionError("NASA supervised pretraining requires SOH labels")
    model = DynamicsTCN(len(features), config)
    model.load_state_dict({name: value.detach().clone() for name, value in ssl_state.items()})
    model = l2sp_source_fit(
        model, source_x, source_y, ssl_state, 0.0, seed + 1, config, device,
    )
    return snapshot(model)


def uniform_policy(bank: ReferenceBank, blend_weight: float,
                   validation_mae: float) -> dict[str, Any]:
    return {
        "mode": "uniform_median",
        "source_units": list(bank.unit_ids),
        "gamma": 0.0,
        "blend_weight": float(blend_weight),
        "selection_validation_mae": float(validation_mae),
        "support_gated": False,
        "fallback_to_uniform": False,
    }


def apply_lifetime_policy(raw: pd.DataFrame, bank: ReferenceBank,
                          policy: dict[str, Any],
                          maximum_lifetime: float) -> pd.DataFrame:
    output = raw.copy()
    units = tuple(map(str, policy["source_units"]))
    blend = float(policy["blend_weight"])
    if policy["mode"] in {"uniform_median", "uniform_fallback"}:
        selected_columns = [f"reference_rul__{unit}" for unit in units]
        output["intercell_rul"] = np.nanmedian(
            output[selected_columns].to_numpy(float), axis=1,
        )
        output["support_factor"] = 1.0
        output["effective_blend_weight"] = blend
        for unit in bank.unit_ids:
            output[f"source_weight__{unit}"] = (
                1.0 / len(units) if unit in units else 0.0
            )
    else:
        gamma = float(policy["gamma"])
        indices = [bank.unit_ids.index(unit) for unit in units]
        known_eol = bank.eol[indices].astype(float)
        reference_rul = output[
            [f"reference_rul__{unit}" for unit in units]
        ].to_numpy(float)
        time = output.time.to_numpy(float)
        predicted_eol = np.maximum(reference_rul + time[:, None], time[:, None])
        base_eol = np.maximum(output.base_rul.to_numpy(float) + time, time + 1.0e-3)
        log_gap = np.abs(
            np.log((base_eol[:, None] + 1.0e-3) / (known_eol[None, :] + 1.0e-3))
        )
        logits = -gamma * log_gap
        logits -= np.nanmax(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
        aggregated_eol = np.exp(
            np.sum(weights * np.log(np.maximum(predicted_eol, time[:, None] + 1.0e-3)), axis=1)
        )
        output["intercell_rul"] = np.maximum(aggregated_eol - time, 0.0)
        # Reduce transfer strength whenever the target estimate lies outside
        # the selected source lifetime support in either direction.  The
        # original five-cell experiment only needed the upper-tail guard;
        # ALT26 introduces 200--540 EFC sources, so a lower-tail guard is
        # essential for 50--120 EFC Basilisk devices.
        lower_deficit = np.maximum(
            np.log((float(np.min(known_eol)) + 1.0e-3) / (base_eol + 1.0e-3)),
            0.0,
        )
        upper_deficit = np.maximum(
            np.log((base_eol + 1.0e-3) / (float(np.max(known_eol)) + 1.0e-3)),
            0.0,
        )
        deficit = lower_deficit + upper_deficit
        support = np.exp(-gamma * deficit)
        output["support_factor"] = support
        output["effective_blend_weight"] = blend * support
        for unit in bank.unit_ids:
            if unit in units:
                output[f"source_weight__{unit}"] = weights[:, units.index(unit)]
            else:
                output[f"source_weight__{unit}"] = 0.0
    cap = np.maximum(maximum_lifetime - output.time.to_numpy(float), 0.0)
    effective = output.effective_blend_weight.to_numpy(float)
    output["predicted_rul_raw"] = np.clip(
        (1.0 - effective) * output.base_rul.to_numpy(float)
        + effective * output.intercell_rul.to_numpy(float),
        0.0, cap,
    )
    weight_columns = [f"source_weight__{unit}" for unit in bank.unit_ids]
    weights = output[weight_columns].to_numpy(float)
    output["effective_source_count"] = 1.0 / np.maximum(
        np.sum(weights**2, axis=1), 1.0e-8,
    )
    output["selected_source_count"] = len(units)
    output["selected_gamma"] = float(policy["gamma"])
    return output


def candidate_policy(raw: pd.DataFrame, bank: ReferenceBank,
                     units: tuple[str, ...], gamma: float, blend: float,
                     maximum_lifetime: float) -> tuple[dict[str, Any], float]:
    policy = {
        "mode": "lifetime_support_gate",
        "source_units": list(units),
        "gamma": float(gamma),
        "blend_weight": float(blend),
        "support_gated": True,
        "fallback_to_uniform": False,
    }
    prediction = apply_lifetime_policy(raw, bank, policy, maximum_lifetime)
    mae = float(np.mean(np.abs(
        prediction.predicted_rul_raw - prediction.true_rul_cycles
    )))
    policy["selection_validation_mae"] = mae
    return policy, mae


def select_lifetime_policy(raw: pd.DataFrame, bank: ReferenceBank,
                           uniform: dict[str, Any],
                           maximum_lifetime: float) -> dict[str, Any]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for units in source_subset_candidates(bank):
        for gamma in GAMMA_GRID:
            for blend in BLEND_GRID:
                policy, mae = candidate_policy(
                    raw, bank, tuple(units), gamma, blend, maximum_lifetime,
                )
                candidates.append((mae, policy))
    minimum = min(value for value, _ in candidates)
    near = [
        policy for value, policy in candidates
        if value <= minimum + SUBSET_TIE_TOLERANCE
    ]
    selected = sorted(
        near,
        key=lambda policy: (
            -len(policy["source_units"]),
            float(policy["gamma"]),
            -float(policy["blend_weight"]),
            tuple(policy["source_units"]),
        ),
    )[0]
    if (
        float(uniform["selection_validation_mae"])
        - float(selected["selection_validation_mae"])
        < SUBSET_REQUIRED_IMPROVEMENT
    ):
        fallback = dict(uniform)
        fallback["mode"] = "uniform_fallback"
        fallback["fallback_to_uniform"] = True
        fallback["best_adaptive_validation_mae"] = float(
            selected["selection_validation_mae"]
        )
        return fallback
    return selected


def source_subset_candidates(bank: ReferenceBank) -> list[tuple[str, ...]]:
    """Generate exact subsets for small banks and lifetime bands for large ones.

    Exhaustively enumerating all subsets is appropriate for the original five
    NASA cells but intractable for the 26-pack source.  For larger banks we
    retain the full bank, contiguous source-lifetime bands, and evenly spaced
    lifetime-support subsets.  Selection still uses inner validation only.
    """
    count = len(bank.unit_ids)
    if count <= 10:
        return [
            tuple(units)
            for size in range(MINIMUM_SUBSET_SIZE, count + 1)
            for units in itertools.combinations(bank.unit_ids, size)
        ]
    ordered = [
        unit
        for _, unit in sorted(
            zip(bank.eol.astype(float), bank.unit_ids),
            key=lambda pair: (pair[0], pair[1]),
        )
    ]
    candidates: set[tuple[str, ...]] = {tuple(ordered)}
    sizes = sorted({
        MINIMUM_SUBSET_SIZE, 5, 8, 12, 16, count,
    })
    for size in sizes:
        if size > count:
            continue
        for start in range(count - size + 1):
            candidates.add(tuple(ordered[start : start + size]))
        positions = np.round(np.linspace(0, count - 1, size)).astype(int)
        candidates.add(tuple(ordered[index] for index in positions))
    return sorted(candidates, key=lambda units: (len(units), units))


def select_bank_and_policy(
    train: FlatTarget, validation: FlatTarget, bank: ReferenceBank,
    vector_state: Any, maximum_lifetime: float, seed: int,
    config: IntercellConfig, device: torch.device, checkpoint: Path,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    model, epoch, uniform_blend, uniform_mae = train_intercell_select(
        train, validation, bank, bank, vector_state, maximum_lifetime,
        seed, config, device, checkpoint,
    )
    raw = predict_intercell(
        model, validation, bank, vector_state, maximum_lifetime, config, device,
    )
    uniform = uniform_policy(bank, uniform_blend, uniform_mae)
    adaptive = select_lifetime_policy(raw, bank, uniform, maximum_lifetime)
    return epoch, uniform, adaptive


def fit_and_predict_policy(
    train: FlatTarget, evaluation: FlatTarget, bank: ReferenceBank,
    vector_state: Any, maximum_lifetime: float, seed: int, epochs: int,
    policy: dict[str, Any], config: IntercellConfig, device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, ReferenceBank]:
    selected = tuple(map(str, policy["source_units"]))
    selected_bank = subset_bank(bank, selected)
    model = train_intercell_fixed(
        train, selected_bank, vector_state, maximum_lifetime,
        seed, epochs, config, device,
    )
    raw = predict_intercell(
        model, evaluation, selected_bank, vector_state,
        maximum_lifetime, config, device,
    )
    prediction = apply_lifetime_policy(
        raw, selected_bank, policy, maximum_lifetime,
    )
    return model, prediction, selected_bank


def run_fold(
    seed: int, fold: int, source: pd.DataFrame, target: pd.DataFrame,
    source_units: tuple[str, ...],
    target_knees: pd.DataFrame, features: list[str], reference: dict[str, Any],
    fold_map: dict[str, int], selection_seed: int,
    history_config: HistoryConfig, intercell_config: IntercellConfig,
    smoke: bool, output: Path, device: torch.device,
    include_nasa_finetune: bool = False,
    nasa_pretrain_mode: str = "soh",
    adaptive_v2: bool = False,
    minimal_v2: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if minimal_v2 and not (adaptive_v2 and include_nasa_finetune):
        raise ValueError("Minimal V2 requires NASA fine-tuning and adaptive V2")
    test_units = sorted(unit for unit, value in fold_map.items() if value == fold)
    train_units = sorted(set(fold_map) - set(test_units))
    inner_train_units, inner_validation_units = inner_split(
        train_units, target, selection_seed, fold,
        0.20 if smoke else 0.10,
    )
    outer_train, outer_test = subset(target, train_units), subset(target, test_units)
    inner_train = subset(target, inner_train_units)
    inner_validation = subset(target, inner_validation_units)
    config = build_config(reference, 24, smoke)
    maximum_lifetime = maximum_lifetime_without_test(outer_train)
    run_seed = seed + 100 * fold
    if nasa_pretrain_mode not in {"soh", "dynamics"}:
        raise ValueError(f"Unknown NASA pretraining mode: {nasa_pretrain_mode}")
    finetune_method = (
        "nasa_dynamics_finetune_bstats"
        if nasa_pretrain_mode == "dynamics"
        else "nasa_finetune_bstats"
    )
    ssl_state, target_state = target_ssl_state(
        outer_train, features, config, run_seed, device,
    )
    if set(map(str, target_state.fit_units)) != set(train_units):
        raise AssertionError("Target scaler/SSL must use exactly outer-train devices")

    selected_tcn, tcn_epoch, _, _ = train_configuration(
        "target_ssl", inner_train, inner_validation, features, target_state,
        ssl_state, maximum_lifetime, config, run_seed, device,
    )
    for parameter in selected_tcn.parameters():
        parameter.requires_grad = False
    selection_train_bundle = make_bundle(
        inner_train, target_knees, features, target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device, inner_train_units,
    )
    selection_validation_bundle = make_bundle(
        inner_validation, target_knees, features,
        target_state.median, target_state.iqr,
        selected_tcn, config, maximum_lifetime, device,
        inner_validation_units,
    )
    fit_statistics(selection_train_bundle, selection_validation_bundle)
    selection_head, bstats_epoch, base_inner_mae = train_variant(
        "B_stats", selection_train_bundle, selection_validation_bundle,
        maximum_lifetime, run_seed, history_config, device,
        output / f"checkpoint_selection_bstats_seed_{seed}_fold_{fold}.pt",
    )
    nasa_state: dict[str, torch.Tensor] | None = None
    nasa_tcn_epoch: int | None = None
    nasa_bstats_epoch: int | None = None
    nasa_inner_mae: float | None = None
    nasa_intercell_epoch: int | None = None
    nasa_intercell_policy: dict[str, Any] | None = None
    nasa_v2_policy: dict[str, Any] | None = None
    nasa_v2_selection_audit: dict[str, Any] | None = None
    if include_nasa_finetune:
        if nasa_pretrain_mode == "dynamics":
            nasa_state = nasa_dynamics_initial_state(
                source, outer_train, features, ssl_state, config, run_seed,
                device,
                output / f"nasa_dynamics_pretrain_seed_{seed}_fold_{fold}.json",
            )
        else:
            nasa_state = nasa_supervised_initial_state(
                source, outer_train, features, ssl_state, config, run_seed, device,
            )
        nasa_selected_tcn, nasa_tcn_epoch, _, _ = train_configuration(
            "nasa_pretrain_finetune", inner_train, inner_validation, features,
            target_state, nasa_state, maximum_lifetime, config, run_seed, device,
        )
        for parameter in nasa_selected_tcn.parameters():
            parameter.requires_grad = False
        nasa_selection_train_bundle = make_bundle(
            inner_train, target_knees, features, target_state.median,
            target_state.iqr, nasa_selected_tcn, config, maximum_lifetime,
            device, inner_train_units,
        )
        nasa_selection_validation_bundle = make_bundle(
            inner_validation, target_knees, features, target_state.median,
            target_state.iqr, nasa_selected_tcn, config, maximum_lifetime,
            device, inner_validation_units,
        )
        fit_statistics(
            nasa_selection_train_bundle, nasa_selection_validation_bundle,
        )
        nasa_selection_head, nasa_bstats_epoch, nasa_inner_mae = train_variant(
            "B_stats", nasa_selection_train_bundle,
            nasa_selection_validation_bundle, maximum_lifetime, run_seed,
            history_config, device,
            output / f"checkpoint_selection_{finetune_method}_seed_{seed}_fold_{fold}.pt",
        )
        nasa_selection_train_prediction = predict_bundle(
            nasa_selection_head, nasa_selection_train_bundle,
            maximum_lifetime, device, history_config.batch_devices,
        )
        nasa_selection_validation_prediction = predict_bundle(
            nasa_selection_head, nasa_selection_validation_bundle,
            maximum_lifetime, device, history_config.batch_devices,
        )
        nasa_selection_train_flat = flat_target(
            nasa_selection_train_bundle, nasa_selection_train_prediction,
        )
        nasa_selection_validation_flat = flat_target(
            nasa_selection_validation_bundle,
            nasa_selection_validation_prediction,
        )
        nasa_selection_vector_state = robust_vector_fit(
            nasa_selection_train_flat.representation,
            nasa_selection_train_flat.frame.unit_id.astype(str),
            "target_inner_train_after_nasa_pretrain",
        )
        nasa_dynamics_selection_bank = nasa_reference_bank(
            source, features, nasa_selected_tcn, config, source_units,
            intercell_config.progress_knots, device,
        )
        nasa_intercell_epoch, _nasa_intercell_uniform, nasa_intercell_policy = (
            select_bank_and_policy(
                nasa_selection_train_flat, nasa_selection_validation_flat,
                nasa_dynamics_selection_bank, nasa_selection_vector_state,
                maximum_lifetime, run_seed, intercell_config, device,
                output / f"checkpoint_selection_{finetune_method}_intercell_seed_{seed}_fold_{fold}.pt",
            )
        )
    selection_train_prediction = predict_bundle(
        selection_head, selection_train_bundle,
        maximum_lifetime, device, history_config.batch_devices,
    )
    selection_validation_prediction = predict_bundle(
        selection_head, selection_validation_bundle,
        maximum_lifetime, device, history_config.batch_devices,
    )
    selection_train_flat = flat_target(
        selection_train_bundle, selection_train_prediction,
    )
    selection_validation_flat = flat_target(
        selection_validation_bundle, selection_validation_prediction,
    )
    selection_vector_state = robust_vector_fit(
        selection_train_flat.representation,
        selection_train_flat.frame.unit_id.astype(str),
        "target_inner_train",
    )
    nasa_selection_bank = nasa_reference_bank(
        source, features, selected_tcn, config, source_units,
        intercell_config.progress_knots, device,
    )
    target_selection_units = choose_target_references(inner_train, 5)
    target_selection_bank = target_reference_bank(
        selection_train_bundle, selection_train_flat,
        selection_vector_state, target_selection_units,
        intercell_config.progress_knots,
    )
    nasa_epoch, nasa_uniform, nasa_adaptive = select_bank_and_policy(
        selection_train_flat, selection_validation_flat,
        nasa_selection_bank, selection_vector_state, maximum_lifetime,
        run_seed, intercell_config, device,
        output / f"checkpoint_selection_nasa_seed_{seed}_fold_{fold}.pt",
    )
    target_epoch, _target_uniform, target_adaptive = select_bank_and_policy(
        selection_train_flat, selection_validation_flat,
        target_selection_bank, selection_vector_state, maximum_lifetime,
        run_seed, intercell_config, device,
        output / f"checkpoint_selection_target_control_seed_{seed}_fold_{fold}.pt",
    )
    target_v2_policy: dict[str, Any] | None = None
    target_v2_selection_audit: dict[str, Any] | None = None
    if adaptive_v2:
        from bhump_v10_adaptive_gate import AdaptiveV2Config, select_adaptive_v2
        v2_config = AdaptiveV2Config(
            source_epochs=2 if smoke else 20,
            target_epochs=2 if smoke else 15,
            gate_epochs=2 if smoke else 30,
            batch_size=intercell_config.batch_size,
        )
        target_v2_policy, target_v2_selection_audit = select_adaptive_v2(
            selection_train_flat, selection_validation_flat,
            target_selection_bank, target_selection_bank,
            selection_vector_state, maximum_lifetime, run_seed,
            target_epoch, intercell_config, v2_config, device,
        )
        if include_nasa_finetune:
            if nasa_intercell_epoch is None or nasa_intercell_policy is None:
                raise AssertionError("NASA V2 requires selected inter-cell configuration")
            nasa_v2_policy, nasa_v2_selection_audit = select_adaptive_v2(
                nasa_selection_train_flat, nasa_selection_validation_flat,
                nasa_dynamics_selection_bank, nasa_dynamics_selection_bank,
                nasa_selection_vector_state, maximum_lifetime, run_seed,
                nasa_intercell_epoch, intercell_config, v2_config, device,
            )

    refit_tcn = fit_multitask_fixed_epochs(
        ssl_state, outer_train, features, target_state, maximum_lifetime,
        config, run_seed, tcn_epoch, device,
    )
    for parameter in refit_tcn.parameters():
        parameter.requires_grad = False
    train_bundle = make_bundle(
        outer_train, target_knees, features,
        target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device, train_units,
    )
    test_bundle = make_bundle(
        outer_test, target_knees, features,
        target_state.median, target_state.iqr,
        refit_tcn, config, maximum_lifetime, device, test_units,
    )
    stats_median, stats_iqr = fit_statistics(train_bundle, test_bundle)
    refit_head = fit_bstats_fixed_epochs(
        train_bundle, maximum_lifetime, run_seed,
        bstats_epoch, history_config, device,
    )
    assert_causal(refit_head, test_bundle, maximum_lifetime, device)
    train_prediction = predict_bundle(
        refit_head, train_bundle, maximum_lifetime,
        device, history_config.batch_devices,
    )
    test_prediction = predict_bundle(
        refit_head, test_bundle, maximum_lifetime,
        device, history_config.batch_devices,
    )
    train_flat = flat_target(train_bundle, train_prediction)
    test_flat = flat_target(test_bundle, test_prediction)
    vector_state = robust_vector_fit(
        train_flat.representation, train_flat.frame.unit_id.astype(str),
        "target_outer_train",
    )
    nasa_bank = nasa_reference_bank(
        source, features, refit_tcn, config, source_units,
        intercell_config.progress_knots, device,
    )
    target_reference_units = choose_target_references(outer_train, 5)
    target_bank = target_reference_bank(
        train_bundle, train_flat, vector_state, target_reference_units,
        intercell_config.progress_knots,
    )
    target_adaptive_refit = remap_reference_policy(
        target_adaptive, target_selection_bank, target_bank,
    )

    frames: list[pd.DataFrame] = []
    base = test_flat.frame.copy()
    base["predicted_rul_raw"] = base.pop("base_rul")
    base["method"] = "target_ssl_bstats"
    base["blend_weight"] = 0.0
    base["effective_blend_weight"] = 0.0
    base["effective_source_count"] = 0.0
    base["intercell_rul"] = base["predicted_rul_raw"]
    base["reference_prediction_std"] = 0.0
    frames.append(base)
    rows: dict[str, Any] = {}
    if include_nasa_finetune:
        assert nasa_state is not None
        assert nasa_tcn_epoch is not None and nasa_bstats_epoch is not None
        nasa_refit_tcn = fit_multitask_fixed_epochs(
            nasa_state, outer_train, features, target_state, maximum_lifetime,
            config, run_seed, nasa_tcn_epoch, device,
        )
        for parameter in nasa_refit_tcn.parameters():
            parameter.requires_grad = False
        nasa_train_bundle = make_bundle(
            outer_train, target_knees, features, target_state.median,
            target_state.iqr, nasa_refit_tcn, config, maximum_lifetime,
            device, train_units,
        )
        nasa_test_bundle = make_bundle(
            outer_test, target_knees, features, target_state.median,
            target_state.iqr, nasa_refit_tcn, config, maximum_lifetime,
            device, test_units,
        )
        nasa_stats_median, nasa_stats_iqr = fit_statistics(
            nasa_train_bundle, nasa_test_bundle,
        )
        nasa_refit_head = fit_bstats_fixed_epochs(
            nasa_train_bundle, maximum_lifetime, run_seed,
            nasa_bstats_epoch, history_config, device,
        )
        assert_causal(
            nasa_refit_head, nasa_test_bundle, maximum_lifetime, device,
        )
        nasa_train_prediction = predict_bundle(
            nasa_refit_head, nasa_train_bundle, maximum_lifetime,
            device, history_config.batch_devices,
        )
        nasa_prediction = predict_bundle(
            nasa_refit_head, nasa_test_bundle, maximum_lifetime,
            device, history_config.batch_devices,
        )
        if not minimal_v2:
            nasa_prediction["method"] = finetune_method
            nasa_prediction["blend_weight"] = 0.0
            nasa_prediction["effective_blend_weight"] = 0.0
            nasa_prediction["effective_source_count"] = float(source.unit_id.nunique())
            frames.append(nasa_prediction)
            rows[finetune_method] = {
                "selection_tcn_epoch": int(nasa_tcn_epoch),
                "selection_bstats_epoch": int(nasa_bstats_epoch),
                "inner_validation_mae": float(nasa_inner_mae),
                "outer_rul_mae": float(regression_metrics(
                    nasa_prediction, "predicted_rul_raw",
                )["rul_mae"]),
                "source_units": list(source_units),
                "nasa_pretrain_mode": nasa_pretrain_mode,
            }
        if nasa_intercell_epoch is None or nasa_intercell_policy is None:
            raise AssertionError("NASA-pretrained inter-cell policy was not selected")
        nasa_train_flat = flat_target(
            nasa_train_bundle, nasa_train_prediction,
        )
        nasa_test_flat = flat_target(
            nasa_test_bundle, nasa_prediction,
        )
        nasa_vector_state = robust_vector_fit(
            nasa_train_flat.representation,
            nasa_train_flat.frame.unit_id.astype(str),
            "target_outer_train_after_nasa_pretrain",
        )
        nasa_dynamics_bank = nasa_reference_bank(
            source, features, nasa_refit_tcn, config, source_units,
            intercell_config.progress_knots, device,
        )
        if not minimal_v2:
            pretrained_intercell_method = (
                "nasa_dynamics_adaptive"
                if nasa_pretrain_mode == "dynamics"
                else "nasa_soh_pretrain_adaptive"
            )
            pretrained_intercell_model, pretrained_intercell_prediction, selected_pretrained_bank = (
                fit_and_predict_policy(
                    nasa_train_flat, nasa_test_flat, nasa_dynamics_bank,
                    nasa_vector_state, maximum_lifetime, run_seed,
                    nasa_intercell_epoch, nasa_intercell_policy,
                    intercell_config, device,
                )
            )
            pretrained_intercell_prediction["method"] = pretrained_intercell_method
            pretrained_intercell_prediction["blend_weight"] = float(
                nasa_intercell_policy["blend_weight"]
            )
            frames.append(pretrained_intercell_prediction)
            rows[pretrained_intercell_method] = {
                "selection_epoch": int(nasa_intercell_epoch),
                "policy": nasa_intercell_policy,
                "outer_rul_mae": float(regression_metrics(
                    pretrained_intercell_prediction, "predicted_rul_raw",
                )["rul_mae"]),
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in pretrained_intercell_model.parameters()
                ),
                "refit_source_units": selected_pretrained_bank.unit_ids,
                "nasa_pretrain_mode": nasa_pretrain_mode,
            }
            torch.save({
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in pretrained_intercell_model.state_dict().items()
                },
                "method": pretrained_intercell_method,
                "seed": seed, "outer_fold": fold,
                "refit_units": train_units, "outer_test_units": test_units,
                "selected_epoch": nasa_intercell_epoch,
                "selected_policy": nasa_intercell_policy,
                "reference_units": selected_pretrained_bank.unit_ids,
                "nasa_pretrain_mode": nasa_pretrain_mode,
                "sealed_accessed": False,
            }, output / f"checkpoint_refit_{pretrained_intercell_method}_seed_{seed}_fold_{fold}.pt")
        save_base_checkpoint(
            output / f"checkpoint_refit_{finetune_method}_tcn_seed_{seed}_fold_{fold}.pt",
            nasa_refit_tcn, target_state, train_units, features, config,
            seed, fold, nasa_tcn_epoch,
            f"NASA5_{nasa_pretrain_mode}_pretrain_then_target_refit",
        )
        torch.save({
            "model_state": {
                name: value.detach().cpu()
                for name, value in nasa_refit_head.state_dict().items()
            },
            "variant": f"NASA5_{nasa_pretrain_mode}_finetune_B_stats", "seed": seed,
            "outer_fold": fold, "refit_units": train_units,
            "outer_test_units": test_units,
            "selected_epoch": nasa_bstats_epoch,
            "statistics_median": nasa_stats_median,
            "statistics_iqr": nasa_stats_iqr,
            "source_units": list(source_units),
            "nasa_pretrain_mode": nasa_pretrain_mode,
        }, output / f"checkpoint_refit_{finetune_method}_seed_{seed}_fold_{fold}.pt")
        if adaptive_v2:
            from bhump_v10_adaptive_gate import checkpoint_payload, refit_adaptive_v2
            if nasa_v2_policy is None:
                raise AssertionError("NASA V2 policy was not selected")
            nasa_v2_prediction, nasa_v2_dynamic, nasa_v2_gate, nasa_v2_diagnostics = (
                refit_adaptive_v2(
                    nasa_train_flat, nasa_test_flat, nasa_dynamics_bank,
                    nasa_dynamics_bank, nasa_vector_state, maximum_lifetime,
                    run_seed, nasa_intercell_epoch, nasa_v2_policy,
                    intercell_config, v2_config, device,
                )
            )
            nasa_v2_prediction["method"] = "nasa_dynamics_adaptive_v2"
            frames.append(nasa_v2_prediction)
            rows["nasa_dynamics_adaptive_v2"] = {
                "selection_epoch": int(nasa_intercell_epoch),
                "policy": nasa_v2_policy,
                "selection_audit": nasa_v2_selection_audit,
                "outer_rul_mae": float(regression_metrics(
                    nasa_v2_prediction, "predicted_rul_raw",
                )["rul_mae"]),
                **nasa_v2_diagnostics,
            }
            torch.save(
                checkpoint_payload(
                    nasa_v2_dynamic, nasa_v2_gate, nasa_v2_policy,
                    v2_config, train_units, test_units, nasa_dynamics_bank,
                    "nasa_dynamics_adaptive_v2",
                ),
                output / f"checkpoint_refit_nasa_dynamics_adaptive_v2_seed_{seed}_fold_{fold}.pt",
            )
    specifications = () if minimal_v2 else (
        ("nasa_all5_uniform", nasa_bank, nasa_epoch, nasa_uniform),
        ("target_reference_control", target_bank, target_epoch, target_adaptive_refit),
        ("nasa_adaptive", nasa_bank, nasa_epoch, nasa_adaptive),
    )
    parameter_count: int | None = None
    for method, bank, epoch, policy in specifications:
        model, prediction, selected_bank = fit_and_predict_policy(
            train_flat, test_flat, bank, vector_state, maximum_lifetime,
            run_seed, epoch, policy, intercell_config, device,
        )
        current_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count is None:
            parameter_count = current_count
        elif parameter_count != current_count:
            raise AssertionError("Inter-cell comparison parameter counts differ")
        prediction["method"] = method
        prediction["blend_weight"] = float(policy["blend_weight"])
        frames.append(prediction)
        outer_mae = regression_metrics(
            prediction, "predicted_rul_raw",
        )["rul_mae"]
        rows[method] = {
            "selection_epoch": int(epoch),
            "policy": policy,
            "outer_rul_mae": float(outer_mae),
            "parameter_count": current_count,
            "refit_source_units": selected_bank.unit_ids,
        }
        torch.save({
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "method": method, "seed": seed, "outer_fold": fold,
            "refit_units": train_units, "outer_test_units": test_units,
            "selected_epoch": epoch, "selected_policy": policy,
            "reference_units": selected_bank.unit_ids,
            "parameter_count": current_count,
            "sealed_accessed": False,
        }, output / f"checkpoint_refit_{method}_seed_{seed}_fold_{fold}.pt")

    if adaptive_v2:
        from bhump_v10_adaptive_gate import checkpoint_payload, refit_adaptive_v2
        if target_v2_policy is None:
            raise AssertionError("Target dynamics gate policy was not selected")
        target_v2_prediction, target_v2_dynamic, target_v2_gate, target_v2_diagnostics = (
            refit_adaptive_v2(
                train_flat, test_flat, target_bank, target_bank,
                vector_state, maximum_lifetime, run_seed, target_epoch,
                target_v2_policy, intercell_config, v2_config, device,
            )
        )
        target_v2_prediction["method"] = "target_dynamics_gate_control"
        frames.append(target_v2_prediction)
        rows["target_dynamics_gate_control"] = {
            "selection_epoch": int(target_epoch), "policy": target_v2_policy,
            "selection_audit": target_v2_selection_audit,
            "outer_rul_mae": float(regression_metrics(
                target_v2_prediction, "predicted_rul_raw",
            )["rul_mae"]),
            **target_v2_diagnostics,
        }
        torch.save(
            checkpoint_payload(
                target_v2_dynamic, target_v2_gate, target_v2_policy,
                v2_config, train_units, test_units, target_bank,
                "target_dynamics_gate_control",
            ),
            output / f"checkpoint_refit_target_dynamics_gate_control_seed_{seed}_fold_{fold}.pt",
        )

    save_base_checkpoint(
        output / f"checkpoint_refit_targetssl_tcn_seed_{seed}_fold_{fold}.pt",
        refit_tcn, target_state, train_units, features, config,
        seed, fold, tcn_epoch, "fresh_outer_train_fixed_epoch_refit",
    )
    torch.save({
        "model_state": {
            name: value.detach().cpu()
            for name, value in refit_head.state_dict().items()
        },
        "variant": "B_stats", "seed": seed, "outer_fold": fold,
        "refit_units": train_units, "outer_test_units": test_units,
        "selected_epoch": bstats_epoch,
        "statistics_median": stats_median,
        "statistics_iqr": stats_iqr,
    }, output / f"checkpoint_refit_targetssl_bstats_seed_{seed}_fold_{fold}.pt")
    nasa_bank.metadata.to_csv(
        output / f"nasa_reference_nodes_seed_{seed}_fold_{fold}.csv",
        index=False,
    )
    result = {
        "seed": seed, "outer_fold": fold,
        "outer_train_devices": len(train_units),
        "outer_test_devices": len(test_units),
        "inner_train_devices": len(inner_train_units),
        "inner_validation_devices": len(inner_validation_units),
        "tcn_selected_epoch": int(tcn_epoch),
        "bstats_selected_epoch": int(bstats_epoch),
        "base_inner_validation_mae": float(base_inner_mae),
        "nasa_finetune_enabled": bool(include_nasa_finetune),
        "nasa_pretrain_mode": nasa_pretrain_mode if include_nasa_finetune else None,
        "adaptive_v2_enabled": bool(adaptive_v2),
        "minimal_v2_enabled": bool(minimal_v2),
        "methods": rows,
        "target_reference_units": target_reference_units,
        "target_scaler_fit_units": list(map(str, target_state.fit_units)),
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.insert(0, "outer_fold", fold)
    combined.insert(0, "seed", seed)
    return combined, result


def q4_metrics(frame: pd.DataFrame) -> dict[str, float]:
    life = frame.groupby("unit_id").true_eol_cycle.max()
    cutoff = float(life.quantile(0.75))
    group = frame.loc[frame.unit_id.isin(life.loc[life.ge(cutoff)].index)]
    values = regression_metrics(group, "predicted_rul_raw")
    return {
        "q4_rul_mae": float(values["rul_mae"]),
        "q4_rul_bias": float(values["rul_bias"]),
    }


def confirmation_audit(
    predictions: pd.DataFrame, ensemble: pd.DataFrame,
    fold_results: list[dict[str, Any]], fold_map: dict[str, int],
    seeds: tuple[int, ...], expected_samples: int,
    expected_train_devices: int = 256,
    expected_test_devices: int = 64,
) -> dict[str, Any]:
    baseline = "target_ssl_bstats"
    control = "target_reference_control"
    candidate = "nasa_adaptive"
    metrics = metric_rows(ensemble, "ensemble").set_index("method")
    candidate_mae = float(metrics.loc[candidate].rul_mae)
    baseline_mae = float(metrics.loc[baseline].rul_mae)
    control_mae = float(metrics.loc[control].rul_mae)
    gain_baseline = (baseline_mae - candidate_mae) / baseline_mae
    gain_control = (control_mae - candidate_mae) / control_mae
    bootstrap_baseline = paired_device_bootstrap(
        ensemble, candidate, baseline, seed=202608,
    )
    bootstrap_control = paired_device_bootstrap(
        ensemble, candidate, control, seed=202609,
    )
    fold_wins = 0
    for fold in sorted(ensemble.outer_fold.unique()):
        values = metric_rows(
            ensemble.loc[ensemble.outer_fold.eq(fold)], "ensemble",
        ).set_index("method")
        fold_wins += int(
            values.loc[candidate].rul_mae < values.loc[baseline].rul_mae
            and values.loc[candidate].rul_mae < values.loc[control].rul_mae
        )
    by_seed = metric_rows(predictions, "seed").pivot(
        index="seed", columns="method", values="rul_mae",
    )
    seed_wins = int(((
        by_seed[candidate] < by_seed[baseline]
    ) & (
        by_seed[candidate] < by_seed[control]
    )).sum())
    baseline_q4 = q4_metrics(ensemble.loc[ensemble.method.eq(baseline)])
    candidate_q4 = q4_metrics(ensemble.loc[ensemble.method.eq(candidate)])
    q4_bias_gain = (
        abs(baseline_q4["q4_rul_bias"])
        - abs(candidate_q4["q4_rul_bias"])
    ) / max(abs(baseline_q4["q4_rul_bias"]), 1.0e-8)
    nasa = predictions.loc[predictions.method.eq(candidate)]
    median_blend = float(nasa.effective_blend_weight.median())
    median_sources = float(nasa.effective_source_count.median())
    conditions = {
        "gain_vs_target_ssl_at_least_10_percent": gain_baseline >= 0.10,
        "gain_vs_target_control_at_least_2_percent": gain_control >= 0.02,
        "bootstrap_vs_target_ssl_upper_below_zero":
            bootstrap_baseline["mae_difference_p975"] < 0.0,
        "bootstrap_vs_target_control_upper_below_zero":
            bootstrap_control["mae_difference_p975"] < 0.0,
        "at_least_4_of_5_folds_beat_both": fold_wins >= 4,
        "at_least_2_of_3_seeds_beat_both": seed_wins >= 2,
        "q4_bias_improves_at_least_20_percent": q4_bias_gain >= 0.20,
        "worst_device_not_worse_than_10_percent":
            float(metrics.loc[candidate].worst_device_mae)
            <= 1.10 * float(metrics.loc[baseline].worst_device_mae),
        "soh_mae_not_worse_than_1_percent":
            float(metrics.loc[candidate].soh_mae)
            <= 1.01 * float(metrics.loc[baseline].soh_mae),
        "median_nasa_blend_at_least_0p25": median_blend >= 0.25,
        "median_effective_sources_at_least_2": median_sources >= 2.0,
    }
    conditions = {name: bool(value) for name, value in conditions.items()}
    for method in METHODS:
        group = predictions.loc[predictions.method.eq(method)]
        if len(group) != expected_samples * len(seeds):
            raise AssertionError(f"Incomplete OOF coverage for {method}")
        if group.groupby(["seed", "unit_id", "time"]).size().ne(1).any():
            raise AssertionError(f"Duplicate OOF predictions for {method}")
        if not np.isfinite(
            group[["predicted_soh", "predicted_rul_raw"]].to_numpy(float)
        ).all():
            raise AssertionError(f"Non-finite predictions for {method}")
    for row in fold_results:
        if (
            row["outer_train_devices"] != expected_train_devices
            or row["outer_test_devices"] != expected_test_devices
        ):
            raise AssertionError("Formal fold device counts changed")
        test_units = {
            unit for unit, fold in fold_map.items()
            if fold == row["outer_fold"]
        }
        if set(row["target_scaler_fit_units"]) & test_units:
            raise AssertionError("Outer-test device entered target scaler")
    return {
        "oof_passed": bool(all(conditions.values())),
        "candidate_rul_mae": candidate_mae,
        "target_ssl_rul_mae": baseline_mae,
        "target_control_rul_mae": control_mae,
        "relative_gain_vs_target_ssl": float(gain_baseline),
        "relative_gain_vs_target_control": float(gain_control),
        "positive_fold_count_vs_both": fold_wins,
        "positive_seed_count_vs_both": seed_wins,
        "q4_bias_relative_gain": float(q4_bias_gain),
        "median_effective_blend_weight": median_blend,
        "median_effective_source_count": median_sources,
        "conditions": conditions,
        "bootstrap_vs_target_ssl": bootstrap_baseline,
        "bootstrap_vs_target_control": bootstrap_control,
        "passed_structural_audit": True,
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }


def fold_paths(output: Path, seed: int, fold: int) -> tuple[Path, Path]:
    return (
        output / f"fold_predictions_seed_{seed}_fold_{fold}.csv",
        output / f"fold_result_seed_{seed}_fold_{fold}.json",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        getattr(args, "include_nasa_finetune", False)
        and getattr(args, "nasa_pretrain_mode", "soh") == "dynamics"
        and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve()
    ):
        # Never overwrite the registered 4.3656 EFC baseline directory.
        args.output_dir = DEFAULT_DYNAMICS_OUTPUT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    reference = json.loads(
        (args.reference_run / "experiment_manifest.json").read_text(
            encoding="utf-8",
        )
    )
    source, target, features, _report, target_knees = prepare_oof_data(
        args.data_root, args.source_file, args.expected_target_units,
    )
    source_units = tuple(sorted(source.unit_id.astype(str).unique()))
    if len(features) != 16 or any(
        any(token in feature.lower() for token in FORBIDDEN_INPUT_TOKENS)
        for feature in features
    ):
        raise ValueError("Confirmation requires the frozen leak-free 16 features")
    seeds = tuple(int(value) for value in args.formal_seeds.split(","))
    if args.smoke:
        keep = sorted(target.unit_id.astype(str).unique())[:args.smoke_units]
        target = subset(target, keep)
        target_knees = target_knees.loc[
            target_knees.unit_id.astype(str).isin(keep)
        ].copy()
        folds, seeds = 2, (52,)
    else:
        folds = args.outer_folds
        expected_seed_count = 5 if getattr(args, "adaptive_v2", False) else 3
        if folds != 5 or len(seeds) != expected_seed_count:
            raise ValueError(
                f"Formal confirmation requires five folds and exactly {expected_seed_count} seeds"
            )
    fold_map = seeded_stratified_outer_folds(
        target, folds, args.outer_fold_seed,
    )
    history_config = HistoryConfig(
        epochs=3 if args.smoke else args.history_epochs,
        patience=2 if args.smoke else args.patience,
        batch_devices=args.batch_devices,
    )
    intercell_config = IntercellConfig(
        progress_knots=args.progress_knots,
        epochs=3 if args.smoke else args.intercell_epochs,
        patience=2 if args.smoke else args.patience,
        batch_size=args.intercell_batch_size,
    )
    frames: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in range(folds):
            prediction_path, result_path = fold_paths(
                args.output_dir, seed, fold,
            )
            if args.resume and prediction_path.exists() and result_path.exists():
                prediction = pd.read_csv(prediction_path)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                expected_mode = (
                    getattr(args, "nasa_pretrain_mode", "soh")
                    if getattr(args, "include_nasa_finetune", False)
                    else None
                )
                if result.get("nasa_pretrain_mode") != expected_mode:
                    raise RuntimeError(
                        "Resume output was created with a different NASA pretraining mode"
                    )
                if bool(result.get("adaptive_v2_enabled", False)) != bool(
                    getattr(args, "adaptive_v2", False)
                ):
                    raise RuntimeError(
                        "Resume output was created with a different adaptive V2 mode"
                    )
                if bool(result.get("minimal_v2_enabled", False)) != bool(
                    getattr(args, "minimal_v2", False)
                ):
                    raise RuntimeError(
                        "Resume output was created with a different minimal V2 mode"
                    )
            else:
                prediction, result = run_fold(
                    seed, fold, source, target, source_units, target_knees, features,
                    reference, fold_map, args.selection_seed,
                    history_config, intercell_config, args.smoke,
                    args.output_dir, device,
                    include_nasa_finetune=getattr(args, "include_nasa_finetune", False),
                    nasa_pretrain_mode=getattr(args, "nasa_pretrain_mode", "soh"),
                    adaptive_v2=getattr(args, "adaptive_v2", False),
                    minimal_v2=getattr(args, "minimal_v2", False),
                )
                prediction.to_csv(prediction_path, index=False)
                result_path.write_text(
                    json.dumps(result, indent=2), encoding="utf-8",
                )
            frames.append(prediction)
            results.append(result)
            pd.concat(frames, ignore_index=True).to_csv(
                args.output_dir / "oof_predictions_by_seed.csv", index=False,
            )
            print(json.dumps({
                "seed": seed, "outer_fold": fold,
                "methods": result["methods"],
            }), flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    ensemble = ensemble_predictions(predictions)
    metrics = metric_rows(ensemble, "ensemble")
    seed_metrics = metric_rows(predictions, "seed")
    device_metrics = per_device_results(ensemble)
    dynamics_transfer_audit: dict[str, Any] | None = None
    if "nasa_dynamics_adaptive" in set(metrics.method.astype(str)):
        indexed = metrics.set_index("method")
        candidate_mae = float(indexed.loc["nasa_dynamics_adaptive"].rul_mae)
        comparisons: dict[str, Any] = {}
        for index, baseline_method in enumerate((
            "target_ssl_bstats", "target_reference_control", "nasa_adaptive",
        )):
            baseline_mae = float(indexed.loc[baseline_method].rul_mae)
            comparison = {
                "baseline_rul_mae": baseline_mae,
                "candidate_rul_mae": candidate_mae,
                "relative_gain": (baseline_mae - candidate_mae) / baseline_mae,
            }
            if not args.smoke:
                comparison["paired_device_bootstrap"] = paired_device_bootstrap(
                    ensemble, "nasa_dynamics_adaptive", baseline_method,
                    seed=202620 + index,
                )
            comparisons[baseline_method] = comparison
        dynamics_transfer_audit = {
            "method": "nasa_dynamics_adaptive",
            "pretraining_targets": ["current_SOH", "delta_SOH_1", "delta_SOH_4",
                                    "delta_SOH_8", "delta_SOH_16",
                                    "log_local_degradation_rate",
                                    "rate_acceleration"],
            "comparisons": comparisons,
            "target_labels_used_during_nasa_pretraining": False,
            "sealed_accessed": False,
        }
        (args.output_dir / "dynamics_transfer_audit.json").write_text(
            json.dumps(dynamics_transfer_audit, indent=2), encoding="utf-8",
        )
    predictions.to_csv(
        args.output_dir / "oof_predictions_by_seed.csv", index=False,
    )
    ensemble.to_csv(
        args.output_dir / "oof_ensemble_predictions.csv", index=False,
    )
    metrics.to_csv(args.output_dir / "oof_metrics.csv", index=False)
    seed_metrics.to_csv(
        args.output_dir / "oof_metrics_by_seed.csv", index=False,
    )
    device_metrics.to_csv(
        args.output_dir / "oof_results_per_device.csv", index=False,
    )
    pd.DataFrame([{
        "seed": row["seed"], "outer_fold": row["outer_fold"],
        "tcn_selected_epoch": row["tcn_selected_epoch"],
        "bstats_selected_epoch": row["bstats_selected_epoch"],
        **(
            {
                "nasa_v2_policy": json.dumps(
                    row["methods"]["nasa_dynamics_adaptive_v2"]["policy"],
                ),
                "target_v2_policy": json.dumps(
                    row["methods"]["target_dynamics_gate_control"]["policy"],
                ),
            }
            if getattr(args, "minimal_v2", False)
            else {
                "nasa_uniform_policy": json.dumps(
                    row["methods"]["nasa_all5_uniform"]["policy"],
                ),
                "nasa_adaptive_policy": json.dumps(
                    row["methods"]["nasa_adaptive"]["policy"],
                ),
                "target_control_policy": json.dumps(
                    row["methods"]["target_reference_control"]["policy"],
                ),
            }
        ),
    } for row in results]).to_csv(
        args.output_dir / "selected_source_policies.csv", index=False,
    )

    if args.smoke:
        audit = {
            "mode": "smoke", "passed": True,
            "devices": int(ensemble.unit_id.nunique()),
            "validation_accessed": False,
            "sealed_features_accessed": False,
            "sealed_labels_accessed": False,
        }
    elif getattr(args, "minimal_v2", False):
        required_methods = {
            "target_ssl_bstats", "target_dynamics_gate_control",
            "nasa_dynamics_adaptive_v2",
        }
        if set(predictions.method.astype(str).unique()) != required_methods:
            raise AssertionError("Minimal V2 emitted an unexpected method set")
        expected_points = len(target.loc[target.time.ge(7.0)]) * len(seeds)
        for method in required_methods:
            group = predictions.loc[predictions.method.eq(method)]
            if len(group) != expected_points:
                raise AssertionError(f"Incomplete minimal OOF coverage for {method}")
            if group.groupby(["seed", "unit_id", "time"]).size().ne(1).any():
                raise AssertionError(f"Duplicate minimal OOF rows for {method}")
        for row in results:
            if row["outer_train_devices"] != 256 or row["outer_test_devices"] != 64:
                raise AssertionError("Minimal formal fold device counts changed")
            test_units = {
                unit for unit, current_fold in fold_map.items()
                if current_fold == row["outer_fold"]
            }
            if set(row["target_scaler_fit_units"]) & test_units:
                raise AssertionError("Outer-test device entered minimal target scaler")
        audit = {
            "mode": "formal_minimal_v2", "passed_structural_audit": True,
            "methods": sorted(required_methods), "outer_folds": folds,
            "formal_seeds": list(seeds), "outer_train_devices": 256,
            "outer_test_devices": 64, "validation_accessed": False,
            "sealed_features_accessed": False, "sealed_labels_accessed": False,
            "validation_confirmation_permitted": False,
            "stage_a_decision": "evaluate_adaptive_v2_oof_only",
        }
    else:
        audit = confirmation_audit(
            predictions, ensemble, results, fold_map, seeds,
            len(target.loc[target.time.ge(7.0)]),
            expected_train_devices=(folds - 1) * len(fold_map) // folds,
            expected_test_devices=len(fold_map) // folds,
        )
        audit["outer_fold_seed"] = args.outer_fold_seed
        audit["selection_seed"] = args.selection_seed
        audit["formal_seeds"] = list(seeds)
        # The validation confirmation is intentionally locked until OOF passes.
        audit["validation_confirmation_permitted"] = bool(
            audit["oof_passed"] and args.confirm_validation_once
        )
        audit["validation_accessed"] = False
        audit["validation_devices_expected_after_unlock"] = 80
        audit["stage_a_decision"] = (
            "run_frozen_validation_confirmation"
            if audit["validation_confirmation_permitted"]
            else "expand_public_source_domain"
        )
    (args.output_dir / "confirmation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8",
    )
    stage_decision = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "oof_passed": bool(audit.get("oof_passed", False)),
        "validation_confirmation_permitted": bool(
            audit.get("validation_confirmation_permitted", False)
        ),
        "validation_accessed": bool(audit.get("validation_accessed", False)),
        "next_stage": audit.get(
            "stage_a_decision", "smoke_test_only",
        ),
        "reason": (
            "All pre-registered OOF conditions passed."
            if audit.get("oof_passed", False)
            else "At least one pre-registered OOF condition failed; validation remains locked."
        ),
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }
    (args.output_dir / "stage_a_decision.json").write_text(
        json.dumps(stage_decision, indent=2), encoding="utf-8",
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "fresh-fold NASA source screening and lifetime support",
        "mode": "smoke" if args.smoke else "formal",
        "data_version": args.data_version,
        "feature_count": len(features), "features": features,
        "tcn_window": 24, "history_statistics": 38,
        "source_units": list(source_units),
        "source_device_count": len(source_units),
        "source_file": str(args.source_file or "nasa_source_rich.csv"),
        "legacy_method_name_note": (
            "nasa_all5_uniform is retained as a historical output key; "
            "it uses all devices in the configured source file"
        ),
        "minimum_source_subset": MINIMUM_SUBSET_SIZE,
        "gamma_grid": list(GAMMA_GRID),
        "blend_grid": list(BLEND_GRID),
        "outer_fold_seed": args.outer_fold_seed,
        "selection_seed": args.selection_seed,
        "formal_seeds": list(seeds),
        "history_configuration": asdict(history_config),
        "intercell_configuration": asdict(intercell_config),
        "nasa_finetune_in_same_folds": bool(
            getattr(args, "include_nasa_finetune", False)
        ),
        "nasa_pretrain_mode": getattr(args, "nasa_pretrain_mode", "soh"),
        "adaptive_v2_enabled": bool(getattr(args, "adaptive_v2", False)),
        "minimal_v2_enabled": bool(getattr(args, "minimal_v2", False)),
        "dynamics_transfer_audit": dynamics_transfer_audit,
        "audit": audit,
        "sealed_features_accessed": False,
        "sealed_labels_accessed": False,
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--source-file",
        type=Path,
        default=None,
        help=(
            "Exact-RUL NASA source CSV. Relative paths are resolved under "
            "--data-root; omit for nasa_source_rich.csv."
        ),
    )
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--expected-target-units", type=int, default=320)
    parser.add_argument("--data-version", default="Basilisk V1.0 unchanged")
    parser.add_argument("--outer-fold-seed", type=int, default=202608)
    parser.add_argument("--selection-seed", type=int, default=202609)
    parser.add_argument("--formal-seeds", default="52,53,54")
    parser.add_argument("--progress-knots", type=int, default=32)
    parser.add_argument("--history-epochs", type=int, default=45)
    parser.add_argument("--intercell-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-devices", type=int, default=32)
    parser.add_argument("--intercell-batch-size", type=int, default=256)
    parser.add_argument("--include-nasa-finetune", action="store_true")
    parser.add_argument("--adaptive-v2", action="store_true")
    parser.add_argument("--minimal-v2", action="store_true")
    parser.add_argument(
        "--nasa-pretrain-mode", choices=("soh", "dynamics"), default="soh",
        help="Use ordinary source SOH or multi-horizon source dynamics pretraining.",
    )
    parser.add_argument("--confirm-validation-once", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-units", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
