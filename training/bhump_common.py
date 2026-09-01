"""Shared, leakage-safe curve features for NASA and Basilisk battery domains."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


VOLTAGE_GRID = np.round(np.arange(4.0, 2.999, -0.05), 2)
LEGACY_FEATURES = (
    "voltage_mean", "voltage_std", "voltage_min", "voltage_max",
    "current_mean", "current_std", "current_abs_mean",
    "temperature_mean", "temperature_std", "temperature_max", "ambient_temperature_c",
)
FORBIDDEN_INPUT_TOKENS = (
    "capacity", "soh", "rul", "eol", "future", "resistance_truth",
    "internal_resistance", "soc_truth", "failure_time",
)


def summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [5, 25, 50, 75, 95])
    nearly_constant = float(np.std(values)) < 1.0e-12
    skew_value = (
        float(skew(values, bias=False))
        if len(values) >= 3 and not nearly_constant
        else 0.0
    )
    kurtosis_value = (
        float(kurtosis(values, fisher=True, bias=False))
        if len(values) >= 4 and not nearly_constant
        else 0.0
    )
    # Constant-current reference tests have mathematically undefined
    # standardized moments.  They carry no shape information, so encode them
    # as zero rather than leaking NaNs into an otherwise valid curve.
    if not math.isfinite(skew_value):
        skew_value = 0.0
    if not math.isfinite(kurtosis_value):
        kurtosis_value = 0.0
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(q[2]),
        f"{prefix}_p05": float(q[0]),
        f"{prefix}_p25": float(q[1]),
        f"{prefix}_p75": float(q[3]),
        f"{prefix}_p95": float(q[4]),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_skew": skew_value,
        f"{prefix}_kurtosis": kurtosis_value,
    }


def linear_slope(time: np.ndarray, signal: np.ndarray) -> float:
    centered = time - time.mean()
    denominator = float(centered @ centered)
    return float(centered @ (signal - signal.mean()) / denominator) if denominator > 1e-12 else 0.0


def first_crossing(time: np.ndarray, voltage: np.ndarray, signal: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(voltage <= level)
    if not len(indices) or indices[0] == 0:
        raise ValueError(f"missing_voltage_crossing_{level:.2f}")
    right = int(indices[0])
    left = right - 1
    delta_voltage = voltage[right] - voltage[left]
    weight = 0.0 if abs(delta_voltage) < 1e-12 else (level - voltage[left]) / delta_voltage
    return float(signal[left] + weight * (signal[right] - signal[left]))


def curve_features(voltage: np.ndarray, current: np.ndarray, temperature: np.ndarray,
                   time: np.ndarray, ambient: float) -> dict[str, float]:
    arrays = [np.asarray(x, dtype=float).reshape(-1) for x in (voltage, current, temperature, time)]
    voltage, current, temperature, time = arrays
    if min(map(len, arrays)) < 20:
        raise ValueError("curve_too_short")
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("curve_length_mismatch")
    numeric = np.column_stack(arrays)
    if not np.isfinite(numeric).all() or not math.isfinite(float(ambient)):
        raise ValueError("non_finite_curve")
    if np.any(np.diff(time) < 0) or time[-1] <= time[0]:
        raise ValueError("invalid_sample_time")
    if voltage[0] < VOLTAGE_GRID[0] or np.min(voltage) > VOLTAGE_GRID[-1]:
        raise ValueError("insufficient_voltage_coverage")

    elapsed = time - time[0]
    duration = float(elapsed[-1])
    normalized_time = elapsed / duration
    result: dict[str, float] = {}
    result.update(summary("voltage", voltage))
    result.update(summary("current", current))
    result.update(summary("temperature", temperature))
    result["ambient_temperature_c"] = float(ambient)
    result["voltage_slope_per_second"] = linear_slope(elapsed, voltage)
    for index, section in enumerate(np.array_split(np.arange(len(voltage)), 3), 1):
        result[f"voltage_slope_segment_{index}"] = linear_slope(elapsed[section], voltage[section])
    result["voltage_area_normalized"] = float(np.trapezoid(voltage, normalized_time))
    result["temperature_area_normalized"] = float(np.trapezoid(temperature, normalized_time))
    result["temperature_rise"] = float(temperature[-1] - temperature[0])
    delta_time = np.diff(elapsed)
    valid_delta = delta_time > 1e-9
    temperature_rate = np.diff(temperature)[valid_delta] / delta_time[valid_delta]
    result["temperature_max_rise_rate"] = float(np.max(temperature_rate)) if len(temperature_rate) else 0.0
    result["voltage_current_correlation"] = (
        float(np.corrcoef(voltage, current)[0, 1]) if np.std(current) > 1e-8 else 0.0
    )
    result["voltage_temperature_correlation"] = (
        float(np.corrcoef(voltage, temperature)[0, 1]) if np.std(temperature) > 1e-8 else 0.0
    )
    result["current_excitation_std"] = float(np.std(current))
    result["resistance_proxy_dv_di"] = abs(linear_slope(current, voltage)) if np.std(current) > 1e-8 else 0.0

    crossing_time: dict[float, float] = {}
    for level in VOLTAGE_GRID:
        suffix = f"{level:.2f}".replace(".", "p")
        crossing_time[level] = first_crossing(elapsed, voltage, elapsed, level)
        result[f"cross_time_v{suffix}"] = crossing_time[level]
        result[f"cross_current_v{suffix}"] = first_crossing(elapsed, voltage, current, level)
        result[f"cross_temperature_v{suffix}"] = first_crossing(elapsed, voltage, temperature, level)
    for high, low in zip(VOLTAGE_GRID[:-1], VOLTAGE_GRID[1:]):
        high_suffix = f"{high:.2f}".replace(".", "p")
        low_suffix = f"{low:.2f}".replace(".", "p")
        result[f"passage_time_v{high_suffix}_to_v{low_suffix}"] = crossing_time[low] - crossing_time[high]
    return result


def add_causal_baseline_deltas(frame: pd.DataFrame, metadata: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    metadata_set = set(metadata)
    base_features = [column for column in frame.columns if column not in metadata_set]
    ordered = frame.sort_values(["unit_id", "time"]).copy()
    first = ordered.groupby("unit_id", sort=False)[base_features].transform("first")
    delta = ordered[base_features] - first
    delta.columns = [f"delta_{column}" for column in base_features]
    ordered = pd.concat([ordered, delta], axis=1)
    if not np.isfinite(ordered[base_features + list(delta.columns)].to_numpy(float)).all():
        raise ValueError("non_finite_engineered_feature")
    return ordered, base_features + list(delta.columns)


def assert_feature_contract(features: Iterable[str]) -> None:
    failures = {
        feature: token
        for feature in features
        for token in FORBIDDEN_INPUT_TOKENS
        if token in feature.lower()
    }
    if failures:
        raise ValueError(f"Leakage-prone features: {failures}")
