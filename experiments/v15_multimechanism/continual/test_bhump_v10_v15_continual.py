from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_bhump_v10_v15_continual import (
    DEFAULT_NASA_PARENT, DomainSamples, domain_weights, l2sp_loss, load_parent,
    state_sha256, stratified_units,
)
from train_bhump_v10_history_ablation import causal_statistics, local_windows


ROOT = Path(__file__).resolve().parent


class ContinualTransferTests(unittest.TestCase):
    def test_anchor_weight_is_exactly_half(self) -> None:
        weights = domain_weights(("v10", "a", "b", "c"), 0.5)
        self.assertAlmostEqual(weights["v10"], 0.5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["a"], weights["b"])
        dro = domain_weights(("v10", "a", "b"), 0.5, {"a": 3.0, "b": 1.0})
        self.assertAlmostEqual(dro["v10"], 0.5)
        self.assertAlmostEqual(dro["a"], 0.375)
        self.assertAlmostEqual(dro["b"], 0.125)

    def test_unit_balanced_sampler_is_reproducible(self) -> None:
        unit_ids = np.asarray(["u0"] * 20 + ["u1"] * 2, object)
        size = len(unit_ids)
        samples = DomainSamples(
            "x", unit_ids, np.zeros((size, 24, 16), np.float32),
            np.zeros((size, 38), np.float32), np.arange(size, dtype=np.float32),
            np.ones(size, np.float32), np.ones(size, np.float32),
            {"u0": np.arange(20), "u1": np.arange(20, 22)},
        )
        first = samples.sample(16, np.random.default_rng(7), torch.device("cpu"))[2]
        second = samples.sample(16, np.random.default_rng(7), torch.device("cpu"))[2]
        self.assertTrue(torch.equal(first, second))
        # Uniform unit selection means the short device is not starved by its
        # much smaller cycle count.
        self.assertGreater(int((first >= 20).sum()), 2)

    def test_stratified_split_is_reproducible_and_unique(self) -> None:
        rows = []
        for unit in range(24):
            eol = 40 + unit
            for time in range(20):
                rows.append({
                    "unit_id": f"u{unit:02d}", "time": time,
                    "target_soh": 1.0 - 0.2 * time / eol,
                    "true_eol_cycle": eol,
                })
        frame = pd.DataFrame(rows)
        first = stratified_units(frame, 8, 202608)
        second = stratified_units(frame, 8, 202608)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_parent_load_is_bitwise_and_seed_locked(self) -> None:
        path = DEFAULT_NASA_PARENT / "checkpoint_full320_nasa_dynamics_adaptive_seed_52.pt"
        model, payload, _config, _history = load_parent(path, 52, torch.device("cpu"))
        expected = {
            **{f"tcn.{key}": value for key, value in payload["model_state_tcn"].items()},
            **{f"head.{key}": value for key, value in payload["model_state_bstats"].items()},
        }
        self.assertEqual(state_sha256(model.state_dict()), state_sha256(expected))
        with self.assertRaises(AssertionError):
            load_parent(path, 53, torch.device("cpu"))

    def test_l2sp_is_zero_at_parent_and_positive_after_change(self) -> None:
        path = DEFAULT_NASA_PARENT / "checkpoint_full320_nasa_dynamics_adaptive_seed_52.pt"
        model, _payload, _config, _history = load_parent(path, 52, torch.device("cpu"))
        anchor = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.assertAlmostEqual(float(l2sp_loss(model, anchor).detach()), 0.0, places=12)
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        self.assertGreater(float(l2sp_loss(model, anchor).detach()), 0.0)

    def test_microdomain_configs_use_natural_eol_contract(self) -> None:
        target = ROOT.parent / "battery_target_domain"
        for path in sorted(target.glob("simulation_config_v15_micro_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["role_counts"]["train"], 40)
            self.assertEqual(payload["role_counts"]["validation"], 8)
            self.assertEqual(payload["maximum_efc"], 140)
            bins = payload["lifetime_bins_efc"]
            self.assertEqual(min(item["minimum"] for item in bins), 35)
            self.assertEqual(max(item["maximum"] for item in bins), 130)

    def test_future_changes_do_not_change_current_window_or_statistics(self) -> None:
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(50, 16)).astype(np.float32)
        local_soh = np.linspace(1.0, 0.8, 50, dtype=np.float32)
        times = np.arange(50, dtype=float)
        changed_raw, changed_soh = raw.copy(), local_soh.copy()
        changed_raw[31:] = 1.0e4
        changed_soh[31:] = 0.0
        before_windows = local_windows(raw, 24)
        after_windows = local_windows(changed_raw, 24)
        before_stats = causal_statistics(raw, local_soh, times, 130.0)
        after_stats = causal_statistics(changed_raw, changed_soh, times, 130.0)
        self.assertTrue(np.array_equal(before_windows[:31], after_windows[:31]))
        self.assertTrue(np.array_equal(before_stats[:31], after_stats[:31]))


if __name__ == "__main__":
    unittest.main()
