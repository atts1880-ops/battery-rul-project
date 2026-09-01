from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from train_bhump_v10_v15_generalization import (
    GeneralizationHead, HeadConfig, balanced_weights, stratified_folds,
)


class GeneralizationExperimentTests(unittest.TestCase):
    def test_multi_expert_and_wide_are_parameter_matched(self) -> None:
        multi = GeneralizationHead("multi_expert", HeadConfig())
        wide = GeneralizationHead("wide", HeadConfig())
        a = sum(parameter.numel() for parameter in multi.parameters())
        b = sum(parameter.numel() for parameter in wide.parameters())
        self.assertLessEqual(abs(a - b) / a, 0.05)

    def test_gate_is_finite_soft_and_bounded(self) -> None:
        model = GeneralizationHead("multi_expert", HeadConfig()).eval()
        local = torch.randn(3, 12, 16)
        soh = torch.sigmoid(torch.randn(3, 12))
        stats = torch.randn(3, 12, 38)
        times = torch.arange(12).repeat(3, 1).float()
        with torch.no_grad():
            output = model(local, soh, stats, times, 100.0)
        self.assertTrue(torch.isfinite(output["rul"]).all())
        self.assertTrue(torch.allclose(output["gate_probability"].sum(-1), torch.ones(3, 12)))
        self.assertTrue(bool((output["rul"] >= 0).all() and (output["rul"] <= 100).all()))

    def test_future_representation_cannot_change_current_prediction(self) -> None:
        model = GeneralizationHead("multi_expert", HeadConfig()).eval()
        local = torch.randn(1, 20, 16)
        soh = torch.sigmoid(torch.randn(1, 20))
        stats = torch.randn(1, 20, 38)
        times = torch.arange(20).repeat(1, 1).float()
        changed_local, changed_soh, changed_stats = local.clone(), soh.clone(), stats.clone()
        changed_local[:, 11:] = 1.0e4
        changed_soh[:, 11:] = 0.0
        changed_stats[:, 11:] = -1.0e4
        with torch.no_grad():
            before = model(local, soh, stats, times, 100.0)
            after = model(changed_local, changed_soh, changed_stats, times, 100.0)
        self.assertTrue(torch.equal(before["rul"][:, :11], after["rul"][:, :11]))
        self.assertTrue(torch.equal(
            before["gate_probability"][:, :11], after["gate_probability"][:, :11],
        ))

    def test_balanced_weights_equalize_domains_and_units(self) -> None:
        rows = []
        for domain, counts in (("v10", (3, 5)), ("v15", (2, 6))):
            for index, count in enumerate(counts):
                for time in range(count):
                    rows.append({"evaluation_domain": domain, "unit_id": f"{domain}::{index}", "time": time})
        frame = balanced_weights(pd.DataFrame(rows))
        domain = frame.groupby("evaluation_domain").sample_weight.sum()
        unit = frame.groupby(["evaluation_domain", "unit_id"]).sample_weight.sum()
        self.assertAlmostEqual(float(domain.iloc[0]), float(domain.iloc[1]), places=6)
        for _, group in unit.groupby(level=0):
            self.assertAlmostEqual(float(group.max()), float(group.min()), places=6)

    def test_fold_assignment_is_complete_and_reproducible(self) -> None:
        frame = pd.DataFrame({
            "unit_id": [f"v10::{i:03d}" for i in range(40)],
            "true_eol_cycle": np.repeat(np.arange(40) + 50, 1),
        })
        report = pd.DataFrame({
            "unit_id": frame.unit_id, "has_knee": [i % 3 == 0 for i in range(40)],
            "knee_progress": [0.3 if i % 2 else 0.7 for i in range(40)],
        })
        first = stratified_folds(frame, report, 5, 202608)
        second = stratified_folds(frame, report, 5, 202608)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(frame.unit_id))
        self.assertEqual(set(first.values()), set(range(5)))


if __name__ == "__main__":
    unittest.main()
