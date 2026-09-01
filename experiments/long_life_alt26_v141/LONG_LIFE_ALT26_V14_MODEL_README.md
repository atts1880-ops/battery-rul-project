# ALT26 → Basilisk V1.4.1 independent long-life model

Status: **UNTRAINED**. This package contains model/training code and frozen
contracts only. It contains no checkpoint and makes no performance claim.

## Domains

- Source: 12 NASA ALT26 batteries with complete EOL, 202.0–540.8 EFC.
- Target: Basilisk V1.4.1, 500 devices, actual EOL 202–538 EFC.
- Model input: frozen 16-dimensional V-I-T curve features only.

## Architecture

- causal TCN, window 24, channels 16/24/24, dilations 1/2/4;
- 38-dimensional causal full-history B_stats;
- 32-knot SOH degradation-progress synchronization;
- inter-cell target/reference/difference projection predicting delta-log(EOL);
- RUL = max(predicted EOL - current EFC, 0).

## Required comparison

1. `target_ssl_bstats`;
2. `target_reference_control`;
3. `alt26_direct` (internal historical key: `nasa_all5_uniform`);
4. `alt26_progress_intercell` (internal key: `nasa_adaptive`).

The ALT26 candidate must beat both target controls under strict device-level
OOF before any validation/sealed claim is made. V1.0's 4.3456 EFC result is not
comparable because the lifetime scale is different.

## Commands

```powershell
python train_bhump_v14_alt26_transfer.py --smoke --device cuda
python train_bhump_v14_alt26_transfer.py --outer-folds 5 --formal-seeds 42,43,44 --device cuda --resume
```
