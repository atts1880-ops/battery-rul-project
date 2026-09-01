# ALT26 / Basilisk V1.4.1 long-life training dataset

This package is paired with the untrained long-life model package. Extract both
archives into the same parent directory so they merge into
`battery_longlife_alt26_v14_workspace/`.

Contents:

- `target_v14_public/`: raw public V-I-T curves, train/validation labels,
  sealed 30-EFC prefixes, legacy11 tables and all public audits;
- `bhump_transfer_v14_alt26_diverse_data/`: compact 16-feature target tables,
  ALT26-only source table, frozen feature contract, lifetime assignments and
  domain-shift audit;
- `generation/`: V1.4.1 generator, inherited V1.3 physics generator, configs,
  tests and data-generation documentation.

The package deliberately excludes custodian-private oracle fields, complete
sealed labels, internal truth caches and model checkpoints. Public validation
labels are included for frozen independent evaluation. The source table contains
only the 12 NASA ALT26 batteries with complete EOL (202.0–540.8 EFC).

Target formal result: 500 devices, actual EOL 202–538 EFC; 320 train, 80
validation, 50 sealed ID, 25 temperature OOD and 25 load OOD. Every lifetime
stratum contains all five degradation mechanisms, three knee modes and four
stress profiles.
