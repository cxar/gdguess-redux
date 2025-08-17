# AGENTS RULES

- Docs-drift: any PR touching `src/` must update relevant `docs/`. CI enforces via scripts/check_docs_changed.py. Exemptions: label `DOCS-EXEMPT` or commit tag `[docs-exempt]`.
- Reproducibility: set seeds; log artifacts/run_config.yaml and git SHA in training runs.
- No label QA: do not implement NN label checks or down-weighting.
- Tests: generate synthetic fixtures at test-time; do not commit large audio assets.
- Determinism: augmentations and dataset ops must be seedable and deterministic under fixed seeds.
- Tooling: use black/ruff/mypy/pytest; keep stubs compiling; prefer Parquet for tables.
- PR hygiene: link related docs diffs; append to docs/DECISIONS.md and docs/CHANGELOG.md when thresholds/toggles change.
