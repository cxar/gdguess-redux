# Docs Policy

Rule: Any PR that changes files under `src/` must also change at least one relevant doc under `docs/`. CI will fail otherwise.

Exemptions:
- Pure comment/typo changes marked with PR label `DOCS-EXEMPT` or commit tag `[docs-exempt]`.

Use `scripts/check_docs_changed.py` in CI to enforce this policy.
