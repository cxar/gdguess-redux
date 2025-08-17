#!/usr/bin/env python3
import json
import os
import subprocess
import sys

EXEMPT_LABEL = "DOCS-EXEMPT"
EXEMPT_TAG = "[docs-exempt]"


def get_changed_files() -> list[str]:
    base = os.environ.get("GITHUB_BASE_REF", "origin/main")
    head = os.environ.get("GITHUB_SHA")
    if not head:
        # Fallback to compare with HEAD
        cmd = ["git", "diff", "--name-only", f"{base}...HEAD"]
    else:
        cmd = ["git", "diff", "--name-only", f"{base}...{head}"]
    out = subprocess.check_output(cmd, text=True)
    return [p.strip() for p in out.splitlines() if p.strip()]


def is_exempt() -> bool:
    # Allow commit tag exemption in latest commit message
    try:
        msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True)
        if EXEMPT_TAG in msg:
            return True
    except Exception:
        pass
    # Allow PR label via GitHub env (requires workflow to set DOCS_EXEMPT=1)
    return os.environ.get("DOCS_EXEMPT", "0") == "1"


def main() -> int:
    if is_exempt():
        print("Docs-drift: exempted")
        return 0
    changed = get_changed_files()
    changed_src = any(p.startswith("src/") for p in changed)
    changed_docs = any(p.startswith("docs/") for p in changed)
    if changed_src and not changed_docs:
        print("Docs-drift failure: src/ changed but no docs/ changes detected. See docs/DOCS_POLICY.md.")
        return 2
    print("Docs-drift: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
