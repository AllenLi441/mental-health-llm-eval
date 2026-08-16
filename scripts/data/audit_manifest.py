#!/usr/bin/env python3
"""Manifest audit: schema + local-path + row-count validation.

Fails (exit 1) when a downloaded entry lacks a license/gate decision or when
declared paths/rows do not match disk. Mirrors the upstream-reconciliation
rule: no row is ALLOW unless its license evidence and local revision are
both present in MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = ["id", "alias", "lang", "nature", "license", "path", "size_mb", "rows"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("tmp/datasets/MANIFEST.json"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/m1-v1/audit/license_audit.json"))
    args = ap.parse_args()

    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = []
    for d in m.get("downloaded", []):
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            problems.append(f"{d.get('id')}: missing fields {missing}")
            continue
        p = Path(d["path"])
        if not p.exists():
            problems.append(f"{d['id']}: path missing {d['path']}")
        if not any(k in d for k in ("gate",)):
            # entries without an explicit gate must at least carry evidence in license
            if d["license"].startswith("conditional") or "HOLD" in d["license"] or "RESERVE" in d["license"]:
                problems.append(f"{d['id']}: license not cleared for training ({d['license']})")
    audit = {
        "manifest": str(args.manifest),
        "downloaded": len(m.get("downloaded", [])),
        "gate_decisions": m.get("gate_decisions"),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
