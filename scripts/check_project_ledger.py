#!/usr/bin/env python3
"""Offline structural guard for the public project ledger."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "project-ledger"
REQUIRED = [
    LEDGER / "README.md",
    LEDGER / "CURRENT_STATUS.md",
    LEDGER / "CHANGELOG.md",
    LEDGER / "ERROR_CASES.md",
    LEDGER / "templates" / "UPDATE_TEMPLATE.md",
]
FORBIDDEN_MARKERS = [
    "sk-",
    "api_key=",
    "authorization: bearer",
    "private-diagnostics.npz",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit(f"project ledger missing required files: {missing}")

    updates = sorted((LEDGER / "updates").glob("????-??-??-*.md"))
    if not updates:
        raise SystemExit("project ledger has no dated update entries")

    changelog = (LEDGER / "CHANGELOG.md").read_text(encoding="utf-8")
    current = (LEDGER / "CURRENT_STATUS.md").read_text(encoding="utf-8")
    errors = (LEDGER / "ERROR_CASES.md").read_text(encoding="utf-8")
    for update in updates:
        relative = update.relative_to(LEDGER).as_posix()
        if relative not in changelog:
            raise SystemExit(f"dated update is not linked from CHANGELOG.md: {relative}")

    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in LEDGER.rglob("*.md")
    )
    for marker in FORBIDDEN_MARKERS:
        if marker in combined:
            raise SystemExit(f"forbidden secret/private marker in project ledger: {marker}")

    if "88.11%" not in current or "93.43%" not in current:
        raise SystemExit("CURRENT_STATUS.md must retain the cross-split warning metrics")
    if "answer: b" not in errors.lower() or "16,000" not in errors:
        raise SystemExit("ERROR_CASES.md is missing the parser or call-count regression")

    print(
        f"project ledger check PASS: {len(REQUIRED)} required files, "
        f"{len(updates)} dated update, changelog links, public-safe markers"
    )


if __name__ == "__main__":
    main()
