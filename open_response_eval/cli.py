from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .core import load_cpcd_tasks, load_esconv_test, validate_preregistration
from .runner import portable_tree_sha256, protocol_sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PREREGISTRATION = PACKAGE_ROOT / "preregistration.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_preregistration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preregistration must be a JSON object")
    validate_preregistration(value)
    return value


def _cpcd_hash_files(eval_root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in ("srg", "memory_recall", "TCR", "full_session"):
        files.extend((eval_root / directory).glob("*.json"))
    for directory in ("srg", "memory_recall", "TCR"):
        files.append(eval_root / directory / "rubric.md")
    if not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise FileNotFoundError(f"missing CPCD evaluation assets: {missing}")
    return files


def validate_protocol_assets(
    preregistration: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    sources = preregistration["sources"]
    prompt_status = {}
    for language, spec in preregistration["prompts"].items():
        path = repository_root / spec["path"]
        actual = _sha256(path)
        prompt_status[language] = {
            "path": str(path),
            "sha256": actual,
            "sha256_ok": actual == spec["sha256"],
        }
        if actual != spec["sha256"]:
            raise ValueError(f"frozen {language} prompt hash mismatch")

    cpcd_spec = sources["cpcd"]
    cpcd_checkout = repository_root / cpcd_spec["checkout"]
    cpcd_head = _git_head(cpcd_checkout)
    if cpcd_head != cpcd_spec["commit"]:
        raise ValueError(f"CPCD checkout is {cpcd_head}, expected {cpcd_spec['commit']}")
    eval_root = cpcd_checkout / cpcd_spec["eval_root"]
    cpcd_hash = portable_tree_sha256(eval_root, _cpcd_hash_files(eval_root))
    if cpcd_hash != cpcd_spec["portable_eval_tree_sha256"]:
        raise ValueError("CPCD evaluation tree hash mismatch")
    cpcd_tasks = load_cpcd_tasks(eval_root)
    cpcd_counts = {
        family: sum(task.family == family for task in cpcd_tasks)
        for family in ("srg", "mr", "tcr")
    }
    if cpcd_counts != preregistration["cpcd"]["expected_task_counts"]:
        raise ValueError(f"CPCD task count mismatch: {cpcd_counts}")

    esconv_spec = sources["esconv"]
    esconv_checkout = repository_root / esconv_spec["checkout"]
    esconv_head = _git_head(esconv_checkout)
    if esconv_head != esconv_spec["commit"]:
        raise ValueError(
            f"ESConv checkout is {esconv_head}, expected {esconv_spec['commit']}"
        )
    esconv_test = esconv_checkout / esconv_spec["test_file"]
    esconv_hash = _sha256(esconv_test)
    if esconv_hash != esconv_spec["test_sha256"]:
        raise ValueError("ESConv official test hash mismatch")
    esconv_rows = load_esconv_test(esconv_test)
    if len(esconv_rows) != preregistration["esconv"]["expected_rows"]:
        raise ValueError(f"ESConv row count mismatch: {len(esconv_rows)}")

    augesc_spec = sources["augesc"]
    augesc_checkout = repository_root / augesc_spec["checkout"]
    augesc_head = _git_head(augesc_checkout)
    if augesc_head != augesc_spec["commit"]:
        raise ValueError(
            f"AugESC checkout is {augesc_head}, expected {augesc_spec['commit']}"
        )

    return {
        "protocol_id": preregistration["protocol_id"],
        "protocol_sha256": protocol_sha256(preregistration),
        "prompts": prompt_status,
        "cpcd": {
            "commit": cpcd_head,
            "portable_eval_tree_sha256": cpcd_hash,
            "task_counts": cpcd_counts,
        },
        "esconv": {
            "commit": esconv_head,
            "test_sha256": esconv_hash,
            "rows": len(esconv_rows),
        },
        "augesc": {
            "commit": augesc_head,
            "independent_test_accuracy": False,
        },
    }


def build_dry_run_summary(
    preregistration: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    cpcd_calls = sum(preregistration["cpcd"]["expected_task_counts"].values())
    esconv_calls = int(preregistration["esconv"]["expected_rows"])
    arm_count = len(preregistration["target_arms"])
    return {
        "protocol_id": preregistration["protocol_id"],
        "protocol_sha256": protocol_sha256(preregistration),
        "target_calls_per_arm": {"cpcd": cpcd_calls, "esconv": esconv_calls},
        "target_calls_total": arm_count * (cpcd_calls + esconv_calls),
        "judge_calls": {"cpcd": arm_count * cpcd_calls},
        "reports_single_accuracy": False,
        "reporting": {
            "cpcd": "0-5 means plus normalized quality percent",
            "esconv_strategy": "Accuracy, Macro-F1, Weighted-F1, invalid rate",
            "esconv_response": "BLEU-2/4, ROUGE-L, Distinct-1/2/3",
            "augesc": "No separate accuracy; downstream utility uses ESConv test",
        },
        "repository_root": str(repository_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen CPCD/ESConv comparison for Jingshi DeepSeek and Qwen3.6-27B"
    )
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="verify prompts, commits, hashes, and counts")
    subparsers.add_parser("dry-run", help="show the frozen call and metric plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preregistration = load_preregistration(args.prereg)
    if args.command == "validate":
        result = validate_protocol_assets(preregistration, REPOSITORY_ROOT)
    elif args.command == "dry-run":
        result = build_dry_run_summary(preregistration, REPOSITORY_ROOT)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
