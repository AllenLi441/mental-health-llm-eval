from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .core import (
    build_cpcd_target_messages,
    build_esconv_target_messages,
    load_cpcd_tasks,
    load_esconv_test,
    parse_strategy_response,
    validate_preregistration,
)
from .runner import (
    EndpointConfig,
    JsonlRunStore,
    OpenAICompatibleClient,
    load_cpcd_full_history,
    portable_tree_sha256,
    protocol_sha256,
)


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


def _load_explicit_env(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is not None:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                raise ValueError(f"invalid env line at {path}:{line_number}")
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _arm_spec(preregistration: dict[str, Any], arm_id: str) -> dict[str, Any]:
    for spec in preregistration["target_arms"]:
        if spec["id"] == arm_id:
            return spec
    raise ValueError(f"unknown target arm: {arm_id}")


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_generation_output(
    preregistration: dict[str, Any], arm_id: str, benchmark: str
) -> Path:
    return (
        PACKAGE_ROOT
        / "results"
        / preregistration["protocol_id"]
        / f"{arm_id}.{benchmark}.generations.jsonl"
    )


def _generation_items(
    preregistration: dict[str, Any],
    repository_root: Path,
    benchmark: str,
    family: str | None,
) -> tuple[list[Any], Path | None]:
    source = preregistration["sources"][benchmark]
    checkout = repository_root / source["checkout"]
    if benchmark == "cpcd":
        eval_root = checkout / source["eval_root"]
        tasks = load_cpcd_tasks(eval_root)
        if family:
            tasks = [task for task in tasks if task.family == family]
        return tasks, eval_root / "full_session"
    if family:
        raise ValueError("--family is only valid for CPCD")
    if benchmark == "esconv":
        return load_esconv_test(checkout / source["test_file"]), None
    raise ValueError(f"unsupported generation benchmark: {benchmark}")


def run_generation(
    preregistration: dict[str, Any],
    repository_root: Path,
    arm_id: str,
    benchmark: str,
    family: str | None,
    limit: int | None,
    output: Path | None,
    env_file: Path | None,
    concurrency: int,
) -> dict[str, Any]:
    validate_protocol_assets(preregistration, repository_root)
    protocol_hash = protocol_sha256(preregistration)
    spec = _arm_spec(preregistration, arm_id)
    endpoint = EndpointConfig.from_spec(spec, _load_explicit_env(env_file))
    client = OpenAICompatibleClient(endpoint)
    items, full_session_dir = _generation_items(
        preregistration, repository_root, benchmark, family
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        items = items[:limit]
    output_path = output or _default_generation_output(
        preregistration, arm_id, benchmark
    )
    store = JsonlRunStore(output_path)
    zh_prompt = (repository_root / preregistration["prompts"]["zh"]["path"]).read_text(
        encoding="utf-8"
    )
    en_prompt = (repository_root / preregistration["prompts"]["en"]["path"]).read_text(
        encoding="utf-8"
    )

    pending = []
    for item in items:
        item_id = item.id
        run_id = f"{preregistration['protocol_id']}:{benchmark}:{arm_id}:{item_id}"
        existing = store.index.get(run_id)
        if existing:
            if existing.get("protocol_sha256") != protocol_hash:
                raise ValueError(f"resume protocol hash mismatch for {run_id}")
            continue
        pending.append((run_id, item))

    def generate_one(entry: tuple[str, Any]) -> dict[str, Any]:
        run_id, item = entry
        if benchmark == "cpcd":
            full_history = load_cpcd_full_history(item, full_session_dir)  # type: ignore[arg-type]
            messages = build_cpcd_target_messages(item, full_history, zh_prompt)
            max_tokens = int(preregistration["cpcd"]["target_max_output_tokens"][item.family])
        else:
            full_history = None
            messages = build_esconv_target_messages(item, en_prompt)
            max_tokens = int(preregistration["esconv"]["target_max_output_tokens"])

        api_result = client.complete(messages, max_tokens=max_tokens)
        raw_response = api_result.pop("text")
        record: dict[str, Any] = {
            "run_id": run_id,
            "protocol_id": preregistration["protocol_id"],
            "protocol_sha256": protocol_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": benchmark,
            "arm_id": arm_id,
            "task_id": item.id,
            "messages_sha256": _messages_sha256(messages),
            "input_chars": sum(len(message["content"]) for message in messages),
            **api_result,
        }
        if benchmark == "cpcd":
            record.update({"family": item.family, "response": raw_response})
        else:
            parsed = parse_strategy_response(raw_response)
            attempts = 1
            if parsed.error and preregistration["generation"]["invalid_output_retry_count"]:
                correction = [
                    *messages,
                    {"role": "assistant", "content": raw_response},
                    {
                        "role": "user",
                        "content": (
                            "The previous output did not match the required strict JSON schema. "
                            "Return only the required object with one exact strategy label and a non-empty response."
                        ),
                    },
                ]
                retry_result = client.complete(correction, max_tokens=max_tokens)
                raw_response = retry_result.pop("text")
                parsed = parse_strategy_response(raw_response)
                attempts += 1
                record.update(retry_result)
            record.update(
                {
                    "gold_strategy": item.gold_strategy,
                    "predicted_strategy": parsed.strategy,
                    "response": parsed.response,
                    "format_error": parsed.error,
                    "format_attempts": attempts,
                    "raw_response": raw_response,
                }
            )
        return record

    started = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {executor.submit(generate_one, entry): entry[0] for entry in pending}
        for future in concurrent.futures.as_completed(future_map):
            record = future.result()
            store.append(record)
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"generated {completed}/{len(pending)} -> {output_path}", flush=True)

    return {
        "protocol_id": preregistration["protocol_id"],
        "arm_id": arm_id,
        "benchmark": benchmark,
        "selected": len(items),
        "already_complete": len(items) - len(pending),
        "generated_now": completed,
        "output": str(output_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen CPCD/ESConv comparison for Jingshi DeepSeek and Qwen3.6-27B"
    )
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="verify prompts, commits, hashes, and counts")
    subparsers.add_parser("dry-run", help="show the frozen call and metric plan")
    generate = subparsers.add_parser("generate", help="generate frozen CPCD or ESConv outputs")
    generate.add_argument("--arm", required=True)
    generate.add_argument("--benchmark", required=True, choices=("cpcd", "esconv"))
    generate.add_argument("--family", choices=("srg", "mr", "tcr"))
    generate.add_argument("--limit", type=int)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--env-file", type=Path)
    generate.add_argument("--concurrency", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preregistration = load_preregistration(args.prereg)
    if args.command == "validate":
        result = validate_protocol_assets(preregistration, REPOSITORY_ROOT)
    elif args.command == "dry-run":
        result = build_dry_run_summary(preregistration, REPOSITORY_ROOT)
    elif args.command == "generate":
        result = run_generation(
            preregistration=preregistration,
            repository_root=REPOSITORY_ROOT,
            arm_id=args.arm,
            benchmark=args.benchmark,
            family=args.family,
            limit=args.limit,
            output=args.output,
            env_file=args.env_file,
            concurrency=args.concurrency,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
