from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .core import (
    ESConvExample,
    STRATEGIES,
    build_cpcd_judge_messages,
    build_cpcd_target_messages,
    build_esconv_target_messages,
    load_cpcd_tasks,
    load_esconv_test,
    parse_cpcd_judgement,
    parse_strategy_response,
    read_jsonl_index,
    response_overlap_metrics_tokenized,
    score_strategy_predictions,
    stable_blind_order,
    summarize_cpcd_judgements,
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
from .provenance import (
    PROVIDER_LEVEL,
    SELF_HOSTED_LEVEL,
    load_pinned_deployment_manifest,
    provider_alias_provenance,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_PREREGISTRATION = PACKAGE_ROOT / "preregistration.json"
DEFAULT_ESCONV_METHOD = PACKAGE_ROOT / "esconv_method_v2.json"


_PROVENANCE_FIELDS = (
    "deployment_manifest_sha256",
    "deployment_id",
    "deployment_verification_level",
    "resolved_revision",
    "artifact_set_sha256",
    "runtime_engine",
    "runtime_engine_version",
    "runtime_image_digest",
    "quantization_mode",
    "weight_dtype",
    "served_model_alias",
    "chat_template_sha256",
    "launch_args_sha256",
)


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


def load_esconv_method_contract(
    path: Path = DEFAULT_ESCONV_METHOD,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("method_id"):
        raise ValueError("ESConv method contract must be a named JSON object")
    source_symbols = {
        "ESConvExample.from_tsv_line": ESConvExample.from_tsv_line,
        "build_esconv_target_messages": build_esconv_target_messages,
        "parse_strategy_response": parse_strategy_response,
        "score_strategy_predictions": score_strategy_predictions,
        "response_overlap_metrics_tokenized": response_overlap_metrics_tokenized,
        "_esconv_correction_messages": _esconv_correction_messages,
        "_validate_esconv_generation_content": _validate_esconv_generation_content,
        "_official_esconv_tokens": _official_esconv_tokens,
    }
    expected_hashes = value.get("implementation_source_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        source_symbols
    ):
        raise ValueError("ESConv method contract has incomplete implementation hashes")
    if tuple(value.get("strategy_labels") or ()) != STRATEGIES:
        raise ValueError("ESConv method contract has the wrong ordered strategy labels")
    for name, symbol in source_symbols.items():
        actual = hashlib.sha256(inspect.getsource(symbol).encode("utf-8")).hexdigest()
        if actual != expected_hashes[name]:
            raise ValueError(f"ESConv frozen implementation hash mismatch for {name}")
    return value


def esconv_method_identity() -> dict[str, str]:
    contract = load_esconv_method_contract()
    return {
        "method_id": str(contract["method_id"]),
        "method_sha256": protocol_sha256(contract),
    }


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
    esconv_method = esconv_method_identity()

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
            **esconv_method,
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
        "esconv_method": esconv_method_identity(),
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


def _target_provenance(
    spec: dict[str, Any], environment: dict[str, str]
) -> dict[str, Any]:
    if spec.get("revision"):
        path_env = str(spec.get("deployment_manifest_path_env") or "").strip()
        raw_path = str(environment.get(path_env, "") if path_env else "").strip()
        if not raw_path:
            label = path_env or "deployment manifest path"
            raise ValueError(f"revision-pinned target requires {label}")
        return load_pinned_deployment_manifest(Path(raw_path), spec)
    return provider_alias_provenance(spec)


def _require_complete_api_result(
    api_result: dict[str, Any], preregistration: dict[str, Any], label: str
) -> None:
    generation = preregistration.get("generation") or {}
    if not bool(generation.get("require_complete_termination")):
        return
    wire_api = api_result.get("wire_api")
    if wire_api == "chat":
        accepted = set(generation.get("accepted_chat_finish_reasons") or ("stop",))
        observed = api_result.get("finish_reason")
        if observed not in accepted:
            raise ValueError(
                f"{label} did not terminate completely: finish_reason={observed!r}"
            )
        return
    if wire_api == "responses":
        accepted = set(generation.get("accepted_responses_statuses") or ("completed",))
        observed = api_result.get("response_status")
        if observed not in accepted:
            raise ValueError(
                f"{label} did not terminate completely: response_status={observed!r}"
            )
        return
    raise ValueError(f"{label} has unsupported wire_api={wire_api!r}")


def _validate_record_completion(
    record: dict[str, Any], preregistration: dict[str, Any], label: str
) -> None:
    _require_complete_api_result(record, preregistration, label)


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _esconv_correction_messages(
    messages: list[dict[str, str]], raw_response: str
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": raw_response},
        {
            "role": "user",
            "content": (
                "The previous output did not match the required strict JSON schema. "
                "Return only the required object with one exact strategy label and a "
                "non-empty response."
            ),
        },
    ]


def _default_generation_output(
    preregistration: dict[str, Any], arm_id: str, benchmark: str
) -> Path:
    output_benchmark = (
        "esconv_strategy_conditioned" if benchmark == "esconv" else benchmark
    )
    return (
        PACKAGE_ROOT
        / "results"
        / preregistration["protocol_id"]
        / f"{arm_id}.{output_benchmark}.generations.jsonl"
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
    environment = _load_explicit_env(env_file)
    provenance = _target_provenance(spec, environment)
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
    esconv_method = esconv_method_identity() if benchmark == "esconv" else {}

    def target_request(item: Any) -> tuple[list[dict[str, str]], int]:
        if benchmark == "cpcd":
            full_history = load_cpcd_full_history(
                item, full_session_dir  # type: ignore[arg-type]
            )
            messages = build_cpcd_target_messages(item, full_history, zh_prompt)
            max_tokens = int(
                preregistration["cpcd"]["target_max_output_tokens"][item.family]
            )
        else:
            messages = build_esconv_target_messages(item, en_prompt)
            max_tokens = int(preregistration["esconv"]["target_max_output_tokens"])
        return messages, max_tokens

    pending: list[tuple[str, Any, list[dict[str, str]], int]] = []
    for item in items:
        item_id = item.id
        run_id = f"{preregistration['protocol_id']}:{benchmark}:{arm_id}:{item_id}"
        messages, max_tokens = target_request(item)
        existing = store.index.get(run_id)
        if existing:
            _validate_target_generation_record(existing, preregistration, benchmark)
            _validate_record_completion(existing, preregistration, run_id)
            if existing.get("max_output_tokens_requested") != max_tokens:
                raise ValueError(f"resume output budget mismatch for {run_id}")
            for field, expected in provenance.items():
                if existing.get(field) != expected:
                    raise ValueError(f"resume provenance mismatch for {run_id}: {field}")
            if existing.get("messages_sha256") != _messages_sha256(messages):
                raise ValueError(f"resume target prompt mismatch for {run_id}")
            if benchmark == "cpcd":
                if existing.get("family") != item.family:
                    raise ValueError(f"resume CPCD family mismatch for {run_id}")
                if not isinstance(existing.get("response"), str):
                    raise ValueError(f"resume CPCD response missing for {run_id}")
            else:
                _validate_esconv_generation_content(
                    existing, item, messages, preregistration
                )
            continue
        pending.append((run_id, item, messages, max_tokens))

    def generate_one(
        entry: tuple[str, Any, list[dict[str, str]], int]
    ) -> dict[str, Any]:
        run_id, item, messages, max_tokens = entry
        api_result = client.complete(messages, max_tokens=max_tokens)
        _require_complete_api_result(api_result, preregistration, run_id)
        raw_response = api_result.pop("text")
        format_attempt_trace = [
            {
                "messages_sha256": _messages_sha256(messages),
                "raw_response": raw_response,
                **api_result,
            }
        ]
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
            "max_output_tokens_requested": max_tokens,
            **esconv_method,
            **provenance,
            **api_result,
        }
        if benchmark == "cpcd":
            record.update({"family": item.family, "response": raw_response})
        else:
            parsed = parse_strategy_response(raw_response)
            attempts = 1
            if parsed.error and preregistration["generation"]["invalid_output_retry_count"]:
                correction = _esconv_correction_messages(messages, raw_response)
                retry_result = client.complete(correction, max_tokens=max_tokens)
                _require_complete_api_result(
                    retry_result, preregistration, f"{run_id}:format-retry"
                )
                raw_response = retry_result.pop("text")
                format_attempt_trace.append(
                    {
                        "messages_sha256": _messages_sha256(correction),
                        "raw_response": raw_response,
                        **retry_result,
                    }
                )
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
                    "format_attempt_trace": format_attempt_trace,
                    "scored_messages_sha256": format_attempt_trace[-1][
                        "messages_sha256"
                    ],
                    "raw_response": raw_response,
                }
            )
        return record

    started = time.monotonic()
    completed = 0
    if pending:
        endpoint = EndpointConfig.from_spec(spec, environment)
        client = OpenAICompatibleClient(endpoint)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, concurrency)
        ) as executor:
            future_map = {
                executor.submit(generate_one, entry): entry[0] for entry in pending
            }
            first_error: Exception | None = None
            for future in concurrent.futures.as_completed(future_map):
                try:
                    record = future.result()
                except Exception as error:
                    if first_error is None:
                        first_error = error
                    continue
                store.append(record)
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(pending):
                    print(
                        f"generated {completed}/{len(pending)} -> {output_path}",
                        flush=True,
                    )
            if first_error is not None:
                raise first_error

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


_CPCD_RUBRIC_DIRECTORIES = {
    "srg": "srg",
    "mr": "memory_recall",
    "tcr": "TCR",
}


def _default_cpcd_judgement_output(preregistration: dict[str, Any]) -> Path:
    return (
        PACKAGE_ROOT
        / "results"
        / preregistration["protocol_id"]
        / "cpcd.judgements.jsonl"
    )


def _load_run_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluation input does not exist: {path}")
        records.extend(read_jsonl_index(path).values())
    return records


def _require_frozen_record(
    record: dict[str, Any],
    preregistration: dict[str, Any],
    benchmark: str,
) -> None:
    if record.get("protocol_id") != preregistration["protocol_id"]:
        raise ValueError(f"record {record.get('run_id')!r} has the wrong protocol id")
    expected_hash = protocol_sha256(preregistration)
    if record.get("protocol_sha256") != expected_hash:
        raise ValueError(f"record {record.get('run_id')!r} has the wrong protocol hash")
    if record.get("benchmark") != benchmark:
        raise ValueError(f"record {record.get('run_id')!r} is not a {benchmark} record")


def _looks_like_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _validate_target_generation_record(
    record: dict[str, Any],
    preregistration: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    _require_frozen_record(record, preregistration, benchmark)
    arm_id = str(record.get("arm_id") or "")
    try:
        spec = next(
            value
            for value in preregistration["target_arms"]
            if value["id"] == arm_id
        )
    except StopIteration as error:
        raise ValueError(f"unknown target arm in {benchmark} generation: {arm_id!r}") from error
    task_id = str(record.get("task_id") or "")
    expected_run_id = (
        f"{preregistration['protocol_id']}:{benchmark}:{arm_id}:{task_id}"
    )
    if record.get("run_id") != expected_run_id:
        raise ValueError(f"broken generation run id for {arm_id}/{task_id}")
    if record.get("endpoint_id") != spec["id"]:
        raise ValueError(f"wrong generation endpoint for {arm_id}/{task_id}")
    if record.get("requested_model") != spec["model"]:
        raise ValueError(f"wrong requested model for {arm_id}/{task_id}")
    if record.get("response_model") != spec["model"]:
        raise ValueError(f"wrong served model for {arm_id}/{task_id}")
    if not _looks_like_sha256(record.get("messages_sha256")):
        raise ValueError(f"generation {arm_id}/{task_id} lacks a prompt hash")
    _validate_record_completion(
        record, preregistration, f"generation {arm_id}/{task_id}"
    )
    required_level = (
        SELF_HOSTED_LEVEL if spec.get("revision") else PROVIDER_LEVEL
    )
    if record.get("deployment_verification_level") != required_level:
        raise ValueError(f"wrong deployment verification level for {arm_id}/{task_id}")
    if spec.get("revision"):
        if not spec.get("deployment_manifest_sha256"):
            raise ValueError(f"target {arm_id} has no frozen deployment manifest")
        if record.get("deployment_manifest_sha256") != spec.get(
            "deployment_manifest_sha256"
        ):
            raise ValueError(f"wrong deployment manifest for {arm_id}/{task_id}")
        if record.get("resolved_revision") != spec.get("revision"):
            raise ValueError(f"wrong resolved revision for {arm_id}/{task_id}")
    elif record.get("resolved_revision") is not None:
        raise ValueError(f"provider-managed target {arm_id} cannot claim a revision")
    if benchmark == "esconv":
        expected_method = esconv_method_identity()
        if any(record.get(key) != expected_method[key] for key in expected_method):
            raise ValueError(f"wrong ESConv method contract for {arm_id}/{task_id}")
    return spec


def _validate_esconv_generation_content(
    record: dict[str, Any],
    example: Any,
    target_messages: list[dict[str, str]],
    preregistration: dict[str, Any],
) -> None:
    arm_id = str(record["arm_id"])
    task_id = str(record["task_id"])
    spec = _arm_spec(preregistration, arm_id)
    target_hash = _messages_sha256(target_messages)
    raw_response = record.get("raw_response")
    if not isinstance(raw_response, str):
        raise ValueError(f"ESConv generation {arm_id}/{task_id} lacks raw output")
    format_attempts = record.get("format_attempts")
    max_retries = int(
        (preregistration.get("generation") or {}).get(
            "invalid_output_retry_count", 1
        )
    )
    if (
        isinstance(format_attempts, bool)
        or not isinstance(format_attempts, int)
        or not 1 <= format_attempts <= 1 + max_retries
    ):
        raise ValueError(f"ESConv generation {arm_id}/{task_id} has invalid attempt count")

    trace = record.get("format_attempt_trace")
    if trace is None:
        if format_attempts != 1:
            raise ValueError(f"ESConv retry trace missing for {arm_id}/{task_id}")
        if record.get("scored_messages_sha256") not in (None, target_hash):
            raise ValueError(f"ESConv scored prompt mismatch for {arm_id}/{task_id}")
    else:
        if not isinstance(trace, list) or len(trace) != format_attempts:
            raise ValueError(f"ESConv attempt trace length mismatch for {arm_id}/{task_id}")
        expected_messages = target_messages
        for index, attempt in enumerate(trace):
            if not isinstance(attempt, dict):
                raise ValueError(f"ESConv attempt trace is invalid for {arm_id}/{task_id}")
            attempt_raw = attempt.get("raw_response")
            if not isinstance(attempt_raw, str):
                raise ValueError(f"ESConv attempt raw output missing for {arm_id}/{task_id}")
            if attempt.get("messages_sha256") != _messages_sha256(expected_messages):
                raise ValueError(f"ESConv attempt prompt mismatch for {arm_id}/{task_id}")
            if attempt.get("endpoint_id") != spec["id"]:
                raise ValueError(f"ESConv attempt endpoint mismatch for {arm_id}/{task_id}")
            if attempt.get("requested_model") != spec["model"]:
                raise ValueError(f"ESConv attempt model mismatch for {arm_id}/{task_id}")
            if attempt.get("response_model") != spec["model"]:
                raise ValueError(f"ESConv attempt served model mismatch for {arm_id}/{task_id}")
            if index < len(trace) - 1:
                if parse_strategy_response(attempt_raw).error is None:
                    raise ValueError(
                        f"ESConv retry was not justified for {arm_id}/{task_id}"
                    )
                expected_messages = _esconv_correction_messages(
                    target_messages, attempt_raw
                )
        if trace[-1]["raw_response"] != raw_response:
            raise ValueError(f"ESConv scored raw output mismatch for {arm_id}/{task_id}")
        if record.get("scored_messages_sha256") != trace[-1]["messages_sha256"]:
            raise ValueError(f"ESConv scored prompt mismatch for {arm_id}/{task_id}")
        for field in (
            "endpoint_id",
            "requested_model",
            "response_model",
            "fingerprint",
            "response_id",
            "wire_api",
        ):
            if record.get(field) != trace[-1].get(field):
                raise ValueError(
                    f"ESConv final {field} differs from attempt trace for "
                    f"{arm_id}/{task_id}"
                )

    parsed = parse_strategy_response(raw_response)
    if (
        parsed.strategy != record.get("predicted_strategy")
        or parsed.response != record.get("response")
        or parsed.error != record.get("format_error")
    ):
        raise ValueError(
            f"ESConv derived fields differ from raw output for {arm_id}/{task_id}"
        )
    if record.get("gold_strategy") != example.gold_strategy:
        raise ValueError(f"ESConv gold label mismatch for {arm_id}/{task_id}")


def _cpcd_generation_matrix(
    preregistration: dict[str, Any],
    repository_root: Path,
    generation_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]], Path]:
    source = preregistration["sources"]["cpcd"]
    eval_root = repository_root / source["checkout"] / source["eval_root"]
    tasks = {task.id: task for task in load_cpcd_tasks(eval_root)}
    zh_prompt = (
        repository_root / preregistration["prompts"]["zh"]["path"]
    ).read_text(encoding="utf-8")
    full_session_dir = eval_root / "full_session"
    arm_ids = [str(spec["id"]) for spec in preregistration["target_arms"]]
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm_id: {} for arm_id in arm_ids}

    for record in _load_run_records(generation_paths):
        _validate_target_generation_record(record, preregistration, "cpcd")
        arm_id = str(record.get("arm_id") or "")
        task_id = str(record.get("task_id") or "")
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown CPCD task in generation: {task_id!r}")
        if record.get("family") != task.family:
            raise ValueError(f"CPCD family mismatch for {task_id}")
        full_history = load_cpcd_full_history(task, full_session_dir)
        expected_messages = build_cpcd_target_messages(task, full_history, zh_prompt)
        if record.get("messages_sha256") != _messages_sha256(expected_messages):
            raise ValueError(f"CPCD target prompt mismatch for {arm_id}/{task_id}")
        expected_budget = int(
            preregistration["cpcd"]["target_max_output_tokens"][task.family]
        )
        if record.get("max_output_tokens_requested") != expected_budget:
            raise ValueError(f"CPCD output budget mismatch for {arm_id}/{task_id}")
        if not isinstance(record.get("response"), str) or not record["response"].strip():
            raise ValueError(f"CPCD generation {arm_id}/{task_id} has no response")
        if task_id in by_arm[arm_id]:
            raise ValueError(f"duplicate CPCD generation for {arm_id}/{task_id}")
        by_arm[arm_id][task_id] = record

    task_sets = [set(by_arm[arm_id]) for arm_id in arm_ids]
    if (
        not task_sets
        or not task_sets[0]
        or any(value != task_sets[0] for value in task_sets[1:])
    ):
        raise ValueError(
            "both target arms must contain the same CPCD task set before judging"
        )
    selected_tasks = {task_id: tasks[task_id] for task_id in task_sets[0]}
    return selected_tasks, by_arm, eval_root


def run_cpcd_judging(
    preregistration: dict[str, Any],
    repository_root: Path,
    generation_paths: Sequence[Path],
    output: Path,
    env_file: Path | None,
    concurrency: int,
) -> dict[str, Any]:
    if output.resolve() in {path.resolve() for path in generation_paths}:
        raise ValueError("CPCD judgement output must differ from generation inputs")
    validate_protocol_assets(preregistration, repository_root)
    protocol_hash = protocol_sha256(preregistration)
    tasks, generations_by_arm, eval_root = _cpcd_generation_matrix(
        preregistration, repository_root, generation_paths
    )
    arm_ids = [str(spec["id"]) for spec in preregistration["target_arms"]]
    seed = int(preregistration["randomization"]["blind_order_seed"])
    store = JsonlRunStore(output)

    rubrics = {
        family: (eval_root / directory / "rubric.md").read_text(encoding="utf-8")
        for family, directory in _CPCD_RUBRIC_DIRECTORIES.items()
    }
    full_session_dir = eval_root / "full_session"
    official_order = [task.id for task in load_cpcd_tasks(eval_root) if task.id in tasks]
    pending: list[
        tuple[
            str,
            Any,
            str,
            str,
            dict[str, Any],
            str,
            list[dict[str, str]],
            str,
        ]
    ] = []
    for task_id in official_order:
        task = tasks[task_id]
        full_history = load_cpcd_full_history(task, full_session_dir)
        blind_order = stable_blind_order(task_id, arm_ids, seed)
        for blind_index, arm_id in enumerate(blind_order):
            blind_id = f"candidate-{chr(ord('A') + blind_index)}"
            generation = generations_by_arm[arm_id][task_id]
            candidate_hash = hashlib.sha256(
                generation["response"].encode("utf-8")
            ).hexdigest()
            messages = build_cpcd_judge_messages(
                task=task,
                response=generation["response"],
                full_history=full_history,
                rubric=rubrics[task.family],
                blind_id=blind_id,
            )
            messages_hash = _messages_sha256(messages)
            run_id = (
                f"{preregistration['protocol_id']}:cpcd-judge:{task_id}:{blind_id}"
            )
            existing = store.index.get(run_id)
            if existing:
                if existing.get("protocol_sha256") != protocol_hash:
                    raise ValueError(f"resume protocol hash mismatch for {run_id}")
                if existing.get("arm_id") != arm_id:
                    raise ValueError(f"resume blind assignment mismatch for {run_id}")
                if existing.get("candidate_response_sha256") != candidate_hash:
                    raise ValueError(f"resume candidate response mismatch for {run_id}")
                if existing.get("messages_sha256") != messages_hash:
                    raise ValueError(f"resume judge prompt mismatch for {run_id}")
                if existing.get("record_type") != "judgement":
                    raise ValueError(f"resume record type mismatch for {run_id}")
                if existing.get("task_id") != task.id or existing.get("family") != task.family:
                    raise ValueError(f"resume CPCD task metadata mismatch for {run_id}")
                if existing.get("blind_id") != blind_id:
                    raise ValueError(f"resume blind id mismatch for {run_id}")
                if existing.get("candidate_generation_run_id") != generation["run_id"]:
                    raise ValueError(f"resume generation linkage mismatch for {run_id}")
                if existing.get("endpoint_id") != preregistration["judge"]["id"]:
                    raise ValueError(f"resume judge endpoint mismatch for {run_id}")
                if existing.get("requested_model") != preregistration["judge"]["model"]:
                    raise ValueError(f"resume judge model mismatch for {run_id}")
                if existing.get("response_model") != preregistration["judge"]["model"]:
                    raise ValueError(f"resume served judge model mismatch for {run_id}")
                if existing.get("candidate_endpoint_id") != generation.get("endpoint_id"):
                    raise ValueError(f"resume candidate endpoint mismatch for {run_id}")
                if existing.get("candidate_requested_model") != generation.get(
                    "requested_model"
                ):
                    raise ValueError(f"resume candidate requested model mismatch for {run_id}")
                if existing.get("candidate_response_model") != generation.get(
                    "response_model"
                ):
                    raise ValueError(f"resume candidate served model mismatch for {run_id}")
                if existing.get("candidate_messages_sha256") != generation.get(
                    "messages_sha256"
                ):
                    raise ValueError(f"resume candidate prompt mismatch for {run_id}")
                raw_judgement = existing.get("judge_raw_response")
                if not isinstance(raw_judgement, str):
                    raise ValueError(f"resume raw judgement missing for {run_id}")
                parsed = parse_cpcd_judgement(task.family, raw_judgement)
                if parsed["scores"] != existing.get("scores"):
                    raise ValueError(f"resume judgement score mismatch for {run_id}")
                continue
            pending.append(
                (
                    run_id,
                    task,
                    arm_id,
                    blind_id,
                    generation,
                    candidate_hash,
                    messages,
                    messages_hash,
                )
            )

    started = time.monotonic()
    completed = 0
    if pending:
        endpoint = EndpointConfig.from_spec(
            preregistration["judge"], _load_explicit_env(env_file)
        )
        client = OpenAICompatibleClient(endpoint)
        max_tokens = int(preregistration["judge"]["max_output_tokens"])

        def judge_one(
            entry: tuple[
                str,
                Any,
                str,
                str,
                dict[str, Any],
                str,
                list[dict[str, str]],
                str,
            ]
        ) -> dict[str, Any]:
            (
                run_id,
                task,
                arm_id,
                blind_id,
                generation,
                candidate_hash,
                messages,
                messages_hash,
            ) = entry
            api_result = client.complete(messages, max_tokens=max_tokens)
            raw_judgement = api_result.pop("text")
            try:
                parsed = parse_cpcd_judgement(task.family, raw_judgement)
            except ValueError as error:
                raise ValueError(
                    f"invalid CPCD judgement for {task.id}/{blind_id}: {error}"
                ) from error
            return {
                "run_id": run_id,
                "protocol_id": preregistration["protocol_id"],
                "protocol_sha256": protocol_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "benchmark": "cpcd",
                "record_type": "judgement",
                "task_id": task.id,
                "family": task.family,
                "arm_id": arm_id,
                "blind_id": blind_id,
                "candidate_generation_run_id": generation["run_id"],
                "candidate_response_sha256": candidate_hash,
                "candidate_endpoint_id": generation["endpoint_id"],
                "candidate_requested_model": generation["requested_model"],
                "candidate_response_model": generation["response_model"],
                "candidate_messages_sha256": generation["messages_sha256"],
                "messages_sha256": messages_hash,
                "judge_raw_response": raw_judgement,
                **parsed,
                **api_result,
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, concurrency)
        ) as executor:
            future_map = {
                executor.submit(judge_one, entry): entry[0] for entry in pending
            }
            first_error: Exception | None = None
            for future in concurrent.futures.as_completed(future_map):
                try:
                    record = future.result()
                except Exception as error:
                    if first_error is None:
                        first_error = error
                    continue
                store.append(record)
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(pending):
                    print(
                        f"judged {completed}/{len(pending)} -> {output}", flush=True
                    )
            if first_error is not None:
                raise first_error

    selected = len(tasks) * len(arm_ids)
    return {
        "protocol_id": preregistration["protocol_id"],
        "benchmark": "cpcd",
        "selected": selected,
        "already_complete": selected - len(pending),
        "generated_now": completed,
        "output": str(output),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _official_esconv_tokens(value: str) -> list[str]:
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError("NLTK is required for official ESConv tokenization") from error
    try:
        return list(nltk.word_tokenize(value.lower()))
    except LookupError as error:
        raise RuntimeError(
            "NLTK punkt resources are required for official ESConv tokenization; "
            "run `python -m nltk.downloader punkt punkt_tab`"
        ) from error


def _field_distribution(
    records: Sequence[dict[str, Any]], field: str
) -> dict[str, int]:
    values = [
        str(record[field]) if record.get(field) is not None else "<missing>"
        for record in records
    ]
    return {value: values.count(value) for value in sorted(set(values))}


def summarize_results(
    preregistration: dict[str, Any],
    repository_root: Path,
    cpcd_judgement_paths: Sequence[Path],
    esconv_generation_paths: Sequence[Path],
    cpcd_generation_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    if not cpcd_generation_paths:
        raise ValueError("CPCD summary requires both target generation files")
    cpcd_generation_tasks, cpcd_generations_by_arm, _ = _cpcd_generation_matrix(
        preregistration, repository_root, cpcd_generation_paths
    )
    arm_ids = {str(spec["id"]) for spec in preregistration["target_arms"]}
    result: dict[str, Any] = {
        "protocol_id": preregistration["protocol_id"],
        "protocol_sha256": protocol_sha256(preregistration),
        "esconv_method": esconv_method_identity(),
        "cpcd": {"score_type": "quality_not_accuracy", "arms": {}},
        "esconv": {
            "score_type": "strategy_classification_and_response_quality",
            "arms": {},
        },
        "claim_boundaries": {
            "cpcd": (
                "Paired frozen comparison with an adapted reference-aware judge prompt; "
                "not an exact reproduction of published leaderboard rows."
            ),
            "esconv": (
                "Official pinned test rows and strategy-conditioned history under a "
                "zero-shot chat prompt; do not directly rank against papers using other "
                "splits, preprocessing, label spaces, or supervised models."
            ),
        },
    }

    cpcd_source = preregistration["sources"]["cpcd"]
    cpcd_eval_root = (
        repository_root / cpcd_source["checkout"] / cpcd_source["eval_root"]
    )
    official_cpcd = {task.id: task for task in load_cpcd_tasks(cpcd_eval_root)}
    cpcd_full_session_dir = cpcd_eval_root / "full_session"
    zh_prompt = (
        repository_root / preregistration["prompts"]["zh"]["path"]
    ).read_text(encoding="utf-8")
    target_specs = {
        str(spec["id"]): spec for spec in preregistration["target_arms"]
    }
    cpcd_rubrics = {
        family: (cpcd_eval_root / directory / "rubric.md").read_text(
            encoding="utf-8"
        )
        for family, directory in _CPCD_RUBRIC_DIRECTORIES.items()
    }
    cpcd_records_by_arm: dict[str, list[dict[str, Any]]] = {}
    cpcd_seen: set[tuple[str, str]] = set()
    for record in _load_run_records(cpcd_judgement_paths):
        _require_frozen_record(record, preregistration, "cpcd")
        arm_id = str(record.get("arm_id") or "")
        task_id = str(record.get("task_id") or "")
        if arm_id not in arm_ids:
            raise ValueError(f"unknown target arm in CPCD judgement: {arm_id!r}")
        task = official_cpcd.get(task_id)
        if task is None:
            raise ValueError(f"unknown CPCD task in judgement: {task_id!r}")
        if record.get("record_type") != "judgement":
            raise ValueError(f"CPCD record {record.get('run_id')!r} is not a judgement")
        if record.get("family") != task.family:
            raise ValueError(f"CPCD family mismatch for judgement {arm_id}/{task_id}")
        judge = preregistration["judge"]
        if record.get("endpoint_id") != judge["id"]:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} used the wrong endpoint")
        if record.get("requested_model") != judge["model"]:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} used the wrong judge model")
        if record.get("response_model") != judge["model"]:
            raise ValueError(
                f"CPCD judgement {arm_id}/{task_id} used the wrong served judge model"
            )
        expected_generation_id = (
            f"{preregistration['protocol_id']}:cpcd:{arm_id}:{task_id}"
        )
        if record.get("candidate_generation_run_id") != expected_generation_id:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} has broken generation linkage")
        if not _looks_like_sha256(record.get("candidate_response_sha256")):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} lacks a response hash")
        generation = cpcd_generations_by_arm.get(arm_id, {}).get(task_id)
        if generation is None:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} lacks its generation")
        actual_candidate_hash = hashlib.sha256(
            generation["response"].encode("utf-8")
        ).hexdigest()
        if record.get("candidate_response_sha256") != actual_candidate_hash:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} has wrong response hash")
        target_spec = target_specs[arm_id]
        if record.get("candidate_endpoint_id") != target_spec["id"]:
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} has wrong candidate endpoint")
        if record.get("candidate_requested_model") != target_spec["model"]:
            raise ValueError(
                f"CPCD judgement {arm_id}/{task_id} has wrong candidate requested model"
            )
        if record.get("candidate_response_model") != target_spec["model"]:
            raise ValueError(
                f"CPCD judgement {arm_id}/{task_id} has wrong candidate served model"
            )
        full_history = load_cpcd_full_history(task, cpcd_full_session_dir)
        target_messages = build_cpcd_target_messages(task, full_history, zh_prompt)
        if record.get("candidate_messages_sha256") != _messages_sha256(
            target_messages
        ):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} has wrong target prompt")
        if not _looks_like_sha256(record.get("messages_sha256")):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} lacks a prompt hash")
        judge_messages = build_cpcd_judge_messages(
            task=task,
            response=generation["response"],
            full_history=full_history,
            rubric=cpcd_rubrics[task.family],
            blind_id=str(record.get("blind_id") or ""),
        )
        if record.get("messages_sha256") != _messages_sha256(judge_messages):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} has wrong judge prompt")
        raw_judgement = record.get("judge_raw_response")
        if not isinstance(raw_judgement, str):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} lacks raw judge output")
        parsed = parse_cpcd_judgement(task.family, raw_judgement)
        if parsed["scores"] != record.get("scores"):
            raise ValueError(f"CPCD judgement {arm_id}/{task_id} scores differ from raw output")
        key = (arm_id, task_id)
        if key in cpcd_seen:
            raise ValueError(f"duplicate CPCD judgement for {arm_id}/{task_id}")
        cpcd_seen.add(key)
        cpcd_records_by_arm.setdefault(arm_id, []).append(record)

    if set(cpcd_records_by_arm) != arm_ids:
        raise ValueError("CPCD summary requires both target arms")
    cpcd_task_sets = {
        arm_id: {str(record["task_id"]) for record in records}
        for arm_id, records in cpcd_records_by_arm.items()
    }
    first_cpcd_set = next(iter(cpcd_task_sets.values()))
    if not first_cpcd_set or any(
        task_ids != first_cpcd_set for task_ids in cpcd_task_sets.values()
    ):
        raise ValueError("both target arms must contain the same CPCD task set")
    if first_cpcd_set != set(cpcd_generation_tasks):
        raise ValueError("CPCD judgements and generations must contain the same task set")
    arm_order = [str(spec["id"]) for spec in preregistration["target_arms"]]
    blind_seed = int(preregistration["randomization"]["blind_order_seed"])
    for task_id in first_cpcd_set:
        task_records = [
            record
            for records in cpcd_records_by_arm.values()
            for record in records
            if record["task_id"] == task_id
        ]
        expected_blind_ids = {
            arm_id: f"candidate-{chr(ord('A') + index)}"
            for index, arm_id in enumerate(
                stable_blind_order(task_id, arm_order, blind_seed)
            )
        }
        if any(
            record.get("blind_id") != expected_blind_ids[record["arm_id"]]
            for record in task_records
        ):
            raise ValueError(f"CPCD blind assignment is invalid for {task_id}")

    expected_cpcd = preregistration["cpcd"]["expected_task_counts"]
    for arm_id, records in sorted(cpcd_records_by_arm.items()):
        summary = summarize_cpcd_judgements(records)
        summary["derived_nonofficial_equal_family_macro"] = summary.pop("overall")
        by_family = {
            family: sum(record.get("family") == family for record in records)
            for family in ("srg", "mr", "tcr")
        }
        valid_by_family = {
            family: int(summary["families"].get(family, {}).get("n", 0))
            for family in ("srg", "mr", "tcr")
        }
        summary["coverage"] = {
            "observed": len(records),
            "valid_observed": sum(valid_by_family.values()),
            "expected": sum(int(value) for value in expected_cpcd.values()),
            "by_family": by_family,
            "valid_by_family": valid_by_family,
            "complete": all(
                valid_by_family[family] == int(expected_cpcd[family])
                for family in ("srg", "mr", "tcr")
            ),
        }
        target_records = list(cpcd_generations_by_arm[arm_id].values())
        summary["target_provenance"] = {
            "requested_models": _field_distribution(target_records, "requested_model"),
            "response_models": _field_distribution(target_records, "response_model"),
            "fingerprints": _field_distribution(target_records, "fingerprint"),
            "declared_revision": target_specs[arm_id].get("revision"),
            "weight_revision_verification": (
                "requires_external_server_weight_manifest"
                if target_specs[arm_id].get("revision")
                else "provider_managed_model"
            ),
        }
        summary["judge_provenance"] = {
            "requested_models": _field_distribution(records, "requested_model"),
            "response_models": _field_distribution(records, "response_model"),
            "fingerprints": _field_distribution(records, "fingerprint"),
        }
        result["cpcd"]["arms"][arm_id] = summary
    result["cpcd"]["paired_task_count"] = len(first_cpcd_set)
    result["cpcd"]["coverage_complete"] = all(
        arm_summary["coverage"]["complete"]
        for arm_summary in result["cpcd"]["arms"].values()
    )
    cpcd_revision_verified = all(
        not spec.get("revision") for spec in preregistration["target_arms"]
    )
    result["cpcd"]["revision_verification_complete"] = cpcd_revision_verified
    result["cpcd"]["formal_claim_ready"] = (
        result["cpcd"]["coverage_complete"] and cpcd_revision_verified
    )

    esconv_source = preregistration["sources"]["esconv"]
    esconv_path = (
        repository_root / esconv_source["checkout"] / esconv_source["test_file"]
    )
    official_examples = load_esconv_test(esconv_path)
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError("NLTK 3.9.4 is required for ESConv evaluation") from error

    result["esconv"]["evaluator"] = {
        "tokenizer": "nltk.word_tokenize(lowercase=True)",
        "nltk_version": nltk.__version__,
        "bleu": "nltk.corpus_bleu with SmoothingFunction.method3",
    }
    en_prompt = (
        repository_root / preregistration["prompts"]["en"]["path"]
    ).read_text(encoding="utf-8")
    expected_esconv_rows = int(preregistration["esconv"]["expected_rows"])
    if len(official_examples) != expected_esconv_rows:
        raise ValueError(
            f"official ESConv row count is {len(official_examples)}, "
            f"expected {expected_esconv_rows}"
        )
    official_by_id = {example.id: example for example in official_examples}
    esconv_records_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for record in _load_run_records(esconv_generation_paths):
        _validate_target_generation_record(record, preregistration, "esconv")
        arm_id = str(record.get("arm_id") or "")
        task_id = str(record.get("task_id") or "")
        if arm_id not in arm_ids:
            raise ValueError(f"unknown target arm in ESConv generation: {arm_id!r}")
        example = official_by_id.get(task_id)
        if example is None:
            raise ValueError(f"unknown ESConv task in generation: {task_id!r}")
        expected_messages = build_esconv_target_messages(example, en_prompt)
        if record.get("messages_sha256") != _messages_sha256(expected_messages):
            raise ValueError(f"ESConv target prompt mismatch for {arm_id}/{task_id}")
        predicted = record.get("predicted_strategy")
        if predicted is not None and predicted not in STRATEGIES:
            raise ValueError(f"invalid stored ESConv strategy for {arm_id}/{task_id}")
        if not isinstance(record.get("response"), str):
            raise ValueError(f"ESConv generation {arm_id}/{task_id} has no response")
        if record.get("max_output_tokens_requested") != int(
            preregistration["esconv"]["target_max_output_tokens"]
        ):
            raise ValueError(f"ESConv output budget mismatch for {arm_id}/{task_id}")
        _validate_esconv_generation_content(
            record, example, expected_messages, preregistration
        )
        arm_records = esconv_records_by_arm.setdefault(arm_id, {})
        if task_id in arm_records:
            raise ValueError(f"duplicate ESConv generation for {arm_id}/{task_id}")
        arm_records[task_id] = record

    if set(esconv_records_by_arm) != arm_ids:
        raise ValueError("ESConv summary requires both target arms")
    esconv_task_sets = {
        arm_id: set(records_by_id)
        for arm_id, records_by_id in esconv_records_by_arm.items()
    }
    first_esconv_set = next(iter(esconv_task_sets.values()))
    if not first_esconv_set or any(
        task_ids != first_esconv_set for task_ids in esconv_task_sets.values()
    ):
        raise ValueError("both target arms must contain the same ESConv task set")

    official_order = [example.id for example in official_examples]
    for arm_id, records_by_id in sorted(esconv_records_by_arm.items()):
        ordered_ids = [task_id for task_id in official_order if task_id in records_by_id]
        pairs = [
            (
                official_by_id[task_id].gold_strategy,
                records_by_id[task_id].get("predicted_strategy"),
            )
            for task_id in ordered_ids
        ]
        strategy = score_strategy_predictions(pairs)
        strategy["invalid_rate"] = (
            strategy["invalid"] / strategy["n"] if strategy["n"] else 0.0
        )
        references = [
            _official_esconv_tokens(official_by_id[task_id].gold_response)
            for task_id in ordered_ids
        ]
        hypotheses = [
            _official_esconv_tokens(records_by_id[task_id]["response"])
            for task_id in ordered_ids
        ]
        response_metrics = response_overlap_metrics_tokenized(references, hypotheses)
        observed_ids = set(ordered_ids)
        result["esconv"]["arms"][arm_id] = {
            "strategy": strategy,
            "response": response_metrics,
            "coverage": {
                "observed": len(observed_ids),
                "expected": expected_esconv_rows,
                "complete": observed_ids == set(official_by_id),
            },
            "target_provenance": {
                "requested_models": _field_distribution(
                    list(records_by_id.values()), "requested_model"
                ),
                "response_models": _field_distribution(
                    list(records_by_id.values()), "response_model"
                ),
                "fingerprints": _field_distribution(
                    list(records_by_id.values()), "fingerprint"
                ),
                "declared_revision": target_specs[arm_id].get("revision"),
                "weight_revision_verification": (
                    "requires_external_server_weight_manifest"
                    if target_specs[arm_id].get("revision")
                    else "provider_managed_model"
                ),
            },
        }
    result["esconv"]["paired_task_count"] = len(first_esconv_set)
    result["esconv"]["coverage_complete"] = all(
        arm_summary["coverage"]["complete"]
        for arm_summary in result["esconv"]["arms"].values()
    )
    esconv_revision_verified = all(
        not spec.get("revision") for spec in preregistration["target_arms"]
    )
    result["esconv"]["revision_verification_complete"] = esconv_revision_verified
    result["esconv"]["formal_claim_ready"] = (
        result["esconv"]["coverage_complete"] and esconv_revision_verified
    )
    return result


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
    judge = subparsers.add_parser(
        "judge-cpcd", help="blindly judge paired CPCD generations with the frozen judge"
    )
    judge.add_argument(
        "--generation",
        dest="generation_paths",
        action="append",
        type=Path,
        help="CPCD generation JSONL; repeat once per target arm",
    )
    judge.add_argument("--output", type=Path)
    judge.add_argument("--env-file", type=Path)
    judge.add_argument("--concurrency", type=int, default=4)
    summarize = subparsers.add_parser(
        "summarize", help="summarize frozen CPCD judgements and ESConv generations"
    )
    summarize.add_argument(
        "--cpcd-judgement",
        "--cpcd-judgements",
        dest="cpcd_judgement_paths",
        action="append",
        type=Path,
    )
    summarize.add_argument(
        "--cpcd-generation",
        dest="cpcd_generation_paths",
        action="append",
        type=Path,
        help="CPCD generation JSONL; repeat once per target arm",
    )
    summarize.add_argument(
        "--esconv-generation",
        dest="esconv_generation_paths",
        action="append",
        type=Path,
    )
    summarize.add_argument("--output", type=Path)
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
    elif args.command == "judge-cpcd":
        generation_paths = args.generation_paths or [
            _default_generation_output(preregistration, spec["id"], "cpcd")
            for spec in preregistration["target_arms"]
        ]
        result = run_cpcd_judging(
            preregistration=preregistration,
            repository_root=REPOSITORY_ROOT,
            generation_paths=generation_paths,
            output=args.output or _default_cpcd_judgement_output(preregistration),
            env_file=args.env_file,
            concurrency=args.concurrency,
        )
    elif args.command == "summarize":
        validate_protocol_assets(preregistration, REPOSITORY_ROOT)
        cpcd_paths = args.cpcd_judgement_paths or [
            _default_cpcd_judgement_output(preregistration)
        ]
        cpcd_generation_paths = args.cpcd_generation_paths or [
            _default_generation_output(preregistration, spec["id"], "cpcd")
            for spec in preregistration["target_arms"]
        ]
        esconv_paths = args.esconv_generation_paths or [
            _default_generation_output(preregistration, spec["id"], "esconv")
            for spec in preregistration["target_arms"]
        ]
        if args.output and args.output.resolve() in {
            path.resolve()
            for path in [*cpcd_paths, *cpcd_generation_paths, *esconv_paths]
        }:
            raise ValueError("summary output must differ from all JSONL inputs")
        result = summarize_results(
            preregistration=preregistration,
            repository_root=REPOSITORY_ROOT,
            cpcd_judgement_paths=cpcd_paths,
            esconv_generation_paths=esconv_paths,
            cpcd_generation_paths=cpcd_generation_paths,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
