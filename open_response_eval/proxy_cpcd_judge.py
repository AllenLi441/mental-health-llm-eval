from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import (
    build_cpcd_judge_messages,
    load_cpcd_tasks,
    parse_cpcd_judgement,
)
from .runner import (
    EndpointConfig,
    JsonlRunStore,
    OpenAICompatibleClient,
    load_cpcd_full_history,
    read_jsonl_index,
)


MODEL_INPUTS = {
    "deepseek_v4_pro": Path("outputs/provider_v2_deepseek_cpcd_20260808.jsonl"),
    "deepseek_v4_flash": Path("outputs/provider_v2_flash_cpcd_20260809.jsonl"),
    "qwen_3_6_27b": Path("outputs/provider_v2_qwen_cpcd_20260808.jsonl"),
    "qwen_3_6_27b_siliconflow": Path(
        "outputs/provider_sf_qwen_cpcd_20260813.jsonl"
    ),
}


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("tmp/official_benchmarks/psy_chronicle/eval_task_info"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cpcd_proxy_flash_judgements_20260809.jsonl"),
    )
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()

    env = load_env(args.env_file)
    endpoint = EndpointConfig(
        id="cpcd_proxy_deepseek_v4_flash",
        model="deepseek-v4-flash",
        base_url=env.get("EVAL_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=env["EVAL_API_KEY"],
        wire_api=env.get("EVAL_WIRE_API", "chat"),
        timeout_seconds=240,
        max_retries=3,
        request_overrides={
            "temperature": 0,
            "thinking": {"type": "disabled"},
        },
    )
    client = OpenAICompatibleClient(endpoint)

    tasks = {task.id: task for task in load_cpcd_tasks(args.eval_root)}
    full_session_dir = args.eval_root / "full_session"
    histories = {
        task.id: load_cpcd_full_history(task, full_session_dir)
        for task in tasks.values()
    }
    rubric_dirs = {"srg": "srg", "mr": "memory_recall", "tcr": "TCR"}
    rubrics = {
        family: (args.eval_root / directory / "rubric.md").read_text(encoding="utf-8")
        for family, directory in rubric_dirs.items()
    }

    pending: list[tuple[str, dict[str, Any]]] = []
    existing = read_jsonl_index(args.output) if args.output.exists() else {}
    for model_id, relative_path in MODEL_INPUTS.items():
        for generation in load_jsonl(relative_path):
            run_id = f"cpcd-proxy-flash:{model_id}:{generation['task_id']}"
            if run_id not in existing:
                pending.append((model_id, generation))

    store = JsonlRunStore(args.output)

    def judge_one(model_id: str, generation: dict[str, Any]) -> dict[str, Any]:
        task = tasks[generation["task_id"]]
        blind_id = hashlib.sha256(
            f"{model_id}:{task.id}".encode("utf-8")
        ).hexdigest()[:16]
        base_messages = build_cpcd_judge_messages(
            task,
            generation["response"],
            histories[task.id],
            rubrics[task.family],
            blind_id,
        )
        traces: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for format_attempt in range(1, 4):
            messages = list(base_messages)
            if format_attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous result was not valid under the required JSON "
                            "schema. Return only the JSON object required by the rubric."
                        ),
                    }
                )
            api_result = client.complete(messages, max_tokens=2200)
            raw = api_result["text"]
            try:
                parsed = parse_cpcd_judgement(task.family, raw)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                last_error = error
                traces.append(
                    {
                        "format_attempt": format_attempt,
                        "raw_response": raw,
                        "usage": api_result.get("usage"),
                        "error": str(error),
                    }
                )
                continue
            return {
                "run_id": f"cpcd-proxy-flash:{model_id}:{task.id}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task_id": task.id,
                "family": task.family,
                "target_model_id": model_id,
                "target_requested_model": generation.get("requested_model"),
                "target_response_model": generation.get("response_model"),
                "target_generation_run_id": generation.get("run_id"),
                "blind_id": blind_id,
                "judge_id": endpoint.id,
                "judge_requested_model": endpoint.model,
                "judge_response_model": api_result.get("response_model"),
                "judge_fingerprint": api_result.get("fingerprint"),
                "judge_mode": "proxy_non_thinking",
                "rubric_source": str(
                    args.eval_root / rubric_dirs[task.family] / "rubric.md"
                ),
                "parsed_judgement": parsed,
                "raw_response": raw,
                "usage": api_result.get("usage"),
                "latency_ms": api_result.get("latency_ms"),
                "format_attempts": format_attempt,
                "format_attempt_trace": traces,
            }
        raise RuntimeError(
            f"judge format failed for {model_id}/{task.id}: {last_error}"
        )

    errors: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(judge_one, model_id, generation): (model_id, generation["task_id"])
            for model_id, generation in pending
        }
        for future in concurrent.futures.as_completed(futures):
            model_id, task_id = futures[future]
            try:
                store.append(future.result())
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(pending):
                    print(f"judged {completed}/{len(pending)} -> {args.output}", flush=True)
            except Exception as error:  # Preserve all successful records for resume.
                errors.append(f"{model_id}/{task_id}: {error}")

    print(
        json.dumps(
            {
                "already_complete": len(existing),
                "judged_now": completed,
                "errors": len(errors),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if errors:
        print("\n".join(errors[:10]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
