from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


STRATEGIES = (
    "Questions",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Other",
)

_STRATEGY_BY_CASEFOLD = {label.casefold(): label for label in STRATEGIES}
_ESCONV_SEGMENT = re.compile(
    r"^\s*(?P<loss>[01](?:\.0)?)\s+(?P<role>[01])\s+(?P<turn>\d+)\s+(?P<body>.*)\s*$"
)
_STRATEGY_PREFIX = re.compile(r"^\[(?P<strategy>[^\]]+)\]\s*(?P<content>.*)$")


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _canonical_strategy(value: str) -> str | None:
    return _STRATEGY_BY_CASEFOLD.get(_normalize_space(value).casefold())


@dataclass(frozen=True)
class ESConvExample:
    id: str
    context: list[dict[str, str]]
    gold_strategy: str
    gold_response: str

    @classmethod
    def from_tsv_line(cls, line: str, line_number: int) -> "ESConvExample":
        raw_segments = re.split(r"\s+EOS\s+", line.strip())
        if not raw_segments or not raw_segments[-1].strip():
            raise ValueError(f"ESConv row {line_number} is empty")

        parsed: list[tuple[int, str | None, str]] = []
        for raw_segment in raw_segments:
            match = _ESCONV_SEGMENT.match(raw_segment)
            if not match:
                raise ValueError(
                    f"ESConv row {line_number} has an invalid turn segment: {raw_segment[:80]!r}"
                )
            role = int(match.group("role"))
            body = _normalize_space(match.group("body"))
            strategy = None
            if role == 1:
                strategy_match = _STRATEGY_PREFIX.match(body)
                if not strategy_match:
                    raise ValueError(
                        f"ESConv row {line_number} supporter turn lacks a strategy label"
                    )
                strategy = _canonical_strategy(strategy_match.group("strategy"))
                if strategy is None:
                    raise ValueError(
                        f"ESConv row {line_number} has unknown strategy "
                        f"{strategy_match.group('strategy')!r}"
                    )
                body = _normalize_space(strategy_match.group("content"))
            parsed.append((role, strategy, body))

        target_role, target_strategy, target_text = parsed[-1]
        if target_role != 1 or target_strategy is None:
            raise ValueError(f"ESConv row {line_number} target must be a supporter turn")
        if not target_text:
            raise ValueError(f"ESConv row {line_number} target supporter response is empty")

        context = [
            {"role": "assistant" if role == 1 else "user", "content": text}
            for role, _strategy, text in parsed[:-1]
        ]
        return cls(
            id=f"esconv-test-{line_number:06d}",
            context=context,
            gold_strategy=target_strategy,
            gold_response=target_text,
        )


@dataclass(frozen=True)
class StrategyResponse:
    strategy: str | None
    response: str
    error: str | None


@dataclass(frozen=True)
class CPCDTask:
    id: str
    family: str
    case_id: str
    payload: dict[str, Any]
    source_file: str


def load_esconv_test(path: Path) -> list[ESConvExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(ESConvExample.from_tsv_line(line, line_number))
    return rows


def _task_records(path: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for source_file in sorted(path.glob("*.json")):
        value = json.loads(source_file.read_text(encoding="utf-8"))
        records = value if isinstance(value, list) else [value]
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{source_file} contains a non-object task")
            yield source_file, record


def load_cpcd_tasks(eval_task_info: Path) -> list[CPCDTask]:
    tasks: list[CPCDTask] = []
    families = (
        ("srg", eval_task_info / "srg"),
        ("mr", eval_task_info / "memory_recall"),
        ("tcr", eval_task_info / "TCR"),
    )
    for family, directory in families:
        if not directory.is_dir():
            raise FileNotFoundError(f"missing CPCD task directory: {directory}")
        for source_file, record in _task_records(directory):
            task_id = str(record.get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"{source_file} contains a task without task_id")
            tasks.append(
                CPCDTask(
                    id=task_id,
                    family=family,
                    case_id=str(record.get("case_id") or source_file.stem),
                    payload=record,
                    source_file=source_file.name,
                )
            )

    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("CPCD task_id values are not globally unique")
    return tasks


def _format_context(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _render_history(value: Any) -> str:
    if not isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, indent=2)
    lines = []
    for index, message in enumerate(value, start=1):
        if isinstance(message, dict):
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", ""))
            lines.append(f"[{index:04d}] {role}: {content}")
        else:
            lines.append(f"[{index:04d}] {json.dumps(message, ensure_ascii=False)}")
    return "\n".join(lines)


def build_cpcd_target_messages(
    task: CPCDTask,
    full_history: Any,
    jingshi_system_prefix: str,
) -> list[dict[str, str]]:
    payload = task.payload
    if task.family == "srg":
        inp = payload.get("input_to_model") or {}
        official_system = (
            "你正在完成 CPCD-Bench 的下一轮校园心理支持回复任务。"
            "请根据学生画像、历史会话、当前事件、当前上下文和学生最新发言，"
            "生成下一轮回复。优先准确共情并承接历史，保持专业边界和风险敏感性；"
            "不要保证、评判、说教、编造信息或替学生做决定。"
            "只输出回复本身，不要输出分析、标题、评分或 JSON。"
        )
        system = f"{jingshi_system_prefix.strip()}\n\n{official_system}".strip()
        user = f"""请生成下一轮回复。

【task_id】
{task.id}

【student_profile_summary】
{inp.get('student_profile_summary', '')}

【history_until_previous_session】
{inp.get('history_until_previous_session', '')}

【current_session_event】
{inp.get('current_session_event', '')}

【current_session_context】
{_format_context(inp.get('current_session_context', ''))}

【current_student_utterance】
{inp.get('current_student_utterance', '')}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if task.family not in {"mr", "tcr"}:
        raise ValueError(f"unsupported CPCD task family: {task.family}")
    if full_history is None:
        raise ValueError(f"{task.id} requires a complete consultation history")

    if task.family == "mr":
        system = (
            "请只依据提供的完整咨询历史回答事实回忆问题。不要编造材料中没有的"
            "事实、人物、数值或因果关系。证据不足时明确说材料中未明确提到。"
        )
    else:
        system = (
            "你正在完成校园心理咨询案例的时序和因果推理测试。请只依据提供的"
            "完整咨询历史，按题目要求分析事件顺序、相互影响和核心困扰的演化。"
            "不要编造事实、人物、诊断、数值或因果关系。证据不足时明确说明。"
        )
    user = f"""【task_id】
{task.id}

【完整咨询历史】
{_render_history(full_history)}

【问题】
{payload.get('question', '')}

请直接、准确作答，不要输出评分。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_esconv_target_messages(
    example: ESConvExample,
    jingshi_system_prefix: str,
) -> list[dict[str, str]]:
    strategy_json = json.dumps(STRATEGIES)
    system = f"""{jingshi_system_prefix.strip()}

You are being evaluated on the ESConv next-supporter-turn task. Select exactly one
support strategy from this JSON list: {strategy_json}
Then write the next supportive response. Return one strict JSON object only:
{{"strategy":"<exact label>","response":"<supportive response>"}}
Do not include Markdown, analysis, or any other keys.""".strip()
    transcript = "\n".join(
        f"{'Supporter' if message['role'] == 'assistant' else 'Seeker'}: {message['content']}"
        for message in example.context
    )
    user = f"Conversation so far:\n{transcript}\n\nReturn the next supporter turn."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_strategy_response(raw: str) -> StrategyResponse:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return StrategyResponse(None, "", "invalid_json")
    if not isinstance(value, dict) or set(value) != {"strategy", "response"}:
        return StrategyResponse(None, "", "invalid_schema")
    if not isinstance(value["strategy"], str) or not isinstance(value["response"], str):
        return StrategyResponse(None, "", "invalid_schema")
    strategy = _canonical_strategy(value["strategy"])
    response = value["response"].strip()
    if strategy is None:
        return StrategyResponse(None, response, "invalid_strategy")
    if not response:
        return StrategyResponse(strategy, "", "empty_response")
    return StrategyResponse(strategy, response, None)


def _f1_for_label(pairs: Sequence[tuple[str, str | None]], label: str) -> float:
    true_positive = sum(gold == label and predicted == label for gold, predicted in pairs)
    false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
    false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else 0.0


def score_strategy_predictions(
    pairs: Sequence[tuple[str, str | None]],
) -> dict[str, Any]:
    n = len(pairs)
    correct = sum(gold == predicted for gold, predicted in pairs)
    invalid = sum(predicted is None for _gold, predicted in pairs)
    per_class = {label: _f1_for_label(pairs, label) for label in STRATEGIES}
    support = {label: sum(gold == label for gold, _predicted in pairs) for label in STRATEGIES}
    macro = sum(per_class.values()) / len(STRATEGIES)
    weighted = (
        sum(per_class[label] * support[label] for label in STRATEGIES) / n if n else 0.0
    )
    confusion = {
        gold: {
            predicted: sum(g == gold and p == predicted for g, p in pairs)
            for predicted in (*STRATEGIES, "<INVALID>")
        }
        for gold in STRATEGIES
    }
    for gold in STRATEGIES:
        confusion[gold]["<INVALID>"] = sum(g == gold and p is None for g, p in pairs)
    return {
        "n": n,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class_f1": per_class,
        "support": support,
        "confusion": confusion,
    }


def normalized_quality_score(mean_score: float) -> dict[str, float]:
    if not 0 <= mean_score <= 5:
        raise ValueError("mean score must be between 0 and 5")
    return {
        "mean_score_0_to_5": mean_score,
        "normalized_quality_percent": round(mean_score * 20, 6),
    }


def stable_blind_order(task_id: str, arm_ids: Sequence[str], seed: int) -> list[str]:
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm ids must be unique")
    return sorted(
        arm_ids,
        key=lambda arm_id: hashlib.sha256(
            f"{seed}\0{task_id}\0{arm_id}".encode("utf-8")
        ).digest(),
    )


def validate_preregistration(preregistration: dict[str, Any]) -> None:
    arms = preregistration.get("target_arms") or []
    if len(arms) != 2:
        raise ValueError("exactly two target arms are required")
    arm_ids = [str(arm.get("id") or "") for arm in arms]
    arm_models = [str(arm.get("model") or "") for arm in arms]
    if not all(arm_ids) or len(set(arm_ids)) != 2:
        raise ValueError("target arm ids must be non-empty and unique")
    if not all(arm_models):
        raise ValueError("target model ids must be non-empty")
    judge_model = str((preregistration.get("judge") or {}).get("model") or "")
    if not judge_model:
        raise ValueError("an independent judge model is required")
    if judge_model.casefold() in {model.casefold() for model in arm_models}:
        raise ValueError("judge model must be independent from both target models")
    if not bool((preregistration.get("cpcd") or {}).get("include_reference_answer")):
        raise ValueError("paper-aligned CPCD judging requires the reference answer")


def build_chat_request(
    wire_api: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any]:
    if wire_api == "chat":
        return {
            "path": "/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        }
    if wire_api == "responses":
        return {
            "path": "/responses",
            "body": {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
            },
        }
    raise ValueError(f"unsupported wire API: {wire_api}")
