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

        context = []
        for role, strategy, text in parsed[:-1]:
            message = {
                "role": "assistant" if role == 1 else "user",
                "content": text,
            }
            if strategy is not None:
                message["strategy"] = strategy
            context.append(message)
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
    transcript_lines = []
    for message in example.context:
        if message["role"] == "assistant":
            strategy = message.get("strategy")
            prefix = f"[{strategy}] " if strategy else ""
            transcript_lines.append(f"Supporter: {prefix}{message['content']}")
        else:
            transcript_lines.append(f"Seeker: {message['content']}")
    transcript = "\n".join(transcript_lines)
    user = f"Conversation so far:\n{transcript}\n\nReturn the next supporter turn."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_CPCD_SCORE_SPECS = {
    "srg": ({"empathy", "coherence", "professionalism"}, 1, 5),
    "mr": (
        {"accuracy", "completeness", "temporal_consistency", "no_hallucination"},
        0,
        5,
    ),
    "tcr": (
        {"temporal_accuracy", "causal_coherence", "completeness", "no_hallucination"},
        0,
        5,
    ),
}


def build_cpcd_judge_messages(
    task: CPCDTask,
    response: str,
    full_history: Any,
    rubric: str,
    blind_id: str,
) -> list[dict[str, str]]:
    if task.family not in _CPCD_SCORE_SPECS:
        raise ValueError(f"unsupported CPCD task family: {task.family}")
    payload = task.payload
    inp = payload.get("input_to_model") or {}

    if task.family == "srg":
        system = (
            "你是严格、审慎的心理咨询回复评测员。请基于评分规则、任务焦点、"
            "学生画像、历史上下文、参考答案和匿名候选回复评分。不要因语言流畅"
            "自动高分，也不要要求候选回复逐字匹配参考答案。只输出严格 JSON。"
        )
        evidence = f"""【student_profile_summary】
{inp.get('student_profile_summary', '')}

【history_until_previous_session】
{inp.get('history_until_previous_session', '')}

【current_session_event】
{inp.get('current_session_event', '')}

【current_session_context】
{_format_context(inp.get('current_session_context', ''))}

【current_student_utterance】
{inp.get('current_student_utterance', '')}"""
        schema = (
            '{"scores":{"empathy":1,"coherence":1,"professionalism":1},'
            '"rationales":{"empathy":"...","coherence":"...",'
            '"professionalism":"..."}}'
        )
    else:
        if full_history is None:
            raise ValueError(f"{task.id} requires full history for judging")
        system = (
            "你是严格的咨询历史事实评测员。请依据评分规则、参考答案和完整咨询"
            "历史，对匿名候选答案评分。不要因文风不同扣分，只在事实错误、遗漏、"
            "顺序或因果问题、幻觉时扣分。只输出严格 JSON。"
        )
        evidence = f"""【question】
{payload.get('question', '')}

【完整咨询历史】
{_render_history(full_history)}"""
        if task.family == "mr":
            schema = (
                '{"scores":{"accuracy":0,"completeness":0,'
                '"temporal_consistency":0,"no_hallucination":0},'
                '"rationales":{"accuracy":"...","completeness":"...",'
                '"temporal_consistency":"...","no_hallucination":"..."}}'
            )
        else:
            schema = (
                '{"scores":{"temporal_accuracy":0,"causal_coherence":0,'
                '"completeness":0,"no_hallucination":0},'
                '"rationales":{"temporal_accuracy":"...",'
                '"causal_coherence":"...","completeness":"...",'
                '"no_hallucination":"..."}}'
            )

    user = f"""【匿名输出 ID】
{blind_id}

【评分规则】
{rubric}

【任务证据】
{evidence}

【evaluation_focus】
{json.dumps(payload.get('evaluation_focus', {}), ensure_ascii=False, indent=2)}

【reference_answer，高分参考方向，不要求逐字相似】
{payload.get('reference_answer', '')}

【匿名候选回复】
{response}

请只输出以下结构，所有分数必须是整数：
{schema}"""
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
        return StrategyResponse(None, "", "empty_response")
    return StrategyResponse(strategy, response, None)


def _extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge output is not valid JSON")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("judge output is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    return value


def parse_cpcd_judgement(family: str, raw: str) -> dict[str, Any]:
    try:
        keys, minimum, maximum = _CPCD_SCORE_SPECS[family]
    except KeyError as error:
        raise ValueError(f"unsupported CPCD task family: {family}") from error
    value = _extract_json_object(raw)
    raw_scores = value.get("scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != keys:
        raise ValueError(f"judge scores must contain exactly {sorted(keys)}")
    scores: dict[str, int] = {}
    for key in sorted(keys):
        score_value = raw_scores[key]
        if isinstance(score_value, dict):
            score_value = score_value.get("score")
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise ValueError(f"judge score {key} is not numeric")
        if int(score_value) != score_value:
            raise ValueError(f"judge score {key} must be an integer")
        score = int(score_value)
        if not minimum <= score <= maximum:
            raise ValueError(f"judge score {key} is outside the {minimum}-{maximum} range")
        scores[key] = score
    return {
        "scores": scores,
        "rationales": value.get("rationales") or {},
        "average_score": sum(scores.values()) / len(scores),
    }


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


def summarize_cpcd_judgements(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in ("srg", "mr", "tcr"):
        family_records = [record for record in records if record.get("family") == family]
        if not family_records:
            continue
        expected_keys = _CPCD_SCORE_SPECS[family][0]
        valid = [
            record
            for record in family_records
            if isinstance(record.get("scores"), dict)
            and set(record["scores"]) == expected_keys
        ]
        if not valid:
            continue
        metric_means = {
            key: sum(float(record["scores"][key]) for record in valid) / len(valid)
            for key in sorted(expected_keys)
        }
        item_means = [
            sum(float(value) for value in record["scores"].values())
            / len(record["scores"])
            for record in valid
        ]
        mean_score = sum(item_means) / len(item_means)
        families[family] = {
            "n": len(valid),
            "failed_or_missing": len(family_records) - len(valid),
            "metric_means_0_to_5": metric_means,
            **normalized_quality_score(mean_score),
        }
    family_means = [entry["mean_score_0_to_5"] for entry in families.values()]
    macro_family_mean = sum(family_means) / len(family_means) if family_means else 0.0
    return {
        "families": families,
        "overall": {
            "macro_family_mean_score_0_to_5": macro_family_mean,
            "macro_family_normalized_quality_percent": macro_family_mean * 20,
        },
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
    strategy_labels = tuple((preregistration.get("esconv") or {}).get("strategy_labels") or ())
    if strategy_labels and strategy_labels != STRATEGIES:
        raise ValueError("preregistered ESConv strategy labels must exactly match STRATEGIES")


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


def extract_api_response(wire_api: str, payload: dict[str, Any]) -> dict[str, Any]:
    text = ""
    termination: dict[str, Any]
    if wire_api == "chat":
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("chat response does not contain message content") from error
        status = choice.get("finish_reason")
        termination = {
            "termination_status": status,
            "termination_details": None,
            "finish_reason": status,
        }
    elif wire_api == "responses":
        if isinstance(payload.get("output_text"), str):
            text = payload["output_text"]
        else:
            chunks = []
            for item in payload.get("output") or []:
                if item.get("type") != "message":
                    continue
                for content in item.get("content") or []:
                    if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                        chunks.append(content["text"])
            text = "".join(chunks)
        status = payload.get("status")
        details = payload.get("incomplete_details")
        termination = {
            "termination_status": status,
            "termination_details": details,
            "response_status": status,
            "incomplete_details": details,
        }
    else:
        raise ValueError(f"unsupported wire API: {wire_api}")
    if not text.strip():
        raise ValueError("API response contains no answer text")
    return {
        "text": text.strip(),
        "response_model": payload.get("model"),
        "fingerprint": payload.get("system_fingerprint"),
        "usage": payload.get("usage") or {},
        "response_id": payload.get("id"),
        **termination,
    }


def read_jsonl_index(
    path: Path,
    id_key: str = "run_id",
    *,
    recover_truncated_tail: bool = False,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    raw_lines = path.read_bytes().splitlines(keepends=True)
    offset = 0
    for line_index, raw_line in enumerate(raw_lines):
        line_number = line_index + 1
        final_unterminated_line = (
            line_index == len(raw_lines) - 1
            and not raw_line.endswith((b"\n", b"\r"))
        )
        try:
            line = raw_line.decode("utf-8")
            record = json.loads(line) if line.strip() else None
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if recover_truncated_tail and final_unterminated_line:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        offset += len(raw_line)
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError(f"non-object JSONL record at {path}:{line_number}")
        run_id = str(record.get(id_key) or "")
        if not run_id:
            raise ValueError(f"missing {id_key} at {path}:{line_number}")
        if run_id in index:
            raise ValueError(f"duplicate {id_key} {run_id!r} in {path}")
        index[run_id] = record
    return index


def write_jsonl_record(path: Path, record: dict[str, Any], id_key: str = "run_id") -> None:
    run_id = str(record.get(id_key) or "")
    if not run_id:
        raise ValueError(f"record is missing {id_key}")
    existing = read_jsonl_index(
        path, id_key=id_key, recover_truncated_tail=True
    )
    if run_id in existing:
        raise ValueError(f"duplicate {id_key} {run_id!r} in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with path.open("ab+") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size:
            handle.seek(-1, 2)
            if handle.read(1) not in (b"\n", b"\r"):
                handle.write(b"\n")
        handle.write(encoded + b"\n")


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def response_overlap_metrics_tokenized(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
) -> dict[str, float]:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must have the same non-zero length")
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as error:
        raise RuntimeError("NLTK is required for official ESConv BLEU") from error
    corpus_refs = [[list(reference)] for reference in references]
    metrics: dict[str, float] = {}
    for k in range(1, 5):
        weights = tuple([1 / k] * k + [0.0] * (4 - k))
        metrics[f"bleu_{k}"] = (
            corpus_bleu(
                corpus_refs,
                hypotheses,
                weights=weights,
                smoothing_function=SmoothingFunction().method3,
            )
            * 100
        )

    rouge_scores = []
    beta = 1.2
    for reference, hypothesis in zip(references, hypotheses):
        lcs = _lcs_length(reference, hypothesis)
        precision = lcs / len(hypothesis) if hypothesis else 0.0
        recall = lcs / len(reference) if reference else 0.0
        if precision and recall:
            rouge_scores.append(
                ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
            )
        else:
            rouge_scores.append(0.0)
    metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores) * 100

    for k in range(1, 4):
        ngrams: set[tuple[str, ...]] = set()
        total = 0
        for hypothesis in hypotheses:
            # Reproduce the released ESConv implementation, including its exclusive
            # upper bound (the final possible n-gram is not counted).
            for index in range(max(0, len(hypothesis) - k)):
                ngrams.add(tuple(hypothesis[index : index + k]))
                total += 1
        metrics[f"distinct_{k}"] = (len(ngrams) / total * 100) if total else 0.0
    metrics["mean_length_tokens"] = sum(len(hypothesis) for hypothesis in hypotheses) / len(hypotheses)
    return metrics
