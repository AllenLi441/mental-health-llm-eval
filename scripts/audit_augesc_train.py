#!/usr/bin/env python3
"""Create a reproducible, training-oriented quality audit for AugESC train."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9']+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

TOPIC_PATTERNS = {
    "grief_loss": re.compile(r"\b(died|death|dead|passed away|loss|lost my|grief|funeral)\b"),
    "family_relationship": re.compile(r"\b(parent|mother|father|mom|dad|wife|husband|girlfriend|boyfriend|daughter|son|family|marriage|divorce|friend)\b"),
    "work_school": re.compile(r"\b(work|job|boss|coworker|school|college|student|exam|test|grade|homework|class)\b"),
    "finance_housing": re.compile(r"\b(money|rent|loan|debt|bill|financial|afford|landlord|house|apartment|mortgage)\b"),
    "health_medical": re.compile(r"\b(doctor|hospital|pain|sick|illness|medication|medicine|pills?|injury|health|stomach)\b"),
    "anxiety_mood": re.compile(r"\b(anxiety|anxious|depress|depressed|stress|stressed|sad|worried|nervous|panic|worthless)\b"),
    "social_loneliness": re.compile(r"\b(lonely|alone|no friends|isolated|social|talk to|miss having)\b"),
    "pets_animals": re.compile(r"\b(dog|cat|pet|puppy|kitten|animal)\b"),
    "accident_safety": re.compile(r"\b(accident|wreck|crash|robbed|robbery|fire|tornado|snake|danger|scared|attack)\b"),
}

SAFETY_PATTERNS = {
    "self_harm_or_suicide": re.compile(r"\b(suicid|kill myself|end my life|want to die|self[- ]?harm|not worth living)\b"),
    "violence_or_crime": re.compile(r"\b(kill|murder|stab|shoot|weapon|gun|robbed|steal|stole|crime|assault|rage)\b"),
    "substance_use": re.compile(r"\b(alcohol|drunk|drinking|drug|cocaine|heroin|overdose)\b"),
    "medical_risk": re.compile(r"\b(doctor|hospital|medication|medicine|pills?|dose|diagnos|severe pain)\b"),
    "abuse_or_coercion": re.compile(r"\b(abuse|abusive|rape|raped|molest|coerc|domestic violence)\b"),
}

SUSPICIOUS_SYSTEM_PATTERNS = {
    "blanket_approval": re.compile(r"^(that's|that is|its|it's) (okay|ok|fine)[.! ]*$"),
    "dismissive": re.compile(r"\b(just get over it|not a big deal|you are overreacting|stop worrying)\b"),
    "medical_certainty": re.compile(r"\b(you (definitely|clearly) have|you are diagnosed with|stop taking your medication)\b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip().lower())


def normalized_key(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", normalize_text(text)).strip()


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(normalize_text(text)))


def percentile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def simhash64(tokens: set[str]) -> int:
    vector = [0] * 64
    for token in tokens:
        value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def candidate_near_duplicate_pairs(starters: list[str]) -> tuple[int, int]:
    token_sets = [set(TOKEN_RE.findall(value)) for value in starters]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    signatures = [simhash64(tokens) for tokens in token_sets]
    for index, signature in enumerate(signatures):
        for band in range(4):
            buckets[(band, (signature >> (band * 16)) & 0xFFFF)].append(index)
    candidates: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) > 300:
            continue
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                candidates.add((left, right))
    matches = 0
    affected: set[int] = set()
    for left, right in candidates:
        a, b = token_sets[left], token_sets[right]
        if not a or not b:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity >= 0.90 and starters[left] != starters[right]:
            matches += 1
            affected.update((left, right))
    return matches, len(affected)


def parse_dialogue(raw: Any) -> list[list[str]]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, list):
        raise ValueError("dialogue is not a list")
    parsed: list[list[str]] = []
    for turn in value:
        if not isinstance(turn, (list, tuple)) or len(turn) != 2:
            raise ValueError("turn is not [role, text]")
        parsed.append([str(turn[0]), str(turn[1])])
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if "text" not in frame.columns:
        raise ValueError(f"expected a text column, found {list(frame.columns)}")

    records: list[dict[str, Any]] = []
    exact_dialogue_first: dict[str, int] = {}
    normalized_dialogue_first: dict[str, int] = {}
    starter_groups: dict[str, list[int]] = defaultdict(list)
    topic_counts = Counter()
    safety_counts = Counter()
    issue_counts = Counter()
    turn_counts: list[int] = []
    user_turn_counts: list[int] = []
    system_turn_counts: list[int] = []
    word_counts: list[int] = []
    starters: list[str] = []
    sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, raw in enumerate(frame["text"].tolist()):
        conversation_id = f"augesc-train-{index + 1:06d}"
        flags: set[str] = set()
        dialogue: list[list[str]] = []
        try:
            dialogue = parse_dialogue(raw)
        except Exception:
            flags.add("parse_error")

        roles = [role for role, _ in dialogue]
        texts = [text for _, text in dialogue]
        canonical = json.dumps(dialogue, ensure_ascii=False, separators=(",", ":"))
        normalized_dialogue = json.dumps(
            [[role.lower().strip(), normalized_key(text)] for role, text in dialogue],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        starter = next((text for role, text in dialogue if role == "usr"), "")
        starter_key = normalized_key(starter)
        starters.append(starter_key)
        starter_hash = hashlib.sha256(starter_key.encode()).hexdigest()[:16]
        starter_groups[starter_hash].append(index)

        if not dialogue:
            flags.add("empty_dialogue")
        if any(role not in {"usr", "sys"} for role in roles):
            flags.add("invalid_role")
        if roles and roles[0] != "usr":
            flags.add("does_not_start_with_usr")
        if roles and roles[-1] != "sys":
            flags.add("does_not_end_with_sys")
        if any(left == right for left, right in zip(roles, roles[1:])):
            flags.add("consecutive_same_role")
        if any(not normalize_text(text) for text in texts):
            flags.add("empty_utterance")
        if any(role == "sys" and token_count(text) <= 3 for role, text in dialogue):
            flags.add("very_short_system_turn")

        if canonical in exact_dialogue_first:
            flags.add("exact_dialogue_duplicate")
        else:
            exact_dialogue_first[canonical] = index
        if normalized_dialogue in normalized_dialogue_first:
            flags.add("normalized_dialogue_duplicate")
        else:
            normalized_dialogue_first[normalized_dialogue] = index

        joined = normalize_text(" ".join(texts))
        starter_joined = normalize_text(starter)
        matched_topics = [name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(joined)]
        if not matched_topics:
            matched_topics = ["other"]
        topic_counts.update(matched_topics)

        matched_safety = [name for name, pattern in SAFETY_PATTERNS.items() if pattern.search(joined)]
        safety_counts.update(matched_safety)
        if matched_safety:
            flags.add("safety_review_candidate")

        suspicious_system = []
        for role, text in dialogue:
            if role != "sys":
                continue
            for name, pattern in SUSPICIOUS_SYSTEM_PATTERNS.items():
                if pattern.search(normalize_text(text)):
                    suspicious_system.append(name)
        if suspicious_system:
            flags.add("suspicious_system_response")

        hard_exclude = bool(
            flags
            & {
                "parse_error",
                "empty_dialogue",
                "invalid_role",
                "empty_utterance",
                "exact_dialogue_duplicate",
                "normalized_dialogue_duplicate",
            }
        )
        review_required = bool(
            flags
            & {
                "does_not_start_with_usr",
                "does_not_end_with_sys",
                "consecutive_same_role",
                "very_short_system_turn",
                "safety_review_candidate",
                "suspicious_system_response",
            }
        )

        turn_counts.append(len(dialogue))
        user_turn_counts.append(sum(role == "usr" for role in roles))
        system_turn_counts.append(sum(role == "sys" for role in roles))
        word_counts.append(sum(token_count(text) for text in texts))
        issue_counts.update(flags)

        record = {
            "conversation_id": conversation_id,
            "source": "AugESC",
            "original_split": "train",
            "language": "en",
            "starter_group_id": f"starter-{starter_hash}",
            "turn_count": len(dialogue),
            "user_turn_count": user_turn_counts[-1],
            "system_turn_count": system_turn_counts[-1],
            "word_count": word_counts[-1],
            "topics": matched_topics,
            "safety_flags": matched_safety,
            "quality_flags": sorted(flags),
            "hard_exclude": hard_exclude,
            "review_required": review_required,
            "allowed_for_training_preliminary": not hard_exclude,
            "first_user_message": starter,
        }
        records.append(record)
        for flag in flags:
            if len(sample_rows[flag]) < 25:
                sample_rows[flag].append(record)

    duplicate_starter_groups = [indices for indices in starter_groups.values() if len(indices) > 1]
    near_pair_count, near_affected_count = candidate_near_duplicate_pairs(starters)

    for record in records:
        group_size = len(starter_groups[record["starter_group_id"].removeprefix("starter-")])
        record["starter_group_size"] = group_size
        if group_size > 1:
            record["quality_flags"] = sorted(set(record["quality_flags"]) | {"repeated_starter_group"})
            record["review_required"] = True

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "audit_id": "augesc-train-quality-audit-v1-20260813",
        "created_date": "2026-08-13",
        "source": {
            "dataset": "thu-coai/augesc",
            "split": "train",
            "url": "https://huggingface.co/datasets/thu-coai/augesc",
            "parquet_url": "https://huggingface.co/api/datasets/thu-coai/augesc/parquet/default/train/0.parquet",
            "license": "CC BY-NC 4.0",
            "input_path": str(args.input.resolve()),
            "input_sha256": sha256_file(args.input),
        },
        "scope": "Automated first-pass audit; safety and semantic coherence flags are review candidates, not validated error rates.",
        "counts": {
            "conversations": len(records),
            "utterances": sum(turn_counts),
            "user_turns": sum(user_turn_counts),
            "system_turns": sum(system_turn_counts),
            "hard_exclude": sum(record["hard_exclude"] for record in records),
            "review_required": sum(record["review_required"] for record in records),
            "preliminary_train_allowed": sum(record["allowed_for_training_preliminary"] for record in records),
        },
        "length_distribution": {
            "turns_min": min(turn_counts),
            "turns_median": statistics.median(turn_counts),
            "turns_mean": statistics.fmean(turn_counts),
            "turns_p90": percentile(turn_counts, 0.90),
            "turns_p95": percentile(turn_counts, 0.95),
            "turns_max": max(turn_counts),
            "words_median": statistics.median(word_counts),
            "words_mean": statistics.fmean(word_counts),
            "words_p95": percentile(word_counts, 0.95),
            "words_max": max(word_counts),
        },
        "duplicates": {
            "exact_dialogue_duplicates": issue_counts["exact_dialogue_duplicate"],
            "normalized_dialogue_duplicates": issue_counts["normalized_dialogue_duplicate"],
            "unique_starter_groups": len(starter_groups),
            "repeated_starter_groups": len(duplicate_starter_groups),
            "conversations_in_repeated_starter_groups": sum(len(group) for group in duplicate_starter_groups),
            "approx_near_duplicate_starter_pairs_jaccard_gte_0_90": near_pair_count,
            "approx_near_duplicate_starter_conversations": near_affected_count,
            "split_rule": "Assign starter_group_id as an indivisible group; never random-split individual conversations.",
        },
        "quality_flag_counts": dict(sorted(issue_counts.items())),
        "topic_counts_multilabel": dict(sorted(topic_counts.items())),
        "safety_candidate_counts_multilabel": dict(sorted(safety_counts.items())),
        "sample_rows_by_flag": dict(sample_rows),
        "filter_policy": {
            "automatic_exclude": [
                "parse_error",
                "empty_dialogue",
                "invalid_role",
                "empty_utterance",
                "duplicate copies after the first canonical record",
            ],
            "manual_review": [
                "role sequence anomalies",
                "very short system turns",
                "safety-sensitive conversations",
                "suspicious system responses",
                "repeated or near-duplicate starter groups",
            ],
            "warning": "Do not treat automated lexical safety flags as confirmed unsafe content.",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
