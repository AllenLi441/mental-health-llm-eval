#!/usr/bin/env python3
"""Validate and compare the four frozen PsySUICIDE validation arms.

The output is aggregate-only. It never copies counseling text, case ids,
per-case gold/prediction assignments, raw model output, or credentials into the
analysis file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_N = 1459
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260727
LABELS = [
    "与自杀/自伤/攻击行为无关",
    "被动自杀意图",
    "主动自杀意图",
    "关于自杀的探索",
    "自杀计划",
    "自杀准备行为",
    "自杀未遂",
    "自伤意图",
    "自伤行为",
    "用户攻击行为",
    "他人攻击行为",
]
ARMS = {
    "A_flash_baseline": {
        "model": "deepseek-v4-flash",
        "profile": "baseline",
        "suffix": "flash-baseline",
    },
    "B_pro_baseline": {
        "model": "deepseek-v4-pro",
        "profile": "baseline",
        "suffix": "pro-baseline",
    },
    "C_pro_taxonomy": {
        "model": "deepseek-v4-pro",
        "profile": "taxonomy",
        "suffix": "pro-taxonomy",
    },
    "D_pro_hierarchical": {
        "model": "deepseek-v4-pro",
        "profile": "hierarchical",
        "suffix": "pro-hierarchical",
    },
}
CONTRASTS = [
    ("model_effect", "A_flash_baseline", "B_pro_baseline"),
    ("taxonomy_effect", "B_pro_baseline", "C_pro_taxonomy"),
    ("hierarchical_effect", "B_pro_baseline", "D_pro_hierarchical"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path.name}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def unique(rows: list[dict], field: str) -> list:
    values = {row.get(field) for row in rows}
    return sorted(values, key=lambda value: str(value))


def validate_arm(path: Path, arm_name: str, batch_id: str) -> tuple[list[dict], dict]:
    config = ARMS[arm_name]
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"{path.name}: expected {EXPECTED_N} rows, found {len(rows)}")
    expected_run_id = f"{batch_id}-full-{config['suffix']}"
    required = {
        "id",
        "gold",
        "predicted",
        "ok",
        "invalid",
        "error",
        "split",
        "prompt_profile",
        "requested_model",
        "response_model",
        "provider",
        "fingerprint",
        "run_id",
        "seed",
        "thinking",
        "reasoning_effort",
        "case_sha256",
        "prompt_template_sha256",
        "dataset_manifest_sha256",
        "usage",
    }
    seen = set()
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise SystemExit(f"{path.name}:{index}: missing fields {missing}")
        row_id = str(row["id"])
        if row_id in seen:
            raise SystemExit(f"{path.name}:{index}: duplicate normalized id")
        seen.add(row_id)
        if row["gold"] not in LABELS:
            raise SystemExit(f"{path.name}:{index}: unknown gold label")
        if row["predicted"] is not None and row["predicted"] not in LABELS:
            raise SystemExit(f"{path.name}:{index}: unknown predicted label")
        if row["error"]:
            raise SystemExit(f"{path.name}:{index}: API error prevents complete paired analysis")
        if not isinstance(row["usage"], dict):
            raise SystemExit(f"{path.name}:{index}: missing token usage")
        if row["ok"] is not (row["predicted"] == row["gold"]):
            raise SystemExit(f"{path.name}:{index}: ok flag disagrees with exact label match")
        expected = {
            "split": "valid",
            "prompt_profile": config["profile"],
            "requested_model": config["model"],
            "response_model": config["model"],
            "provider": "deepseek",
            "run_id": expected_run_id,
            "seed": 42,
            "thinking": "enabled",
            "reasoning_effort": "high",
        }
        for field, value in expected.items():
            if row[field] != value:
                raise SystemExit(
                    f"{path.name}:{index}: {field} expected {value!r}, got {row[field]!r}"
                )
        if not row["fingerprint"]:
            raise SystemExit(f"{path.name}:{index}: missing model fingerprint")
    for field in (
        "prompt_template_sha256",
        "dataset_manifest_sha256",
        "requested_model",
        "response_model",
        "run_id",
    ):
        if len(unique(rows, field)) != 1:
            raise SystemExit(f"{path.name}: mixed {field} values")
    return rows, {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "prompt_template_sha256": unique(rows, "prompt_template_sha256")[0],
        "dataset_manifest_sha256": unique(rows, "dataset_manifest_sha256")[0],
        "fingerprints": unique(rows, "fingerprint"),
        "run_id": expected_run_id,
    }


def f1_metrics(golds: list[str], predictions: list[str | None]) -> dict:
    per_class = {}
    weighted = 0.0
    macro = 0.0
    n = len(golds)
    for label in LABELS:
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, predictions))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, predictions))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, predictions))
        support = sum(gold == label for gold in golds)
        f1 = (2 * tp / (2 * tp + fp + fn)) if tp else 0.0
        per_class[label] = {"support": support, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        macro += f1 / len(LABELS)
        weighted += (support / n) * f1
    accuracy = sum(gold == pred for gold, pred in zip(golds, predictions)) / n
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_class": per_class,
    }


def usage_cost(rows: list[dict], pricing: dict, model: str) -> dict:
    rates = pricing["models"][model]
    prompt_tokens = cache_hit = cache_miss = completion_tokens = 0
    reasoning_tokens = 0
    for row in rows:
        usage = row["usage"]
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        cache_hit += int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        cache_miss += int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
    if cache_hit + cache_miss == 0:
        cache_miss = prompt_tokens
        cache_basis = "prompt_tokens_all_priced_as_cache_miss"
    else:
        cache_basis = "provider_cache_hit_and_miss_fields"
    input_cost = (
        cache_hit * rates["input_cache_hit"] + cache_miss * rates["input_cache_miss"]
    ) / 1_000_000
    output_cost = completion_tokens * rates["output"] / 1_000_000
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_pricing_basis": cache_basis,
        "estimated_usd": input_cost + output_cost,
    }


def exact_mcnemar(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(candidate_only, reference_only) + 1)
    )
    return min(1.0, 2.0 * tail / (1 << discordant))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_metric_deltas(
    golds: list[str],
    reference: list[str | None],
    candidate: list[str | None],
    repetitions: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    n = len(golds)
    label_index = {label: index for index, label in enumerate(LABELS)}
    gold_indices = [label_index[gold] for gold in golds]
    reference_indices = [
        label_index[prediction] if prediction in label_index else -1
        for prediction in reference
    ]
    candidate_indices = [
        label_index[prediction] if prediction in label_index else -1
        for prediction in candidate
    ]
    macro_deltas = []
    weighted_deltas = []
    for _ in range(repetitions):
        support = [0] * len(LABELS)
        reference_predicted = [0] * len(LABELS)
        candidate_predicted = [0] * len(LABELS)
        reference_tp = [0] * len(LABELS)
        candidate_tp = [0] * len(LABELS)
        for _ in range(n):
            index = rng.randrange(n)
            gold = gold_indices[index]
            reference_prediction = reference_indices[index]
            candidate_prediction = candidate_indices[index]
            support[gold] += 1
            if reference_prediction >= 0:
                reference_predicted[reference_prediction] += 1
                if reference_prediction == gold:
                    reference_tp[gold] += 1
            if candidate_prediction >= 0:
                candidate_predicted[candidate_prediction] += 1
                if candidate_prediction == gold:
                    candidate_tp[gold] += 1

        def aggregate(predicted: list[int], true_positive: list[int]) -> tuple[float, float]:
            macro = weighted = 0.0
            for label in range(len(LABELS)):
                denominator = support[label] + predicted[label]
                f1 = (2 * true_positive[label] / denominator) if denominator else 0.0
                macro += f1 / len(LABELS)
                weighted += (support[label] / n) * f1
            return macro, weighted

        reference_macro, reference_weighted = aggregate(
            reference_predicted, reference_tp
        )
        candidate_macro, candidate_weighted = aggregate(
            candidate_predicted, candidate_tp
        )
        macro_deltas.append(candidate_macro - reference_macro)
        weighted_deltas.append(candidate_weighted - reference_weighted)
    return {
        "macro_f1_delta_ci95": [
            quantile(macro_deltas, 0.025),
            quantile(macro_deltas, 0.975),
        ],
        "weighted_f1_delta_ci95": [
            quantile(weighted_deltas, 0.025),
            quantile(weighted_deltas, 0.975),
        ],
        "seed": seed,
        "repetitions": repetitions,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = running
    return adjusted


def top_confusions(golds: list[str], predictions: list[str | None]) -> list[dict]:
    counts = Counter(
        (gold, pred)
        for gold, pred in zip(golds, predictions)
        if pred is not None and gold != pred
    )
    return [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in counts.most_common(10)
    ]


def analyze(
    arm_rows: dict[str, list[dict]],
    arm_meta: dict[str, dict],
    pricing: dict,
    bootstrap_reps: int,
) -> dict:
    reference_name = next(iter(ARMS))
    reference_by_id = {str(row["id"]): row for row in arm_rows[reference_name]}
    reference_ids = set(reference_by_id)
    for arm_name, rows in arm_rows.items():
        by_id = {str(row["id"]): row for row in rows}
        if set(by_id) != reference_ids:
            raise SystemExit(f"{arm_name}: result ID set differs from {reference_name}")
        for row_id, reference in reference_by_id.items():
            row = by_id[row_id]
            if row["gold"] != reference["gold"]:
                raise SystemExit(f"{arm_name}: gold mismatch on paired case")
            if row["case_sha256"] != reference["case_sha256"]:
                raise SystemExit(f"{arm_name}: case commitment mismatch on paired case")
        if arm_meta[arm_name]["dataset_manifest_sha256"] != arm_meta[reference_name][
            "dataset_manifest_sha256"
        ]:
            raise SystemExit(f"{arm_name}: dataset manifest differs from {reference_name}")

    ordered_ids = sorted(reference_ids)
    golds = [reference_by_id[row_id]["gold"] for row_id in ordered_ids]
    predictions = {
        arm_name: [
            {str(row["id"]): row for row in rows}[row_id]["predicted"]
            for row_id in ordered_ids
        ]
        for arm_name, rows in arm_rows.items()
    }
    arms = {}
    for arm_name, rows in arm_rows.items():
        metrics = f1_metrics(golds, predictions[arm_name])
        cost = usage_cost(rows, pricing, ARMS[arm_name]["model"])
        arms[arm_name] = {
            **arm_meta[arm_name],
            "model": ARMS[arm_name]["model"],
            "prompt_profile": ARMS[arm_name]["profile"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "invalid": sum(bool(row["invalid"]) for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "per_class": metrics["per_class"],
            "top_confusions": top_confusions(golds, predictions[arm_name]),
            "usage_and_cost": cost,
        }

    contrasts = {}
    raw_p_values = {}
    for contrast_index, (name, reference, candidate) in enumerate(CONTRASTS):
        reference_predictions = predictions[reference]
        candidate_predictions = predictions[candidate]
        reference_ok = [
            gold == pred for gold, pred in zip(golds, reference_predictions)
        ]
        candidate_ok = [
            gold == pred for gold, pred in zip(golds, candidate_predictions)
        ]
        candidate_only = sum(
            candidate_value and not reference_value
            for reference_value, candidate_value in zip(reference_ok, candidate_ok)
        )
        reference_only = sum(
            reference_value and not candidate_value
            for reference_value, candidate_value in zip(reference_ok, candidate_ok)
        )
        p_value = exact_mcnemar(candidate_only, reference_only)
        raw_p_values[name] = p_value
        bootstrap = bootstrap_metric_deltas(
            golds,
            reference_predictions,
            candidate_predictions,
            bootstrap_reps,
            BOOTSTRAP_SEED + contrast_index,
        )
        contrasts[name] = {
            "reference": reference,
            "candidate": candidate,
            "delta_candidate_minus_reference": {
                "accuracy": arms[candidate]["accuracy"] - arms[reference]["accuracy"],
                "macro_f1": arms[candidate]["macro_f1"] - arms[reference]["macro_f1"],
                "weighted_f1": arms[candidate]["weighted_f1"]
                - arms[reference]["weighted_f1"],
                "estimated_usd": arms[candidate]["usage_and_cost"]["estimated_usd"]
                - arms[reference]["usage_and_cost"]["estimated_usd"],
            },
            "paired_correctness": {
                "both_correct": sum(r and c for r, c in zip(reference_ok, candidate_ok)),
                "candidate_only_correct": candidate_only,
                "reference_only_correct": reference_only,
                "neither_correct": sum(
                    not r and not c for r, c in zip(reference_ok, candidate_ok)
                ),
            },
            "exact_mcnemar_two_sided_p": p_value,
            "paired_bootstrap": bootstrap,
        }
    adjusted = holm_adjust(raw_p_values)
    for name, p_value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = p_value
        contrasts[name]["accuracy_difference_after_holm_alpha_0_05"] = p_value < 0.05

    def selection_key(arm_name: str) -> tuple:
        arm = arms[arm_name]
        return (
            round(arm["macro_f1"], 3),
            round(arm["weighted_f1"], 3),
            -arm["invalid"],
            -arm["usage_and_cost"]["estimated_usd"],
        )

    ranking = sorted(ARMS, key=selection_key, reverse=True)
    return {
        "schema_version": 1,
        "scope": "PsySUICIDE official valid split; developmental selection only",
        "n": EXPECTED_N,
        "labels": len(LABELS),
        "pricing_snapshot": pricing,
        "arms": arms,
        "prespecified_contrasts": contrasts,
        "selection": {
            "winner": ranking[0],
            "ranking": ranking,
            "rule": "macro-F1 rounded to 0.001, then weighted-F1 rounded to 0.001, lower invalid, lower estimated cost",
            "confirmatory_claim_allowed": False,
            "next_gate": "prepare and commit one exact test preregistration; do not alter the winner after test predictions",
        },
        "publishing_boundary": "Aggregate development analysis only; no text, ids, per-case gold/prediction assignments, or raw output are copied.",
    }


def result_path(results_dir: Path, batch_id: str, arm_name: str) -> Path:
    config = ARMS[arm_name]
    run_id = f"{batch_id}-full-{config['suffix']}"
    return (
        results_dir
        / f"psysuicide-valid-{config['profile']}-{config['model']}-{run_id}.jsonl"
    )


def selftest() -> None:
    assert exact_mcnemar(3, 0) == 0.25
    assert exact_mcnemar(0, 0) == 1.0
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.08
    metrics = f1_metrics(LABELS, LABELS)
    assert metrics["accuracy"] == 1.0
    assert math.isclose(metrics["macro_f1"], 1.0)
    bootstrap = bootstrap_metric_deltas(
        LABELS * 2,
        LABELS * 2,
        LABELS * 2,
        repetitions=100,
        seed=1,
    )
    assert bootstrap["macro_f1_delta_ci95"] == [0.0, 0.0]
    serialized = json.dumps(
        {
            "scope": "valid",
            "arms": {"a": {"accuracy": 1.0}},
            "publishing_boundary": "aggregate only",
        }
    )
    for forbidden in ("text", "predicted", '"gold"', '"id"', "raw", "api_key"):
        assert forbidden not in serialized
    pricing = {
        "schema_version": 1,
        "currency": "USD",
        "unit": "per_1m_tokens",
        "models": {
            "deepseek-v4-flash": {
                "input_cache_hit": 0.0028,
                "input_cache_miss": 0.14,
                "output": 0.28,
            },
            "deepseek-v4-pro": {
                "input_cache_hit": 0.003625,
                "input_cache_miss": 0.435,
                "output": 0.87,
            },
        },
    }
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        batch_id = "fixture"
        arm_rows = {}
        arm_meta = {}
        wrong_modulus = {
            "A_flash_baseline": 4,
            "B_pro_baseline": 5,
            "C_pro_taxonomy": 6,
            "D_pro_hierarchical": 7,
        }
        for arm_name, config in ARMS.items():
            rows = []
            for index in range(EXPECTED_N):
                gold = LABELS[index % len(LABELS)]
                wrong = index % wrong_modulus[arm_name] == 0
                predicted = LABELS[(index + 1) % len(LABELS)] if wrong else gold
                rows.append(
                    {
                        "id": f"fixture:{index}",
                        "gold": gold,
                        "predicted": predicted,
                        "ok": not wrong,
                        "invalid": False,
                        "error": None,
                        "split": "valid",
                        "prompt_profile": config["profile"],
                        "requested_model": config["model"],
                        "response_model": config["model"],
                        "provider": "deepseek",
                        "fingerprint": "fixture-fingerprint",
                        "run_id": f"{batch_id}-full-{config['suffix']}",
                        "seed": 42,
                        "thinking": "enabled",
                        "reasoning_effort": "high",
                        "case_sha256": hashlib.sha256(
                            f"fixture:{index}".encode()
                        ).hexdigest(),
                        "prompt_template_sha256": hashlib.sha256(
                            arm_name.encode()
                        ).hexdigest(),
                        "dataset_manifest_sha256": "d" * 64,
                        "usage": {
                            "prompt_tokens": 10,
                            "prompt_cache_miss_tokens": 10,
                            "completion_tokens": 2,
                        },
                    }
                )
            path = directory / f"{arm_name}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            validated, meta = validate_arm(path, arm_name, batch_id)
            arm_rows[arm_name] = validated
            arm_meta[arm_name] = meta
        analysis = analyze(arm_rows, arm_meta, pricing, bootstrap_reps=100)
        assert analysis["selection"]["winner"] == "D_pro_hierarchical"
        assert set(analysis["prespecified_contrasts"]) == {
            "model_effect",
            "taxonomy_effect",
            "hierarchical_effect",
        }
        assert not analysis["selection"]["confirmatory_claim_allowed"]
    print(
        "PsySUICIDE valid analyzer selftest PASS: full paired validation, metrics, exact McNemar, Holm, bootstrap, selection, redaction"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--batch-id")
    parser.add_argument("--pricing", type=Path, default=ROOT / "lib" / "deepseek_v4_pricing_2026-07-27.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.batch_id:
        raise SystemExit("--batch-id is required")
    if not args.out:
        raise SystemExit("--out is required")
    if not (100 <= args.bootstrap_reps <= 100_000):
        raise SystemExit("--bootstrap-reps must be between 100 and 100000")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing analysis: {args.out}")
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    arm_rows = {}
    arm_meta = {}
    for arm_name in ARMS:
        path = result_path(args.results_dir.resolve(), args.batch_id, arm_name)
        rows, meta = validate_arm(path, arm_name, args.batch_id)
        arm_rows[arm_name] = rows
        arm_meta[arm_name] = meta
    result = analyze(arm_rows, arm_meta, pricing, args.bootstrap_reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    winner = result["selection"]["winner"]
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "winner": winner,
                "macro_f1": result["arms"][winner]["macro_f1"],
                "weighted_f1": result["arms"][winner]["weighted_f1"],
                "confirmatory_claim_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
