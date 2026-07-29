#!/usr/bin/env python3
"""Build a redacted CPsyExam V4 paired-outcome release and fixed inference.

The public rows contain commitments and binary outcomes only. They never copy
questions, options, labels, predictions, raw model text, source ids, or keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_N = 3902
ALPHA = 0.05
EQUIVALENCE_MARGIN = 0.02
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPS = 20_000


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_cases(dataset_root: Path, revision: str) -> list[dict]:
    data_dir = dataset_root / "CPsyExam" / "data" / "extracted_with_answer" / "test"
    if not data_dir.is_dir():
        raise SystemExit(f"CPsyExam test directory not found under {dataset_root}")
    occurrences: Counter[str] = Counter()
    cases = []
    for path in sorted(data_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit(f"{path}: expected a JSON array")
        relative_file = path.relative_to(dataset_root).as_posix()
        for row_index, question in enumerate(payload):
            source_id = str(question.get("id"))
            occurrences[source_id] += 1
            occurrence = occurrences[source_id]
            normalized_id = source_id if occurrence == 1 else f"{source_id}#dup{occurrence}"
            letters = re.findall(r"[A-E]", str(question.get("answer", "")).upper())
            if not letters:
                raise SystemExit(f"{relative_file}:{row_index}: answer has no A-E label")
            gold = "".join(sorted(set(letters)))
            qtype = "single" if question.get("question_type") == "single" else "multiple"
            kind = question.get("kind")
            group = f"{'KG' if kind == 'knowledge' else 'CA'}-{'SCQ' if qtype == 'single' else 'MAQ'}"
            commitment_payload = {
                "dataset": "CPsyExam",
                "dataset_revision": revision,
                "relative_file": relative_file,
                "row_index": row_index,
                "row": question,
            }
            cases.append(
                {
                    "id": normalized_id,
                    "gold": gold,
                    "group": group,
                    "case_sha256": sha256_bytes(canonical_bytes(commitment_payload)),
                }
            )
    if len(cases) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} CPsyExam cases, found {len(cases)}")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("normalized CPsyExam ids are not unique")
    commitments = [case["case_sha256"] for case in cases]
    if len(set(commitments)) != len(commitments):
        raise SystemExit("canonical CPsyExam case commitments are not unique")
    return cases


def validate_arm(path: Path, expected_model: str, cases: list[dict]) -> tuple[dict[str, dict], dict]:
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"{path.name}: expected {EXPECTED_N} rows, found {len(rows)}")
    case_by_id = {case["id"]: case for case in cases}
    by_id = {}
    required = {
        "id",
        "gold",
        "ok",
        "invalid",
        "error",
        "requested_model",
        "response_model",
        "provider",
        "fingerprint",
        "run_id",
        "attempt",
        "request_started_at",
        "completed_at",
        "case_sha256",
        "prompt_template_sha256",
        "dataset_manifest_sha256",
        "usage",
    }
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            raise SystemExit(f"{path.name}:{index}: missing provenance fields {missing}")
        row_id = str(row["id"])
        if row_id in by_id:
            raise SystemExit(f"{path.name}: duplicate normalized id at row {index}")
        case = case_by_id.get(row_id)
        if not case:
            raise SystemExit(f"{path.name}:{index}: result id absent from authorized dataset")
        if str(row["gold"]) != case["gold"]:
            raise SystemExit(f"{path.name}:{index}: gold mismatch against authorized dataset")
        if row["requested_model"] != expected_model or row["response_model"] != expected_model:
            raise SystemExit(
                f"{path.name}:{index}: model identity mismatch "
                f"({row['requested_model']!r} -> {row['response_model']!r})"
            )
        if row["provider"] != "deepseek":
            raise SystemExit(f"{path.name}:{index}: unexpected provider {row['provider']!r}")
        if not row["fingerprint"]:
            raise SystemExit(f"{path.name}:{index}: missing model fingerprint")
        if not isinstance(row["usage"], dict):
            raise SystemExit(f"{path.name}:{index}: missing token usage object")
        by_id[row_id] = row
    expected_ids = set(case_by_id)
    if set(by_id) != expected_ids:
        raise SystemExit(f"{path.name}: result id set is not the complete dataset id set")

    def unique(field):
        values = sorted({row.get(field) for row in rows if row.get(field) is not None})
        return values

    meta = {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "correct": sum(bool(row["ok"]) for row in rows),
        "invalid": sum(bool(row["invalid"]) for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "requested_models": unique("requested_model"),
        "response_models": unique("response_model"),
        "providers": unique("provider"),
        "fingerprints": unique("fingerprint"),
        "run_ids": unique("run_id"),
        "prompt_template_sha256": unique("prompt_template_sha256"),
        "dataset_manifest_sha256": unique("dataset_manifest_sha256"),
        "thinking": unique("thinking"),
        "reasoning_effort": unique("reasoning_effort"),
        "first_request_started_at": min(row["request_started_at"] for row in rows),
        "last_completed_at": max(row["completed_at"] for row in rows),
    }
    if len(meta["prompt_template_sha256"]) != 1 or len(meta["dataset_manifest_sha256"]) != 1:
        raise SystemExit(f"{path.name}: run mixes prompt or dataset manifests")
    return by_id, meta


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * (tail / (1 << discordant)))


def paired_normal(differences: list[int]) -> dict:
    n = len(differences)
    mean = statistics.fmean(differences)
    variance = statistics.variance(differences) if n > 1 else 0.0
    standard_error = math.sqrt(variance / n)
    normal = statistics.NormalDist()

    def interval(z):
        return [mean - z * standard_error, mean + z * standard_error]

    ci95 = interval(normal.inv_cdf(0.975))
    ci90 = interval(normal.inv_cdf(0.95))
    if standard_error == 0:
        p_lower = 0.0 if mean > -EQUIVALENCE_MARGIN else 1.0
        p_upper = 0.0 if mean < EQUIVALENCE_MARGIN else 1.0
    else:
        z_lower = (mean + EQUIVALENCE_MARGIN) / standard_error
        z_upper = (mean - EQUIVALENCE_MARGIN) / standard_error
        p_lower = 1.0 - normal.cdf(z_lower)
        p_upper = normal.cdf(z_upper)
    return {
        "delta": mean,
        "standard_error": standard_error,
        "ci95": ci95,
        "ci90": ci90,
        "tost": {
            "margin": EQUIVALENCE_MARGIN,
            "alpha": ALPHA,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "equivalent": p_lower < ALPHA and p_upper < ALPHA,
        },
    }


def quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(differences: list[int], reps: int) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(differences)
    estimates = []
    for _ in range(reps):
        estimates.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def analyze(cases: list[dict], pro: dict[str, dict], flash: dict[str, dict], bootstrap_reps: int) -> tuple[dict, list[dict]]:
    public_rows = []
    differences = []
    both_correct = pro_only = flash_only = neither = 0
    groups = {}
    for case in cases:
        pro_row = pro[case["id"]]
        flash_row = flash[case["id"]]
        pro_ok = bool(pro_row["ok"])
        flash_ok = bool(flash_row["ok"])
        differences.append(int(pro_ok) - int(flash_ok))
        if pro_ok and flash_ok:
            both_correct += 1
        elif pro_ok:
            pro_only += 1
        elif flash_ok:
            flash_only += 1
        else:
            neither += 1
        group = groups.setdefault(case["group"], {"n": 0, "v4_pro_correct": 0, "v4_flash_correct": 0})
        group["n"] += 1
        group["v4_pro_correct"] += int(pro_ok)
        group["v4_flash_correct"] += int(flash_ok)
        public_rows.append(
            {
                "case_sha256": case["case_sha256"],
                "group": case["group"],
                "v4_pro": {
                    "ok": pro_ok,
                    "invalid": bool(pro_row["invalid"]),
                    "error": bool(pro_row["error"]),
                    "requested_model": pro_row["requested_model"],
                    "response_model": pro_row["response_model"],
                    "fingerprint": pro_row["fingerprint"],
                },
                "v4_flash": {
                    "ok": flash_ok,
                    "invalid": bool(flash_row["invalid"]),
                    "error": bool(flash_row["error"]),
                    "requested_model": flash_row["requested_model"],
                    "response_model": flash_row["response_model"],
                    "fingerprint": flash_row["fingerprint"],
                },
            }
        )
    public_rows.sort(key=lambda row: row["case_sha256"])
    normal = paired_normal(differences)
    bootstrap_ci = paired_bootstrap(differences, bootstrap_reps)
    mcnemar_p = exact_mcnemar(pro_only, flash_only)
    api_errors = sum(row["v4_pro"]["error"] or row["v4_flash"]["error"] for row in public_rows)
    if api_errors:
        difference_claim = "NOT_EVALUABLE_API_ERRORS"
        equivalence_claim = "NOT_EVALUABLE_API_ERRORS"
    else:
        if mcnemar_p < ALPHA and normal["ci95"][0] > 0 and bootstrap_ci[0] > 0:
            difference_claim = "V4_PRO_SUPERIOR"
        elif mcnemar_p < ALPHA and normal["ci95"][1] < 0 and bootstrap_ci[1] < 0:
            difference_claim = "V4_FLASH_SUPERIOR"
        else:
            difference_claim = "NO_DETECTED_DIFFERENCE_NOT_A_TIE"
        equivalence_claim = "EQUIVALENT_WITHIN_2PP" if normal["tost"]["equivalent"] else "EQUIVALENCE_NOT_DEMONSTRATED"
    for values in groups.values():
        values["v4_pro_accuracy"] = values["v4_pro_correct"] / values["n"]
        values["v4_flash_accuracy"] = values["v4_flash_correct"] / values["n"]
        values["delta"] = values["v4_pro_accuracy"] - values["v4_flash_accuracy"]
    inference = {
        "n": len(cases),
        "v4_pro_correct": both_correct + pro_only,
        "v4_flash_correct": both_correct + flash_only,
        "v4_pro_accuracy": (both_correct + pro_only) / len(cases),
        "v4_flash_accuracy": (both_correct + flash_only) / len(cases),
        "contingency": {
            "both_correct": both_correct,
            "v4_pro_only_correct_b": pro_only,
            "v4_flash_only_correct_c": flash_only,
            "neither_correct": neither,
        },
        "delta_v4_pro_minus_flash": normal["delta"],
        "paired_normal_ci95": normal["ci95"],
        "paired_bootstrap_ci95": bootstrap_ci,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": bootstrap_reps},
        "exact_mcnemar": {"two_sided_p": mcnemar_p, "alpha": ALPHA},
        "equivalence_tost": normal["tost"],
        "difference_claim": difference_claim,
        "equivalence_claim": equivalence_claim,
        "groups_descriptive_only": dict(sorted(groups.items())),
    }
    return inference, public_rows


def write_release(args, cases, inference, public_rows, pro_meta, flash_meta):
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "cpsyexam-v4-paired-row-outcomes.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_rows), encoding="utf-8")

    harness_root = args.harness_root.resolve()
    protocol_files = [
        harness_root / "lib.mjs",
        harness_root / "tasks" / "cpsyexam.mjs",
        harness_root / "scripts" / "cpsyexam_paired_inference.py",
        harness_root / "reports" / "cpsyexam_v4_full_preregistration.md",
    ]
    for path in protocol_files:
        if not path.is_file():
            raise SystemExit(f"protocol file missing: {path}")
    dataset_commitment = sha256_bytes("\n".join(case["case_sha256"] for case in cases).encode("ascii"))
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": args.preregistration_url,
        "runner_commit": args.runner_commit,
        "dataset": {
            "name": "CPsyExam",
            "revision": args.dataset_revision,
            "license": "CC-BY-4.0",
            "source": "https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyExam",
            "rows": len(cases),
            "public_case_commitment_manifest_sha256": dataset_commitment,
        },
        "arms": {"v4_pro": pro_meta, "v4_flash": flash_meta},
        "inference": inference,
        "publishing_boundary": "No question, option, gold, prediction, raw output, source id, API key, or key fingerprint is published.",
        "protocol_files": {path.relative_to(harness_root).as_posix(): sha256_file(path) for path in protocol_files},
    }
    summary_path = out_dir / "cpsyexam-v4-paired-analysis.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    notice = """# CPsyExam attribution and release boundary

CPsyExam is attributed to CAS-SIAT-XinHai and its contributors. Dataset card:
https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyExam

This release treats the dataset as CC BY 4.0. It publishes only SHA-256 case
commitments and binary model outcomes. It excludes source questions, options,
answers, predictions, raw model outputs, source IDs, and credentials.

Benchmark results are research measurements, not clinical validation or medical advice.
"""
    (out_dir / "NOTICE.md").write_text(notice, encoding="utf-8")
    readme = """# CPsyExam V4 full paired outcome artifact

This package contains 3,902 redacted paired row outcomes for DeepSeek V4-Pro
(thinking/high) and V4-Flash (non-thinking), the preregistered inference, source
and private-input hashes, licensing notice, and SHA-256 checksums.

Recompute the contingency table and tests from the JSONL `ok` flags. Authorized
dataset holders can independently recreate the case commitments using the
registered builder and the exact dataset revision. No source text or answer is
included here.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest_payload = {
        "schema_version": 1,
        "runner_commit": args.runner_commit,
        "preregistration": args.preregistration_url,
        "private_inputs": {
            "v4_pro": {"filename": pro_meta["file"], "sha256": pro_meta["sha256"]},
            "v4_flash": {"filename": flash_meta["file"], "sha256": flash_meta["sha256"]},
        },
        "dataset_revision": args.dataset_revision,
        "dataset_commitment_manifest_sha256": dataset_commitment,
        "public_assets": {},
    }
    for filename in ["cpsyexam-v4-paired-row-outcomes.jsonl", "cpsyexam-v4-paired-analysis.json", "NOTICE.md", "README.md"]:
        path = out_dir / filename
        manifest_payload["public_assets"][filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    checksum_files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    checksums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files)
    (out_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "inference": inference}, ensure_ascii=False, indent=2))


def selftest():
    assert exact_mcnemar(3, 0) == 0.25
    assert exact_mcnemar(0, 0) == 1.0
    normal = paired_normal([0, 0, 0, 0])
    assert normal["tost"]["equivalent"]
    row = {
        "case_sha256": "0" * 64,
        "group": "KG-SCQ",
        "v4_pro": {"ok": True, "invalid": False, "error": False},
        "v4_flash": {"ok": False, "invalid": False, "error": False},
    }
    serialized = json.dumps(row)
    for forbidden in ("question", "options", "gold", "predicted", "raw", "sourceId", "api_key"):
        assert forbidden not in serialized
    print("cpsyexam paired inference selftest PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--pro-results", type=Path)
    parser.add_argument("--flash-results", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--harness-root", type=Path)
    parser.add_argument("--runner-commit")
    parser.add_argument("--preregistration-url")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    required = [
        "pro_results",
        "flash_results",
        "dataset_root",
        "dataset_revision",
        "harness_root",
        "runner_commit",
        "preregistration_url",
        "out_dir",
    ]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.runner_commit):
        raise SystemExit("--runner-commit must be a full 40-character lowercase Git SHA")
    if args.bootstrap_reps != BOOTSTRAP_REPS:
        raise SystemExit(f"confirmatory run requires exactly {BOOTSTRAP_REPS} bootstrap repetitions")
    cases = load_cases(args.dataset_root.resolve(), args.dataset_revision)
    pro, pro_meta = validate_arm(args.pro_results.resolve(), "deepseek-v4-pro", cases)
    flash, flash_meta = validate_arm(args.flash_results.resolve(), "deepseek-v4-flash", cases)
    inference, public_rows = analyze(cases, pro, flash, args.bootstrap_reps)
    write_release(args, cases, inference, public_rows, pro_meta, flash_meta)


if __name__ == "__main__":
    main()
