#!/usr/bin/env python3
"""
audit_results.py — 从逐条 JSONL 独立重算所有任务指标,与 summary 对账(固化口径)

回应复查审计建议 #1/#5:把"独立脚本重算 19/19 一致"落成可追踪脚本,并固化 suite overview。

对每个 results/*-v1.jsonl(以及 EmoBench 的独立结果):重算 n / correct / accuracy /
weighted-F1(如适用),与对应 *.summary.json 比对,输出 PASS/FAIL 表 + audit_recompute.json。

用法: python3 scripts/audit_results.py   (在 eval-suite/ 目录下)
"""
import argparse, json, os, glob, collections, sys, hashlib
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
EMO = os.path.join(HERE, "..", "..", "EmoBench", "eval", "results")

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

EVIDENCE_FIELDS = {
    "id": ("id",),
    "gold": ("gold",),
    "prediction": ("predicted", "prediction"),
    "error": ("error",),
    "requested_model": ("requested_model",),
    "response_model": ("response_model",),
    "api_model": ("api_model",),
    "provider": ("provider",),
    "system_fingerprint": ("system_fingerprint", "fingerprint"),
    "prompt_sha256": ("prompt_sha256",),
    "dataset_sha256": ("dataset_sha256",),
    "seed": ("seed",),
    "run_id": ("run_id",),
    "attempt": ("attempt",),
    "timestamp": ("timestamp",),
}

def first_value(row, candidates):
    for field in candidates:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def redacted_row_evidence(path, rows):
    """Return audit metadata only; never copy prompts, outputs, gold text, or row ids."""
    ids = [str(first_value(row, ("id",))) for row in rows if first_value(row, ("id",)) is not None]
    id_counts = collections.Counter(ids)
    field_coverage = {
        logical: sum(first_value(row, candidates) is not None for row in rows)
        for logical, candidates in EVIDENCE_FIELDS.items()
    }

    def counts(logical):
        values = collections.Counter(
            str(first_value(row, EVIDENCE_FIELDS[logical]))
            for row in rows
            if first_value(row, EVIDENCE_FIELDS[logical]) is not None
        )
        return dict(sorted(values.items()))

    return {
        "file": os.path.basename(path) if path else "<selftest>",
        "sha256": sha256_file(path) if path else None,
        "rows": len(rows),
        "unique_ids": len(id_counts),
        "duplicate_rows": sum(count - 1 for count in id_counts.values() if count > 1),
        "error_rows": sum(bool(row.get("error")) for row in rows),
        "invalid_rows": sum(bool(row.get("invalid")) or first_value(row, ("predicted", "prediction")) is None for row in rows),
        "field_coverage": field_coverage,
        "requested_models": counts("requested_model"),
        "response_models": counts("response_model"),
        "api_models": counts("api_model"),
        "providers": counts("provider"),
        "fingerprints_nonempty": field_coverage["system_fingerprint"],
    }

def f1(rows, lab):
    tp = sum(1 for r in rows if r["predicted"] == lab and r["gold"] == lab)
    fp = sum(1 for r in rows if r["predicted"] == lab and r["gold"] != lab)
    fn = sum(1 for r in rows if r["predicted"] != lab and r["gold"] == lab)
    return (2 * tp) / (2 * tp + fp + fn) if tp else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="validate published aggregate summaries without raw row-level data")
    parser.add_argument(
        "--manifest-out",
        help="write a redacted row-evidence manifest (counts, field coverage, model/provider aggregates, file hashes; never row text or outputs)",
    )
    parser.add_argument("--results-dir", default=RES, help="authorized row-level results directory (default: ./results)")
    parser.add_argument("--emobench-results-dir", default=EMO, help="authorized EmoBench row-level results directory")
    parser.add_argument("--audit-out", help="override audit JSON output path")
    args = parser.parse_args()
    if args.selftest:
        summary_dir = os.path.join(HERE, "..", "results-summary")
        files = sorted(glob.glob(os.path.join(summary_dir, "*.summary.json")))
        if not files:
            raise SystemExit("no published summaries found")
        for path in files:
            data = json.load(open(path, encoding="utf-8"))
            if not isinstance(data, (dict, list)):
                raise SystemExit(f"invalid summary JSON: {path}")
        overview = json.load(open(os.path.join(summary_dir, "all-v2.summary.json"), encoding="utf-8"))
        if len(overview) != 19:
            raise SystemExit(f"expected 19 overview tasks, got {len(overview)}")
        fixture = [{
            "id": "private-id", "gold": "a", "predicted": "a", "text": "must-not-leak",
            "raw": "must-not-leak", "requested_model": "model-a", "response_model": "model-a",
            "provider": "provider-a", "system_fingerprint": "fp-a", "run_id": "run-a",
        }]
        redacted = redacted_row_evidence(None, fixture)
        serialized = json.dumps(redacted, ensure_ascii=False)
        if "private-id" in serialized or "must-not-leak" in serialized:
            raise SystemExit("redacted evidence manifest leaks row-level content or identifiers")
        if redacted["field_coverage"]["requested_model"] != 1 or redacted["unique_ids"] != 1:
            raise SystemExit("redacted evidence manifest selftest failed")
        print(f"aggregate audit selftest PASS: {len(files)} summary files, overview=19; redacted-manifest schema PASS; raw recomputation intentionally not claimed")
        return
    results_dir = os.path.abspath(args.results_dir)
    emo_dir = os.path.abspath(args.emobench_results_dir)
    if not os.path.isdir(results_dir) or not glob.glob(os.path.join(results_dir, "*.jsonl")):
        print("Raw row-level results are not published in this repository (license/sensitivity boundary).", file=sys.stderr)
        print("Run with --selftest for aggregate schema checks, or pass --results-dir for authorized metric recomputation.", file=sys.stderr)
        raise SystemExit(2)
    report, fails = [], []
    for p in sorted(glob.glob(os.path.join(results_dir, "*-v1.jsonl"))):
        name = os.path.basename(p).replace("-v1.jsonl", "")
        rows = load(p)
        if not rows or "gold" not in rows[0]:
            continue
        n = len(rows); acc = sum(r["ok"] for r in rows) / n
        labels = sorted({r["gold"] for r in rows})
        wf1 = sum((sum(1 for r in rows if r["gold"] == lab) / n) * f1(rows, lab) for lab in labels)
        sp = p.replace(".jsonl", ".summary.json")
        s = json.load(open(sp)) if os.path.exists(sp) else {}
        ok = abs(acc - s.get("accuracy", -1)) < 1e-9 and \
             (s.get("weightedF1") is None or abs(wf1 - s["weightedF1"]) < 1e-6)
        report.append({"task": name, "n": n, "acc_recomputed": round(acc, 5),
                       "acc_summary": s.get("accuracy"), "wf1_recomputed": round(wf1, 5),
                       "wf1_summary": s.get("weightedF1"), "match": ok})
        if not ok: fails.append(name)
        print(f"[{'✓' if ok else '✗'}] {name:34s} n={n:5d} acc={acc*100:6.2f}%")
    # EmoBench (independent harness, by_language summaries)
    for p in sorted(glob.glob(os.path.join(emo_dir, "*.summary.json"))):
        s = json.load(open(p)); name = os.path.basename(p).replace(".summary.json", "")
        jl = p.replace(".summary.json", ".jsonl")
        if not os.path.exists(jl): continue
        rows = load(jl)
        by = collections.defaultdict(list)
        for r in rows: by[r["lang"]].append(r)
        ok = all(abs(sum(x["correct"] for x in v)/len(v) - s["by_language"][k]["accuracy"]) < 1e-9
                 for k, v in by.items())
        report.append({"task": "emobench/" + name, "n": len(rows), "match": ok,
                       "by_language": {k: round(sum(x["correct"] for x in v)/len(v), 4) for k, v in by.items()}})
        if not ok: fails.append(name)
        print(f"[{'✓' if ok else '✗'}] emobench/{name:24s} n={len(rows):5d}")
    evidence = [redacted_row_evidence(path, load(path)) for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))]
    outp = os.path.abspath(args.audit_out) if args.audit_out else os.path.join(results_dir, "audit_recompute.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump({"all_match": not fails, "fails": fails, "tasks": report, "row_evidence": evidence},
              open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if args.manifest_out:
        manifest_path = os.path.abspath(args.manifest_out)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "publishing_boundary": "redacted metadata only; raw benchmark rows, prompts, outputs, gold text, and row ids are excluded",
            "files": evidence,
        }
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"redacted evidence manifest -> {manifest_path}")
    print(f"\n{'全部一致 ✓' if not fails else '不一致: ' + str(fails)}  ({len(report)} 项)")
    print(f"saved -> {outp}")

if __name__ == "__main__":
    main()
