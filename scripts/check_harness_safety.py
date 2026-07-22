#!/usr/bin/env python3
"""Deterministic safety/portability gate for the public benchmark harness."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def text(relative):
    return (ROOT / relative).read_text(encoding="utf-8")

lib = text("lib.mjs")
required = [
    "const apiKey = pick('EVAL_API_KEY')",
    "credentialScope: 'EVAL_API_KEY'",
    "resume file contains duplicate ids",
    "EVAL_IGNORE_DOTENV",
    "fatal HTTP ${res.status}; run aborted before recording this case",
    "requested_model: cfg.model",
    "dataset_manifest_sha256",
    "prompt_template_sha256",
]
for marker in required:
    if marker not in lib:
        raise SystemExit(f"lib.mjs missing safety marker: {marker}")

code_files = [ROOT / "lib.mjs", ROOT / "contamination_probe.py", ROOT / "emobench-official" / "eval.mjs"]
for path in code_files:
    body = path.read_text(encoding="utf-8")
    for forbidden in ('process.env.DEEPSEEK_API_KEY', 'os.environ["DEEPSEEK_API_KEY"]'):
        if forbidden in body:
            raise SystemExit(f"{path.relative_to(ROOT)} reads production credential: {forbidden}")
if "EMOBENCH_DATA_DIR" not in text("emobench-official/eval.mjs"):
    raise SystemExit("official EmoBench harness has no portable data-root override")

runtime_files = [ROOT / "lib.mjs", ROOT / "run.mjs", *sorted((ROOT / "tasks").glob("*.mjs"))]
for path in runtime_files:
    body = path.read_text(encoding="utf-8")
    if "/Users/allenli/Desktop/静室" in body or "../静室" in body:
        raise SystemExit(f"{path.relative_to(ROOT)} contains a private app path")

readme = text("README.md")
if "多数类 72.4" in readme or "只统计本次新跑" in readme:
    raise SystemExit("README contains a stale baseline or resume claim")
if "EVAL_MODEL=deepseek-chat" in text(".env.example"):
    raise SystemExit(".env.example uses the deprecated deepseek-chat alias")
scoreboard = text("scripts/scoreboard.py")
if "赢过" in scoreboard or "同口径对照" in scoreboard:
    raise SystemExit("scoreboard contains an unsupported win/same-protocol claim")

emo_spec = text("emobench-official/SPEC.md")
if "faithful reproduction" in emo_spec or "Reproduce the OFFICIAL protocol exactly" in emo_spec:
    raise SystemExit("EmoBench deterministic proxy is mislabeled as an exact paper-protocol reproduction")

print("harness safety check PASS: dedicated credentials, safe resume, portable runtime, honest claims")
