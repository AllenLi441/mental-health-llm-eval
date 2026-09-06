#!/usr/bin/env python3
"""Materialize verified SDDP/ERC features for canonical ESConv train/dev.

Without ``--execute`` this command performs only the canonical data audit and
prints the planned unique feature count.  It has no test split or generic
dataset argument.  Execution writes a new, non-overwriting run directory only
after the full feature table and all hashes validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_response_eval import esconv_emodynamix_features as FEATURES  # noqa: E402
from scripts import train_esconv_emodynamix_clean as DATA  # noqa: E402


DEFAULT_UPSTREAM_REPO = ROOT / "tmp/official_benchmarks/emodynamix-v2"
DEFAULT_ASSET_ROOT = (
    DEFAULT_UPSTREAM_REPO
    / "recovered/pre_trained_models_official_drive_1KNsoWp1_20260820/pre_trained_models"
)
DEFAULT_BASE_MODEL = ROOT / "tmp/official_benchmarks/roberta-base-e2da-materialized"
DEFAULT_RUN_DIR = ROOT / "tmp/emodynamix-clean-features/canonical-train-dev-v1"
def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True
        )
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return (rendered + "\n").encode("utf-8")


def _validate_publish_target(run_dir: Path) -> None:
    if not run_dir.is_absolute():
        raise ValueError("feature run directory must be an absolute path")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"feature run directory already exists: {run_dir}")
    existing_ancestor = run_dir.parent
    while not existing_ancestor.exists():
        existing_ancestor = existing_ancestor.parent
    if existing_ancestor.is_symlink():
        raise ValueError("feature run directory has a symlinked ancestor")


def execute_feature_materialization(
    *,
    prepared: dict[str, Any],
    runtime: Any,
    runtime_receipt: dict[str, Any],
    run_dir: Path,
    batch_size: int,
    feature_implementation_path: Path,
    materializer_path: Path,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Generate, validate, and atomically publish a new feature run."""

    run_dir = Path(run_dir)
    _validate_publish_target(run_dir)
    expected_runtime_receipt_sha256 = FEATURES.canonical_json_sha256(
        runtime_receipt
    )
    if (
        getattr(runtime, "runtime_receipt_sha256", None)
        != expected_runtime_receipt_sha256
    ):
        raise ValueError("feature runtime is not bound to the supplied runtime receipt")
    combined_records = list(prepared["train_records"]) + list(
        prepared["dev_records"]
    )
    unique_inputs = FEATURES.unique_causal_inputs(combined_records)
    feature_implementation_path = Path(feature_implementation_path)
    materializer_path = Path(materializer_path)
    generator_manifest = FEATURES.build_generator_manifest(
        runtime_receipt=runtime_receipt,
        implementation_sha256=FEATURES.sha256_file(feature_implementation_path),
        device=str(runtime_receipt["device"]),
        batch_size=batch_size,
    )
    generator_manifest.update(
        {
            "materializer_sha256": FEATURES.sha256_file(materializer_path),
            "data_builder_sha256": FEATURES.sha256_file(Path(DATA.__file__)),
            "source_audit": prepared["source_audit"],
            "overlap_audit": prepared["audit"],
            "unique_input_keys_sha256": FEATURES.canonical_json_sha256(
                [row["model_input_sha256"] for row in unique_inputs]
            ),
            "unique_input_count": len(unique_inputs),
        }
    )
    manifest_sha = FEATURES.generator_manifest_sha256(generator_manifest)
    feature_rows = FEATURES.generate_verified_feature_rows(
        unique_inputs,
        runtime=runtime,
        generator_manifest_sha256=manifest_sha,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    required_keys = {row["model_input_sha256"] for row in unique_inputs}
    actual_keys: set[str] = set()
    for row in feature_rows:
        DATA.validate_feature_row(
            row,
            expected_input_sha256=row["model_input_sha256"],
            expected_generator_manifest_sha256=manifest_sha,
        )
        key = row["model_input_sha256"]
        if key in actual_keys:
            raise ValueError(f"duplicate generated feature key: {key}")
        actual_keys.add(key)
    if actual_keys != required_keys:
        raise ValueError("generated feature keys do not close over train/dev inputs")
    feature_table_sha = DATA.feature_table_commitment(feature_rows)

    features_payload = b"".join(
        _canonical_json_bytes(row) for row in feature_rows
    )
    manifest_payload = _canonical_json_bytes(generator_manifest, pretty=True)
    summary = {
        "schema_version": "esconv-emodynamix-feature-run-v1",
        "status": "VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION",
        "training_records": len(prepared["train_records"]),
        "development_records": len(prepared["dev_records"]),
        "feature_rows": len(feature_rows),
        "feature_batch_size": batch_size,
        "feature_table_sha256": feature_table_sha,
        "features_jsonl_sha256": hashlib.sha256(features_payload).hexdigest(),
        "generator_manifest_sha256": manifest_sha,
        "generator_manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "frozen_test_accessed": False,
        "target_or_label_fields_in_features": [],
        "selectable_model_produced": False,
    }
    summary_payload = _canonical_json_bytes(summary, pretty=True)

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{run_dir.name}.staging-", dir=str(run_dir.parent)
        )
    )
    try:
        (staging / "features.jsonl").write_bytes(features_payload)
        (staging / "generator_manifest.json").write_bytes(manifest_payload)
        (staging / "feature_run_summary.json").write_bytes(summary_payload)
        if run_dir.exists():
            raise FileExistsError(
                f"feature run directory already exists before publish: {run_dir}"
            )
        os.rename(staging, run_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or materialize clean canonical EmoDynamiX features"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DATA.DEFAULT_DATA_ROOT / DATA.OFFICIAL_SPLITS["train"]["filename"],
    )
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=DATA.DEFAULT_DATA_ROOT / DATA.OFFICIAL_SPLITS["dev"]["filename"],
    )
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--expected-base-tree-sha256",
        default=DATA.EXPECTED_CLEAN_CONTRACT.get(
            "base_model_tree_sha256",
            "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c",
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = DATA.prepare_canonical_data(args.train_file, args.dev_file)
    unique_inputs = FEATURES.unique_causal_inputs(
        list(prepared["train_records"]) + list(prepared["dev_records"])
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "FEATURE_MATERIALIZATION_AUDIT_ONLY",
                    "execute": False,
                    "training_records": len(prepared["train_records"]),
                    "development_records": len(prepared["dev_records"]),
                    "planned_unique_feature_rows": len(unique_inputs),
                    "model_loaded": False,
                    "artifact_written": False,
                    "frozen_test_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    device = _resolve_device(args.device)
    FEATURES.validate_generation_environment(device)
    runtime, receipt = FEATURES.load_verified_feature_runtime(
        upstream_repo=args.upstream_repo,
        asset_root=args.asset_root,
        base_model_path=args.base_model.resolve(),
        expected_base_tree_sha256=args.expected_base_tree_sha256,
        device=device,
    )

    last_reported = -1

    def report_progress(progress: dict[str, int]) -> None:
        nonlocal last_reported
        completed = progress["completed"]
        # Batch 8 is ~3 seconds here; print roughly once a minute and at end.
        if completed == progress["total"] or completed - last_reported >= 160:
            print(json.dumps({"feature_progress": progress}), flush=True)
            last_reported = completed

    summary = execute_feature_materialization(
        prepared=prepared,
        runtime=runtime,
        runtime_receipt=receipt,
        run_dir=args.run_dir.resolve(),
        batch_size=args.feature_batch_size,
        feature_implementation_path=Path(FEATURES.__file__),
        materializer_path=Path(__file__),
        progress_callback=report_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
