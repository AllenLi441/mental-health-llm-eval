from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA = "jingshi.model-deployment/v1"
SELF_HOSTED_LEVEL = "content_hash_verified"
PROVIDER_LEVEL = "provider_alias_only"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"deployment manifest {label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"deployment manifest {label} must be non-empty text")
    return value.strip()


def _require_sha256(value: Any, label: str) -> str:
    text = _require_text(value, label).casefold()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"deployment manifest {label} must be a SHA-256 hex digest")
    return text


def _artifact_entries(artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    index = _require_object(artifacts.get("index"), "resolved_deployment.artifacts.index")
    entries.append(index)
    for field in ("files", "required_config_files"):
        values = artifacts.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"deployment manifest resolved_deployment.artifacts.{field} "
                "must be a non-empty array"
            )
        entries.extend(values)

    normalized = []
    for entry in entries:
        item = _require_object(entry, "artifact entry")
        path = _require_text(item.get("path"), "artifact path")
        role = _require_text(item.get("role"), "artifact role")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("deployment manifest artifact size_bytes must be positive")
        normalized.append(
            {
                "path": path,
                "role": role,
                "size_bytes": size,
                "sha256": _require_sha256(item.get("sha256"), "artifact sha256"),
            }
        )
    paths = [entry["path"] for entry in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("deployment manifest artifact paths must be unique")
    return sorted(normalized, key=lambda entry: entry["path"])


def validate_deployment_manifest(
    manifest: Mapping[str, Any], target_spec: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"deployment manifest schema_version must be {MANIFEST_SCHEMA}")
    if manifest.get("deployment_mode") != "self_hosted":
        raise ValueError("confirmatory Qwen evaluation requires a self_hosted deployment")
    if manifest.get("verification_level") != SELF_HOSTED_LEVEL:
        raise ValueError(
            "confirmatory Qwen evaluation requires content_hash_verified provenance"
        )
    deployment_id = _require_text(manifest.get("deployment_id"), "deployment_id")

    expected = _require_object(manifest.get("expected_target"), "expected_target")
    resolved = _require_object(
        manifest.get("resolved_deployment"), "resolved_deployment"
    )
    expected_repo = str(target_spec.get("canonical_repo_id") or target_spec.get("model"))
    expected_revision = _require_text(target_spec.get("revision"), "target revision")
    for label, value in (
        ("expected_target.repo_id", expected.get("repo_id")),
        ("resolved_deployment.repo_id", resolved.get("repo_id")),
    ):
        if value != expected_repo:
            raise ValueError(f"deployment manifest {label} does not match {expected_repo}")
    for label, value in (
        ("expected_target.revision", expected.get("revision")),
        ("resolved_deployment.revision", resolved.get("revision")),
    ):
        if value != expected_revision:
            raise ValueError(
                f"deployment manifest {label} does not match the frozen revision"
            )
    if resolved.get("weights_identity") != "official_snapshot":
        raise ValueError("deployment manifest must load the official_snapshot weights")
    if resolved.get("artifact_format") != "safetensors":
        raise ValueError("deployment manifest artifact_format must be safetensors")

    artifacts = _require_object(resolved.get("artifacts"), "resolved_deployment.artifacts")
    normalized_artifacts = _artifact_entries(artifacts)
    artifact_set_sha256 = _require_sha256(
        artifacts.get("artifact_set_sha256"), "artifact_set_sha256"
    )
    if canonical_sha256(normalized_artifacts) != artifact_set_sha256:
        raise ValueError("deployment manifest artifact_set_sha256 does not match artifacts")

    serving = _require_object(manifest.get("serving"), "serving")
    served_alias = _require_text(serving.get("served_model_alias"), "served_model_alias")
    if served_alias != target_spec.get("model"):
        raise ValueError("deployment manifest served_model_alias does not match target model")
    if serving.get("wire_api") != "openai_chat_completions":
        raise ValueError("deployment manifest wire_api must be openai_chat_completions")
    runtime = _require_object(serving.get("runtime"), "serving.runtime")
    engine = _require_text(runtime.get("engine"), "runtime.engine")
    engine_version = _require_text(runtime.get("engine_version"), "runtime.engine_version")
    image_digest = _require_text(
        runtime.get("container_image_digest"), "runtime.container_image_digest"
    )
    if not image_digest.startswith("sha256:"):
        raise ValueError("runtime.container_image_digest must be immutable sha256")
    _require_sha256(image_digest.removeprefix("sha256:"), "container image digest")

    quantization = _require_object(serving.get("quantization"), "serving.quantization")
    expected_quantization = target_spec.get("expected_quantization") or {
        "mode": "none",
        "weight_dtype": "bfloat16",
    }
    if quantization.get("mode") != expected_quantization.get("mode"):
        raise ValueError("deployment manifest quantization mode differs from protocol")
    if quantization.get("weight_dtype") != expected_quantization.get("weight_dtype"):
        raise ValueError("deployment manifest weight dtype differs from protocol")

    effective = _require_object(
        serving.get("effective_config"), "serving.effective_config"
    )
    if effective.get("enable_thinking") is not False:
        raise ValueError("deployment manifest must disable Qwen thinking")
    chat_template_sha256 = _require_sha256(
        effective.get("chat_template_sha256"), "chat_template_sha256"
    )
    launch_args_sha256 = _require_sha256(
        effective.get("launch_args_sha256"), "launch_args_sha256"
    )

    verification = _require_object(manifest.get("verification"), "verification")
    if verification.get("artifact_hashes_verified") is not True:
        raise ValueError("deployment manifest must attest artifact hash verification")
    if verification.get("models_endpoint_observed_alias") != served_alias:
        raise ValueError("deployment manifest /models alias differs from served alias")
    if verification.get("smoke_requested_model") != served_alias:
        raise ValueError("deployment manifest smoke requested_model differs from alias")
    if verification.get("smoke_response_model") != served_alias:
        raise ValueError("deployment manifest smoke response_model differs from alias")

    return {
        "deployment_id": deployment_id,
        "deployment_verification_level": SELF_HOSTED_LEVEL,
        "resolved_revision": expected_revision,
        "artifact_set_sha256": artifact_set_sha256,
        "runtime_engine": engine,
        "runtime_engine_version": engine_version,
        "runtime_image_digest": image_digest,
        "quantization_mode": str(quantization["mode"]),
        "weight_dtype": str(quantization["weight_dtype"]),
        "served_model_alias": served_alias,
        "chat_template_sha256": chat_template_sha256,
        "launch_args_sha256": launch_args_sha256,
    }


def load_pinned_deployment_manifest(
    path: Path, target_spec: Mapping[str, Any]
) -> dict[str, Any]:
    expected_hash = str(target_spec.get("deployment_manifest_sha256") or "").strip()
    if not expected_hash:
        raise ValueError(
            "target revision is not yet frozen to a deployment manifest; "
            "set deployment_manifest_sha256 in the preregistration"
        )
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError("deployment manifest hash differs from preregistration")
    value = json.loads(path.read_text(encoding="utf-8"))
    manifest = _require_object(value, "root")
    return {
        "deployment_manifest_sha256": actual_hash,
        **validate_deployment_manifest(manifest, target_spec),
    }


def provider_alias_provenance(target_spec: Mapping[str, Any]) -> dict[str, Any]:
    if target_spec.get("revision"):
        raise ValueError("a revision-pinned target requires a deployment manifest")
    return {
        "deployment_manifest_sha256": None,
        "deployment_id": str(target_spec.get("id") or ""),
        "deployment_verification_level": PROVIDER_LEVEL,
        "resolved_revision": None,
        "artifact_set_sha256": None,
        "runtime_engine": "provider_managed",
        "runtime_engine_version": None,
        "runtime_image_digest": None,
        "quantization_mode": "undisclosed",
        "weight_dtype": "undisclosed",
        "served_model_alias": str(target_spec.get("model") or ""),
        "chat_template_sha256": None,
        "launch_args_sha256": None,
    }
