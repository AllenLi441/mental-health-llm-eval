import json
import tempfile
import unittest
from pathlib import Path

from open_response_eval.provenance import (
    MANIFEST_SCHEMA,
    canonical_sha256,
    load_pinned_deployment_manifest,
    provider_alias_provenance,
    validate_deployment_manifest,
)


class DeploymentManifestTests(unittest.TestCase):
    @staticmethod
    def target_spec() -> dict:
        return {
            "id": "qwen_3_6_27b_base",
            "model": "Qwen/Qwen3.6-27B",
            "canonical_repo_id": "Qwen/Qwen3.6-27B",
            "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "expected_quantization": {"mode": "none", "weight_dtype": "bfloat16"},
        }

    @staticmethod
    def manifest() -> dict:
        entries = [
            {
                "path": "config.json",
                "role": "config",
                "size_bytes": 100,
                "sha256": "1" * 64,
            },
            {
                "path": "model-00001-of-00015.safetensors",
                "role": "weight_shard",
                "size_bytes": 1000,
                "sha256": "2" * 64,
            },
            {
                "path": "model.safetensors.index.json",
                "role": "weight_index",
                "size_bytes": 200,
                "sha256": "3" * 64,
            },
        ]
        normalized = sorted(entries, key=lambda entry: entry["path"])
        return {
            "schema_version": MANIFEST_SCHEMA,
            "deployment_id": "qwen36-test-deployment",
            "created_at": "2026-08-06T00:00:00Z",
            "deployment_mode": "self_hosted",
            "verification_level": "content_hash_verified",
            "expected_target": {
                "repo_id": "Qwen/Qwen3.6-27B",
                "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            },
            "resolved_deployment": {
                "repo_id": "Qwen/Qwen3.6-27B",
                "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                "weights_identity": "official_snapshot",
                "artifact_format": "safetensors",
                "artifacts": {
                    "index": entries[2],
                    "files": [entries[1]],
                    "required_config_files": [entries[0]],
                    "artifact_set_sha256": canonical_sha256(normalized),
                },
            },
            "serving": {
                "served_model_alias": "Qwen/Qwen3.6-27B",
                "wire_api": "openai_chat_completions",
                "runtime": {
                    "engine": "vllm",
                    "engine_version": "0.10.0",
                    "container_image_digest": f"sha256:{'4' * 64}",
                },
                "quantization": {"mode": "none", "weight_dtype": "bfloat16"},
                "effective_config": {
                    "enable_thinking": False,
                    "chat_template_sha256": "5" * 64,
                    "launch_args_sha256": "6" * 64,
                },
            },
            "verification": {
                "artifact_hashes_verified": True,
                "models_endpoint_observed_alias": "Qwen/Qwen3.6-27B",
                "smoke_requested_model": "Qwen/Qwen3.6-27B",
                "smoke_response_model": "Qwen/Qwen3.6-27B",
            },
        }

    def test_accepts_content_hash_verified_official_deployment(self):
        provenance = validate_deployment_manifest(self.manifest(), self.target_spec())

        self.assertEqual(provenance["resolved_revision"], self.target_spec()["revision"])
        self.assertEqual(provenance["deployment_verification_level"], "content_hash_verified")
        self.assertEqual(provenance["quantization_mode"], "none")

    def test_rejects_quantized_weights_for_frozen_bfloat16_arm(self):
        manifest = self.manifest()
        manifest["serving"]["quantization"] = {
            "mode": "awq",
            "weight_dtype": "int4",
        }

        with self.assertRaisesRegex(ValueError, "quantization"):
            validate_deployment_manifest(manifest, self.target_spec())

    def test_load_requires_manifest_bytes_pinned_in_preregistration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            path.write_text(json.dumps(self.manifest(), sort_keys=True), encoding="utf-8")
            spec = self.target_spec()
            spec["deployment_manifest_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_pinned_deployment_manifest(path, spec)

    def test_provider_alias_cannot_impersonate_a_revision_pinned_target(self):
        with self.assertRaisesRegex(ValueError, "requires a deployment manifest"):
            provider_alias_provenance(self.target_spec())

        provenance = provider_alias_provenance(
            {"id": "deepseek", "model": "deepseek-v4-pro"}
        )
        self.assertEqual(provenance["deployment_verification_level"], "provider_alias_only")
        self.assertIsNone(provenance["resolved_revision"])


if __name__ == "__main__":
    unittest.main()
