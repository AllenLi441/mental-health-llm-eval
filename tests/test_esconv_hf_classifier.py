import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_esconv_hf_classifier.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("esconv_hf_classifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = load_evaluator()


def sample_frozen_rows():
    return [
        "1.0 0 0 Hello EOS 1.0 1 1 [Questions] How are you feeling?\n",
        (
            "1.0 0 0 Hello EOS "
            "1.0 1 1 [Questions] How are you feeling? EOS "
            "1.0 0 2 Bad EOS "
            "1.0 1 3 [Information] SECRET TARGET RESPONSE\n"
        ),
        "1.0 0 0 New dialogue EOS 1.0 1 1 [Other] SECRET TWO\n",
    ]


def valid_training_provenance_audit():
    return {
        "schema_version": "esconv-checkpoint-training-provenance-audit-v1",
        "audit_status": "COMPLETE",
        "frozen_test": {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "rows": EVAL.EXPECTED_FROZEN_ROWS,
        },
        "train_test_overlap": {"rows": 0},
    }


def campaign_context(limit=None):
    return {
        "campaign_id": "esconv-tinyllama-001",
        "trusted_commit": "1" * 40,
        "run_name": "tinyllama-frozen-campaign-001",
        "profile_key": "tinyllama-augesc-context",
        "model_id": "heegyu/TinyLlama-augesc-context",
        "revision": "4fd4cdc278812afd572e34040ecf433a31c8623e",
        "weights_manifest_sha256": "a" * 64,
        "model_tree_sha256": "b" * 64,
        "tokenizer_tree_sha256": "c" * 64,
        "checkpoint_training_provenance": "frozen_train_dev_only",
        "training_provenance_audit_path": "campaign/training-audit.json",
        "training_provenance_audit_sha256": "d" * 64,
        "evaluator_path": "scripts/eval_esconv_hf_classifier.py",
        "evaluator_sha256": "e" * 64,
        "evaluator_commit": "1" * 40,
        "selection_path": "campaign/selection.json",
        "selection_sha256": "f" * 64,
        "authorization_path": "campaign/authorization.json",
        "authorization_sha256": "0" * 64,
        "receipt_path": "campaign/consumption-receipt.json",
        "receipt_armed_sha256": "2" * 64,
        "limit": limit,
        "run_dir": "/authorized/tinyllama-frozen-campaign-001",
    }


def valid_campaign_authorization(limit=None, context=None):
    context = context or campaign_context(limit=limit)
    return {
        "schema_version": "esconv-frozen-campaign-authorization-v2",
        "authorization_status": "AUTHORIZED",
        "authorization_scope": "one_frozen_test_prediction_run",
        "campaign_id": context["campaign_id"],
        "candidate_frozen_before_test": True,
        "unique_candidate_count": 1,
        "authorized_prediction_runs": 1,
        "selection_artifact_path": context["selection_path"],
        "frozen_test": {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "rows": EVAL.EXPECTED_FROZEN_ROWS,
        },
        "candidate": {
            "profile_key": context["profile_key"],
            "model_id": context["model_id"],
            "revision": context["revision"],
            "weights_manifest_sha256": context["weights_manifest_sha256"],
            "model_tree_sha256": context["model_tree_sha256"],
            "tokenizer_tree_sha256": context["tokenizer_tree_sha256"],
        },
        "checkpoint_training_provenance": context[
            "checkpoint_training_provenance"
        ],
        "training_provenance_audit": {
            "path": context["training_provenance_audit_path"],
            "sha256": context["training_provenance_audit_sha256"],
        },
        "evaluator": {
            "path": context["evaluator_path"],
            "sha256": context["evaluator_sha256"],
            "commit": context["evaluator_commit"],
        },
        "formal_run": {
            "run_name": context["run_name"],
            "limit": context["limit"],
            "run_dir": context["run_dir"],
        },
        "consumption_receipt": {
            "path": context["receipt_path"],
            "armed_sha256": context["receipt_armed_sha256"],
        },
    }


def valid_candidate_selection(context=None):
    context = context or campaign_context()
    return {
        "schema_version": "esconv-frozen-candidate-selection-v1",
        "selection_status": "FROZEN",
        "campaign_id": context["campaign_id"],
        "unique_candidate_count": 1,
        "frozen_test": {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "rows": EVAL.EXPECTED_FROZEN_ROWS,
        },
        "candidate": valid_campaign_authorization(context=context)["candidate"],
        "checkpoint_training_provenance": context[
            "checkpoint_training_provenance"
        ],
        "training_provenance_audit": {
            "path": context["training_provenance_audit_path"],
            "sha256": context["training_provenance_audit_sha256"],
        },
        "evaluator": valid_campaign_authorization(context=context)["evaluator"],
        "formal_run": valid_campaign_authorization(context=context)["formal_run"],
        "authorization_artifact": {
            "path": context["authorization_path"],
            "sha256": context["authorization_sha256"],
        },
        "consumption_receipt": valid_campaign_authorization(context=context)[
            "consumption_receipt"
        ],
    }


def valid_armed_receipt(context=None):
    context = context or campaign_context()
    return {
        "schema_version": "esconv-frozen-campaign-consumption-v1",
        "campaign_id": context["campaign_id"],
        "authorization_sha256": context["authorization_sha256"],
        "run_name": context["run_name"],
        "run_dir": context["run_dir"],
        "status": "ARMED",
        "claimed_at_utc": None,
        "claim_nonce": None,
    }


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def init_git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )


class HFClassifierEvaluatorTests(unittest.TestCase):
    def test_profiles_pin_model_revision_config_card_weight_and_labels(self):
        tiny = EVAL.PROFILES["tinyllama-augesc-context"]
        self.assertEqual(tiny["model_id"], "heegyu/TinyLlama-augesc-context")
        self.assertEqual(
            tiny["revision"], "4fd4cdc278812afd572e34040ecf433a31c8623e"
        )
        self.assertEqual(
            tiny["expected_config_sha256"],
            "46a30993f08d288f45efe40cd41ed1c35f48360a28dd5842cf1d90bc0ddb4e87",
        )
        self.assertEqual(
            tiny["expected_card_sha256"],
            "84a4f2cecb97a5f39f876455cf4579b782a16c35a504612e1f284eba74bafa16",
        )
        self.assertEqual(
            tiny["expected_weights"],
            {
                "model.safetensors": {
                    "sha256": "be1b01ac696c8a4f8397286e5a9accd14bcff0aaa0b8f43ef3ae5a37b82fc483",
                    "size": 4_138_138_032,
                }
            },
        )
        self.assertEqual(tiny["source_label_order"], EVAL.HEEGYU_SOURCE_LABELS)
        self.assertEqual(tiny["canonical_label_order"], EVAL.CANONICAL_LABELS)
        self.assertEqual(tiny["max_length"], 2_048)
        self.assertEqual(tiny["truncation_side"], "right")
        self.assertEqual(
            EVAL.PROFILES["esconv-xlm-roberta-base"]["revision"],
            "8a21d3f0e1ec06aa147faa652e64b938a55c37dd",
        )
        self.assertEqual(
            EVAL.PROFILES["esconv-xlm-roberta-large"]["revision"],
            "0c1955b5f0d1aceb31cd4067432ed29bc936caa9",
        )

    def test_tinyllama_card_template_excludes_target_strategy_and_response(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        records = EVAL.prepare_frozen_lines(
            sample_frozen_rows(), profile=profile, expected_rows=None
        )
        self.assertEqual(records[0]["conversation_id"], "esconv-test-0001")
        self.assertEqual(records[1]["conversation_id"], "esconv-test-0001")
        self.assertEqual(records[2]["conversation_id"], "esconv-test-0002")
        self.assertEqual(records[1]["gold"], "Information")
        self.assertEqual(
            records[1]["_model_input"],
            "usr: Hello\nsys: How are you feeling?\nusr: Bad",
        )
        self.assertNotIn("SECRET TARGET RESPONSE", records[1]["_model_input"])
        self.assertNotIn("Information", records[1]["_model_input"])
        self.assertNotIn("[Questions]", records[1]["_model_input"])
        self.assertNotIn("target_response", records[1])

    def test_xlm_card_template_includes_only_prior_supporter_strategy(self):
        profile = copy.deepcopy(EVAL.PROFILES["esconv-xlm-roberta-base"])
        records = EVAL.prepare_frozen_lines(
            sample_frozen_rows(), profile=profile, expected_rows=None
        )
        self.assertEqual(
            records[1]["_model_input"],
            "usr: Hello\nsys[Question]: How are you feeling?\nusr: Bad",
        )
        self.assertNotIn("sys[Information]", records[1]["_model_input"])
        self.assertNotIn("SECRET TARGET RESPONSE", records[1]["_model_input"])

    def test_xlm_profiles_are_blocked_without_immutable_frozen_input_contract(self):
        for key in ("esconv-xlm-roberta-base", "esconv-xlm-roberta-large"):
            self.assertFalse(EVAL.PROFILES[key]["enabled"])
            self.assertIsNone(EVAL.PROFILES[key]["input_template_source"])
            with self.assertRaisesRegex(ValueError, "frozen input contract"):
                EVAL.get_enabled_profile(key)

    def test_tinyllama_profile_has_immutable_input_template_source(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        source = profile["input_template_source"]
        self.assertEqual(source["kind"], "immutable_model_card")
        self.assertEqual(source["revision"], profile["revision"])
        self.assertEqual(source["sha256"], profile["expected_card_sha256"])
        profile_summary = EVAL._profile_summary(
            "tinyllama-augesc-context", profile
        )
        self.assertEqual(profile_summary["input_template_source"], source)

    def test_limit_is_a_prefix_smoke_without_changing_conversation_identity(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        records = EVAL.prepare_frozen_lines(
            sample_frozen_rows(), profile=profile, expected_rows=None, limit=2
        )
        self.assertEqual([record["item_id"] for record in records], [0, 1])
        self.assertEqual(
            [record["conversation_id"] for record in records],
            ["esconv-test-0001", "esconv-test-0001"],
        )

    def test_config_validation_rejects_causal_lm_head_before_loading_weights(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        config = {
            "architectures": ["LlamaForCausalLM"],
            "id2label": {str(index): f"LABEL_{index}" for index in range(8)},
            "label2id": {f"LABEL_{index}": index for index in range(8)},
        }
        with self.assertRaisesRegex(ValueError, "CausalLM"):
            EVAL.validate_config_contract(config, profile)

    def test_config_validation_rejects_unapproved_or_ambiguous_label_map(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        config = {
            "architectures": ["LlamaForSequenceClassification"],
            "id2label": {str(index): f"LABEL_{index}" for index in range(8)},
            "label2id": {f"LABEL_{index}": index for index in range(8)},
        }
        config["id2label"]["7"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "id2label"):
            EVAL.validate_config_contract(config, profile)

    def test_modernbert_profiles_are_registered_but_blocked(self):
        key = "modernbert-multiturn-upsampling-v3"
        profile = EVAL.PROFILES[key]
        self.assertTrue(profile["model_id"].endswith("multiturn-upsampling-v3"))
        self.assertEqual(
            profile["revision"], "f4057d6e1cfaf62184b369e014ad6e0b26dec13a"
        )
        self.assertEqual(profile["source_label_order"], EVAL.MODERNBERT_SOURCE_LABELS)
        with self.assertRaisesRegex(ValueError, "input template"):
            EVAL.get_enabled_profile(key)

    def test_logits_must_be_exactly_eight_way(self):
        EVAL.validate_logits_shape((3, 8), batch_size=3)
        with self.assertRaisesRegex(ValueError, "8-class"):
            EVAL.validate_logits_shape((3, 32_000), batch_size=3)

    def test_metrics_keep_invalid_rows_in_accuracy_denominator(self):
        records = [
            {"gold": "Questions", "prediction": "Questions", "invalid": False},
            {"gold": "Other", "prediction": "Questions", "invalid": False},
            {"gold": None, "prediction": None, "invalid": True},
        ]
        metrics = EVAL.compute_metrics(records)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["invalid"], 1)
        self.assertAlmostEqual(metrics["invalid_rate"], 1 / 3)
        self.assertEqual(
            metrics["confusion_matrix"]["labels"], list(EVAL.CANONICAL_LABELS)
        )

    def test_public_prediction_rows_drop_private_model_input(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        record = EVAL.prepare_frozen_lines(
            sample_frozen_rows(), profile=profile, expected_rows=None, limit=1
        )[0]
        record.update(
            {
                "prediction": "Questions",
                "prediction_id": 0,
                "correct": True,
                "invalid": False,
                "logits": {label: 0.0 for label in EVAL.CANONICAL_LABELS},
                "probabilities": {
                    label: 0.125 for label in EVAL.CANONICAL_LABELS
                },
            }
        )
        public = EVAL.public_prediction(record)
        serialized = json.dumps(public)
        self.assertNotIn("_model_input", public)
        self.assertNotIn("SECRET", serialized)
        self.assertEqual(public["conversation_id"], "esconv-test-0001")
        self.assertEqual(public["input_sha256"], EVAL.sha256_text("usr: Hello"))

    def test_revision_detection_rejects_mixed_huggingface_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            (metadata / "config.json.metadata").write_text(
                "a" * 40 + "\nblob\ntime\n", encoding="utf-8"
            )
            (metadata / "model.safetensors.metadata").write_text(
                "b" * 40 + "\nblob\ntime\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "mixed revisions"):
                EVAL.detect_local_hf_revision(root)

    def test_config_map_is_not_inferred_from_label_names(self):
        profile = copy.deepcopy(EVAL.PROFILES["esconv-xlm-roberta-base"])
        profile["expected_config_id2label"] = None
        config = {
            "architectures": ["XLMRobertaForSequenceClassification"],
            "id2label": {str(index): f"LABEL_{index}" for index in range(8)},
            "label2id": {f"LABEL_{index}": index for index in range(8)},
        }
        with self.assertRaisesRegex(ValueError, "explicitly approved"):
            EVAL.validate_config_contract(config, profile)

    def test_formal_cli_requires_anchored_campaign_and_has_no_limit_or_overwrite(self):
        parser = EVAL.build_parser()
        actions = {item.dest: item for item in parser._actions}
        for destination in (
            "checkpoint_training_provenance",
            "campaign_trusted_commit",
            "campaign_selection_manifest",
            "campaign_selection_expected_sha256",
            "campaign_authorization_manifest",
            "campaign_authorization_expected_sha256",
            "campaign_consumption_receipt",
            "campaign_consumption_receipt_expected_sha256",
            "run_dir",
        ):
            self.assertIn(destination, actions)
            self.assertTrue(actions[destination].required)
        self.assertNotIn("limit", actions)
        self.assertNotIn("overwrite", actions)

    def test_json_document_and_hash_are_derived_from_one_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            original = b'{"authorization_status":"AUTHORIZED"}\n'
            path.write_bytes(original)
            with mock.patch.object(EVAL, "sha256_file", return_value="0" * 64):
                artifact = EVAL.load_json_artifact(
                    path, description="authorization"
                )
        self.assertEqual(artifact["sha256"], sha256_bytes(original))
        self.assertEqual(
            artifact["document"], {"authorization_status": "AUTHORIZED"}
        )

    def test_committed_artifact_gate_rejects_untracked_dirty_and_wrong_expected_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_git_repo(repo)
            tracked = repo / "campaign.json"
            payload = b'{"status":"FROZEN"}\n'
            tracked.write_bytes(payload)
            subprocess.run(["git", "-C", str(repo), "add", "campaign.json"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "campaign"], check=True
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            artifact = EVAL.load_committed_json_artifact(
                tracked,
                description="campaign",
                repo_root=repo,
                trusted_commit=commit,
                expected_sha256=sha256_bytes(payload),
            )
            self.assertEqual(artifact["trusted_commit"], commit)

            with self.assertRaisesRegex(ValueError, "expected SHA-256"):
                EVAL.load_committed_json_artifact(
                    tracked,
                    description="campaign",
                    repo_root=repo,
                    trusted_commit=commit,
                    expected_sha256="0" * 64,
                )
            tracked.write_text('{"status":"CHANGED"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean|committed"):
                EVAL.load_committed_json_artifact(
                    tracked,
                    description="campaign",
                    repo_root=repo,
                    trusted_commit=commit,
                    expected_sha256=sha256_bytes(payload),
                )
            untracked = repo / "temporary.json"
            untracked.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tracked|committed"):
                EVAL.load_committed_json_artifact(
                    untracked,
                    description="temporary authorization",
                    repo_root=repo,
                    trusted_commit=commit,
                    expected_sha256=sha256_bytes(b"{}\n"),
                )

    def test_model_snapshot_covers_tokenizer_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "README.md").write_text("card\n", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            tokenizer = model_dir / "tokenizer.json"
            tokenizer.write_bytes(b"tokenizer-v1")
            before = EVAL.snapshot_model_tree(model_dir)
            tokenizer.write_bytes(b"tokenizer-v2")
            after = EVAL.snapshot_model_tree(model_dir)
            self.assertNotEqual(
                before["tree_manifest_sha256"], after["tree_manifest_sha256"]
            )
            self.assertNotEqual(
                before["tokenizer_manifest_sha256"],
                after["tokenizer_manifest_sha256"],
            )
            os.symlink(model_dir / "README.md", model_dir / "linked-card")
            with self.assertRaisesRegex(ValueError, "symlink"):
                EVAL.snapshot_model_tree(model_dir)

    def test_loaded_model_is_reaudited_before_inference(self):
        profile = EVAL.get_enabled_profile("tinyllama-augesc-context")
        before = {
            "tree_manifest_sha256": "a" * 64,
            "tokenizer_manifest_sha256": "b" * 64,
        }
        changed = {
            "tree_manifest_sha256": "c" * 64,
            "tokenizer_manifest_sha256": "b" * 64,
        }
        with (
            mock.patch.object(
                EVAL,
                "load_hf_model",
                return_value=(object(), object(), "cpu", {}),
            ),
            mock.patch.object(
                EVAL,
                "audit_model_directory",
                return_value=(changed, {}),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "changed.*load"):
                EVAL.load_hf_model_verified(
                    Path("/model"),
                    profile=profile,
                    requested_device="cpu",
                    declared_revision=profile["revision"],
                    pre_load_artifact=before,
                )

    def test_selection_and_authorization_bind_every_frozen_input(self):
        context = campaign_context()
        selection = valid_candidate_selection(context)
        authorization = valid_campaign_authorization(context=context)
        EVAL.validate_campaign_documents(
            selection, authorization, campaign_context=context
        )
        mutations = (
            (selection, ("authorization_artifact", "sha256"), "9" * 64),
            (selection, ("checkpoint_training_provenance",), "known_overlap"),
            (authorization, ("training_provenance_audit", "sha256"), "9" * 64),
            (authorization, ("candidate", "tokenizer_tree_sha256"), "9" * 64),
            (authorization, ("candidate", "model_tree_sha256"), "9" * 64),
            (authorization, ("evaluator", "commit"), "9" * 40),
            (authorization, ("formal_run", "limit"), 1),
        )
        for original, path, value in mutations:
            mutated_selection = copy.deepcopy(selection)
            mutated_authorization = copy.deepcopy(authorization)
            target = (
                mutated_selection if original is selection else mutated_authorization
            )
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(ValueError, "binding|formal full"):
                EVAL.validate_campaign_documents(
                    mutated_selection,
                    mutated_authorization,
                    campaign_context=context,
                )

    def test_eligibility_requires_full_2775_metrics_and_two_campaign_documents(self):
        context = campaign_context()
        dataset = {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "full_rows": EVAL.EXPECTED_FROZEN_ROWS,
            "evaluated_rows": EVAL.EXPECTED_FROZEN_ROWS,
            "limit": None,
        }
        valid = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact=dataset,
            metrics={"invalid": 0, "total": EVAL.EXPECTED_FROZEN_ROWS},
            training_provenance_audit=valid_training_provenance_audit(),
            campaign_selection_manifest=valid_candidate_selection(context),
            campaign_authorization_manifest=valid_campaign_authorization(
                context=context
            ),
            campaign_context=context,
        )
        self.assertTrue(valid["frozen_leaderboard_eligible"])
        for broken_dataset, broken_metrics in (
            ({**dataset, "evaluated_rows": 1}, {"invalid": 0, "total": 1}),
            (dataset, {"invalid": 0, "total": 1}),
            ({**dataset, "limit": 1}, {"invalid": 0, "total": 2775}),
        ):
            result = EVAL.assess_frozen_leaderboard_eligibility(
                checkpoint_training_provenance="frozen_train_dev_only",
                dataset_artifact=broken_dataset,
                metrics=broken_metrics,
                training_provenance_audit=valid_training_provenance_audit(),
                campaign_selection_manifest=valid_candidate_selection(context),
                campaign_authorization_manifest=valid_campaign_authorization(
                    context=context
                ),
                campaign_context=context,
            )
            self.assertFalse(result["frozen_leaderboard_eligible"])

    def test_prediction_integrity_rejects_one_record_fake_full_result(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            records = [
                {
                    "item_id": 0,
                    "gold": "Questions",
                    "prediction": "Questions",
                    "invalid": False,
                }
            ]
            EVAL.write_predictions(predictions, records)
            with self.assertRaisesRegex(ValueError, "2775"):
                EVAL.validate_formal_result_integrity(
                    records=records,
                    dataset_artifact={
                        "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                        "full_rows": EVAL.EXPECTED_FROZEN_ROWS,
                        "evaluated_rows": 1,
                        "limit": None,
                    },
                    metrics=EVAL.compute_metrics(records),
                    predictions_path=predictions,
                )

    def test_campaign_claim_is_atomic_persistent_and_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_git_repo(repo)
            context = campaign_context()
            receipt_path = repo / "campaign/consumption-receipt.json"
            receipt_path.parent.mkdir()
            receipt_document = valid_armed_receipt(context)
            payload = (
                json.dumps(receipt_document, sort_keys=True, indent=2) + "\n"
            ).encode()
            context["receipt_armed_sha256"] = sha256_bytes(payload)
            receipt_path.write_bytes(payload)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "arm campaign"],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            context["trusted_commit"] = commit
            artifact = EVAL.load_committed_json_artifact(
                receipt_path,
                description="consumption receipt",
                repo_root=repo,
                trusted_commit=commit,
                expected_sha256=sha256_bytes(payload),
            )
            claim_path = EVAL.consume_campaign_slot(
                repo_root=repo,
                receipt_artifact=artifact,
                campaign_context=context,
            )
            self.assertTrue(claim_path.is_file())
            consumed = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(consumed["status"], "CONSUMED")
            self.assertTrue(
                subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--porcelain", "--", str(receipt_path)],
                    text=True,
                ).strip()
            )
            with self.assertRaisesRegex(ValueError, "claimed|consumed|single-use"):
                EVAL.consume_campaign_slot(
                    repo_root=repo,
                    receipt_artifact=artifact,
                    campaign_context=context,
                )

    def test_run_rejects_static_errors_and_untrusted_artifacts_before_test_or_model(self):
        base = Namespace(
            profile="tinyllama-augesc-context",
            model_dir=Path("/model"),
            model_revision=None,
            test_file=Path("/frozen/test.tsv"),
            run_dir=Path("/authorized/tinyllama-frozen-campaign-001"),
            run_name="tinyllama-frozen-campaign-001",
            batch_size=1,
            device="cpu",
            limit=None,
            overwrite=False,
            checkpoint_training_provenance="frozen_train_dev_only",
            training_provenance_audit=Path("/campaign/training-audit.json"),
            training_provenance_audit_expected_sha256="d" * 64,
            campaign_trusted_commit="1" * 40,
            campaign_selection_manifest=Path("/campaign/selection.json"),
            campaign_selection_expected_sha256="f" * 64,
            campaign_authorization_manifest=Path("/campaign/authorization.json"),
            campaign_authorization_expected_sha256="0" * 64,
            campaign_consumption_receipt=Path("/campaign/receipt.json"),
            campaign_consumption_receipt_expected_sha256="2" * 64,
        )
        for field, value, message in (
            ("limit", 1, "limit"),
            ("batch_size", 0, "batch"),
            ("checkpoint_training_provenance", "invented", "provenance"),
        ):
            args = Namespace(**vars(base))
            setattr(args, field, value)
            with (
                mock.patch.object(EVAL, "load_json_artifact") as load_artifact,
                mock.patch.object(EVAL, "audit_model_directory") as audit_model,
                mock.patch.object(EVAL, "prepare_frozen_file") as prepare_test,
                mock.patch.object(EVAL, "load_hf_model") as load_model,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    EVAL.run(args)
            load_artifact.assert_not_called()
            audit_model.assert_not_called()
            prepare_test.assert_not_called()
            load_model.assert_not_called()

        with (
            mock.patch.object(
                EVAL,
                "load_committed_json_artifact",
                side_effect=ValueError("artifact is not tracked and committed"),
                create=True,
            ),
            mock.patch.object(EVAL, "audit_model_directory") as audit_model,
            mock.patch.object(EVAL, "prepare_frozen_file") as prepare_test,
            mock.patch.object(EVAL, "load_hf_model") as load_model,
            mock.patch.object(EVAL, "write_predictions") as write_predictions,
            mock.patch.object(EVAL, "write_json") as write_summary,
        ):
            with self.assertRaisesRegex(ValueError, "tracked and committed"):
                EVAL.run(base)
        audit_model.assert_not_called()
        prepare_test.assert_not_called()
        load_model.assert_not_called()
        write_predictions.assert_not_called()
        write_summary.assert_not_called()

    def test_output_directory_rejects_existing_target_and_symlink_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            existing = real_parent / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                EVAL.validate_run_directory(existing)
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                EVAL.validate_run_directory(linked_parent / "new-run")


if __name__ == "__main__":
    unittest.main()
