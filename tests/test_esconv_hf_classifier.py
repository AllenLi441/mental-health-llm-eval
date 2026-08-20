import copy
import importlib.util
import json
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
        "run_name": "tinyllama-frozen-campaign-001",
        "profile_key": "tinyllama-augesc-context",
        "model_id": "heegyu/TinyLlama-augesc-context",
        "revision": "4fd4cdc278812afd572e34040ecf433a31c8623e",
        "weights_manifest_sha256": "a" * 64,
        "limit": limit,
        "predictions_out": "/authorized/predictions.jsonl",
        "summary_out": "/authorized/summary.json",
    }


def valid_campaign_authorization(limit=None, context=None):
    context = context or campaign_context(limit=limit)
    return {
        "schema_version": "esconv-frozen-campaign-authorization-v1",
        "authorization_status": "AUTHORIZED",
        "authorization_scope": "one_frozen_test_prediction_run",
        "campaign_id": "esconv-tinyllama-001",
        "candidate_frozen_before_test": True,
        "unique_candidate_count": 1,
        "authorized_prediction_runs": 1,
        "frozen_test": {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "rows": EVAL.EXPECTED_FROZEN_ROWS,
        },
        "candidate": {
            "profile_key": context["profile_key"],
            "model_id": context["model_id"],
            "revision": context["revision"],
            "weights_manifest_sha256": context["weights_manifest_sha256"],
        },
        "authorized_run_name": context["run_name"],
        "authorized_limit": context["limit"],
        "authorized_outputs": {
            "predictions": context["predictions_out"],
            "summary": context["summary_out"],
        },
    }


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

    def test_checkpoint_training_provenance_cli_is_required_and_closed(self):
        parser = EVAL.build_parser()
        action = next(
            item
            for item in parser._actions
            if item.dest == "checkpoint_training_provenance"
        )
        self.assertTrue(action.required)
        self.assertEqual(
            set(action.choices),
            {"frozen_train_dev_only", "external_unknown", "known_overlap"},
        )
        campaign_action = next(
            item
            for item in parser._actions
            if item.dest == "campaign_authorization_manifest"
        )
        self.assertTrue(campaign_action.required)

    def test_external_unknown_and_known_overlap_are_diagnostic_only(self):
        dataset = {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "limit": None,
        }
        metrics = {"invalid": 0}
        for provenance, status in (
            ("external_unknown", "DIAGNOSTIC_ONLY_EXTERNAL_UNKNOWN"),
            ("known_overlap", "DIAGNOSTIC_ONLY_KNOWN_OVERLAP"),
        ):
            result = EVAL.assess_frozen_leaderboard_eligibility(
                checkpoint_training_provenance=provenance,
                dataset_artifact=dataset,
                metrics=metrics,
                training_provenance_audit=None,
                campaign_authorization_manifest=None,
                campaign_context=campaign_context(),
            )
            self.assertFalse(result["frozen_leaderboard_eligible"])
            self.assertTrue(result["diagnostic_only"])
            self.assertEqual(result["status"], status)

    def test_frozen_train_dev_declaration_without_audit_is_not_eligible(self):
        result = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": None,
            },
            metrics={"invalid": 0},
            training_provenance_audit=None,
            campaign_authorization_manifest=None,
            campaign_context=campaign_context(),
        )
        self.assertFalse(result["frozen_leaderboard_eligible"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(
            result["status"], "DIAGNOSTIC_ONLY_MISSING_PROVENANCE_AUDIT"
        )

    def test_only_audited_zero_overlap_authorized_unique_candidate_is_eligible(self):
        result = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": None,
            },
            metrics={"invalid": 0},
            training_provenance_audit=valid_training_provenance_audit(),
            campaign_authorization_manifest=valid_campaign_authorization(),
            campaign_context=campaign_context(),
        )
        self.assertTrue(result["frozen_leaderboard_eligible"])
        self.assertFalse(result["diagnostic_only"])
        self.assertEqual(
            result["status"], "ELIGIBLE_AUDITED_FROZEN_TRAIN_DEV_ONLY"
        )

    def test_provenance_audit_rejects_wrong_hash_overlap_or_incomplete_status(self):
        dataset = {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "limit": None,
        }
        metrics = {"invalid": 0}
        mutations = (
            (("frozen_test", "sha256"), "0" * 64, "test hash"),
            (("train_test_overlap", "rows"), 1, "overlap"),
            (("audit_status",), "INCOMPLETE", "COMPLETE"),
        )
        for path, value, message in mutations:
            audit = valid_training_provenance_audit()
            target = audit
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(ValueError, message):
                EVAL.assess_frozen_leaderboard_eligibility(
                    checkpoint_training_provenance="frozen_train_dev_only",
                    dataset_artifact=dataset,
                    metrics=metrics,
                    training_provenance_audit=audit,
                    campaign_authorization_manifest=valid_campaign_authorization(),
                    campaign_context=campaign_context(),
                )

    def test_missing_campaign_authorization_is_diagnostic_only(self):
        result = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": None,
            },
            metrics={"invalid": 0},
            training_provenance_audit=valid_training_provenance_audit(),
            campaign_authorization_manifest=None,
            campaign_context=campaign_context(),
        )
        self.assertFalse(result["frozen_leaderboard_eligible"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(
            result["status"], "DIAGNOSTIC_ONLY_MISSING_CAMPAIGN_AUTHORIZATION"
        )
        unknown = valid_campaign_authorization()
        unknown["authorization_status"] = "UNKNOWN"
        unknown_result = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": None,
            },
            metrics={"invalid": 0},
            training_provenance_audit=valid_training_provenance_audit(),
            campaign_authorization_manifest=unknown,
            campaign_context=campaign_context(),
        )
        self.assertFalse(unknown_result["frozen_leaderboard_eligible"])
        self.assertEqual(
            unknown_result["status"], "DIAGNOSTIC_ONLY_UNAUTHORIZED_CAMPAIGN"
        )

    def test_campaign_authorization_must_bind_one_exact_candidate_and_run(self):
        dataset = {
            "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
            "limit": None,
        }
        mutations = (
            (("candidate_frozen_before_test",), False, "frozen before test"),
            (("unique_candidate_count",), 2, "unique candidate"),
            (("authorized_prediction_runs",), 2, "one prediction"),
            (("candidate", "revision"), "0" * 40, "candidate binding"),
            (("authorized_run_name",), "another-run", "run name"),
        )
        for path, value, message in mutations:
            manifest = valid_campaign_authorization()
            target = manifest
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(ValueError, message):
                EVAL.assess_frozen_leaderboard_eligibility(
                    checkpoint_training_provenance="frozen_train_dev_only",
                    dataset_artifact=dataset,
                    metrics={"invalid": 0},
                    training_provenance_audit=valid_training_provenance_audit(),
                    campaign_authorization_manifest=manifest,
                    campaign_context=campaign_context(),
                )

    def test_limit_smoke_and_invalid_full_runs_stay_ineligible_with_valid_audit(self):
        audit = valid_training_provenance_audit()
        smoke = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": 10,
            },
            metrics={"invalid": 0},
            training_provenance_audit=audit,
            campaign_authorization_manifest=valid_campaign_authorization(limit=10),
            campaign_context=campaign_context(limit=10),
        )
        invalid = EVAL.assess_frozen_leaderboard_eligibility(
            checkpoint_training_provenance="frozen_train_dev_only",
            dataset_artifact={
                "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                "limit": None,
            },
            metrics={"invalid": 1},
            training_provenance_audit=audit,
            campaign_authorization_manifest=valid_campaign_authorization(),
            campaign_context=campaign_context(),
        )
        self.assertFalse(smoke["frozen_leaderboard_eligible"])
        self.assertEqual(smoke["status"], "SMOKE_ONLY_NOT_LEADERBOARD_ELIGIBLE")
        self.assertFalse(invalid["frozen_leaderboard_eligible"])
        self.assertEqual(invalid["status"], "DIAGNOSTIC_ONLY_INVALID_PREDICTIONS")

    def test_run_rejects_invalid_campaign_before_test_read_model_load_or_output(self):
        invalid_manifest = valid_campaign_authorization()
        invalid_manifest["authorization_status"] = "UNKNOWN"
        args = Namespace(
            profile="tinyllama-augesc-context",
            model_dir=Path("/model"),
            model_revision=None,
            test_file=Path("/frozen/test.tsv"),
            predictions_out=Path("/authorized/predictions.jsonl"),
            summary_out=Path("/authorized/summary.json"),
            run_name="tinyllama-frozen-campaign-001",
            batch_size=1,
            device="cpu",
            limit=None,
            overwrite=False,
            checkpoint_training_provenance="external_unknown",
            training_provenance_audit=None,
            campaign_authorization_manifest=Path("/authorization.json"),
        )
        fake_model_artifact = {
            "source_model_id": "heegyu/TinyLlama-augesc-context",
            "revision": "4fd4cdc278812afd572e34040ecf433a31c8623e",
            "weights": {"manifest_sha256": "a" * 64},
        }
        artifact = {
            "path": "/authorization.json",
            "sha256": "b" * 64,
            "document": invalid_manifest,
        }
        with (
            mock.patch.object(
                EVAL,
                "audit_model_directory",
                return_value=(fake_model_artifact, {}),
            ),
            mock.patch.object(EVAL, "load_json_artifact", return_value=artifact),
            mock.patch.object(EVAL, "prepare_frozen_file") as prepare_test,
            mock.patch.object(EVAL, "load_hf_model") as load_model,
            mock.patch.object(EVAL, "write_predictions") as write_predictions,
            mock.patch.object(EVAL, "write_json") as write_summary,
        ):
            with self.assertRaisesRegex(ValueError, "AUTHORIZED"):
                EVAL.run(args)
        prepare_test.assert_not_called()
        load_model.assert_not_called()
        write_predictions.assert_not_called()
        write_summary.assert_not_called()

        missing_args = Namespace(**vars(args))
        missing_args.campaign_authorization_manifest = None
        with (
            mock.patch.object(EVAL, "audit_model_directory") as audit_model,
            mock.patch.object(EVAL, "prepare_frozen_file") as prepare_test,
            mock.patch.object(EVAL, "load_hf_model") as load_model,
        ):
            with self.assertRaisesRegex(ValueError, "campaign-authorization-manifest"):
                EVAL.run(missing_args)
        audit_model.assert_not_called()
        prepare_test.assert_not_called()
        load_model.assert_not_called()

    def test_summary_requires_both_audits_before_marking_formal_result_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = directory / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            context = campaign_context()
            context["predictions_out"] = str(predictions.resolve())
            context["summary_out"] = str((directory / "summary.json").resolve())
            training_artifact = {
                "path": str((directory / "training-audit.json").resolve()),
                "sha256": "c" * 64,
                "document": valid_training_provenance_audit(),
            }
            campaign_artifact = {
                "path": str((directory / "campaign.json").resolve()),
                "sha256": "d" * 64,
                "document": valid_campaign_authorization(context=context),
            }
            records = [
                {
                    "gold": "Questions",
                    "prediction": "Questions",
                    "invalid": False,
                }
            ]
            summary = EVAL.build_summary(
                run_name=context["run_name"],
                profile_key=context["profile_key"],
                profile=EVAL.get_enabled_profile(context["profile_key"]),
                model_artifact={
                    "source_model_id": context["model_id"],
                    "revision": context["revision"],
                    "weights": {
                        "manifest_sha256": context["weights_manifest_sha256"]
                    },
                },
                dataset_artifact={
                    "sha256": EVAL.EXPECTED_FROZEN_TEST_SHA256,
                    "limit": None,
                },
                predictions_path=predictions,
                records=records,
                metrics=EVAL.compute_metrics(records),
                runtime={},
                checkpoint_training_provenance="frozen_train_dev_only",
                training_provenance_audit_artifact=training_artifact,
                campaign_authorization_artifact=campaign_artifact,
                campaign_context=context,
            )
        self.assertTrue(summary["eligible_for_frozen_leaderboard"])
        self.assertFalse(summary["diagnostic_only"])
        self.assertEqual(summary["audit_status"], "COMPLETE")
        self.assertEqual(
            summary["profile"]["input_template_source"]["revision"],
            context["revision"],
        )


if __name__ == "__main__":
    unittest.main()
