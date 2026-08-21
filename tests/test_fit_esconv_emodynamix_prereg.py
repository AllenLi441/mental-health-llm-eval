import argparse
import copy
import importlib.util
import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "fit_esconv_emodynamix_clean_for_prereg", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_trainer()


SOURCE_HASHES = {
    "trainer_sha256": "1" * 64,
    "data_builder_sha256": "2" * 64,
    "model_sha256": "3" * 64,
    "metrics_sha256": "4" * 64,
}


def args():
    return argparse.Namespace(
        mode="pilot",
        arm_id="author-control-seed7",
        output_dir=(
            ROOT
            / "tmp/emodynamix-clean-training/pilot-v1/author-control-seed7"
        ),
        expected_base_tree_sha256=FIT.EXPECTED_BASE_MODEL_TREE_SHA256,
        device="mps",
        seed=7,
        epochs=3,
        train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=1e-3,
        warmup_updates=100,
        max_updates=1_600,
        max_grad_norm=1.0,
        loss="author_weighted_ce",
        author_weight_temperature=1.75,
        class_balance_beta=0.999,
        logit_adjustment_tau=1.0,
    )


def document():
    return {
        "schema_version": "jingshi-esconv-emodynamix-clean-dev-pilot-v1",
        "protocol_id": "jingshi-esconv-emodynamix-clean-dev-pilot-v1",
        "parent_protocol_id": "jingshi-esconv-first-v1",
        "status": "FROZEN",
        "frozen_at": "2026-08-21T12:00:00-07:00",
        "frozen_test_policy": "FORBIDDEN_DURING_TRAIN_DEV_PILOT",
        "data_contract": {
            "training_records": 8_433,
            "development_records": 2_985,
            "feature_rows": 11_366,
            "class_counts": list(FIT.CLASS_COUNTS),
        },
        "feature_run_contract": {
            "feature_run_summary_sha256": "a" * 64,
            "training_records": 8_433,
            "development_records": 2_985,
            "feature_rows": 11_366,
            "feature_batch_size": 8,
            "generator_manifest_sha256": "b" * 64,
            "features_jsonl_sha256": "c" * 64,
            "feature_table_sha256": "d" * 64,
        },
        "base_model_contract": {
            "tree_sha256": FIT.EXPECTED_BASE_MODEL_TREE_SHA256,
            "task_checkpoint_initialization": None,
        },
        "source_code_contract": SOURCE_HASHES,
        "shared_training": {
            "device": "mps",
            "epochs": 3,
            "train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "eval_batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 1e-3,
            "warmup_updates": 100,
            "max_updates": 1_600,
            "max_grad_norm": 1.0,
            "author_weight_temperature": 1.75,
            "class_balance_beta": 0.999,
            "logit_adjustment_tau": 1.0,
        },
        "arms": {
            "author-control-seed7": {
                "loss": "author_weighted_ce",
                "seed": 7,
                "output_dir_relative": (
                    "tmp/emodynamix-clean-training/pilot-v1/author-control-seed7"
                ),
            }
        },
        "selection": {
            "primary": "dev_macro_f1",
            "secondary": "dev_accuracy",
            "tertiary": "lower_raw_unweighted_ce",
            "prediction_source": "raw_logits",
        },
    }


class PilotPreregistrationTests(unittest.TestCase):
    def test_main_pilot_uses_frozen_contract_before_loading_features(self):
        prepared = {
            "train_records": [{"model_input_sha256": "1" * 64}],
            "dev_records": [{"model_input_sha256": "2" * 64}],
            "source_audit": {},
            "audit": {},
        }
        feature_contract = document()["feature_run_contract"]
        provenance = {
            "protocol_id": document()["protocol_id"],
            "arm_id": "author-control-seed7",
            "feature_run_contract": feature_contract,
            "execution_validated": True,
            "frozen_test_accessed": False,
        }
        fake_result = {
            "training": {"best_epoch": 1},
            "output_dir": "/tmp/run",
            "artifact_receipt": {"checkpoint_sha256": "c" * 64},
        }
        output = StringIO()
        with mock.patch(
            "scripts.train_esconv_emodynamix_clean.prepare_canonical_data",
            return_value=prepared,
        ), mock.patch.object(
            FIT, "load_pilot_execution_provenance", return_value=provenance
        ) as load_prereg, mock.patch.object(
            FIT,
            "load_verified_feature_run",
            return_value=({"feature": {}}, {"verified_upstream_features": True}),
        ) as load_features, mock.patch.object(
            FIT, "train_emodynamix", return_value=fake_result
        ) as train, redirect_stdout(output):
            result = FIT.main(
                [
                    "--execute",
                    "--mode",
                    "pilot",
                    "--arm-id",
                    "author-control-seed7",
                    "--preregistration",
                    "/tmp/prereg.json",
                    "--preregistration-sha256",
                    "a" * 64,
                ]
            )

        self.assertEqual(result, 0)
        load_prereg.assert_called_once()
        load_features.assert_called_once()
        self.assertEqual(
            load_features.call_args.kwargs["preregistration_contract"],
            feature_contract,
        )
        train.assert_called_once()
        self.assertEqual(
            train.call_args.kwargs["execution_provenance"], provenance
        )
        self.assertEqual(json.loads(output.getvalue())["training"]["best_epoch"], 1)

    def test_source_hash_roster_is_complete_and_missing_cli_anchor_fails(self):
        hashes = FIT.current_pilot_source_hashes()
        self.assertEqual(
            set(hashes),
            {
                "trainer_sha256",
                "data_builder_sha256",
                "model_sha256",
                "metrics_sha256",
            },
        )
        for digest in hashes.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        arguments = args()
        arguments.preregistration = None
        arguments.preregistration_sha256 = None
        with self.assertRaisesRegex(ValueError, "preregistration.*SHA"):
            FIT.load_pilot_execution_provenance(arguments)

    def test_frozen_document_authorizes_one_exact_arm_and_returns_feature_contract(self):
        validated = FIT.validate_pilot_preregistration_document(
            document(), args(), current_source_hashes=SOURCE_HASHES
        )
        self.assertEqual(validated["protocol_id"], document()["protocol_id"])
        self.assertEqual(validated["arm_id"], "author-control-seed7")
        self.assertEqual(
            validated["feature_run_contract"], document()["feature_run_contract"]
        )
        self.assertTrue(validated["execution_validated"])
        self.assertFalse(validated["frozen_test_accessed"])

    def test_any_training_source_or_freeze_drift_fails_closed(self):
        cases = []
        wrong_loss = args()
        wrong_loss.loss = "ce"
        cases.append((document(), wrong_loss, SOURCE_HASHES, "arm"))
        wrong_device = args()
        wrong_device.device = "cpu"
        cases.append((document(), wrong_device, SOURCE_HASHES, "shared training"))
        unfrozen = document()
        unfrozen["frozen_at"] = None
        cases.append((unfrozen, args(), SOURCE_HASHES, "frozen"))
        changed_source = dict(SOURCE_HASHES, trainer_sha256="f" * 64)
        cases.append((document(), args(), changed_source, "source code"))
        wrong_output = args()
        wrong_output.output_dir = ROOT / "tmp/cherry-picked-rerun"
        cases.append((document(), wrong_output, SOURCE_HASHES, "output"))

        for payload, arguments, hashes, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    FIT.validate_pilot_preregistration_document(
                        copy.deepcopy(payload),
                        arguments,
                        current_source_hashes=hashes,
                    )


if __name__ == "__main__":
    unittest.main()
