import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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
        profile = EVAL.get_enabled_profile("esconv-xlm-roberta-base")
        records = EVAL.prepare_frozen_lines(
            sample_frozen_rows(), profile=profile, expected_rows=None
        )
        self.assertEqual(
            records[1]["_model_input"],
            "usr: Hello\nsys[Question]: How are you feeling?\nusr: Bad",
        )
        self.assertNotIn("sys[Information]", records[1]["_model_input"])
        self.assertNotIn("SECRET TARGET RESPONSE", records[1]["_model_input"])

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
        profile = copy.deepcopy(
            EVAL.get_enabled_profile("esconv-xlm-roberta-base")
        )
        profile["expected_config_id2label"] = None
        config = {
            "architectures": ["XLMRobertaForSequenceClassification"],
            "id2label": {str(index): f"LABEL_{index}" for index in range(8)},
            "label2id": {f"LABEL_{index}": index for index in range(8)},
        }
        with self.assertRaisesRegex(ValueError, "explicitly approved"):
            EVAL.validate_config_contract(config, profile)


if __name__ == "__main__":
    unittest.main()
