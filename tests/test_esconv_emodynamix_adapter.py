import importlib.util
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_esconv_emodynamix.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("esconv_emodynamix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_adapter()


class FrozenTsvPreparationTests(unittest.TestCase):
    def test_label_contract_matches_emodynamix(self):
        self.assertEqual(ADAPTER.CANONICAL_TO_EMODYNAMIX["Questions"], "Question")
        self.assertEqual(ADAPTER.CANONICAL_TO_EMODYNAMIX["Other"], "Others")
        self.assertEqual(ADAPTER.EMODYNAMIX_LABEL_TO_ID["Question"], 2)
        self.assertEqual(ADAPTER.EMODYNAMIX_LABEL_TO_ID["Others"], 7)

    def test_frozen_rows_reconstruct_upstream_five_turn_input_without_target(self):
        rows = [
            "1.0 0 0 Hello EOS 1.0 1 1 [Questions] How are you feeling?\n",
            (
                "1.0 0 0 Hello EOS 1.0 1 1 [Questions] How are you feeling? "
                "EOS 1.0 0 2 Bad EOS 1.0 1 3 [Self-disclosure] I understand.\n"
            ),
            (
                "0.0 1 1 [Questions] How are you feeling? EOS 0.0 0 2 Bad "
                "EOS 0.0 1 3 [Self-disclosure] I understand. EOS 0.0 0 4 Still bad "
                "EOS 1.0 1 5 [Other] SECRET TARGET RESPONSE\n"
            ),
        ]

        records = ADAPTER.prepare_frozen_lines(rows, expected_rows=None)

        self.assertEqual(len(records), 3)
        first = records[0]
        self.assertEqual(first["model_input"]["dialogue_history"], "<START> </s> Hello")
        self.assertEqual(first["model_input"]["strategy_history"], "[-1, -1]")
        self.assertEqual(first["model_input"]["speaker_turn"], "None seeker")
        self.assertEqual(first["gold"], "Questions")
        self.assertEqual(first["gold_emodynamix"], "Question")

        last = records[-1]
        self.assertEqual(
            last["model_input"]["dialogue_history"],
            "Hello </s> How are you feeling? </s> Bad </s> I understand. </s> Still bad",
        )
        self.assertEqual(last["model_input"]["strategy_history"], "[-1, 2, -1, 1, -1]")
        self.assertEqual(
            last["model_input"]["speaker_turn"],
            "seeker supporter seeker supporter seeker",
        )
        self.assertEqual(last["gold"], "Other")
        self.assertEqual(last["gold_emodynamix"], "Others")
        self.assertNotIn("SECRET TARGET RESPONSE", json.dumps(last["model_input"]))
        self.assertNotIn("gold", last["model_input"])

    def test_dialogue_reset_gets_new_cluster_and_start_token(self):
        rows = [
            "1.0 0 0 First EOS 1.0 1 1 [Questions] Target one\n",
            "1.0 0 0 Second EOS 1.0 1 1 [Other] Target two\n",
        ]

        records = ADAPTER.prepare_frozen_lines(rows, expected_rows=None)

        self.assertEqual(records[0]["conversation_id"], "esconv-test-dialog-0001")
        self.assertEqual(records[1]["conversation_id"], "esconv-test-dialog-0002")
        self.assertEqual(
            records[1]["model_input"]["dialogue_history"], "<START> </s> Second"
        )

    def test_expected_frozen_count_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "expected 2775"):
            ADAPTER.prepare_frozen_lines(
                ["1.0 0 0 Hello EOS 1.0 1 1 [Questions] Target\n"],
                expected_rows=2775,
            )


class AuthorPicklePreparationTests(unittest.TestCase):
    def test_pickle_requires_explicit_unsafe_opt_in_and_excludes_target(self):
        payload = [
            {
                "dialogue_history": "<START> </s> Hello",
                "strategy_history": "[-1, -1]",
                "speaker_turn": "None seeker",
                "gold_standard": "SECRET AUTHOR TARGET",
                "label": 2,
                "parsed_dialogue": {(0, 1, 16)},
                "erc_logits": [[0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle)

            with self.assertRaisesRegex(ValueError, "unsafe pickle"):
                ADAPTER.prepare_author_pickle(source, allow_unsafe_pickle=False)
            records = ADAPTER.prepare_author_pickle(source, allow_unsafe_pickle=True)

        self.assertEqual(records[0]["gold"], "Questions")
        self.assertEqual(records[0]["gold_emodynamix"], "Question")
        self.assertEqual(records[0]["model_input"]["parsed_dialogue"], [[0, 1, 16]])
        serialized_input = json.dumps(records[0]["model_input"])
        self.assertNotIn("SECRET AUTHOR TARGET", serialized_input)
        self.assertNotIn("gold_standard", serialized_input)
        self.assertNotIn("label", records[0]["model_input"])


class InferenceAndScoringTests(unittest.TestCase):
    def test_predictor_receives_only_model_input(self):
        prepared = [
            {
                "item_id": "item-1",
                "line_number": 1,
                "gold": "Questions",
                "gold_emodynamix": "Question",
                "model_input": {
                    "dialogue_history": "Hello",
                    "strategy_history": "[-1]",
                    "speaker_turn": "seeker",
                },
            }
        ]
        seen = []

        def predict_batch(model_inputs):
            seen.extend(model_inputs)
            return [{"prediction": "Question", "logits": [0, 0, 1, 0, 0, 0, 0, 0]}]

        predictions = ADAPTER.run_inference(prepared, predict_batch, batch_size=8)

        self.assertEqual(seen, [prepared[0]["model_input"]])
        self.assertEqual(predictions[0]["prediction"], "Questions")
        self.assertNotIn("gold", predictions[0])

    def test_prediction_normalization_and_invalid_scoring(self):
        prepared = [
            {
                "item_id": "item-1",
                "line_number": 1,
                "conversation_id": "dialog-1",
                "gold": "Questions",
                "gold_emodynamix": "Question",
                "model_input": {},
            },
            {
                "item_id": "item-2",
                "line_number": 2,
                "conversation_id": "dialog-1",
                "gold": "Other",
                "gold_emodynamix": "Others",
                "model_input": {},
            },
        ]
        raw = [
            {"item_id": "item-1", "prediction": "Question", "logits": [0] * 8},
            {"item_id": "item-2", "prediction": "not-a-label", "error": None},
        ]

        audited, metrics = ADAPTER.score_predictions(prepared, raw)

        self.assertEqual(audited[0]["prediction"], "Questions")
        self.assertFalse(audited[0]["invalid"])
        self.assertTrue(audited[1]["invalid"])
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["invalid"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["ACC"], 0.5)
        self.assertEqual(metrics["per_class"]["Other"]["support"], 1)
        self.assertEqual(metrics["per_class"]["Other"]["recall"], 0.0)
        self.assertEqual(
            metrics["confusion_matrix"]["invalid_by_gold"]["Other"], 1
        )
        self.assertEqual(
            metrics["confusion_matrix"]["labels"], list(ADAPTER.CANONICAL_LABELS)
        )

    def test_score_rejects_missing_or_extra_item_ids(self):
        prepared = [
            {
                "item_id": "item-1",
                "line_number": 1,
                "gold": "Questions",
                "gold_emodynamix": "Question",
                "model_input": {},
            }
        ]
        with self.assertRaisesRegex(ValueError, "prediction item IDs"):
            ADAPTER.score_predictions(prepared, [])


class AuditIdentityTests(unittest.TestCase):
    def test_summary_contains_repo_checkpoint_test_and_prediction_hashes(self):
        metrics = ADAPTER.compute_metrics(
            [
                {
                    "gold": "Questions",
                    "prediction": "Questions",
                    "correct": True,
                    "invalid": False,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            test_source = directory / "test.tsv"
            test_source.write_text("frozen-test\n", encoding="utf-8")
            prepared = directory / "prepared.jsonl"
            prepared.write_text("{}\n", encoding="utf-8")
            predictions = directory / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")

            summary = ADAPTER.build_summary(
                run_name="unit-test",
                source_kind="frozen_tsv",
                test_source=test_source,
                prepared_input=prepared,
                predictions_output=predictions,
                third_party_repo=ROOT,
                checkpoint=checkpoint,
                metrics=metrics,
                checkpoint_training_provenance="unknown",
            )

            expected_dataset_hash = ADAPTER.sha256_file(test_source)
            expected_checkpoint_hash = ADAPTER.sha256_file(checkpoint)
            expected_predictions_hash = ADAPTER.sha256_file(predictions)

        self.assertEqual(summary["metrics"]["accuracy"], 1.0)
        self.assertEqual(summary["dataset"]["sha256"], expected_dataset_hash)
        self.assertEqual(summary["checkpoint"]["sha256"], expected_checkpoint_hash)
        self.assertEqual(
            summary["predictions"]["sha256"], expected_predictions_hash
        )
        self.assertEqual(summary["third_party_repo"]["commit"], ADAPTER.git_commit(ROOT))
        self.assertTrue(summary["protocol"]["target_response_excluded_from_model_input"])
        self.assertTrue(summary["protocol"]["target_strategy_excluded_from_model_input"])
        self.assertFalse(summary["eligibility"]["frozen_leaderboard_eligible"])
        self.assertEqual(summary["eligibility"]["status"], "DIAGNOSTIC_ONLY_UNKNOWN_PROVENANCE")


class SplitContaminationAuditTests(unittest.TestCase):
    @staticmethod
    def _raw_dialogue(index):
        return {
            "dialog": [
                {"content": f"Seeker {index}"},
                {"content": f"Supporter {index}"},
            ]
        }

    def test_seed_13_split_audit_matches_dialogue_prefixes_and_rows(self):
        author_dialogues = [self._raw_dialogue(index) for index in range(20)]
        shuffled = list(range(20))
        random.Random(13).shuffle(shuffled)
        chosen = [shuffled[0], shuffled[3], shuffled[6]]
        lines = [
            (
                f"1.0 0 0 Seeker {index} EOS "
                f"1.0 1 1 [Questions] Supporter {index}\n"
            )
            for index in chosen
        ]

        audit = ADAPTER.audit_split_contamination(
            lines,
            author_dialogues,
            expected_frozen_rows=None,
            expected_author_dialogues=20,
        )

        self.assertEqual(
            audit["matched_dialogues_by_split"],
            {"train": 1, "valid": 1, "test": 1},
        )
        self.assertEqual(
            audit["matched_rows_by_split"],
            {"train": 1, "valid": 1, "test": 1},
        )
        self.assertEqual(audit["unmatched_dialogues"], 0)
        self.assertTrue(audit["released_checkpoint_train_overlap"])
        self.assertEqual(
            audit["eligibility_status"],
            "DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED",
        )

    def test_released_checkpoint_frozen_summary_requires_and_records_audit(self):
        metrics = ADAPTER.compute_metrics([])
        contamination = {
            "audit_status": "COMPLETE",
            "method": "casefold_whitespace_normalized_dialogue_prefix",
            "split_seed": 13,
            "frozen_source": {"sha256": None},
            "matched_rows_by_split": {"train": 2, "valid": 0, "test": 0},
            "released_checkpoint_train_overlap": True,
            "eligibility_status": "DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            files = {}
            for name in ("checkpoint.pth", "test.tsv", "prepared.jsonl", "predictions.jsonl"):
                files[name] = directory / name
                files[name].write_bytes(name.encode("utf-8"))
            contamination["frozen_source"]["sha256"] = ADAPTER.sha256_file(
                files["test.tsv"]
            )

            with self.assertRaisesRegex(ValueError, "contamination audit is required"):
                ADAPTER.build_summary(
                    run_name="missing-audit",
                    source_kind="frozen_tsv",
                    test_source=files["test.tsv"],
                    prepared_input=files["prepared.jsonl"],
                    predictions_output=files["predictions.jsonl"],
                    third_party_repo=ROOT,
                    checkpoint=files["checkpoint.pth"],
                    metrics=metrics,
                    checkpoint_training_provenance="author_seed13_1300",
                )

            summary = ADAPTER.build_summary(
                run_name="contaminated",
                source_kind="frozen_tsv",
                test_source=files["test.tsv"],
                prepared_input=files["prepared.jsonl"],
                predictions_output=files["predictions.jsonl"],
                third_party_repo=ROOT,
                checkpoint=files["checkpoint.pth"],
                metrics=metrics,
                checkpoint_training_provenance="author_seed13_1300",
                contamination_audit=contamination,
            )

        self.assertFalse(summary["eligibility"]["frozen_leaderboard_eligible"])
        self.assertEqual(
            summary["eligibility"]["status"],
            "DIAGNOSTIC_ONLY_TRAIN_CONTAMINATED",
        )
        self.assertEqual(
            summary["contamination_audit"]["matched_rows_by_split"]["train"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
