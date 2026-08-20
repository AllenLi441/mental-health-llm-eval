from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rerank_esconv_emodynamix.py"
SPEC = importlib.util.spec_from_file_location("rerank_esconv_emodynamix", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


def model_input(history: str) -> dict[str, object]:
    return {
        "dialogue_history": "hello",
        "strategy_history": history,
        "speaker_turn": "seeker",
        "parsed_dialogue": [],
        "erc_logits": [[0.0] * 7],
    }


def record(item_id: str, history: str, gold: str, split: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "source_kind": f"emodynamix_author_{split}_pickle",
        "gold": gold,
        "model_input": model_input(history),
        "model_input_sha256": hashlib.sha256(item_id.encode()).hexdigest(),
    }


def raw_prediction(item_id: str, logits: dict[str, float]) -> dict[str, object]:
    return {
        "item_id": item_id,
        "prediction": max(logits, key=logits.get),
        "invalid": False,
        "logits": logits,
    }


class DevelopmentSplitTests(unittest.TestCase):
    def test_test_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and valid"):
            PLANNER.validate_development_split("test")

    def test_prepare_checks_hash_before_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.pkl"
            source.write_bytes(b"not a pickle")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                PLANNER.prepare_development_pickle(
                    source,
                    split="train",
                    expected_sha256="0" * 64,
                    allow_unsafe_pickle=True,
                )

    def test_prepare_excludes_target_from_model_input(self) -> None:
        payload = [
            {
                "dialogue_history": "seeker text",
                "strategy_history": "[-1]",
                "speaker_turn": "seeker",
                "label": 2,
                "parsed_dialogue": [],
                "erc_logits": [[0.0] * 7],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.pkl"
            source.write_bytes(pickle.dumps(payload))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            prepared = PLANNER.prepare_development_pickle(
                source,
                split="train",
                expected_sha256=digest,
                allow_unsafe_pickle=True,
            )
        self.assertEqual(prepared[0]["source_kind"], "emodynamix_author_train_pickle")
        self.assertEqual(prepared[0]["gold"], "Questions")
        self.assertNotIn("gold", prepared[0]["model_input"])
        self.assertNotIn("label", prepared[0]["model_input"])


class TransitionPriorTests(unittest.TestCase):
    def test_smoothed_probabilities_are_exact_and_deterministic(self) -> None:
        train = [
            record("t1", "[-1, 2]", "Questions", "train"),
            record("t2", "[-1, 2]", "Questions", "train"),
            record("t3", "[-1, 2]", "Information", "train"),
        ]
        first = PLANNER.fit_transition_prior(train, history_order=1, smoothing=1.0)
        second = PLANNER.fit_transition_prior(train, history_order=1, smoothing=1.0)
        self.assertEqual(first, second)
        probabilities = PLANNER.prior_for_input(first, model_input("[-1, 2]"))
        labels = list(PLANNER.CANONICAL_LABELS)
        self.assertAlmostEqual(probabilities[labels.index("Questions")], 3 / 11)
        self.assertAlmostEqual(probabilities[labels.index("Information")], 2 / 11)
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_unseen_history_backs_off_to_global_counts(self) -> None:
        train = [
            record("t1", "[-1, 2]", "Questions", "train"),
            record("t2", "[-1, 3]", "Questions", "train"),
            record("t3", "[-1, 4]", "Information", "train"),
        ]
        prior = PLANNER.fit_transition_prior(train, history_order=2, smoothing=1.0)
        probabilities = PLANNER.prior_for_input(prior, model_input("[-1, 7]"))
        labels = list(PLANNER.CANONICAL_LABELS)
        self.assertAlmostEqual(probabilities[labels.index("Questions")], 3 / 11)
        self.assertAlmostEqual(probabilities[labels.index("Information")], 2 / 11)

    def test_fit_rejects_non_train_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "train-only"):
            PLANNER.fit_transition_prior(
                [record("v1", "[-1, 2]", "Questions", "valid")],
                history_order=1,
                smoothing=1.0,
            )


class RerankingTests(unittest.TestCase):
    def test_reranker_interface_cannot_receive_gold(self) -> None:
        train = [record("t1", "[-1, 2]", "Information", "train")]
        prior = PLANNER.fit_transition_prior(train, history_order=1, smoothing=0.1)
        output = PLANNER.rerank_one(
            model_input("[-1, 2]"),
            {label: 0.0 for label in PLANNER.CANONICAL_LABELS},
            prior,
            transition_weight=2.0,
        )
        self.assertEqual(output["prediction"], "Information")
        self.assertNotIn("gold", output)

    def test_selection_uses_macro_f1_then_accuracy_then_simpler_config(self) -> None:
        candidates = [
            {
                "config": {"transition_weight": 0.5, "history_order": 2, "smoothing": 1.0},
                "metrics": {"macro_f1": 0.40, "accuracy": 0.70},
            },
            {
                "config": {"transition_weight": 0.25, "history_order": 3, "smoothing": 2.0},
                "metrics": {"macro_f1": 0.41, "accuracy": 0.60},
            },
            {
                "config": {"transition_weight": 0.1, "history_order": 1, "smoothing": 0.5},
                "metrics": {"macro_f1": 0.41, "accuracy": 0.60},
            },
        ]
        best = PLANNER.choose_best_candidate(candidates)
        self.assertEqual(best["config"]["transition_weight"], 0.1)

    def test_end_to_end_selection_is_valid_only_and_hash_rich(self) -> None:
        train = [
            record("t1", "[-1, 2]", "Information", "train"),
            record("t2", "[-1, 2]", "Information", "train"),
        ]
        valid = [record("v1", "[-1, 2]", "Information", "valid")]
        logits = {label: 0.0 for label in PLANNER.CANONICAL_LABELS}
        logits["Questions"] = 0.1
        raw = [raw_prediction("v1", logits)]
        result = PLANNER.select_configuration(
            train,
            valid,
            raw,
            history_orders=(1,),
            smoothings=(0.1,),
            transition_weights=(0.0, 1.0),
        )
        self.assertEqual(result["best"]["config"]["transition_weight"], 1.0)
        self.assertEqual(result["best"]["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["status"], "DEVELOPMENTAL_AUTHOR_VALID_ONLY")
        self.assertEqual(len(result["commitments"]["train_records_sha256"]), 64)
        self.assertEqual(len(result["commitments"]["valid_predictions_sha256"]), 64)
        json.dumps(result, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
