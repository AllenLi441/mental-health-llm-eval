import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FIT_SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"
DATA_SCRIPT = ROOT / "scripts/train_esconv_emodynamix_clean.py"
MODEL_MODULE = ROOT / "open_response_eval/esconv_emodynamix_model.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_module("fit_esconv_emodynamix_clean_for_dataset", FIT_SCRIPT)
DATA = load_module("train_esconv_emodynamix_clean_for_fit_dataset", DATA_SCRIPT)
MODEL = load_module("esconv_emodynamix_model_for_fit_dataset", MODEL_MODULE)


def records_and_features():
    records = DATA.build_emodynamix_records(
        [
            "1.0 0 0 first EOS 1.0 1 1 [Other] target one\n",
            "1.0 0 0 second EOS 1.0 1 1 [Questions] target two\n",
        ],
        split="train",
        expected_rows=2,
    )
    duplicate = dict(records[0], item_id="duplicate-record")
    all_records = [records[0], duplicate, records[1]]
    features = {
        record["model_input_sha256"]: DATA.make_structural_fixture_feature(
            record["model_input_sha256"], record["model_input"]
        )
        for record in records
    }
    return all_records, features


class CleanDatasetAndCollatorTests(unittest.TestCase):
    def test_duplicate_records_are_preserved_with_many_to_one_feature_join(self):
        records, features = records_and_features()
        dataset = FIT.CleanEmoDynamiXTrainingDataset(records, features)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]["model_input_sha256"], dataset[1]["model_input_sha256"])
        self.assertEqual(dataset[0]["feature"], dataset[1]["feature"])
        self.assertNotEqual(dataset[0]["item_id"], dataset[1]["item_id"])
        self.assertEqual([dataset[index]["label_id"] for index in range(3)], [7, 7, 2])

    def test_collator_keeps_labels_outside_exact_six_field_model_batch(self):
        records, features = records_and_features()
        dataset = FIT.CleanEmoDynamiXTrainingDataset(records, features)
        collate = FIT.make_training_collator(
            model_module=MODEL, device=torch.device("cpu")
        )
        batch = collate([dataset[index] for index in range(3)])

        self.assertEqual(set(batch), {"model_batch", "labels", "item_ids"})
        self.assertEqual(batch["labels"].tolist(), [7, 7, 2])
        self.assertEqual(
            set(batch["model_batch"]),
            {
                "dialogue_history",
                "strategy_history",
                "speaker_turn",
                "parsed_dialogue",
                "erc_probabilities",
                "dialogue_sizes",
            },
        )
        serialized = json.dumps(
            {
                key: value
                for key, value in batch["model_batch"].items()
                if not torch.is_tensor(value)
            },
            sort_keys=True,
        ).casefold()
        for forbidden in ("gold", "label", "target", "response"):
            self.assertNotIn(forbidden, serialized)

    def test_dataset_recomputes_input_hash_and_requires_matching_feature_key(self):
        records, features = records_and_features()
        tampered = [dict(records[0], model_input=dict(records[0]["model_input"]))]
        tampered[0]["model_input"]["dialogue_history"] += " changed"
        with self.assertRaisesRegex(ValueError, "model input SHA"):
            FIT.CleanEmoDynamiXTrainingDataset(tampered, features)

        wrong_feature = dict(features[records[0]["model_input_sha256"]])
        wrong_feature["model_input_sha256"] = records[2]["model_input_sha256"]
        with self.assertRaisesRegex(ValueError, "feature.*SHA"):
            FIT.CleanEmoDynamiXTrainingDataset(
                [records[0]],
                {records[0]["model_input_sha256"]: wrong_feature},
            )

    def test_collator_rejects_unexpected_outer_fields(self):
        records, features = records_and_features()
        dataset = FIT.CleanEmoDynamiXTrainingDataset(records, features)
        collate = FIT.make_training_collator(
            model_module=MODEL, device=torch.device("cpu")
        )
        malicious = dict(dataset[0], gold="Other")
        with self.assertRaisesRegex(ValueError, "training item fields"):
            collate([malicious])


if __name__ == "__main__":
    unittest.main()
