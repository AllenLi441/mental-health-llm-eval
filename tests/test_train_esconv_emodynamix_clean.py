import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_esconv_emodynamix_clean.py"
TRAIN_FILE = (
    ROOT
    / "tmp/official_benchmarks/esconv/codes/dataset/trainWithStrategy_short.tsv"
)
DEV_FILE = (
    ROOT
    / "tmp/official_benchmarks/esconv/codes/dataset/devWithStrategy_short.tsv"
)


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_esconv_emodynamix_clean", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


def synthetic_rows():
    return [
        "1.0 0 0 hello EOS 1.0 1 1 [Other] target-one\n",
        (
            "0.0 0 0 hello EOS 0.0 1 1 [Other] target-one EOS "
            "0.0 0 2 followup EOS 1.0 1 3 [Questions] target-three\n"
        ),
        (
            "0.0 1 1 [Other] target-one EOS 0.0 0 2 followup EOS "
            "0.0 1 3 [Questions] target-three EOS 0.0 0 4 final seeker EOS "
            "1.0 1 5 [Information] SECRET CURRENT TARGET\n"
        ),
    ]


class CanonicalCausalRecordTests(unittest.TestCase):
    def test_upstream_label_ids_are_preserved(self):
        self.assertEqual(TRAINER.EMODYNAMIX_LABEL_TO_ID["Question"], 2)
        self.assertEqual(TRAINER.EMODYNAMIX_LABEL_TO_ID["Others"], 7)
        self.assertEqual(TRAINER.CANONICAL_TO_EMODYNAMIX["Questions"], "Question")

    def test_reconstructs_cross_row_history_without_current_target(self):
        records = TRAINER.build_emodynamix_records(
            synthetic_rows(), split="train", expected_rows=3
        )

        self.assertEqual(len(records), 3)
        first = records[0]
        self.assertEqual(
            first["model_input"]["dialogue_history"], "<START> </s> hello"
        )
        self.assertEqual(first["model_input"]["strategy_history"], "[-1, -1]")
        self.assertEqual(first["model_input"]["speaker_turn"], "None seeker")
        self.assertEqual(first["gold"], "Other")
        self.assertEqual(first["label_id"], 7)

        last = records[-1]
        self.assertEqual(
            last["model_input"]["dialogue_history"],
            "hello </s> target-one </s> followup </s> target-three </s> final seeker",
        )
        self.assertEqual(
            last["model_input"]["strategy_history"], "[-1, 7, -1, 2, -1]"
        )
        self.assertEqual(
            last["model_input"]["speaker_turn"],
            "seeker supporter seeker supporter seeker",
        )
        serialized = json.dumps(last, sort_keys=True)
        self.assertNotIn("SECRET CURRENT TARGET", serialized)
        self.assertNotIn("gold", last["model_input"])
        self.assertEqual(
            last["source_line_sha256"],
            hashlib.sha256(synthetic_rows()[-1].rstrip("\n").encode()).hexdigest(),
        )
        self.assertEqual(
            last["model_input_sha256"],
            TRAINER.canonical_json_sha256(last["model_input"]),
        )

    def test_current_target_only_becomes_later_history(self):
        records = TRAINER.build_emodynamix_records(
            synthetic_rows(), split="dev", expected_rows=3
        )
        self.assertNotIn("target-one", records[0]["model_input"]["dialogue_history"])
        self.assertIn("target-one", records[1]["model_input"]["dialogue_history"])
        self.assertNotIn("target-three", records[1]["model_input"]["dialogue_history"])
        self.assertIn("target-three", records[2]["model_input"]["dialogue_history"])

    def test_conflicting_repeated_turn_is_rejected(self):
        rows = [
            "1.0 0 0 hello EOS 1.0 1 1 [Other] target-one\n",
            (
                "0.0 0 0 CHANGED EOS 0.0 1 1 [Other] target-one EOS "
                "0.0 0 2 followup EOS 1.0 1 3 [Questions] target-three\n"
            ),
        ]
        with self.assertRaisesRegex(ValueError, "conflicts with an earlier copy"):
            TRAINER.build_emodynamix_records(
                rows, split="train", expected_rows=None
            )

    def test_record_inputs_may_repeat_but_overlap_filter_removes_all_matches(self):
        train = [
            {"model_input_sha256": "keep", "line_number": 1, "gold": "Other"},
            {"model_input_sha256": "keep", "line_number": 2, "gold": "Questions"},
            {"model_input_sha256": "shared", "line_number": 3, "gold": "Other"},
            {"model_input_sha256": "shared", "line_number": 4, "gold": "Other"},
        ]
        dev = [
            {"model_input_sha256": "shared", "line_number": 1, "gold": "Other"}
        ]

        derived, audit = TRAINER.derive_train_without_dev_overlap(
            train, dev, expected_contract=None
        )

        self.assertEqual([row["model_input_sha256"] for row in derived], ["keep", "keep"])
        self.assertEqual(audit["removed_train_rows"], 2)
        self.assertEqual(audit["unique_overlap_count"], 1)
        self.assertEqual(audit["post_filter_overlap_count"], 0)


class OfficialCanonicalDataTests(unittest.TestCase):
    @unittest.skipUnless(TRAIN_FILE.is_file() and DEV_FILE.is_file(), "official train/dev absent")
    def test_official_contract_and_clean_mapping_commitments(self):
        prepared = TRAINER.prepare_canonical_data(TRAIN_FILE, DEV_FILE)

        self.assertEqual(len(prepared["raw_train_records"]), 8_562)
        self.assertEqual(len(prepared["train_records"]), 8_433)
        self.assertEqual(len(prepared["dev_records"]), 2_985)
        self.assertEqual(prepared["audit"]["removed_train_rows"], 129)
        self.assertEqual(prepared["audit"]["unique_overlap_count"], 12)
        self.assertEqual(prepared["audit"]["post_filter_overlap_count"], 0)
        self.assertEqual(
            TRAINER.mapping_commitment(prepared["raw_train_records"]),
            "ef6795ef6355909a684129eed027b9ee17afba50bec0175345efcd9a81599726",
        )
        self.assertEqual(
            TRAINER.mapping_commitment(prepared["dev_records"]),
            "1de343795f59694f8ef298874621f6e4532e602ec41bf987516342980be13ee5",
        )
        self.assertEqual(
            prepared["audit"]["overlap_set_sha256"],
            "c2499ae0abbfb3d492b3b9bbf2141d77ba7967bfc399db4092f93768c795a6c2",
        )
        self.assertEqual(
            TRAINER.mapping_commitment(prepared["train_records"]),
            "28fed69839d9f0cf41c73957fdbb1a265f02e1b449a2dbbeb97ba72ba00ca810",
        )
        self.assertEqual(
            len(
                {
                    row["model_input_sha256"]
                    for row in prepared["train_records"] + prepared["dev_records"]
                }
            ),
            11_366,
        )

    def test_wrong_filename_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "renamed.tsv"
            wrong.write_text("not official\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filename"):
                TRAINER.load_official_split(wrong, "train")


class CliBoundaryTests(unittest.TestCase):
    def test_cli_has_only_explicit_train_and_dev_data_inputs(self):
        parser = TRAINER.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--train-file", option_strings)
        self.assertIn("--dev-file", option_strings)
        self.assertNotIn("--test", option_strings)
        self.assertNotIn("--test-file", option_strings)
        self.assertNotIn("--dataset-dir", option_strings)


if __name__ == "__main__":
    unittest.main()
