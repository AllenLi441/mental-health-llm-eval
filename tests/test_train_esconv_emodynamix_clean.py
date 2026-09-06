import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
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


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.record = TRAINER.build_emodynamix_records(
            synthetic_rows()[:1], split="train", expected_rows=1
        )[0]

    def test_fixture_feature_is_deterministic_label_free_and_never_selectable(self):
        first = TRAINER.make_structural_fixture_feature(
            self.record["model_input_sha256"], self.record["model_input"]
        )
        second = TRAINER.make_structural_fixture_feature(
            self.record["model_input_sha256"], self.record["model_input"]
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), TRAINER.FEATURE_ROW_FIELDS)
        serialized = json.dumps(first, sort_keys=True).casefold()
        for forbidden in ("gold", "label", "target", "response", "split", "line_number"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(first["node_count"], 2)
        self.assertEqual(len(first["upstream_erc_softmax_output"]), 2)
        for vector in first["upstream_erc_softmax_output"]:
            self.assertEqual(len(vector), 7)
            self.assertAlmostEqual(sum(vector), 1.0, places=7)
        TRAINER.validate_feature_row(
            first,
            expected_input_sha256=self.record["model_input_sha256"],
            expected_generator_manifest_sha256=TRAINER.FIXTURE_GENERATOR_SHA256,
        )
        self.assertEqual(
            TRAINER.feature_table_status([first]),
            "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE",
        )

    def test_feature_validator_rejects_leakage_shapes_edges_and_mutation(self):
        valid = TRAINER.make_structural_fixture_feature(
            self.record["model_input_sha256"], self.record["model_input"]
        )
        cases = []

        leaked = deepcopy(valid)
        leaked["gold"] = "Other"
        cases.append((leaked, "fields"))

        wrong_key = deepcopy(valid)
        wrong_key["model_input_sha256"] = "0" * 64
        cases.append((wrong_key, "input SHA"))

        wrong_shape = deepcopy(valid)
        wrong_shape["upstream_erc_softmax_output"][0] = [1.0] * 6
        cases.append((wrong_shape, "seven"))

        wrong_probability = deepcopy(valid)
        wrong_probability["upstream_erc_softmax_output"][0] = [0.2] * 7
        cases.append((wrong_probability, "sum"))

        wrong_edge = deepcopy(valid)
        wrong_edge["parsed_dialogue"] = [[1, 2, 17]]
        cases.append((wrong_edge, "relation"))

        unsorted_edges = deepcopy(valid)
        unsorted_edges["parsed_dialogue"] = [[1, 2, 0], [0, 1, 16]]
        cases.append((unsorted_edges, "sorted"))

        for row, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TRAINER.validate_feature_row(
                        row,
                        expected_input_sha256=self.record["model_input_sha256"],
                        expected_generator_manifest_sha256=(
                            TRAINER.FIXTURE_GENERATOR_SHA256
                        ),
                    )

    def test_feature_table_is_unique_but_duplicate_records_join_many_to_one(self):
        feature = TRAINER.make_structural_fixture_feature(
            self.record["model_input_sha256"], self.record["model_input"]
        )
        payload = (json.dumps(feature, sort_keys=True) + "\n").encode()
        expected_commitment = TRAINER.feature_table_commitment([feature])
        features, audit = TRAINER.load_feature_jsonl_bytes(
            payload,
            required_input_sha256={self.record["model_input_sha256"]},
            expected_generator_manifest_sha256=TRAINER.FIXTURE_GENERATOR_SHA256,
            expected_table_sha256=expected_commitment,
            allow_fixture=True,
        )
        duplicate_records = [self.record, dict(self.record, item_id="duplicate")]
        aligned = TRAINER.resolve_record_features(duplicate_records, features)

        self.assertEqual(len(features), 1)
        self.assertEqual(len(aligned), 2)
        self.assertIs(aligned[0], aligned[1])
        self.assertEqual(audit["feature_rows"], 1)
        self.assertEqual(
            audit["status"], "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE"
        )

    def test_feature_table_rejects_duplicate_missing_extra_and_changed_rows(self):
        feature = TRAINER.make_structural_fixture_feature(
            self.record["model_input_sha256"], self.record["model_input"]
        )
        line = json.dumps(feature, sort_keys=True) + "\n"
        common = {
            "expected_generator_manifest_sha256": TRAINER.FIXTURE_GENERATOR_SHA256,
            "allow_fixture": True,
        }
        with self.assertRaisesRegex(ValueError, "duplicate feature key"):
            TRAINER.load_feature_jsonl_bytes(
                (line + line).encode(),
                required_input_sha256={self.record["model_input_sha256"]},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "missing feature"):
            TRAINER.load_feature_jsonl_bytes(
                b"",
                required_input_sha256={self.record["model_input_sha256"]},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "extra feature"):
            TRAINER.load_feature_jsonl_bytes(
                line.encode(),
                required_input_sha256=set(),
                **common,
            )

        expected_commitment = TRAINER.feature_table_commitment([feature])
        changed = deepcopy(feature)
        original_vector = changed["upstream_erc_softmax_output"][0]
        changed["upstream_erc_softmax_output"][0] = (
            original_vector[-1:] + original_vector[:-1]
        )
        with self.assertRaisesRegex(ValueError, "commitment"):
            TRAINER.load_feature_jsonl_bytes(
                (json.dumps(changed, sort_keys=True) + "\n").encode(),
                required_input_sha256={self.record["model_input_sha256"]},
                expected_table_sha256=expected_commitment,
                **common,
            )


if __name__ == "__main__":
    unittest.main()
