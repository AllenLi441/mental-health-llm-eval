import importlib.util
import inspect
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODULE = ROOT / "open_response_eval/esconv_emodynamix_features.py"
TRAINER_SCRIPT = ROOT / "scripts/train_esconv_emodynamix_clean.py"
BASE_MODEL = ROOT / "tmp/official_benchmarks/roberta-base-e2da-materialized"
UPSTREAM_REPO = ROOT / "tmp/official_benchmarks/emodynamix-v2"
ASSET_ROOT = (
    UPSTREAM_REPO
    / "recovered/pre_trained_models_official_drive_1KNsoWp1_20260820/pre_trained_models"
)
TRAIN_FILE = (
    ROOT
    / "tmp/official_benchmarks/esconv/codes/dataset/trainWithStrategy_short.tsv"
)
DEV_FILE = (
    ROOT
    / "tmp/official_benchmarks/esconv/codes/dataset/devWithStrategy_short.tsv"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FEATURES = load_module("esconv_emodynamix_features", FEATURE_MODULE)
TRAINER = load_module("train_esconv_emodynamix_clean_for_features", TRAINER_SCRIPT)


class FakeFeatureRuntime:
    def __init__(self):
        self.dialogues = []
        self.erc_contexts = []

    def parse_batch(self, dialogues):
        self.dialogues.extend(dialogues)
        return [
            {(1, node_count, 0), (0, 1, 16)}
            for node_count in (len(dialogue) for dialogue in dialogues)
        ]

    def erc_batch(self, contexts, node_counts):
        self.erc_contexts.extend(contexts)
        outputs = []
        for node_count in node_counts:
            outputs.append(
                [
                    [0.70 if index == node % 7 else 0.05 for index in range(7)]
                    for node in range(node_count)
                ]
            )
        return outputs


def sample_input_rows():
    records = TRAINER.build_emodynamix_records(
        [
            "1.0 0 0 I feel lost EOS 1.0 1 1 [Questions] TARGET ONE\n",
            (
                "0.0 0 0 I feel lost EOS 0.0 1 1 [Questions] TARGET ONE EOS "
                "0.0 0 2 Still lost EOS 1.0 1 3 [Information] TARGET TWO\n"
            ),
        ],
        split="train",
        expected_rows=2,
    )
    return [
        {
            "model_input_sha256": record["model_input_sha256"],
            "model_input": record["model_input"],
        }
        for record in records
    ]


class FeatureGenerationContractTests(unittest.TestCase):
    def test_upstream_rendering_preserves_raw_whitespace_and_history_only(self):
        first = sample_input_rows()[0]
        rendered = FEATURES.render_upstream_feature_inputs(first["model_input"])

        self.assertEqual(
            rendered["dialogue_for_parsing"],
            [
                {"speaker": "None", "text": "<START> "},
                {"speaker": "seeker", "text": " I feel lost"},
            ],
        )
        self.assertEqual(
            rendered["erc_context"],
            " </s> <START>  </s>  I feel lost",
        )
        serialized = json.dumps(rendered, sort_keys=True)
        self.assertNotIn("TARGET ONE", serialized)
        self.assertNotIn("Questions", serialized)

    def test_unique_inputs_allow_duplicate_records_but_reject_hash_collision(self):
        rows = sample_input_rows()
        duplicated = [rows[0], dict(rows[0]), rows[1]]
        unique = FEATURES.unique_causal_inputs(duplicated)
        self.assertEqual(len(unique), 2)
        self.assertEqual(
            [row["model_input_sha256"] for row in unique],
            sorted(row["model_input_sha256"] for row in rows),
        )

        collision = dict(rows[0], model_input=dict(rows[1]["model_input"]))
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            FEATURES.unique_causal_inputs([collision])

    def test_fake_runtime_generates_sorted_label_free_valid_feature_rows(self):
        runtime = FakeFeatureRuntime()
        rows = sample_input_rows()
        progress = []
        generated = FEATURES.generate_verified_feature_rows(
            rows,
            runtime=runtime,
            generator_manifest_sha256="a" * 64,
            batch_size=2,
            progress_callback=progress.append,
        )

        self.assertEqual(len(generated), 2)
        self.assertEqual(
            [row["model_input_sha256"] for row in generated],
            sorted(row["model_input_sha256"] for row in rows),
        )
        for row in generated:
            TRAINER.validate_feature_row(
                row,
                expected_generator_manifest_sha256="a" * 64,
            )
            self.assertEqual(
                row["feature_backend"], TRAINER.VERIFIED_FEATURE_BACKEND
            )
            self.assertEqual(row["parsed_dialogue"], sorted(row["parsed_dialogue"]))
            serialized = json.dumps(row, sort_keys=True).casefold()
            for forbidden in ("gold", "label", "target", "response", "split"):
                self.assertNotIn(forbidden, serialized)
        self.assertEqual(len(runtime.dialogues), 2)
        self.assertEqual(len(runtime.erc_contexts), 2)
        self.assertEqual(progress, [{"completed": 2, "total": 2}])

    def test_generator_manifest_binds_every_model_and_implementation_asset(self):
        receipt = {
            "upstream_commit": "c9213d718a9684a5e05ce5daa947f9cbbfb7b927",
            "upstream_feature_code_tree_sha256": "1" * 64,
            "base_model_tree_sha256": "2" * 64,
            "sddp_tree_sha256": "3" * 64,
            "sddp_weights_sha256": "4" * 64,
            "erc_weights_sha256": "5" * 64,
        }
        manifest = FEATURES.build_generator_manifest(
            runtime_receipt=receipt,
            implementation_sha256="6" * 64,
            device="cpu",
        )
        self.assertEqual(manifest["runtime_assets"], receipt)
        self.assertEqual(manifest["implementation_sha256"], "6" * 64)
        self.assertEqual(manifest["feature_backend"], "verified_upstream_sddp_erc_v1")
        self.assertEqual(manifest["target_or_label_fields"], [])
        self.assertTrue(manifest["author_double_softmax_preserved"])
        self.assertRegex(FEATURES.generator_manifest_sha256(manifest), r"^[0-9a-f]{64}$")

    def test_runtime_loader_signature_has_no_dataset_or_test_argument(self):
        parameters = inspect.signature(FEATURES.load_verified_feature_runtime).parameters
        self.assertEqual(
            set(parameters),
            {
                "upstream_repo",
                "asset_root",
                "base_model_path",
                "expected_base_tree_sha256",
                "device",
            },
        )


@unittest.skipUnless(
    os.environ.get("RUN_REAL_EMODYNAMIX_FEATURE_SMOKE") == "1"
    and ASSET_ROOT.is_dir()
    and BASE_MODEL.is_dir()
    and TRAIN_FILE.is_file()
    and DEV_FILE.is_file(),
    "real feature smoke is explicit and asset-gated",
)
class RealFeatureSmokeTests(unittest.TestCase):
    def test_two_clean_train_inputs_generate_real_valid_features(self):
        prepared = TRAINER.prepare_canonical_data(TRAIN_FILE, DEV_FILE)
        unique = FEATURES.unique_causal_inputs(prepared["train_records"])
        smoke_inputs = unique[:2]
        runtime, receipt = FEATURES.load_verified_feature_runtime(
            upstream_repo=UPSTREAM_REPO,
            asset_root=ASSET_ROOT,
            base_model_path=BASE_MODEL.resolve(),
            expected_base_tree_sha256=(
                "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c"
            ),
            device="mps" if torch_mps_available() else "cpu",
        )
        manifest = FEATURES.build_generator_manifest(
            runtime_receipt=receipt,
            implementation_sha256=FEATURES.sha256_file(FEATURE_MODULE),
            device=receipt["device"],
        )
        manifest_sha = FEATURES.generator_manifest_sha256(manifest)
        rows = FEATURES.generate_verified_feature_rows(
            smoke_inputs,
            runtime=runtime,
            generator_manifest_sha256=manifest_sha,
            batch_size=2,
        )

        self.assertEqual(len(rows), 2)
        for input_row, feature in zip(smoke_inputs, rows):
            self.assertEqual(
                feature["model_input_sha256"], input_row["model_input_sha256"]
            )
            TRAINER.validate_feature_row(
                feature,
                expected_input_sha256=input_row["model_input_sha256"],
                expected_generator_manifest_sha256=manifest_sha,
            )
        self.assertEqual(receipt["erc_load_missing_keys"], [])
        self.assertEqual(receipt["erc_load_unexpected_keys"], [])
        self.assertEqual(receipt["sddp_load_missing_keys"], [])
        self.assertEqual(receipt["sddp_load_unexpected_keys"], [])


def torch_mps_available():
    import torch

    return torch.backends.mps.is_available()


if __name__ == "__main__":
    unittest.main()
