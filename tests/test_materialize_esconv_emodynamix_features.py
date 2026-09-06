import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_esconv_emodynamix_features.py"
TRAINER_SCRIPT = ROOT / "scripts/train_esconv_emodynamix_clean.py"
FEATURE_MODULE = ROOT / "open_response_eval/esconv_emodynamix_features.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = load_module("materialize_esconv_emodynamix_features", SCRIPT)
TRAINER = load_module("train_esconv_emodynamix_clean_for_materializer", TRAINER_SCRIPT)
FEATURES = load_module("esconv_emodynamix_features_for_materializer", FEATURE_MODULE)


class FakeRuntime:
    def parse_batch(self, dialogues):
        return [
            {(0, 1, 16), (1, len(dialogue), 0)} for dialogue in dialogues
        ]

    def erc_batch(self, contexts, node_counts):
        del contexts
        return [
            [[0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]] * count
            for count in node_counts
        ]


def sample_prepared():
    records = TRAINER.build_emodynamix_records(
        [
            "1.0 0 0 first EOS 1.0 1 1 [Other] target one\n",
            "1.0 0 0 second EOS 1.0 1 1 [Questions] target two\n",
        ],
        split="train",
        expected_rows=2,
    )
    duplicate = dict(records[0], item_id="duplicate-record")
    return {
        "train_records": [records[0], duplicate],
        "dev_records": [records[1]],
        "source_audit": {"train": {"rows": 2}, "dev": {"rows": 1}},
        "audit": {"post_filter_overlap_count": 0},
    }


def runtime_receipt():
    return {
        "upstream_commit": "c9213d718a9684a5e05ce5daa947f9cbbfb7b927",
        "upstream_feature_code_tree_sha256": "1" * 64,
        "base_model_tree_sha256": "2" * 64,
        "sddp_tree_sha256": "3" * 64,
        "sddp_weights_sha256": "4" * 64,
        "erc_weights_sha256": "5" * 64,
        "device": "cpu",
        "sddp_max_num_contexts": 37,
    }


def bound_fake_runtime(receipt):
    runtime = FakeRuntime()
    runtime.runtime_receipt_sha256 = FEATURES.canonical_json_sha256(receipt)
    return runtime


class MaterializerBoundaryTests(unittest.TestCase):
    def test_cli_is_read_only_by_default_and_has_no_test_or_dataset_input(self):
        parser = MATERIALIZER.build_parser()
        args = parser.parse_args([])
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertFalse(args.execute)
        self.assertIn("--train-file", option_strings)
        self.assertIn("--dev-file", option_strings)
        self.assertIn("--run-dir", option_strings)
        self.assertNotIn("--test", option_strings)
        self.assertNotIn("--test-file", option_strings)
        self.assertNotIn("--dataset-dir", option_strings)

    def test_execution_materializes_unique_features_in_atomic_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "feature-run"
            receipt = runtime_receipt()
            summary = MATERIALIZER.execute_feature_materialization(
                prepared=sample_prepared(),
                runtime=bound_fake_runtime(receipt),
                runtime_receipt=receipt,
                run_dir=run_dir,
                batch_size=2,
                feature_implementation_path=FEATURE_MODULE,
                materializer_path=SCRIPT,
            )

            self.assertTrue(run_dir.is_dir())
            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                [
                    "feature_run_summary.json",
                    "features.jsonl",
                    "generator_manifest.json",
                ],
            )
            feature_lines = (run_dir / "features.jsonl").read_text().splitlines()
            self.assertEqual(len(feature_lines), 2)
            features = [json.loads(line) for line in feature_lines]
            self.assertEqual(
                [row["model_input_sha256"] for row in features],
                sorted(row["model_input_sha256"] for row in features),
            )
            self.assertEqual(summary["feature_rows"], 2)
            self.assertEqual(summary["training_records"], 2)
            self.assertEqual(summary["development_records"], 1)
            self.assertEqual(
                summary["status"],
                "VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION",
            )
            self.assertEqual(
                summary["features_jsonl_sha256"],
                hashlib.sha256((run_dir / "features.jsonl").read_bytes()).hexdigest(),
            )
            manifest = json.loads((run_dir / "generator_manifest.json").read_text())
            self.assertEqual(manifest["feature_batch_size"], 2)
            self.assertEqual(manifest["sddp_max_num_contexts"], 37)
            self.assertNotIn("sddp_context_bound_parity", manifest)
            manifest_sha = FEATURES.generator_manifest_sha256(manifest)
            self.assertTrue(
                all(row["generator_manifest_sha256"] == manifest_sha for row in features)
            )
            self.assertEqual(
                summary,
                json.loads((run_dir / "feature_run_summary.json").read_text()),
            )

    def test_runtime_must_be_bound_to_the_exact_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "feature-run"
            runtime = FakeRuntime()
            runtime.runtime_receipt_sha256 = "f" * 64
            with self.assertRaisesRegex(ValueError, "runtime.*receipt"):
                MATERIALIZER.execute_feature_materialization(
                    prepared=sample_prepared(),
                    runtime=runtime,
                    runtime_receipt=runtime_receipt(),
                    run_dir=run_dir,
                    batch_size=2,
                    feature_implementation_path=FEATURE_MODULE,
                    materializer_path=SCRIPT,
                )
            self.assertFalse(run_dir.exists())

    def test_existing_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "feature-run"
            run_dir.mkdir()
            sentinel = run_dir / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                receipt = runtime_receipt()
                MATERIALIZER.execute_feature_materialization(
                    prepared=sample_prepared(),
                    runtime=bound_fake_runtime(receipt),
                    runtime_receipt=receipt,
                    run_dir=run_dir,
                    batch_size=2,
                    feature_implementation_path=FEATURE_MODULE,
                    materializer_path=SCRIPT,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")

    def test_main_audit_never_loads_runtime_or_writes(self):
        output = StringIO()
        with mock.patch.object(
            MATERIALIZER.DATA,
            "prepare_canonical_data",
            return_value=sample_prepared(),
        ), mock.patch.object(
            MATERIALIZER.FEATURES,
            "load_verified_feature_runtime",
        ) as load_runtime, redirect_stdout(output):
            result = MATERIALIZER.main([])

        self.assertEqual(result, 0)
        load_runtime.assert_not_called()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["status"], "FEATURE_MATERIALIZATION_AUDIT_ONLY")
        self.assertFalse(summary["artifact_written"])
        self.assertFalse(summary["model_loaded"])
        self.assertEqual(summary["planned_unique_feature_rows"], 2)

    def test_main_execute_loads_runtime_only_after_explicit_flag(self):
        output = StringIO()
        fake_summary = {
            "status": "VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION"
        }

        def fake_execute(**kwargs):
            kwargs["progress_callback"]({"completed": 2, "total": 2})
            return fake_summary

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MATERIALIZER.DATA,
            "prepare_canonical_data",
            return_value=sample_prepared(),
        ), mock.patch.object(
            MATERIALIZER.FEATURES,
            "load_verified_feature_runtime",
            return_value=(FakeRuntime(), runtime_receipt()),
        ) as load_runtime, mock.patch.object(
            MATERIALIZER,
            "execute_feature_materialization",
            side_effect=fake_execute,
        ) as execute, redirect_stdout(output):
            result = MATERIALIZER.main(
                [
                    "--execute",
                    "--device",
                    "cpu",
                    "--run-dir",
                    str(Path(directory) / "run"),
                ]
            )

        self.assertEqual(result, 0)
        load_runtime.assert_called_once()
        execute.assert_called_once()
        self.assertIn('"completed": 2', output.getvalue())
        self.assertIn("VERIFIED_FEATURES_REQUIRES_TRAINING_PREREGISTRATION", output.getvalue())

    def test_publish_target_must_be_absolute_and_non_symlinked(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            MATERIALIZER._validate_publish_target(Path("relative/run"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked ancestor"):
                MATERIALIZER._validate_publish_target(link / "new-run")


if __name__ == "__main__":
    unittest.main()
