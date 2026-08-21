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
FIT_SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"
DATA_SCRIPT = ROOT / "scripts/train_esconv_emodynamix_clean.py"
MATERIALIZER_SCRIPT = ROOT / "scripts/materialize_esconv_emodynamix_features.py"
FEATURE_MODULE = ROOT / "open_response_eval/esconv_emodynamix_features.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_module("fit_esconv_emodynamix_clean_for_bundle", FIT_SCRIPT)
DATA = load_module("train_esconv_emodynamix_clean_for_fit_bundle", DATA_SCRIPT)
FEATURES = load_module("esconv_emodynamix_features_for_fit_bundle", FEATURE_MODULE)
MATERIALIZER = load_module(
    "materialize_esconv_emodynamix_features_for_fit_bundle", MATERIALIZER_SCRIPT
)


class FakeRuntime:
    def parse_batch(self, dialogues):
        return [{(0, 1, 16), (1, len(dialogue), 0)} for dialogue in dialogues]

    def erc_batch(self, contexts, node_counts):
        del contexts
        return [
            [[0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]] * count
            for count in node_counts
        ]


def sample_prepared():
    records = DATA.build_emodynamix_records(
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


def materialize_fixture(parent, *, runtime_overrides=None):
    prepared = sample_prepared()
    receipt = runtime_receipt()
    receipt.update(runtime_overrides or {})
    runtime = FakeRuntime()
    runtime.runtime_receipt_sha256 = FEATURES.canonical_json_sha256(receipt)
    run_dir = Path(parent) / "feature-run"
    MATERIALIZER.execute_feature_materialization(
        prepared=prepared,
        runtime=runtime,
        runtime_receipt=receipt,
        run_dir=run_dir,
        batch_size=2,
        feature_implementation_path=FEATURE_MODULE,
        materializer_path=MATERIALIZER_SCRIPT,
    )
    summary = json.loads((run_dir / "feature_run_summary.json").read_text())
    contract = {
        "feature_run_summary_sha256": hashlib.sha256(
            (run_dir / "feature_run_summary.json").read_bytes()
        ).hexdigest(),
        "training_records": 2,
        "development_records": 1,
        "feature_rows": 2,
        "feature_batch_size": 2,
        "generator_manifest_sha256": summary["generator_manifest_sha256"],
        "features_jsonl_sha256": summary["features_jsonl_sha256"],
        "feature_table_sha256": summary["feature_table_sha256"],
    }
    return prepared, run_dir, contract


class FeatureBundleGateTests(unittest.TestCase):
    def test_three_file_bundle_closes_over_preregistered_summary_and_all_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(directory)
            features, audit = FIT.load_verified_feature_run(
                run_dir,
                prepared=prepared,
                preregistration_contract=contract,
            )

            self.assertEqual(len(features), 2)
            self.assertEqual(
                audit["feature_run_summary_sha256"],
                contract["feature_run_summary_sha256"],
            )
            self.assertEqual(audit["missing_feature_keys"], [])
            self.assertEqual(audit["extra_feature_keys"], [])
            self.assertTrue(audit["verified_upstream_features"])

    def test_mutation_or_wrong_external_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(directory)
            wrong_contract = dict(contract, feature_run_summary_sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "summary SHA"):
                FIT.load_verified_feature_run(
                    run_dir,
                    prepared=prepared,
                    preregistration_contract=wrong_contract,
                )

            features_path = run_dir / "features.jsonl"
            features_path.write_bytes(features_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "features JSONL SHA"):
                FIT.load_verified_feature_run(
                    run_dir,
                    prepared=prepared,
                    preregistration_contract=contract,
                )

    def test_unknown_summary_fields_are_rejected_even_when_reanchored(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(directory)
            summary_path = run_dir / "feature_run_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["frozen_test_metrics"] = {"accuracy": 1.0}
            payload = (
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            summary_path.write_bytes(payload)
            contract["feature_run_summary_sha256"] = hashlib.sha256(
                payload
            ).hexdigest()

            with self.assertRaisesRegex(ValueError, "summary.*unknown"):
                FIT.load_verified_feature_run(
                    run_dir,
                    prepared=prepared,
                    preregistration_contract=contract,
                )

    def test_runtime_must_prove_that_no_test_dataset_was_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(
                directory, runtime_overrides={"test_dataset_loaded": True}
            )
            with self.assertRaisesRegex(ValueError, "test dataset"):
                FIT.load_verified_feature_run(
                    run_dir,
                    prepared=prepared,
                    preregistration_contract=contract,
                )

    def test_symlinked_bundle_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(directory)
            summary_path = run_dir / "feature_run_summary.json"
            real_path = run_dir / "summary-real.json"
            summary_path.rename(real_path)
            summary_path.symlink_to(real_path.name)
            with self.assertRaisesRegex(ValueError, "symlink"):
                FIT.load_verified_feature_run(
                    run_dir,
                    prepared=prepared,
                    preregistration_contract=contract,
                )

    def test_smoke_contract_is_derived_from_exact_summary_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            _prepared, run_dir, expected = materialize_fixture(directory)
            contract = FIT.derive_smoke_feature_contract(run_dir)
            self.assertEqual(contract, expected)

    def test_main_smoke_loads_feature_bundle_only_after_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, run_dir, contract = materialize_fixture(directory)
            output_dir = Path(directory) / "training-run"
            output = StringIO()
            fake_features = {"k": {"feature": True}}
            fake_audit = {
                "feature_run_summary_sha256": contract["feature_run_summary_sha256"],
                "verified_upstream_features": True,
            }
            fake_result = {
                "training": {"best_epoch": 1},
                "output_dir": str(output_dir),
                "artifact_receipt": {"checkpoint_sha256": "c" * 64},
            }
            with mock.patch(
                "scripts.train_esconv_emodynamix_clean.prepare_canonical_data",
                return_value=prepared,
            ), mock.patch.object(
                FIT,
                "load_verified_feature_run",
                return_value=(fake_features, fake_audit),
            ) as load_bundle, mock.patch.object(
                FIT, "train_emodynamix", return_value=fake_result
            ) as train, redirect_stdout(output):
                result = FIT.main(
                    [
                        "--execute",
                        "--mode",
                        "smoke",
                        "--device",
                        "cpu",
                        "--feature-run-dir",
                        str(run_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            load_bundle.assert_called_once()
            self.assertEqual(
                load_bundle.call_args.kwargs["preregistration_contract"], contract
            )
            train.assert_called_once()
            execution = train.call_args.kwargs["execution_provenance"]
            self.assertFalse(execution["execution_validated"])
            self.assertEqual(execution["protocol_id"], "developmental-smoke-only")
            printed = json.loads(output.getvalue())
            self.assertEqual(printed["training"]["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
