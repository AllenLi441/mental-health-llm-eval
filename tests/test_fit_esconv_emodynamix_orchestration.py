import argparse
import importlib.util
import json
import sys
import tempfile
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


FIT = load_module("fit_esconv_emodynamix_clean_for_orchestration", FIT_SCRIPT)
DATA = load_module("train_esconv_emodynamix_clean_for_orchestration", DATA_SCRIPT)
REAL_MODEL = load_module("esconv_emodynamix_model_for_orchestration", MODEL_MODULE)


class TinyGraphPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 8))

    def forward(self, model_batch):
        batch_size = len(model_batch["dialogue_history"])
        return {"logits": self.logits.unsqueeze(0).expand(batch_size, -1)}


class FakeModelModule:
    collate_clean_model_batch = staticmethod(REAL_MODEL.collate_clean_model_batch)

    @staticmethod
    def load_clean_emodynamix_model(**kwargs):
        return TinyGraphPolicy(), {
            "base_model_tree_sha256": kwargs["expected_base_tree_sha256"],
            "task_checkpoint_sha256": None,
            "initialization": "test_random_clean_components",
        }


def sample_prepared_and_features():
    train = DATA.build_emodynamix_records(
        [
            "1.0 0 0 first EOS 1.0 1 1 [Other] target one\n",
            "1.0 0 0 second EOS 1.0 1 1 [Questions] target two\n",
            "1.0 0 0 third EOS 1.0 1 1 [Information] target three\n",
        ],
        split="train",
        expected_rows=3,
    )
    dev = DATA.build_emodynamix_records(
        [
            "1.0 0 0 dev EOS 1.0 1 1 [Other] target dev\n",
        ],
        split="dev",
        expected_rows=1,
    )
    records = [*train, *dev]
    features = {
        record["model_input_sha256"]: DATA.make_structural_fixture_feature(
            record["model_input_sha256"], record["model_input"]
        )
        for record in records
    }
    return {
        "train_records": train,
        "dev_records": dev,
        "source_audit": {"train": {"rows": 3}, "dev": {"rows": 1}},
        "audit": {"post_filter_overlap_count": 0},
    }, features


class CleanTrainingOrchestrationTests(unittest.TestCase):
    def test_fixture_smoke_trains_saves_and_remains_nonselectable(self):
        prepared, features = sample_prepared_and_features()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                mode="smoke",
                output_dir=Path(directory) / "run",
                base_model_dir=Path(directory) / "fake-base",
                expected_base_tree_sha256="b" * 64,
                device="cpu",
                seed=7,
                epochs=1,
                train_batch_size=2,
                gradient_accumulation_steps=2,
                eval_batch_size=1,
                learning_rate=1e-2,
                weight_decay=1e-3,
                warmup_updates=0,
                max_updates=1,
                max_grad_norm=1.0,
                loss="ce",
                author_weight_temperature=1.75,
                class_balance_beta=0.999,
                logit_adjustment_tau=1.0,
            )
            result = FIT.train_emodynamix(
                args,
                prepared=prepared,
                feature_run={
                    "features_by_input_sha256": features,
                    "audit": {
                        "status": "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE",
                        "feature_run_summary_sha256": "a" * 64,
                        "verified_upstream_features": False,
                    },
                },
                execution_provenance={
                    "protocol_id": "smoke-only",
                    "execution_validated": False,
                },
                model_module=FakeModelModule,
            )

            self.assertEqual(result["training"]["actual_optimizer_updates"], 1)
            manifest = json.loads(
                (args.output_dir / "training_manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE")
            self.assertFalse(manifest["selectable_model_produced"])
            self.assertFalse(manifest["frozen_leaderboard_eligible"])
            self.assertEqual(manifest["feature_run"]["status"], "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE")
            self.assertIsNone(manifest["model_initialization"]["task_checkpoint_sha256"])

    def test_pilot_rejects_unverified_features_or_unfrozen_execution(self):
        prepared, features = sample_prepared_and_features()
        args = argparse.Namespace(mode="pilot")
        with self.assertRaisesRegex(ValueError, "verified feature"):
            FIT.validate_training_execution(
                args,
                prepared=prepared,
                feature_run={
                    "features_by_input_sha256": features,
                    "audit": {"verified_upstream_features": False},
                },
                execution_provenance={"execution_validated": True},
            )
        with self.assertRaisesRegex(ValueError, "frozen preregistration"):
            FIT.validate_training_execution(
                args,
                prepared=prepared,
                feature_run={
                    "features_by_input_sha256": features,
                    "audit": {"verified_upstream_features": True},
                },
                execution_provenance={"execution_validated": False},
            )


if __name__ == "__main__":
    unittest.main()
