import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "fit_esconv_emodynamix_clean_for_artifacts", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_trainer()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CleanTrainingArtifactTests(unittest.TestCase):
    def test_atomic_bundle_is_weights_only_hash_closed_and_never_frozen_eligible(self):
        state = {
            "classifier.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "classifier.bias": torch.arange(4, dtype=torch.float32),
        }
        manifest_base = {
            "protocol_id": "jingshi-esconv-emodynamix-clean-dev-pilot-v1",
            "feature_run_summary_sha256": "a" * 64,
            "best_epoch": 2,
            "best_dev": {
                "macro_f1": 0.3,
                "accuracy": 0.4,
                "selection_loss": 1.1,
            },
            "task_checkpoint_sha256": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            result = FIT.write_training_artifacts(
                output_dir,
                model_state_dict=state,
                manifest_base=manifest_base,
                run_status="DEVELOPMENTAL_SMOKE_NOT_SELECTABLE",
            )

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["artifact_receipt.json", "best_model.pt", "training_manifest.json"],
            )
            checkpoint = torch.load(
                output_dir / "best_model.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(checkpoint),
                {"checkpoint_schema_version", "label_order", "model_state_dict"},
            )
            for name, tensor in state.items():
                self.assertTrue(torch.equal(checkpoint["model_state_dict"][name], tensor))

            manifest = json.loads((output_dir / "training_manifest.json").read_text())
            receipt = json.loads((output_dir / "artifact_receipt.json").read_text())
            self.assertEqual(manifest["status"], "DEVELOPMENTAL_SMOKE_NOT_SELECTABLE")
            self.assertFalse(manifest["selectable_model_produced"])
            self.assertFalse(manifest["frozen_leaderboard_eligible"])
            self.assertIsNone(manifest["task_checkpoint_sha256"])
            self.assertEqual(manifest["checkpoint_sha256"], sha256(output_dir / "best_model.pt"))
            self.assertEqual(receipt["checkpoint_sha256"], manifest["checkpoint_sha256"])
            self.assertEqual(
                receipt["training_manifest_sha256"],
                sha256(output_dir / "training_manifest.json"),
            )
            self.assertEqual(result["artifact_receipt"], receipt)

            class ReloadTarget(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.classifier = torch.nn.Linear(3, 4)

            target = ReloadTarget()
            reload_audit = FIT.load_training_checkpoint_strict(
                output_dir / "best_model.pt",
                expected_sha256=manifest["checkpoint_sha256"],
                model=target,
            )
            self.assertEqual(reload_audit["missing_keys"], [])
            self.assertEqual(reload_audit["unexpected_keys"], [])
            for name, tensor in state.items():
                self.assertTrue(torch.equal(target.state_dict()[name], tensor))
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                FIT.load_training_checkpoint_strict(
                    output_dir / "best_model.pt",
                    expected_sha256="f" * 64,
                    model=ReloadTarget(),
                )

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                FIT.write_training_artifacts(
                    output_dir,
                    model_state_dict={"weight": torch.ones(1)},
                    manifest_base={"task_checkpoint_sha256": None},
                    run_status="DEVELOPMENTAL_SMOKE_NOT_SELECTABLE",
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()
