import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_esconv_policy.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_esconv_policy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


def sample_rows():
    return [
        "1.0 0 0 Hello EOS 1.0 1 1 [Questions] SECRET TARGET ONE\n",
        (
            "1.0 0 0 Hello EOS "
            "1.0 1 1 [Questions] How are you feeling? EOS "
            "1.0 0 2 Not good EOS "
            "1.0 1 3 [Information] SECRET TARGET TWO\n"
        ),
        "1.0 0 0 New dialogue EOS 1.0 1 1 [Other] SECRET TARGET THREE\n",
    ]


class ESConvPolicyTrainerTests(unittest.TestCase):
    def test_official_contract_pins_only_train_and_dev(self):
        self.assertEqual(
            TRAINER.OFFICIAL_SPLITS,
            {
                "train": {
                    "filename": "trainWithStrategy_short.tsv",
                    "sha256": "0ecf37462f8e3fa7f1dc5bfbecd70733abb3501c3bf83b27a957bec5a926ec21",
                    "rows": 8562,
                },
                "dev": {
                    "filename": "devWithStrategy_short.tsv",
                    "sha256": "625b511f40cf9a9285582e808e6bbb4c2f35d0b0cd063f3709db430b8d0c3bdc",
                    "rows": 2985,
                },
            },
        )
        self.assertNotIn("test", json.dumps(TRAINER.OFFICIAL_SPLITS).lower())

    def test_parser_structurally_discards_target_response_and_strategy_from_input(self):
        records = TRAINER.parse_split_lines(
            sample_rows(), split="train", expected_rows=None
        )
        self.assertEqual(records[0]["conversation_id"], "esconv-train-0001")
        self.assertEqual(records[1]["conversation_id"], "esconv-train-0001")
        self.assertEqual(records[2]["conversation_id"], "esconv-train-0002")
        self.assertEqual(records[1]["label"], "Information")
        self.assertEqual(records[1]["label_id"], TRAINER.LABELS.index("Information"))
        self.assertEqual(
            records[1]["input_text"],
            "Seeker: Hello\n"
            "Supporter [Questions]: How are you feeling?\n"
            "Seeker: Not good",
        )
        serialized = json.dumps(records)
        self.assertNotIn("SECRET TARGET", serialized)
        self.assertNotIn("target_response", serialized)
        self.assertNotIn("Supporter [Information]", records[1]["input_text"])

    def test_parser_rejects_nonprior_context_and_unknown_target_label(self):
        nonprior = (
            "1.0 0 2 future EOS 1.0 1 1 [Questions] discarded response\n"
        )
        with self.assertRaisesRegex(ValueError, "non-prior"):
            TRAINER.parse_split_lines([nonprior], split="dev", expected_rows=None)
        unknown = "1.0 0 0 hello EOS 1.0 1 1 [Made up] response\n"
        with self.assertRaisesRegex(ValueError, "unknown ESConv strategy"):
            TRAINER.parse_split_lines([unknown], split="dev", expected_rows=None)

    def test_test_split_and_test_cli_option_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "train/dev"):
            TRAINER.parse_split_lines(sample_rows(), split="test", expected_rows=None)
        parser = TRAINER.build_parser()
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--test-file", "/tmp/test.tsv"])
        for action in parser._actions:
            self.assertNotIn("test", " ".join(action.option_strings).lower())

    def test_official_file_verification_checks_filename_hash_and_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainWithStrategy_short.tsv"
            payload = "row one\nrow two\n"
            path.write_text(payload, encoding="utf-8")
            spec = {
                "filename": path.name,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "rows": 2,
            }
            with mock.patch.dict(TRAINER.OFFICIAL_SPLITS, {"train": spec}):
                audit = TRAINER.verify_official_split_file(path, "train")
            self.assertEqual(audit["rows"], 2)
            self.assertEqual(audit["sha256"], spec["sha256"])

            wrong_name = Path(directory) / "renamed.tsv"
            wrong_name.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filename"):
                TRAINER.verify_official_split_file(wrong_name, "train")
            path.write_text(payload + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                TRAINER.verify_official_split_file(path, "train")

    def test_base_model_must_be_local_regular_tree_with_exact_commitment(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            base.mkdir()
            (base / "config.json").write_text(
                json.dumps({"model_type": "roberta"}), encoding="utf-8"
            )
            (base / "tokenizer.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").write_bytes(b"weights")
            commitment, entries = TRAINER.hash_model_tree(base)
            audit = TRAINER.validate_base_model_dir(base, commitment)
            self.assertEqual(audit["tree_sha256"], commitment)
            self.assertEqual(audit["files"], entries)
            with self.assertRaisesRegex(ValueError, "commitment"):
                TRAINER.validate_base_model_dir(base, "0" * 64)

            symlink = base / "unsafe-link"
            try:
                symlink.symlink_to(base / "config.json")
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "symlink"):
                TRAINER.hash_model_tree(base)

    def test_loss_modes_are_explicit_and_reward_rare_classes(self):
        import torch

        counts = [100, 50, 25, 20, 10, 5, 2, 1]
        weights = TRAINER.class_balanced_weights(counts, beta=0.999)
        self.assertEqual(tuple(weights.shape), (8,))
        self.assertGreater(float(weights[-1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)

        logits = torch.tensor([[2.0, 0.0] + [-1.0] * 6, [0.0, 2.0] + [-1.0] * 6])
        gold = torch.tensor([0, 1])
        ce = TRAINER.policy_loss(logits, gold, mode="ce", class_counts=counts)
        balanced = TRAINER.policy_loss(
            logits, gold, mode="class_balanced", class_counts=counts
        )
        adjusted = TRAINER.policy_loss(
            logits,
            gold,
            mode="logit_adjusted",
            class_counts=counts,
            logit_adjustment_tau=1.0,
        )
        self.assertTrue(torch.isfinite(ce))
        self.assertTrue(torch.isfinite(balanced))
        self.assertTrue(torch.isfinite(adjusted))
        with self.assertRaisesRegex(ValueError, "loss mode"):
            TRAINER.policy_loss(logits, gold, mode="unknown", class_counts=counts)

    def test_dev_selection_is_macro_f1_first(self):
        incumbent = {"macro_f1": 0.40, "accuracy": 0.90, "loss": 0.5}
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.41, "accuracy": 0.10, "loss": 9.0}, incumbent
            )
        )
        self.assertFalse(
            TRAINER.is_better_dev(
                {"macro_f1": 0.39, "accuracy": 0.99, "loss": 0.1}, incumbent
            )
        )
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.40, "accuracy": 0.91, "loss": 9.0}, incumbent
            )
        )

    def test_checkpoint_manifest_and_receipt_bind_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "best_model_state.pt"
            checkpoint.write_bytes(b"checkpoint bytes")
            manifest_path = output / "best_checkpoint_manifest.json"
            result = TRAINER.write_checkpoint_manifest(
                checkpoint,
                manifest_path,
                {
                    "protocol_id": "esconv-policy-roberta-v1",
                    "selection": {"primary": "dev_macro_f1", "selected_epoch": 2},
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint bytes").hexdigest(),
            )
            self.assertEqual(manifest["checkpoint"]["size"], 16)
            self.assertEqual(result["manifest_sha256"], TRAINER.sha256_file(manifest_path))
            receipt = Path(result["receipt_path"]).read_text(encoding="utf-8")
            self.assertEqual(
                receipt,
                f"{result['manifest_sha256']}  {manifest_path.name}\n",
            )

    def test_cli_defaults_are_safe_and_keep_recent_context(self):
        parser = TRAINER.build_parser()
        args = parser.parse_args(
            [
                "--train-file",
                "/data/trainWithStrategy_short.tsv",
                "--dev-file",
                "/data/devWithStrategy_short.tsv",
                "--base-model-dir",
                "/models/roberta",
                "--base-model-sha256",
                "a" * 64,
            ]
        )
        self.assertFalse(args.execute)
        self.assertEqual(args.max_length, 256)
        self.assertEqual(args.truncation_side, "left")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.loss, "ce")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(args.train_batch_size, 8)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.eval_batch_size, 16)
        self.assertEqual(args.class_balance_beta, 0.999)

    def test_selftest_covers_contract_without_loading_transformers(self):
        result = TRAINER.run_selftest()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["target_response_retained"], False)
        self.assertEqual(result["selection_primary"], "dev_macro_f1")

    def test_manual_loop_writes_selected_checkpoint_with_fake_local_model(self):
        import torch

        class FakeTokenizer:
            truncation_side = "right"

            def __call__(self, texts, **_kwargs):
                ids = [[2 + (len(text) % 7), 3, 4] for text in texts]
                return {
                    "input_ids": ids,
                    "attention_mask": [[1] * len(row) for row in ids],
                }

            def pad(self, features, *, padding, return_tensors):
                self.assert_pad_contract(padding, return_tensors)
                return {
                    key: torch.tensor([feature[key] for feature in features])
                    for key in features[0]
                }

            @staticmethod
            def assert_pad_contract(padding, return_tensors):
                if padding is not True or return_tensors != "pt":
                    raise AssertionError("unexpected padding contract")

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(1, len(TRAINER.LABELS))

            def forward(self, input_ids, attention_mask):
                del attention_mask
                feature = input_ids.float().mean(dim=1, keepdim=True)
                return types.SimpleNamespace(logits=self.classifier(feature))

        class Factory:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeModel()

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeTokenizer()

        class ConfigFactory:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return types.SimpleNamespace(model_type="roberta")

        class Scheduler:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def step(self):
                return None

            def get_last_lr(self):
                return [self.optimizer.param_groups[0]["lr"]]

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "5.0.fake"
        fake_transformers.AutoConfig = ConfigFactory
        fake_transformers.AutoModelForSequenceClassification = Factory
        fake_transformers.AutoTokenizer = TokenizerFactory
        fake_transformers.get_linear_schedule_with_warmup = (
            lambda optimizer, **_kwargs: Scheduler(optimizer)
        )
        fake_transformers.utils = types.SimpleNamespace(
            logging=types.SimpleNamespace(disable_progress_bar=lambda: None)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text(
                json.dumps({"model_type": "roberta"}), encoding="utf-8"
            )
            (base / "tokenizer.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").write_bytes(b"immutable base")
            base_hash, _entries = TRAINER.hash_model_tree(base)
            base_audit = TRAINER.validate_base_model_dir(base, base_hash)
            output = root / "output"
            args = TRAINER.build_parser().parse_args(
                [
                    "--train-file",
                    "/data/trainWithStrategy_short.tsv",
                    "--dev-file",
                    "/data/devWithStrategy_short.tsv",
                    "--base-model-dir",
                    str(base),
                    "--base-model-sha256",
                    base_hash,
                    "--output-dir",
                    str(output),
                    "--execute",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--train-batch-size",
                    "8",
                    "--eval-batch-size",
                    "8",
                    "--gradient-accumulation-steps",
                    "1",
                ]
            )
            records = [
                {
                    "input_text": f"Seeker: record {index}",
                    "label": label,
                    "label_id": index,
                }
                for index, label in enumerate(TRAINER.LABELS)
            ]
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                with redirect_stdout(StringIO()):
                    result = TRAINER.train_policy(
                        args,
                        records,
                        records,
                        {"split": "train", "rows": 8},
                        {"split": "dev", "rows": 8},
                        base_audit,
                    )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["primary"], "dev_macro_f1")
            self.assertEqual(manifest["selection"]["selected_epoch"], 1)
            self.assertEqual(manifest["training"]["epochs_completed"], 1)
            self.assertEqual(
                manifest["run_scope"]["run_scope"],
                "developmental_smoke_or_nonfrozen_configuration",
            )
            self.assertEqual(
                result["checkpoint_sha256"],
                TRAINER.sha256_file(output / TRAINER.CHECKPOINT_NAME),
            )


if __name__ == "__main__":
    unittest.main()
