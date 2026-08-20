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


def policy_record(index, label, input_text, *, source_line_sha256=None, split="train"):
    input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    return {
        "item_id": index,
        "line_number": index + 1,
        "conversation_id": f"esconv-{split}-{index + 1:04d}",
        "target_turn": 1,
        "label": label,
        "label_id": TRAINER.LABELS.index(label),
        "input_text": input_text,
        "input_sha256": input_sha256,
        "source_line_sha256": source_line_sha256 or hashlib.sha256(
            f"{split}-source-{index}".encode("utf-8")
        ).hexdigest(),
    }


def records_commitment(records):
    return TRAINER.canonical_json_sha256(
        [
            {
                "source_line_sha256": record["source_line_sha256"],
                "input_sha256": record["input_sha256"],
                "label": record["label"],
                "conversation_id": record["conversation_id"],
            }
            for record in records
        ]
    )


class ESConvPolicyTrainerTests(unittest.TestCase):
    def test_official_dev_overlap_contract_is_frozen(self):
        self.assertEqual(
            TRAINER.EXPECTED_DEV_INPUT_OVERLAP_SHA256,
            (
                "3358ea28648260fd2f0aed0819cdfac9b55d44e55c6a9eead8dd28478dd3751d",
                "3dd472de131aea43923eafc47114def5bebeff71eed52de7ebd2c5a42cbcfb44",
                "4ab4aedefae16b8912287b3440a0325ca219c0ec33c704e9465c9bb729f2289e",
                "58a8f9e38ce5dcea33f500e22442e5c42ba134cbeebb7b1240d95a9c62e5bc4e",
                "62a369522f0374e2c1e39c23ffc01b84cb85ec464210a9ea7e545623906a851a",
                "75635a3047863753ee3993183f260d478bd5ce202d59548bcb83821d26c1384f",
                "89812797e3f4500b0cb41157be48c90ad549a57e5a7e556322ab9c07c61ae3a6",
                "9ad0444929093d274cc1bc0bac2886c5581fb5e5b4811e8cf9f69fb8e17cc65e",
                "b1a29810b05b1a1274a96612b9865386567d46513d77961df98ab575bfa8ee2a",
                "d2ac838857de6b685570a1e894f787be8f2637b956433b2e5d4228a5a86d2930",
                "d47c492b14a0d576a2447e5e3630bd6b2cf0f9789718131d7b5c0c017418798a",
                "f60d89294c3747ef88dcac250145dc0789423aec05b96d0ed784afad40a45341",
            ),
        )
        self.assertEqual(
            TRAINER.EXPECTED_DEV_INPUT_OVERLAP_SET_SHA256,
            "b9e1d891f736f1588d6386214413862c43b812c69466589b7f259e2e6d30f6cd",
        )
        self.assertEqual(TRAINER.EXPECTED_REMOVED_TRAIN_ROWS, 129)
        self.assertEqual(TRAINER.EXPECTED_DERIVED_TRAIN_ROWS, 8433)
        self.assertEqual(
            TRAINER.EXPECTED_DERIVED_TRAIN_RECORDS_SHA256,
            "55c098b7a1cf1c9c9c9e8c9da49d95d4c8484552fbcc3d18319affda353b3480",
        )
        self.assertEqual(
            TRAINER.EXPECTED_EXACT_TSV_OVERLAP_SHA256,
            (
                "77be9b60fae1f61a2bb3c1c733aaaad24b95bd4b505c4c45b43123727c80531c",
                "f1085cc07bcb4b13f49c732932b1f9a1a22cac9c6a77180872b021f92f626b0a",
            ),
        )

    def test_dev_overlap_filter_removes_every_matching_train_row_and_fails_closed(self):
        shared_text = "Seeker: shared development context"
        shared_source = "1" * 64
        train = [
            policy_record(0, TRAINER.LABELS[0], shared_text),
            policy_record(
                1,
                TRAINER.LABELS[1],
                shared_text,
                source_line_sha256=shared_source,
            ),
            policy_record(2, TRAINER.LABELS[2], "Seeker: train only"),
        ]
        dev = [
            policy_record(
                0,
                TRAINER.LABELS[3],
                shared_text,
                source_line_sha256=shared_source,
                split="dev",
            )
        ]
        shared_hash = train[0]["input_sha256"]
        derived = [train[2]]
        frozen = {
            "EXPECTED_DEV_INPUT_OVERLAP_SHA256": (shared_hash,),
            "EXPECTED_DEV_INPUT_OVERLAP_SET_SHA256": TRAINER.canonical_json_sha256(
                [shared_hash]
            ),
            "EXPECTED_REMOVED_TRAIN_ROWS": 2,
            "EXPECTED_DERIVED_TRAIN_ROWS": 1,
            "EXPECTED_DERIVED_TRAIN_RECORDS_SHA256": records_commitment(derived),
            "EXPECTED_EXACT_TSV_OVERLAP_SHA256": (shared_source,),
        }
        with mock.patch.multiple(TRAINER, **frozen):
            actual, audit = TRAINER.derive_train_without_dev_overlap(train, dev)
            self.assertEqual(actual, derived)
            self.assertEqual(audit["removed_train_rows"], 2)
            self.assertEqual(audit["derived_train_rows"], 1)
            self.assertEqual(audit["unique_overlap_input_sha256"], [shared_hash])
            self.assertTrue(audit["mandatory"])
            self.assertFalse(audit["disable_option_exists"])
            self.assertFalse(
                {record["input_sha256"] for record in actual}
                & {record["input_sha256"] for record in dev}
            )

            drifted_dev = dev + [
                policy_record(
                    1,
                    TRAINER.LABELS[4],
                    "Seeker: train only",
                    split="dev",
                )
            ]
            with self.assertRaisesRegex(ValueError, "overlap contract drift"):
                TRAINER.derive_train_without_dev_overlap(train, drifted_dev)

        parser = TRAINER.build_parser()
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--disable-dev-overlap-filter"])
        self.assertFalse(
            any(
                "overlap" in option or "leakage" in option or "disable" in option
                for action in parser._actions
                for option in action.option_strings
            )
        )

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
            train_records = [
                policy_record(
                    index,
                    label,
                    f"Seeker: train-only record {index}",
                )
                for index, label in enumerate(TRAINER.LABELS)
            ]
            dev_records = [
                policy_record(
                    index,
                    label,
                    f"Seeker: dev-only record {index}",
                    split="dev",
                )
                for index, label in enumerate(TRAINER.LABELS)
            ]
            raw_train_records = train_records + [
                policy_record(
                    8,
                    TRAINER.LABELS[0],
                    dev_records[0]["input_text"],
                )
            ]
            shared_hash = dev_records[0]["input_sha256"]
            frozen_overlap = {
                "EXPECTED_DEV_INPUT_OVERLAP_SHA256": (shared_hash,),
                "EXPECTED_DEV_INPUT_OVERLAP_SET_SHA256": (
                    TRAINER.canonical_json_sha256([shared_hash])
                ),
                "EXPECTED_REMOVED_TRAIN_ROWS": 1,
                "EXPECTED_DERIVED_TRAIN_ROWS": 8,
                "EXPECTED_DERIVED_TRAIN_RECORDS_SHA256": records_commitment(
                    train_records
                ),
                "EXPECTED_EXACT_TSV_OVERLAP_SHA256": (),
            }
            with mock.patch.multiple(TRAINER, **frozen_overlap):
                with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                    with redirect_stdout(StringIO()):
                        result = TRAINER.train_policy(
                            args,
                            raw_train_records,
                            dev_records,
                            {"split": "train", "rows": 9},
                            {"split": "dev", "rows": 8},
                            base_audit,
                        )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["primary"], "dev_macro_f1")
            self.assertEqual(manifest["selection"]["selected_epoch"], 1)
            self.assertEqual(manifest["training"]["epochs_completed"], 1)
            self.assertEqual(
                manifest["data"]["dev_overlap_filter"]["removed_train_rows"], 1
            )
            self.assertEqual(
                manifest["data"]["dev_overlap_filter"]["derived_train_rows"], 8
            )
            self.assertEqual(
                manifest["training"]["class_counts_in_label_order"], [1] * 8
            )
            self.assertEqual(
                manifest["implementation"]["determinism"],
                {
                    "determinism_mode": "best_effort_cpu",
                    "torch_deterministic_algorithms_enabled": True,
                    "torch_deterministic_algorithms_warn_only": True,
                    "bitwise_reproducible_not_guaranteed": True,
                },
            )
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
