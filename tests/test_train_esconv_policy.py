import hashlib
import importlib.util
import json
import os
import subprocess
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


def frozen_pilot_document(frozen_at="2026-08-20T02:30:00-07:00"):
    return {
        "protocol_id": "jingshi-esconv-first-v1",
        "frozen_at": frozen_at,
        "training": {
            "first_policy_pilot": {
                "scope": (
                    "single-seed development pilot; cannot select or freeze the "
                    "formal candidate"
                ),
                "base_model": "materialized local roberta-base",
                "base_model_tree_sha256": (
                    "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c"
                ),
                "seed": 42,
                "arms": [
                    {"id": "ce", "loss": "cross_entropy"},
                    {
                        "id": "class_balanced",
                        "loss": "effective_number_class_balanced_cross_entropy",
                        "beta": 0.999,
                    },
                ],
                "shared_hyperparameters": {
                    "device": "mps",
                    "max_length": 256,
                    "truncation_side": "left",
                    "epochs": 3,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "train_batch_size": 8,
                    "gradient_accumulation_steps": 2,
                    "eval_batch_size": 16,
                    "logit_adjustment_tau": 1.0,
                    "max_grad_norm": 1.0,
                    "early_stopping_patience": 3,
                },
                "optimizer": {
                    "name": "torch.optim.AdamW",
                    "bias_and_layer_norm_weight_decay": 0.0,
                    "other_weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "amsgrad": False,
                    "foreach": False,
                },
                "comparison_primary": "best dev Macro-F1",
                "comparison_secondary": "dev Accuracy",
                "comparison_tertiary": "lower dev loss",
                "determinism": {
                    "mode": "best_effort_mps",
                    "torch_deterministic_algorithms": True,
                    "warn_only": True,
                    "bitwise_reproducibility_guaranteed": False,
                    "checkpoint_sha256_role": (
                        "binds the selected artifact from this run; it is not an "
                        "assertion that reruns produce bit-identical bytes"
                    ),
                },
                "frozen_test_access": "prohibited",
            }
        },
    }


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
        import torch.nn.functional as functional

        counts = [100, 50, 25, 20, 10, 5, 2, 1]
        weights = TRAINER.class_balanced_weights(counts, beta=0.999)
        self.assertEqual(tuple(weights.shape), (8,))
        self.assertGreater(float(weights[-1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        raw_effective = torch.tensor(
            [(1 - 0.999) / (1 - 0.999**count) for count in counts]
        )
        expected_weights = raw_effective / raw_effective.mean()
        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-7, rtol=0))

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
        priors = torch.tensor(counts, dtype=logits.dtype)
        priors = priors / priors.sum()
        expected_adjusted = functional.cross_entropy(logits + priors.log(), gold)
        self.assertTrue(torch.allclose(adjusted, expected_adjusted, atol=0, rtol=0))
        for mode in ("ce", "logit_adjusted"):
            numerator, denominator = TRAINER.policy_loss_components(
                logits,
                gold,
                mode=mode,
                class_counts=counts,
                logit_adjustment_tau=1.0,
            )
            self.assertEqual(float(denominator), 2.0)
            self.assertTrue(
                torch.allclose(
                    numerator / denominator,
                    TRAINER.policy_loss(
                        logits,
                        gold,
                        mode=mode,
                        class_counts=counts,
                        logit_adjustment_tau=1.0,
                    ),
                    atol=0,
                    rtol=0,
                )
            )
        self.assertEqual(
            TRAINER.loss_normalization_disclosure("class_balanced"),
            {
                "mode": "class_balanced",
                "numerator": "sum_target_class_weighted_cross_entropy",
                "denominator": "sum_target_class_weights",
                "gradient_accumulation_scope": (
                    "complete_optimizer_update_window_including_final_partial_window"
                ),
                "evaluation_scope": "complete_development_split",
            },
        )
        with self.assertRaisesRegex(ValueError, "loss mode"):
            TRAINER.policy_loss(logits, gold, mode="unknown", class_counts=counts)

    def test_class_balanced_accumulation_matches_one_full_weighted_window(self):
        import torch
        import torch.nn.functional as functional

        counts = [100, 50, 25, 20, 10, 5, 2, 1]
        labels = torch.tensor([0, 7, 1, 6, 7])
        base_logits = torch.tensor(
            [
                [2.0, 0.0, -0.5, -1.0, 0.3, 0.1, -0.2, 0.4],
                [0.0, -0.5, 0.1, 0.2, -0.2, 0.3, 0.4, 1.4],
                [0.3, 1.3, 0.2, -0.4, 0.1, -0.3, 0.7, 0.0],
                [-0.2, 0.4, 0.1, 0.0, -0.5, 0.3, 1.1, 0.2],
                [0.5, -0.3, 0.2, 0.1, 0.0, -0.4, 0.3, 1.0],
            ],
            dtype=torch.float64,
        )
        weights = TRAINER.class_balanced_weights(counts, beta=0.999).to(
            dtype=base_logits.dtype
        )
        analytic_numerator = functional.cross_entropy(
            base_logits,
            labels,
            weight=weights,
            reduction="sum",
        )
        analytic_denominator = weights[labels].sum()

        full_logits = base_logits.clone().requires_grad_(True)
        full_loss = TRAINER.policy_loss(
            full_logits,
            labels,
            mode="class_balanced",
            class_counts=counts,
        )
        full_loss.backward()

        split_logits = base_logits.clone().requires_grad_(True)
        accumulated_numerator = 0.0
        accumulated_denominator = 0.0
        # The last single-row microbatch exercises an odd/final partial window.
        for start, end in ((0, 2), (2, 4), (4, 5)):
            numerator, denominator = TRAINER.policy_loss_components(
                split_logits[start:end],
                labels[start:end],
                mode="class_balanced",
                class_counts=counts,
            )
            numerator.backward()
            accumulated_numerator += float(numerator.detach())
            accumulated_denominator += float(denominator.detach())
        TRAINER.normalize_accumulated_gradients(
            [split_logits], accumulated_denominator
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(accumulated_numerator, dtype=base_logits.dtype),
                analytic_numerator,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(accumulated_denominator, dtype=base_logits.dtype),
                analytic_denominator,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                split_logits.grad,
                full_logits.grad,
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_dev_loss_is_invariant_to_batch_partition(self):
        import torch

        counts = [100, 50, 25, 20, 10, 5, 2, 1]
        logits = torch.tensor(
            [
                [2.0, 0.0, -0.5, -1.0, 0.3, 0.1, -0.2, 0.4],
                [0.0, -0.5, 0.1, 0.2, -0.2, 0.3, 0.4, 1.4],
                [0.3, 1.3, 0.2, -0.4, 0.1, -0.3, 0.7, 0.0],
                [-0.2, 0.4, 0.1, 0.0, -0.5, 0.3, 1.1, 0.2],
                [0.5, -0.3, 0.2, 0.1, 0.0, -0.4, 0.3, 1.0],
            ]
        )
        labels = torch.tensor([0, 7, 1, 6, 7])

        class FixedLogitModel(torch.nn.Module):
            def forward(self, row_ids):
                return types.SimpleNamespace(logits=logits[row_ids])

        def loader(partitions):
            batches = []
            offset = 0
            for size in partitions:
                batches.append(
                    {
                        "row_ids": torch.arange(offset, offset + size),
                        "labels": labels[offset : offset + size].clone(),
                    }
                )
                offset += size
            return batches

        common = {
            "device": torch.device("cpu"),
            "loss_mode": "class_balanced",
            "class_counts": counts,
            "class_balance_beta": 0.999,
            "logit_adjustment_tau": 1.0,
        }
        one_batch = TRAINER.evaluate_model(
            FixedLogitModel(), loader([5]), **common
        )
        odd_split = TRAINER.evaluate_model(
            FixedLogitModel(), loader([2, 2, 1]), **common
        )
        self.assertAlmostEqual(one_batch["loss"], odd_split["loss"], places=7)
        self.assertEqual(one_batch["confusion_matrix"], odd_split["confusion_matrix"])

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
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(args.train_batch_size, 8)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.eval_batch_size, 16)
        self.assertEqual(args.class_balance_beta, 0.999)

    def test_pilot_scope_gate_binds_every_frozen_parameter_and_excludes_la(self):
        parser = TRAINER.build_parser()
        common = [
            "--train-file",
            "/data/trainWithStrategy_short.tsv",
            "--dev-file",
            "/data/devWithStrategy_short.tsv",
            "--base-model-dir",
            "/models/roberta",
            "--base-model-sha256",
            "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c",
            "--preregistration-sha256",
            "a" * 64,
            "--materialization-receipt",
            "/repo/reports/base-receipt.json",
            "--materialization-receipt-sha256",
            "4ed49f1c0558c3765e52095f6f0ea50030c9dba0387ff4ae6a82e94410784df2",
        ]
        ce_args = parser.parse_args(common + ["--loss", "ce"])
        ce_scope = TRAINER.pilot_scope_audit(ce_args)
        self.assertEqual(ce_scope["run_scope"], "preregistered_pilot_arm")
        self.assertEqual(ce_scope["pilot_arm"], "ce")
        self.assertEqual(ce_scope["deviations"], {})
        self.assertEqual(
            set(ce_scope["actual"]),
            {
                "base_model_sha256",
                "device",
                "seed",
                "loss",
                "class_balance_beta",
                "logit_adjustment_tau",
                "max_length",
                "truncation_side",
                "epochs",
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "train_batch_size",
                "gradient_accumulation_steps",
                "eval_batch_size",
                "max_grad_norm",
                "early_stopping_patience",
                "adam_beta1",
                "adam_beta2",
                "adam_eps",
                "adam_amsgrad",
                "adam_foreach",
            },
        )

        cb_args = parser.parse_args(common + ["--loss", "class_balanced"])
        self.assertEqual(
            TRAINER.pilot_scope_audit(cb_args)["run_scope"],
            "preregistered_pilot_arm",
        )
        la_args = parser.parse_args(common + ["--loss", "logit_adjusted"])
        la_scope = TRAINER.pilot_scope_audit(la_args)
        self.assertNotEqual(la_scope["run_scope"], "preregistered_pilot_arm")
        self.assertIsNone(la_scope["pilot_arm"])
        self.assertIn("loss", la_scope["deviations"])

        for nonpilot_device in ("cpu", "auto"):
            device_args = parser.parse_args(
                common + ["--loss", "ce", "--device", nonpilot_device]
            )
            device_scope = TRAINER.pilot_scope_audit(device_args)
            self.assertNotEqual(
                device_scope["run_scope"], "preregistered_pilot_arm"
            )
            self.assertIsNone(device_scope["pilot_arm"])
            self.assertEqual(
                device_scope["deviations"]["device"]["actual"],
                nonpilot_device,
            )

        wrong_base = parser.parse_args(
            [value if value != common[7] else "b" * 64 for value in common]
        )
        self.assertNotEqual(
            TRAINER.pilot_scope_audit(wrong_base)["run_scope"],
            "preregistered_pilot_arm",
        )

    def test_preregistration_document_freezes_pilot_and_tertiary_selection(self):
        document = frozen_pilot_document()
        audit = TRAINER.validate_preregistration_document(document)
        self.assertEqual(audit["parent_protocol_id"], "jingshi-esconv-first-v1")
        self.assertEqual(audit["frozen_at"], "2026-08-20T02:30:00-07:00")
        self.assertEqual(audit["comparison_tertiary"], "lower dev loss")
        self.assertEqual(audit["eligible_pilot_arms"], ["ce", "class_balanced"])

        missing_timestamp = frozen_pilot_document(frozen_at=None)
        with self.assertRaisesRegex(ValueError, "frozen_at"):
            TRAINER.validate_preregistration_document(missing_timestamp)
        changed = frozen_pilot_document()
        changed["training"]["first_policy_pilot"]["shared_hyperparameters"][
            "max_grad_norm"
        ] = 2.0
        with self.assertRaisesRegex(ValueError, "pilot preregistration"):
            TRAINER.validate_preregistration_document(changed)

        determinism_mutations = (
            ("mode", "best_effort_cpu"),
            ("torch_deterministic_algorithms", False),
            ("warn_only", False),
            ("bitwise_reproducibility_guaranteed", True),
            ("checkpoint_sha256_role", "claims bit-identical reruns"),
        )
        for key, value in determinism_mutations:
            with self.subTest(determinism_key=key):
                tampered = frozen_pilot_document()
                tampered["training"]["first_policy_pilot"]["determinism"][
                    key
                ] = value
                with self.assertRaisesRegex(ValueError, "pilot preregistration"):
                    TRAINER.validate_preregistration_document(tampered)
        extra = frozen_pilot_document()
        extra["training"]["first_policy_pilot"]["determinism"][
            "unregistered"
        ] = True
        with self.assertRaisesRegex(ValueError, "pilot preregistration"):
            TRAINER.validate_preregistration_document(extra)

    def test_materialization_receipt_binds_source_revision_license_and_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text(
                json.dumps({"model_type": "roberta"}), encoding="utf-8"
            )
            (base / "tokenizer.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").write_bytes(b"weights")
            base_hash, entries = TRAINER.hash_model_tree(base)
            base_audit = TRAINER.validate_base_model_dir(base, base_hash)
            receipt = {
                "schema_version": "local-model-materialization-receipt-v1",
                "source": {
                    "model_id": "FacebookAI/roberta-base",
                    "immutable_revision": (
                        "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
                    ),
                    "license": "MIT",
                },
                "materialization": {
                    "source_and_materialized_file_bytes_identical": True,
                    "symlinks_in_materialized_tree": False,
                    "local_tree_sha256": base_hash,
                    "files": entries,
                },
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            receipt_hash = TRAINER.sha256_file(receipt_path)
            audit = TRAINER.validate_materialization_receipt(
                receipt_path, receipt_hash, base_audit
            )
            self.assertEqual(audit["source_model_id"], "FacebookAI/roberta-base")
            self.assertEqual(
                audit["source_model_revision"],
                "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            )
            self.assertEqual(audit["source_model_license"], "MIT")
            self.assertEqual(audit["base_model_tree_sha256"], base_hash)
            self.assertEqual(audit["receipt_sha256"], receipt_hash)

            receipt["source"]["model_id"] = "lookalike/roberta-base"
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source model"):
                TRAINER.validate_materialization_receipt(
                    receipt_path, TRAINER.sha256_file(receipt_path), base_audit
                )

    def test_read_only_protocol_audit_binds_preregistration_and_receipt_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            base = repo / "base"
            base.mkdir()
            (base / "config.json").write_text(
                json.dumps({"model_type": "roberta"}), encoding="utf-8"
            )
            (base / "tokenizer.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").write_bytes(b"weights")
            base_hash, entries = TRAINER.hash_model_tree(base)
            base_audit = TRAINER.validate_base_model_dir(base, base_hash)
            prereg = repo / "prereg.json"
            prereg.write_text(
                json.dumps(frozen_pilot_document()) + "\n", encoding="utf-8"
            )
            receipt = repo / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "local-model-materialization-receipt-v1",
                        "source": {
                            "model_id": "FacebookAI/roberta-base",
                            "immutable_revision": (
                                "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
                            ),
                            "license": "MIT",
                        },
                        "materialization": {
                            "source_and_materialized_file_bytes_identical": True,
                            "symlinks_in_materialized_tree": False,
                            "local_tree_sha256": base_hash,
                            "files": entries,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            metrics = repo / "metrics.py"
            metrics.write_text("METRIC = 1\n", encoding="utf-8")
            trainer = repo / "trainer.py"
            trainer.write_text("TRAINER = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=ESConv Test",
                    "-c",
                    "user.email=esconv@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "audit assets",
                ],
                check=True,
            )
            args = types.SimpleNamespace(
                preregistration_sha256=TRAINER.sha256_file(prereg),
                materialization_receipt=receipt,
                materialization_receipt_sha256=TRAINER.sha256_file(receipt),
            )
            with mock.patch.multiple(
                TRAINER,
                ROOT=repo,
                PREREGISTRATION_PATH=prereg,
                METRICS_PATH=metrics,
                __file__=str(trainer),
            ):
                provenance = TRAINER.prepare_protocol_provenance(
                    args, base_audit, require_execution=False
                )
            self.assertFalse(provenance["validated_for_execution"])
            self.assertEqual(
                provenance["parent_preregistration"]["sha256"],
                args.preregistration_sha256,
            )
            public = TRAINER._public_provenance(provenance)
            self.assertNotIn("_repo", public)
            self.assertEqual(
                set(public["launch_asset_snapshot"]["assets"]),
                {
                    "trainer",
                    "metrics",
                    "preregistration",
                    "materialization_receipt",
                },
            )

    def test_committed_asset_gate_records_freeze_and_rejects_related_dirty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            prereg = repo / "prereg.json"
            metrics = repo / "metrics.py"
            trainer = repo / "trainer.py"
            receipt = repo / "receipt.json"
            timestamp = "2026-08-20T02:30:00-07:00"
            prereg.write_text(
                json.dumps(frozen_pilot_document(timestamp)) + "\n",
                encoding="utf-8",
            )
            metrics.write_text("METRIC = 1\n", encoding="utf-8")
            trainer.write_text("TRAINER = 1\n", encoding="utf-8")
            receipt.write_text("{}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            commit_environment = {
                **os.environ,
                "GIT_AUTHOR_DATE": timestamp,
                "GIT_COMMITTER_DATE": timestamp,
            }
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=ESConv Test",
                    "-c",
                    "user.email=esconv@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "freeze",
                ],
                check=True,
                env=commit_environment,
            )
            assets = {
                "preregistration": prereg,
                "trainer": trainer,
                "metrics": metrics,
                "materialization_receipt": receipt,
            }
            snapshot = TRAINER.capture_protocol_assets(repo, assets)
            validated = TRAINER.validate_execution_asset_state(
                snapshot,
                frozen_pilot_document(timestamp),
                expected_preregistration_sha256=TRAINER.sha256_file(prereg),
            )
            self.assertTrue(validated["validated_for_execution"])
            self.assertRegex(validated["freeze_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(validated["freeze_commit_timestamp"], timestamp)
            self.assertEqual(validated["git_status_porcelain"], "")

            (repo / "unrelated.tmp").write_text("not a protocol asset\n")
            unrelated_dirty = TRAINER.capture_protocol_assets(repo, assets)
            still_valid = TRAINER.validate_execution_asset_state(
                unrelated_dirty,
                frozen_pilot_document(timestamp),
                expected_preregistration_sha256=TRAINER.sha256_file(prereg),
            )
            self.assertTrue(still_valid["validated_for_execution"])
            self.assertTrue(still_valid["git_status_porcelain"])
            self.assertEqual(still_valid["related_git_status_porcelain"], "")

            metrics.write_text("METRIC = 2\n", encoding="utf-8")
            dirty = TRAINER.capture_protocol_assets(repo, assets)
            with self.assertRaisesRegex(ValueError, "dirty or uncommitted"):
                TRAINER.validate_execution_asset_state(
                    dirty,
                    frozen_pilot_document(timestamp),
                    expected_preregistration_sha256=TRAINER.sha256_file(prereg),
                )

    def test_training_cannot_start_without_validated_execution_provenance(self):
        with self.assertRaisesRegex(ValueError, "execution provenance"):
            TRAINER.train_policy(
                None,
                [],
                [],
                {},
                {},
                {},
                execution_provenance={"validated_for_execution": False},
            )
        with self.assertRaisesRegex(ValueError, "incomplete execution provenance"):
            TRAINER.train_policy(
                None,
                [],
                [],
                {},
                {},
                {},
                execution_provenance={"validated_for_execution": True},
            )

    def test_optimizer_groups_and_odd_accumulation_window_are_exact(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(2, 2)
                self.LayerNorm = torch.nn.LayerNorm(2)

        groups, audit = TRAINER.optimizer_parameter_groups(
            TinyModel(), weight_decay=0.01
        )
        self.assertEqual(audit["decay_parameter_names"], ["dense.weight"])
        self.assertEqual(
            audit["no_decay_parameter_names"],
            ["LayerNorm.bias", "LayerNorm.weight", "dense.bias"],
        )
        self.assertEqual([group["weight_decay"] for group in groups], [0.01, 0.0])
        self.assertEqual(TRAINER.accumulation_window_divisor(0, 1055, 2), 2)
        self.assertEqual(TRAINER.accumulation_window_divisor(1053, 1055, 2), 2)
        self.assertEqual(TRAINER.accumulation_window_divisor(1054, 1055, 2), 1)
        self.assertEqual(TRAINER.accumulation_window_divisor(0, 1, 2), 1)

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
                ids = [
                    [11 if "train-only" in text else 22, 3, 4]
                    for text in texts
                ]
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
                self.calls = []

            def forward(self, input_ids, attention_mask):
                del attention_mask
                self.calls.append(
                    {
                        "sentinels": set(input_ids[:, 0].detach().cpu().tolist()),
                        "grad_enabled": torch.is_grad_enabled(),
                        "training": self.training,
                    }
                )
                feature = input_ids.float().mean(dim=1, keepdim=True)
                return types.SimpleNamespace(logits=self.classifier(feature))

        class Factory:
            last_model = None

            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                Factory.last_model = FakeModel()
                return Factory.last_model

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
            execution_provenance = {
                "validated_for_execution": True,
                "parent_preregistration": {
                    "parent_protocol_id": "jingshi-esconv-first-v1",
                    "path": "/repo/open_response_eval/preregistration_esconv_first_v1.json",
                    "sha256": "a" * 64,
                    "freeze_commit": "b" * 40,
                    "freeze_commit_timestamp": "2026-08-20T02:30:00-07:00",
                },
                "materialization_receipt": {
                    "path": "/repo/reports/base-receipt.json",
                    "receipt_sha256": "c" * 64,
                },
                "launch_asset_snapshot": {"assets": {}},
            }
            finalized_provenance = {
                **execution_provenance,
                "end_asset_snapshot": {"assets": {}},
                "assets_unchanged": True,
            }
            with mock.patch.multiple(TRAINER, **frozen_overlap):
                with mock.patch.object(
                    TRAINER,
                    "finalize_execution_provenance",
                    return_value=finalized_provenance,
                ):
                    with mock.patch.object(
                        TRAINER,
                        "revalidate_execution_provenance",
                        return_value=None,
                    ):
                        with mock.patch.dict(
                            sys.modules, {"transformers": fake_transformers}
                        ):
                            with redirect_stdout(StringIO()):
                                result = TRAINER.train_policy(
                                    args,
                                    raw_train_records,
                                    dev_records,
                                    {"split": "train", "rows": 9},
                                    {"split": "dev", "rows": 8},
                                    base_audit,
                                    execution_provenance=execution_provenance,
                                )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["primary"], "dev_macro_f1")
            self.assertEqual(manifest["selection"]["selected_epoch"], 1)
            self.assertEqual(
                manifest["selection"]["tie_breaker_2"], "lower_dev_loss"
            )
            self.assertEqual(manifest["training"]["epochs_completed"], 1)
            self.assertEqual(manifest["training"]["actual_optimizer_updates"], 1)
            self.assertEqual(manifest["training"]["actual_scheduler_steps"], 1)
            self.assertEqual(
                manifest["training"]["stop_reason"], "completed_planned_epochs"
            )
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
                    "mode": "best_effort_cpu",
                    "torch_deterministic_algorithms": True,
                    "warn_only": True,
                    "bitwise_reproducibility_guaranteed": False,
                    "checkpoint_sha256_role": (
                        "binds the selected artifact from this run; it is not an "
                        "assertion that reruns produce bit-identical bytes"
                    ),
                },
            )
            self.assertEqual(
                manifest["training"]["loss_normalization"],
                {
                    "mode": "ce",
                    "numerator": "sum_per_sample_cross_entropy",
                    "denominator": "row_count",
                    "gradient_accumulation_scope": (
                        "complete_optimizer_update_window_including_final_partial_window"
                    ),
                    "evaluation_scope": "complete_development_split",
                },
            )
            self.assertEqual(
                manifest["protocol"]["parent_preregistration"][
                    "parent_protocol_id"
                ],
                "jingshi-esconv-first-v1",
            )
            self.assertTrue(manifest["protocol"]["assets_unchanged"])
            self.assertEqual(
                manifest["training"]["optimizer"]["betas"], [0.9, 0.999]
            )
            self.assertFalse(manifest["training"]["optimizer"]["foreach"])
            training_calls = [
                call for call in Factory.last_model.calls if call["grad_enabled"]
            ]
            evaluation_calls = [
                call for call in Factory.last_model.calls if not call["grad_enabled"]
            ]
            self.assertTrue(training_calls)
            self.assertTrue(evaluation_calls)
            self.assertTrue(all(call["training"] for call in training_calls))
            self.assertTrue(all(not call["training"] for call in evaluation_calls))
            self.assertTrue(
                all(call["sentinels"] == {11} for call in training_calls)
            )
            self.assertTrue(
                all(call["sentinels"] == {22} for call in evaluation_calls)
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
