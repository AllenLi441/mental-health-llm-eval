import importlib.util
import math
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("fit_esconv_emodynamix_clean", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


CLASS_COUNTS = (706, 820, 1_523, 1_470, 1_392, 508, 524, 1_490)
CANONICAL_LABELS = (
    "Reflection of feelings",
    "Self-disclosure",
    "Questions",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Restatement or Paraphrasing",
    "Information",
    "Other",
)


class LossContractTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [
                [1.2, -0.2, 0.1, 0.4, -0.5, 0.3, -0.7, 0.0],
                [-0.4, 0.9, 0.2, -0.1, 0.5, -0.8, 0.4, 0.1],
                [0.0, 0.2, -0.5, 1.0, 0.1, 0.3, -0.2, 0.7],
            ],
            dtype=torch.float64,
        )
        self.labels = torch.tensor([0, 5, 7], dtype=torch.long)

    def test_author_weighted_ce_matches_published_formula(self):
        numerator, denominator = TRAINER.loss_components(
            self.logits,
            self.labels,
            mode="author_weighted_ce",
            class_counts=CLASS_COUNTS,
            author_weight_temperature=1.75,
        )
        counts = torch.tensor(CLASS_COUNTS, dtype=self.logits.dtype)
        inverse = counts.sum() / len(CLASS_COUNTS) / counts
        weights = torch.softmax(inverse / 1.75, dim=0)
        nll = functional.cross_entropy(
            self.logits, self.labels, reduction="none"
        )
        self.assertTrue(torch.allclose(numerator, (nll * weights[self.labels]).sum()))
        self.assertTrue(torch.allclose(denominator, weights[self.labels].sum()))

    def test_public_weight_helpers_match_frozen_formulas(self):
        author = TRAINER.author_weighted_ce_weights(
            CLASS_COUNTS, temperature=1.75, dtype=torch.float64
        )
        counts = torch.tensor(CLASS_COUNTS, dtype=torch.float64)
        expected_author = torch.softmax(
            ((counts.sum() / counts.numel()) / counts) / 1.75, dim=0
        )
        self.assertTrue(torch.allclose(author, expected_author))

        balanced = TRAINER.class_balanced_weights(
            CLASS_COUNTS, beta=0.999, dtype=torch.float64
        )
        raw = (1.0 - 0.999) / (1.0 - torch.pow(0.999, counts))
        self.assertTrue(torch.allclose(balanced, raw / raw.mean()))
        self.assertAlmostEqual(balanced.mean().item(), 1.0)

    def test_class_balanced_and_logit_adjusted_formulas_are_exact(self):
        cb_numerator, cb_denominator = TRAINER.loss_components(
            self.logits,
            self.labels,
            mode="class_balanced",
            class_counts=CLASS_COUNTS,
            class_balance_beta=0.999,
        )
        counts = torch.tensor(CLASS_COUNTS, dtype=self.logits.dtype)
        raw_weights = (1.0 - 0.999) / (1.0 - torch.pow(0.999, counts))
        weights = raw_weights / raw_weights.mean()
        expected_cb = functional.cross_entropy(
            self.logits,
            self.labels,
            weight=weights,
            reduction="sum",
        )
        self.assertTrue(torch.allclose(cb_numerator, expected_cb))
        self.assertTrue(torch.allclose(cb_denominator, weights[self.labels].sum()))

        la_numerator, la_denominator = TRAINER.loss_components(
            self.logits,
            self.labels,
            mode="logit_adjusted",
            class_counts=CLASS_COUNTS,
            logit_adjustment_tau=1.0,
        )
        adjusted = self.logits + torch.log(counts / counts.sum())
        expected_la = functional.cross_entropy(
            adjusted, self.labels, reduction="sum"
        )
        self.assertTrue(torch.allclose(la_numerator, expected_la))
        self.assertEqual(la_denominator.item(), 3.0)

    def test_ce_uses_additive_sum_and_row_denominator(self):
        numerator, denominator = TRAINER.loss_components(
            self.logits,
            self.labels,
            mode="ce",
            class_counts=CLASS_COUNTS,
        )
        self.assertTrue(
            torch.allclose(
                numerator,
                functional.cross_entropy(self.logits, self.labels, reduction="sum"),
            )
        )
        self.assertEqual(denominator.item(), 3.0)

    def test_microbatch_weighted_gradient_equals_full_window_gradient(self):
        torch.manual_seed(7)
        full_model = torch.nn.Linear(3, 8, bias=False).double()
        micro_model = torch.nn.Linear(3, 8, bias=False).double()
        micro_model.load_state_dict(full_model.state_dict())
        inputs = torch.tensor(
            [[1.0, 0.0, 0.2], [0.1, 0.3, 0.5], [0.7, 0.2, 0.4]],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 0, 5], dtype=torch.long)

        full_num, full_den = TRAINER.loss_components(
            full_model(inputs),
            labels,
            mode="author_weighted_ce",
            class_counts=CLASS_COUNTS,
        )
        (full_num / full_den).backward()

        denominator = 0.0
        for batch_slice in (slice(0, 2), slice(2, 3)):
            numerator, current_denominator = TRAINER.loss_components(
                micro_model(inputs[batch_slice]),
                labels[batch_slice],
                mode="author_weighted_ce",
                class_counts=CLASS_COUNTS,
            )
            numerator.backward()
            denominator += current_denominator.item()
        TRAINER.normalize_accumulated_gradients(
            micro_model.parameters(), denominator
        )

        self.assertTrue(
            torch.allclose(
                full_model.weight.grad,
                micro_model.weight.grad,
                rtol=1e-12,
                atol=1e-12,
            )
        )

    def test_every_loss_arm_has_full_window_microbatch_gradient_parity(self):
        inputs = torch.tensor(
            [[1.0, 0.0, 0.2], [0.1, 0.3, 0.5], [0.7, 0.2, 0.4]],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 0, 5], dtype=torch.long)
        for mode in TRAINER.LOSS_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(9)
                full_model = torch.nn.Linear(3, 8, bias=False).double()
                micro_model = torch.nn.Linear(3, 8, bias=False).double()
                micro_model.load_state_dict(full_model.state_dict())
                full_num, full_den = TRAINER.loss_components(
                    full_model(inputs), labels, mode=mode, class_counts=CLASS_COUNTS
                )
                (full_num / full_den).backward()

                denominator = 0.0
                for batch_slice in (slice(0, 2), slice(2, 3)):
                    numerator, current_denominator = TRAINER.loss_components(
                        micro_model(inputs[batch_slice]),
                        labels[batch_slice],
                        mode=mode,
                        class_counts=CLASS_COUNTS,
                    )
                    numerator.backward()
                    denominator += current_denominator.item()
                TRAINER.normalize_accumulated_gradients(
                    micro_model.parameters(), denominator
                )
                self.assertTrue(
                    torch.allclose(
                        full_model.weight.grad,
                        micro_model.weight.grad,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )

    def test_loss_rejects_invalid_hyperparameters_and_shapes(self):
        with self.assertRaisesRegex(ValueError, "eight"):
            TRAINER.loss_components(
                self.logits, self.labels, mode="ce", class_counts=CLASS_COUNTS[:-1]
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            TRAINER.loss_components(
                self.logits,
                self.labels,
                mode="author_weighted_ce",
                class_counts=CLASS_COUNTS,
                author_weight_temperature=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TRAINER.loss_components(
                self.logits,
                self.labels,
                mode="logit_adjusted",
                class_counts=CLASS_COUNTS,
                logit_adjustment_tau=-1.0,
            )


class SelectionAndOptimizerTests(unittest.TestCase):
    def test_dev_selection_is_macro_f1_then_accuracy_then_loss(self):
        incumbent = {"macro_f1": 0.3, "accuracy": 0.4, "selection_loss": 1.2}
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.31, "accuracy": 0.1, "selection_loss": 9.0}, incumbent
            )
        )
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.41, "selection_loss": 9.0}, incumbent
            )
        )
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.4, "selection_loss": 1.1}, incumbent
            )
        )
        self.assertFalse(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.4, "selection_loss": 1.2}, incumbent
            )
        )

    def test_adamw_groups_exclude_bias_and_layernorm_from_decay(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.LayerNorm = torch.nn.LayerNorm(2)

        groups, audit = TRAINER.optimizer_parameter_groups(
            SmallModel(), weight_decay=0.001
        )
        self.assertEqual(groups[0]["weight_decay"], 0.001)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertIn("linear.weight", audit["decay_parameter_names"])
        self.assertIn("linear.bias", audit["no_decay_parameter_names"])
        self.assertIn("LayerNorm.weight", audit["no_decay_parameter_names"])
        self.assertIn("LayerNorm.bias", audit["no_decay_parameter_names"])
        self.assertEqual(audit["trainable_parameter_count"], 4)
        self.assertEqual(audit["trainable_parameter_numel"], 12)
        self.assertRegex(audit["parameter_roster_sha256"], r"^[0-9a-f]{64}$")

    def test_metrics_map_internal_ids_and_select_on_common_raw_ce(self):
        metrics = TRAINER.metrics_from_ids(
            [0, 1, 2],
            [0, 2, 2],
            objective_loss=0.7,
            selection_loss=0.9,
        )
        self.assertEqual(TRAINER.EMODYNAMIX_ID_TO_CANONICAL, CANONICAL_LABELS)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["invalid"], 0)
        self.assertEqual(metrics["objective_loss"], 0.7)
        self.assertEqual(metrics["selection_loss"], 0.9)
        self.assertEqual(
            metrics["confusion_matrix"]["labels"], list(CANONICAL_LABELS)
        )

    def test_predictions_always_use_unadjusted_logits(self):
        raw_logits = torch.tensor(
            [[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        self.assertEqual(TRAINER.prediction_ids_from_raw_logits(raw_logits), [0])

    def test_metrics_and_selector_reject_nonfinite_or_noninteger_values(self):
        for bad_ids in ([1.2], ["1"], [True]):
            with self.subTest(bad_ids=bad_ids):
                with self.assertRaisesRegex(ValueError, "integer"):
                    TRAINER.metrics_from_ids(
                        bad_ids,
                        [0],
                        objective_loss=0.7,
                        selection_loss=0.9,
                    )
        incumbent = {"macro_f1": 0.3, "accuracy": 0.4, "selection_loss": 1.2}
        for candidate in (
            {"macro_f1": float("nan"), "accuracy": 0.4, "selection_loss": 1.2},
            {"macro_f1": 1.1, "accuracy": 0.4, "selection_loss": 1.2},
            {"macro_f1": 0.3, "accuracy": -0.1, "selection_loss": 1.2},
            {"macro_f1": 0.3, "accuracy": 0.4, "selection_loss": float("inf")},
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, "dev metric"):
                    TRAINER.is_better_dev(candidate, incumbent)


class CliBoundaryTests(unittest.TestCase):
    def test_cli_defaults_to_audit_and_has_no_test_input(self):
        parser = TRAINER.build_parser()
        args = parser.parse_args([])
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertFalse(args.execute)
        self.assertEqual(args.mode, "audit")
        self.assertEqual(args.loss, "author_weighted_ce")
        self.assertEqual(args.train_file.name, "trainWithStrategy_short.tsv")
        self.assertEqual(args.dev_file.name, "devWithStrategy_short.tsv")
        self.assertIn("--train-file", options)
        self.assertIn("--dev-file", options)
        self.assertIn("--feature-run-dir", options)
        self.assertIn("--expected-base-tree-sha256", options)
        self.assertIn("--preregistration", options)
        self.assertIn("--preregistration-sha256", options)
        self.assertIn("--max-grad-norm", options)
        self.assertNotIn("--test", options)
        self.assertNotIn("--test-file", options)
        self.assertNotIn("--dataset-dir", options)
        self.assertNotIn("--init-checkpoint", options)

    def test_default_main_only_prepares_train_dev_and_prints_data_ready(self):
        prepared = {
            "train_records": [{"label_id": 0}],
            "dev_records": [{"label_id": 1}],
            "source_audit": {"train": {"rows": 1}, "dev": {"rows": 1}},
            "audit": {"post_filter_overlap_count": 0},
        }
        output = StringIO()
        with mock.patch(
            "scripts.train_esconv_emodynamix_clean.prepare_canonical_data",
            return_value=prepared,
        ) as prepare, mock.patch.object(
            TRAINER, "load_verified_feature_run", wraps=TRAINER.load_verified_feature_run
        ) as load_features, redirect_stdout(output):
            result = TRAINER.main([])

        self.assertEqual(result, 0)
        prepare.assert_called_once()
        load_features.assert_not_called()
        summary = __import__("json").loads(output.getvalue())
        self.assertEqual(summary["status"], "DATA_READY")
        self.assertFalse(summary["model_loaded"])
        self.assertFalse(summary["optimizer_created"])
        self.assertFalse(summary["artifact_written"])
        self.assertFalse(summary["frozen_test_accessed"])


if __name__ == "__main__":
    unittest.main()
