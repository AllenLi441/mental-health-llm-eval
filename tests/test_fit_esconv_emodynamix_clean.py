import importlib.util
import math
import sys
import unittest
from pathlib import Path

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


class SelectionAndOptimizerTests(unittest.TestCase):
    def test_dev_selection_is_macro_f1_then_accuracy_then_loss(self):
        incumbent = {"macro_f1": 0.3, "accuracy": 0.4, "loss": 1.2}
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.31, "accuracy": 0.1, "loss": 9.0}, incumbent
            )
        )
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.41, "loss": 9.0}, incumbent
            )
        )
        self.assertTrue(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.4, "loss": 1.1}, incumbent
            )
        )
        self.assertFalse(
            TRAINER.is_better_dev(
                {"macro_f1": 0.3, "accuracy": 0.4, "loss": 1.2}, incumbent
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
        self.assertIn("--train-file", options)
        self.assertIn("--dev-file", options)
        self.assertIn("--feature-run-dir", options)
        self.assertNotIn("--test", options)
        self.assertNotIn("--test-file", options)
        self.assertNotIn("--dataset-dir", options)
        self.assertNotIn("--init-checkpoint", options)


if __name__ == "__main__":
    unittest.main()
