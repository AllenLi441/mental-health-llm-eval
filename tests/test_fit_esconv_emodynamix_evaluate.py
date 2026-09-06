import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "fit_esconv_emodynamix_clean_for_evaluate", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_trainer()


def model_batch(size):
    return {
        "dialogue_history": [f"dialogue-{index}" for index in range(size)],
        "strategy_history": [[-1]] * size,
        "speaker_turn": [["seeker"]] * size,
        "parsed_dialogue": [[] for _ in range(size)],
        "erc_probabilities": torch.zeros((size, 7)),
        "dialogue_sizes": [1] * size,
    }


class ModeAssertingModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.forward_grad_enabled = []
        self.forward_training = []

    def forward(self, batch):
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        self.forward_training.append(self.training)
        self.assert_model_fields = set(batch)
        return {"logits": self.outputs.pop(0)}


class CleanDevEvaluationTests(unittest.TestCase):
    def test_full_dev_uses_eval_no_grad_raw_predictions_and_two_losses(self):
        logits_one = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        logits_two = torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        model = ModeAssertingModel([logits_one, logits_two])
        model.train()
        batches = [
            {
                "model_batch": model_batch(2),
                "labels": torch.tensor([0, 1]),
                "item_ids": ["a", "b"],
            },
            {
                "model_batch": model_batch(1),
                "labels": torch.tensor([2]),
                "item_ids": ["c"],
            },
        ]

        metrics = FIT.evaluate_emodynamix(
            model,
            batches,
            device=torch.device("cpu"),
            loss_mode="logit_adjusted",
            class_counts=FIT.CLASS_COUNTS,
            author_weight_temperature=1.75,
            class_balance_beta=0.999,
            logit_adjustment_tau=1.0,
        )

        combined_logits = torch.cat((logits_one, logits_two), dim=0)
        labels = torch.tensor([0, 1, 2])
        expected_selection = functional.cross_entropy(
            combined_logits, labels, reduction="mean"
        ).item()
        objective_num, objective_den = FIT.loss_components(
            combined_logits,
            labels,
            mode="logit_adjusted",
            class_counts=FIT.CLASS_COUNTS,
        )
        self.assertAlmostEqual(metrics["selection_loss"], expected_selection, places=7)
        self.assertAlmostEqual(
            metrics["objective_loss"],
            (objective_num / objective_den).item(),
            places=7,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["invalid_rate"], 0.0)
        self.assertEqual(model.forward_grad_enabled, [False, False])
        self.assertEqual(model.forward_training, [False, False])
        self.assertEqual(
            model.assert_model_fields,
            {
                "dialogue_history",
                "strategy_history",
                "speaker_turn",
                "parsed_dialogue",
                "erc_probabilities",
                "dialogue_sizes",
            },
        )

    def test_empty_dev_loader_fails(self):
        model = ModeAssertingModel([])
        with self.assertRaisesRegex(ValueError, "development loader is empty"):
            FIT.evaluate_emodynamix(
                model,
                [],
                device=torch.device("cpu"),
                loss_mode="ce",
                class_counts=FIT.CLASS_COUNTS,
                author_weight_temperature=1.75,
                class_balance_beta=0.999,
                logit_adjustment_tau=1.0,
            )


if __name__ == "__main__":
    unittest.main()
