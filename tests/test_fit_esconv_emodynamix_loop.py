import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_esconv_emodynamix_clean.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "fit_esconv_emodynamix_clean_for_loop", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FIT = load_trainer()


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(3, 8, bias=False)
        self.training_modes = []
        self.grad_modes = []

    def forward(self, model_batch):
        self.training_modes.append(self.training)
        self.grad_modes.append(torch.is_grad_enabled())
        self.assert_model_fields = set(model_batch)
        return {"logits": self.classifier(model_batch["erc_probabilities"])}


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def batch(rows, labels):
    size = len(labels)
    return {
        "model_batch": {
            "dialogue_history": ["dialogue"] * size,
            "strategy_history": [[-1]] * size,
            "speaker_turn": [["seeker"]] * size,
            "parsed_dialogue": [[] for _ in range(size)],
            "erc_probabilities": torch.tensor(rows, dtype=torch.float32),
            "dialogue_sizes": [1] * size,
        },
        "labels": torch.tensor(labels, dtype=torch.long),
        "item_ids": [f"item-{index}" for index in range(size)],
    }


class CleanTrainingLoopTests(unittest.TestCase):
    def test_linear_scheduler_matches_huggingface_boundary_sequence(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = FIT._linear_warmup_scheduler(
            optimizer, warmup_updates=2, total_updates=4
        )

        learning_rates_used = []
        for _ in range(4):
            learning_rates_used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertEqual(learning_rates_used, [0.0, 0.5, 1.0, 0.5])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_schedule_horizon_is_epoch_capacity_not_execution_cap(self):
        plan = FIT.training_schedule_plan(
            microbatch_count=2_109,
            gradient_accumulation_steps=4,
            epochs=8,
            max_updates=3_000,
            warmup_updates=500,
        )

        self.assertEqual(
            plan,
            {
                "updates_per_epoch": 528,
                "total_schedule_updates": 4_224,
                "execution_update_cap": 3_000,
                "planned_execution_updates": 3_000,
                "warmup_updates": 500,
            },
        )

    def test_odd_accumulation_windows_step_and_best_epoch_is_not_last(self):
        model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CountingScheduler()
        train_loader = [
            batch([[1.0, 0.0, 0.2]], [0]),
            batch([[0.1, 0.3, 0.5]], [1]),
            batch([[0.7, 0.2, 0.4]], [5]),
        ]
        metric_sequence = [
            {
                "macro_f1": 0.40,
                "accuracy": 0.50,
                "weighted_f1": 0.45,
                "selection_loss": 1.0,
                "objective_loss": 1.1,
            },
            {
                "macro_f1": 0.30,
                "accuracy": 0.60,
                "weighted_f1": 0.40,
                "selection_loss": 0.9,
                "objective_loss": 1.0,
            },
        ]
        saved = []

        def save_best(**kwargs):
            saved.append(
                {
                    "epoch": kwargs["epoch"],
                    "metrics": dict(kwargs["metrics"]),
                    "state": {
                        name: tensor.detach().clone()
                        for name, tensor in kwargs["model"].state_dict().items()
                    },
                    "actual_updates": kwargs["actual_updates"],
                }
            )

        with mock.patch.object(
            FIT, "evaluate_emodynamix", side_effect=metric_sequence
        ) as evaluate:
            result = FIT.run_training_epochs(
                model,
                train_loader,
                dev_loader=[object()],
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                epochs=2,
                gradient_accumulation_steps=2,
                max_updates=20,
                max_grad_norm=1.0,
                loss_mode="author_weighted_ce",
                class_counts=FIT.CLASS_COUNTS,
                author_weight_temperature=1.75,
                class_balance_beta=0.999,
                logit_adjustment_tau=1.0,
                checkpoint_callback=save_best,
            )

        self.assertEqual(result["actual_optimizer_updates"], 4)
        self.assertEqual(result["actual_scheduler_steps"], 4)
        self.assertEqual(result["stop_reason"], "epochs_completed")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["epoch"], 1)
        self.assertEqual(saved[0]["actual_updates"], 2)
        self.assertEqual(scheduler.steps, 4)
        self.assertEqual(evaluate.call_count, 2)
        self.assertTrue(all(model.training_modes))
        self.assertTrue(all(model.grad_modes))
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

    def test_max_updates_stops_after_a_real_optimizer_step(self):
        model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CountingScheduler()
        train_loader = [
            batch([[1.0, 0.0, 0.2]], [0]),
            batch([[0.1, 0.3, 0.5]], [1]),
            batch([[0.7, 0.2, 0.4]], [5]),
        ]
        metrics = {
            "macro_f1": 0.2,
            "accuracy": 0.3,
            "weighted_f1": 0.2,
            "selection_loss": 1.2,
            "objective_loss": 1.2,
        }
        with mock.patch.object(FIT, "evaluate_emodynamix", return_value=metrics):
            result = FIT.run_training_epochs(
                model,
                train_loader,
                dev_loader=[object()],
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                epochs=5,
                gradient_accumulation_steps=2,
                max_updates=1,
                max_grad_norm=1.0,
                loss_mode="ce",
                class_counts=FIT.CLASS_COUNTS,
                author_weight_temperature=1.75,
                class_balance_beta=0.999,
                logit_adjustment_tau=1.0,
                checkpoint_callback=lambda **kwargs: None,
            )
        self.assertEqual(result["actual_optimizer_updates"], 1)
        self.assertEqual(result["actual_scheduler_steps"], 1)
        self.assertEqual(result["stop_reason"], "max_updates_reached")


if __name__ == "__main__":
    unittest.main()
