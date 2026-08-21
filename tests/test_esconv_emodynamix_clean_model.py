import importlib.util
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_MODULE = ROOT / "open_response_eval/esconv_emodynamix_model.py"
BASE_MODEL = ROOT / "tmp/official_benchmarks/roberta-base-e2da-materialized"
BASE_TREE_SHA256 = (
    "1d9faa93557a63a92292cd11dfbca3de8e336ffa60768745a71ecd1ed19aa91c"
)


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "esconv_emodynamix_model", MODEL_MODULE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODEL = load_model_module()


def model_inputs_and_features():
    model_inputs = [
        {
            "dialogue_history": "<START> </s> I feel lost",
            "strategy_history": "[-1, -1]",
            "speaker_turn": "None seeker",
        },
        {
            "dialogue_history": (
                "I need help </s> Have you tried breathing? </s> It did not work"
            ),
            "strategy_history": "[-1, 4, -1]",
            "speaker_turn": "seeker supporter seeker",
        },
    ]
    features = [
        {
            "node_count": 2,
            "parsed_dialogue": [[0, 1, 16], [1, 2, 0]],
            "upstream_erc_softmax_output": [
                [0.70, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.70, 0.05, 0.05, 0.05, 0.05, 0.05],
            ],
        },
        {
            "node_count": 3,
            "parsed_dialogue": [[0, 1, 16], [1, 2, 1], [2, 3, 2]],
            "upstream_erc_softmax_output": [
                [0.05, 0.05, 0.70, 0.05, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05],
            ],
        },
    ]
    return model_inputs, features


class ArchitectureContractTests(unittest.TestCase):
    def test_frozen_architecture_matches_author_lightmode(self):
        config = MODEL.CleanEmoDynamiXConfig()
        self.assertEqual(config.upstream_commit, "c9213d718a9684a5e05ce5daa947f9cbbfb7b927")
        self.assertEqual(config.graph_dim, 512)
        self.assertEqual(config.erc_classes, 7)
        self.assertEqual(config.strategy_classes, 8)
        self.assertEqual(config.relations, 19)
        self.assertEqual(config.graph_layers, 3)
        self.assertEqual(config.erc_temperature, 0.5)
        self.assertTrue(config.erc_mixed)
        self.assertTrue(config.author_double_softmax)

    def test_formal_loader_has_no_released_task_checkpoint_argument(self):
        parameters = inspect.signature(MODEL.load_clean_emodynamix_model).parameters
        self.assertIn("base_model_path", parameters)
        self.assertIn("expected_base_tree_sha256", parameters)
        self.assertNotIn("checkpoint", parameters)
        self.assertNotIn("task_checkpoint", parameters)

    def test_context_rendering_preserves_author_whitespace_semantics(self):
        self.assertEqual(
            MODEL.author_context_string(
                "<START> </s> I feel lost", ["None", "seeker"]
            ),
            "[None] <START>  [seeker]  I feel lost",
        )

    @unittest.skipUnless(BASE_MODEL.is_dir(), "materialized RoBERTa base absent")
    def test_symlink_base_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "base-link"
            link.symlink_to(BASE_MODEL, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                MODEL.load_clean_emodynamix_model(
                    base_model_path=link,
                    expected_base_tree_sha256=BASE_TREE_SHA256,
                )


@unittest.skipUnless(BASE_MODEL.is_dir(), "materialized RoBERTa base absent")
class RealModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        cls.model, cls.receipt = MODEL.load_clean_emodynamix_model(
            base_model_path=BASE_MODEL.resolve(),
            expected_base_tree_sha256=BASE_TREE_SHA256,
        )
        cls.model.to(cls.device)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def test_initialization_receipt_and_parameter_shapes(self):
        model = self.model
        self.assertEqual(self.receipt["base_model_tree_sha256"], BASE_TREE_SHA256)
        self.assertIsNone(self.receipt["task_checkpoint_sha256"])
        self.assertTrue(self.receipt["local_files_only"])
        self.assertEqual(tuple(model.erc_prototypes.shape), (7, 512))
        self.assertEqual(tuple(model.strategy_embedding.weight.shape), (8, 512))
        self.assertEqual(model.conv1.conv.num_relations, 19)
        self.assertEqual(model.conv2.conv.num_relations, 19)
        self.assertEqual(model.conv3.conv.num_relations, 19)
        self.assertEqual(model.classifier.dense.in_features, 1280)
        self.assertEqual(model.classifier.out_proj.out_features, 8)

    def test_real_forward_backward_uses_all_live_architecture_components(self):
        model_inputs, features = model_inputs_and_features()
        batch = MODEL.collate_clean_model_batch(
            model_inputs, features, device=self.device
        )
        labels = torch.tensor([2, 6], dtype=torch.long, device=self.device)
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(batch)
        loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        loss.backward()

        self.assertEqual(tuple(outputs["logits"].shape), (2, 8))
        self.assertTrue(torch.isfinite(outputs["logits"]).all().item())
        self.assertTrue(torch.isfinite(loss).item())
        live_parameters = [
            self.model.roberta.embeddings.word_embeddings.weight,
            self.model.t,
            self.model.erc_prototypes,
            self.model.dummy_embedding,
            self.model.strategy_embedding.weight,
            next(self.model.conv1.conv.parameters()),
            next(self.model.conv2.conv.parameters()),
            next(self.model.conv3.conv.parameters()),
            self.model.classifier.dense.weight,
            self.model.classifier.out_proj.weight,
        ]
        for parameter in live_parameters:
            with self.subTest(shape=tuple(parameter.shape)):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

        # These parameters are present in the author checkpoint but unused by
        # the published forward path; the clean control keeps that behavior.
        self.assertIsNone(self.model.node_position_embedding.weight.grad)
        self.assertIsNone(self.model.conv1.ffn.linear.weight.grad)
        self.assertIsNone(self.model.roberta.pooler.dense.weight.grad)

    def test_collator_rejects_gold_and_malformed_feature_inputs(self):
        model_inputs, features = model_inputs_and_features()
        leaked = dict(model_inputs[0], gold="Questions")
        with self.assertRaisesRegex(ValueError, "model input fields"):
            MODEL.collate_clean_model_batch(
                [leaked], [features[0]], device=self.device
            )
        malformed = dict(features[0])
        malformed["parsed_dialogue"] = [[1, 2, 17]]
        with self.assertRaisesRegex(ValueError, "SDDP relation"):
            MODEL.collate_clean_model_batch(
                [model_inputs[0]], [malformed], device=self.device
            )


if __name__ == "__main__":
    unittest.main()
