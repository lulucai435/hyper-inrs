"""Component checks for the scope documented in README.md."""

import unittest

import torch
from torch import nn

from examples.minimal_components import ComponentDemo, TinyTarget
from hypernet_core import InitWeightHypernet, ModuleDecoder, ModuleEncoder, OptBlock
from model.encoder.transformer import CrossAttn, SelfAttn


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_attention_preserves_query_axis_with_different_memory_length(self):
        query = torch.randn(2, 5, 16)
        memory = torch.randn(2, 7, 16)
        for output in (SelfAttn(16, 32).eval()(query), CrossAttn(16, 32).eval()(query, memory)):
            self.assertEqual(tuple(output.shape), (2, 5, 16))
            self.assertTrue(torch.isfinite(output).all())

    def test_row_codec_non_square_weights_batched_and_unbatched(self):
        target = nn.Linear(3, 5, bias=False)
        encoder = ModuleEncoder(target, 16, 32, 2)
        decoder = ModuleDecoder(target, 16, 32, 2)
        for weight, batch in [(target.weight, 1), (torch.randn(2, 5, 3), 2)]:
            with self.subTest(batch=batch):
                tokens = encoder({"weight": weight})
                restored = decoder(tokens)["weight"]
                self.assertEqual(tuple(tokens.shape), (batch, 5, 16))
                self.assertEqual(tuple(restored.shape), (batch, 5, 3))
                self.assertTrue(torch.isfinite(restored).all())

    def test_row_codec_propagates_gradients_to_input_weights(self):
        target = nn.Linear(3, 5, bias=False)
        encoder = ModuleEncoder(target, 16, 32, 2)
        decoder = ModuleDecoder(target, 16, 32, 2)
        weights = torch.randn(2, 5, 3, requires_grad=True)
        decoder(encoder({"weight": weights}))["weight"].square().mean().backward()
        self.assertIsNotNone(weights.grad)
        self.assertTrue(torch.isfinite(weights.grad).all())
        self.assertGreater(weights.grad.abs().sum().item(), 0)

    def test_column_grouping_is_explicitly_unavailable(self):
        target = nn.Linear(3, 5, bias=False)
        constructors = [
            lambda: ModuleEncoder(target, 16, 32, 2, column_grouping=True),
            lambda: ModuleDecoder(target, 16, 32, 2, column_grouping=True),
            lambda: InitWeightHypernet(TinyTarget(), 16, column_grouping=True),
            lambda: OptBlock(target, 16, 16, 16, 32, 1, column_grouping=True),
        ]
        for constructor in constructors:
            with self.assertRaisesRegex(NotImplementedError, "row grouping"):
                constructor()

    def test_one_update_has_finite_outputs_and_backward_gradients(self):
        model = ComponentDemo().eval()
        tasks = torch.randn(2, 7, 16, requires_grad=True)
        predictions, tokens = model(tasks, torch.randn(2, 11, 3))
        self.assertEqual(tuple(predictions.shape), (2, 11, 2))
        self.assertEqual([tuple(token.shape) for token in tokens], [(2, 5, 16), (2, 2, 16)])
        predictions.square().mean().backward()
        self.assertIsNotNone(tasks.grad)
        self.assertTrue(torch.isfinite(tasks.grad).all())
        self.assertGreater(tasks.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_eval_batches_match_independent_examples(self):
        model = ComponentDemo().eval()
        tasks = torch.randn(2, 7, 16)
        coordinates = torch.randn(2, 11, 3)
        with torch.no_grad():
            batched, _ = model(tasks, coordinates)
            separate = torch.cat([
                model(tasks[index:index + 1], coordinates[index:index + 1])[0]
                for index in range(2)
            ])
        torch.testing.assert_close(batched, separate, atol=2e-6, rtol=2e-5)

    def test_state_dict_roundtrip_preserves_predictions(self):
        model = ComponentDemo().eval()
        restored = ComponentDemo().eval()
        restored.load_state_dict(model.state_dict())
        tasks, coordinates = torch.randn(2, 7, 16), torch.randn(2, 11, 3)
        with torch.no_grad():
            original, _ = model(tasks, coordinates)
            recovered, _ = restored(tasks, coordinates)
        torch.testing.assert_close(original, recovered, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
