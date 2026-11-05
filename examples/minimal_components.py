"""Exercise one component-level update using synthetic task embeddings.

Run from the repository root: python -m examples.minimal_components
This is a wiring and differentiation example, not a paper experiment.
"""

import torch
from torch import nn

from hypernet_core import (
    FIN_FOUT,
    InitWeightHypernet,
    ModuleDecoder,
    ModuleEncoder,
    OptBlock,
    OptLayer,
    ParamLN,
    TargetNet,
)


class TinyTarget(TargetNet):
    """A bias-free two-layer coordinate MLP used only for this example."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 5, bias=False), nn.Linear(5, 2, bias=False)])

    def forward(self, coordinates):
        return self.layers[1](torch.sin(self.layers[0](coordinates)))

    def get_submodules(self):
        return list(self.layers)

    def construct_opt_blocks(
        self, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs
    ):
        blocks = nn.ModuleList(
            OptBlock(
                layer, ftask_dim, weight_dim, weight_dim,
                deriv_hidden_dim, driv_num_layers, **kwargs
            )
            for layer in self.layers
        )
        fin = nn.ModuleList([FIN_FOUT(
            ftask_dim, weight_dim, hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers
        )])
        fout = nn.ModuleList([FIN_FOUT(
            ftask_dim, weight_dim, hidden_dim=deriv_hidden_dim, num_layers=driv_num_layers
        )])
        fin[0].link(blocks[0])
        blocks[0].link(blocks[1])
        blocks[-1].link(fout[0])
        return blocks, fin, fout


class ComponentDemo(nn.Module):
    """Minimal composition of the released components with one update step."""

    def __init__(self):
        super().__init__()
        self.target = TinyTarget()
        dim = 16  # Attention uses 8 heads, so this must be divisible by 8.
        self.initializer = InitWeightHypernet(self.target, dim, hidden_dim=32, num_layers=1)
        self.encoders = nn.ModuleList(
            ModuleEncoder(layer, dim, 32, 2) for layer in self.target.get_submodules()
        )
        self.decoders = nn.ModuleList(
            ModuleDecoder(layer, dim, 32, 2) for layer in self.target.get_submodules()
        )
        self.norms = nn.ModuleList(ParamLN(dim) for _ in self.target.get_submodules())
        self.opt_layer = OptLayer(
            self.target, dim, dim, 32, 1,
            use_hyper_crossattn=True,
            enable_each_layer_lr=True,
            column_grouping=False,
            slice_dl_din=False,
        )

    def forward(self, task_embeddings, coordinates):
        tokens = self.initializer(task_embeddings)
        initial = [decoder(token) for decoder, token in zip(self.decoders, tokens)]
        updates = self.opt_layer(task_embeddings, tokens)
        rates = self.opt_layer.get_lrs()
        weights = [
            {"weight": base["weight"] + rate * decoder(norm(update))["weight"]}
            for base, rate, decoder, norm, update in zip(
                initial, rates, self.decoders, self.norms, updates
            )
        ]
        encoded = [encoder(weight) for encoder, weight in zip(self.encoders, weights)]
        parameters = self.target.merge_submodule_weights(weights)
        predictions = torch.stack([
            torch.func.functional_call(
                self.target,
                {name: value[index] for name, value in parameters.items()},
                (coordinates[index],),
            )
            for index in range(task_embeddings.shape[0])
        ])
        return predictions, encoded


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = ComponentDemo().eval()
    task_embeddings = torch.randn(2, 7, 16, requires_grad=True)
    coordinates = torch.randn(2, 11, 3)
    predictions, tokens = model(task_embeddings, coordinates)
    loss = predictions.square().mean()
    loss.backward()
    assert predictions.shape == (2, 11, 2)
    assert torch.isfinite(predictions).all()
    assert task_embeddings.grad is not None
    assert torch.isfinite(task_embeddings.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    print(f"Prediction shape: {tuple(predictions.shape)}")
    print(f"Re-encoded token shapes: {[tuple(token.shape) for token in tokens]}")
    print("PASS: finite forward outputs and backward gradients on CPU.")


if __name__ == "__main__":
    main()
