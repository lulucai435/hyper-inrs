"""Selected optimization-oriented hypernetwork components.

This initial release contains reusable components from the supplementary code,
not the complete experiment pipeline. See README.md for the supported scope.
"""

from typing import List

import einops
import torch
import torch.nn.functional as F
from torch import nn

from model.encoder.transformer import CrossAttn, SelfAttn

class TargetNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def construct_opt_blocks(
        self, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, *args, **kwargs
    ):
        raise NotImplementedError

    def topo_order(self):
        raise NotImplementedError

    def reverse_topo_order(self):
        raise NotImplementedError

    def get_submodule_names(self) -> List[str]:
        if not hasattr(self, "submodule_names"):
            self.submodule_names = []
            for subm in self.get_submodules():
                for name, module in self.named_modules():
                    if subm is module:
                        self.submodule_names.append(name)
                        break
        return self.submodule_names

    def merge_submodule_weights(self, weight_dicts):
        """
        convert the weight dict of the submodules to the weight dict of the target net.
        """
        weight_dict = {}
        for sub_name, wd in zip(self.get_submodule_names(), weight_dicts):
            for k, v in wd.items():
                weight_dict[sub_name + "." + k] = v

        return weight_dict


class MLP(nn.Module):
    """
    A simple MLP with a variable number of layers and hidden dimensions.
    """

    def __init__(
        self, in_dim, out_dim, hidden_dim, num_layers, activation=F.leaky_relu
    ):
        super().__init__()
        if num_layers == 1:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.fcs = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
            self.fcs.extend(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
            )
            self.fcs.append(nn.Linear(hidden_dim, out_dim))
        self.activation = activation

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == len(self.fcs) - 1:
                x = fc(x)  # Remove the activation function from the last layer
            elif fc.in_features == fc.out_features:
                x = self.activation(fc(x)) + x
            else:
                x = self.activation(fc(x))
        return x


class FIN_FOUT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_dim = args[0]
        self.out_dim = args[1]
        self.hidden_dim = kwargs.get("hidden_dim", 1024)
        self.num_layers = kwargs.get("num_layers", 1)

        self.next_blocks = []
        self.prev_blocks = []
        self.net = SelfAttn(
            self.in_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers
        )

    """
    Dummy function, does not change anything.
    """

    def add2in_dim(self, dim):
        pass

    def link(self, next_block):
        next_block.add2in_dim(self.out_dim)
        self.next_blocks.append(next_block)
        next_block.prev_blocks.append(self)

    def pseudo_forward(self, x):
        res = self.net(x)
        self.z_in = res
        return res

    def pseudo_backward(self, x):
        res = self.net(x)
        self.dldin = res
        return res

    def get_zin(self):
        return self.z_in

    def get_dldin(self, prev_blk):
        return self.dldin


class ModuleEncoder(nn.Module):
    """Encode weight rows; biases are outside this component's output scope."""

    def __init__(
        self, target_net, weight_dim, hidden_dim, num_layers, column_grouping=False
    ):
        super().__init__()
        if column_grouping:
            raise NotImplementedError("This release supports row grouping only.")
        self.name_shape_dict = {k: v.shape for k, v in target_net.named_parameters()}
        weight_shape = self.name_shape_dict["weight"]
        self.out_dim = weight_shape[-2]
        self.in_dim = weight_shape[-1]
        self.weight_dim = weight_dim
        if column_grouping:
            self.num_tokens = self.in_dim 
            self.split_mlp = MLP(in_dim=self.out_dim, out_dim=weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            self.num_tokens = self.out_dim 
            self.split_mlp = MLP(in_dim=self.in_dim, out_dim=weight_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        self.pe = nn.Parameter(torch.randn(1, self.num_tokens, weight_dim))
        self.norm = nn.LayerNorm(weight_dim)

    def forward(self, weight_dict):
        W = weight_dict["weight"] 
        if W.ndim == 2:
            W = W.unsqueeze(0)

        tokens = W # [B, num_tokens, in_dim]
        tokens = self.split_mlp(tokens)  # [B, num_tokens, weight_dim]
        tokens = tokens + self.pe
        tokens = self.norm(tokens)
        return tokens


class ModuleDecoder(nn.Module):
    """Decode row tokens into weight matrices, without predicting biases."""

    def __init__(
        self, target_net, weight_dim, hidden_dim, num_layers, column_grouping=False
    ):
        super().__init__()
        if column_grouping:
            raise NotImplementedError("This release supports row grouping only.")
        self.name_shape_dict = {k: v.shape for k, v in target_net.named_parameters()}
        weight_shape = self.name_shape_dict["weight"]

        self.out_dim = weight_shape[-2]
        self.in_dim = weight_shape[-1]
        if column_grouping:
            self.num_tokens = self.in_dim 
            self.post_mlp = MLP(in_dim=weight_dim, out_dim=self.out_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=F.relu,)
        else:
            self.num_tokens = self.out_dim 
            self.post_mlp = MLP(in_dim=weight_dim, out_dim=self.in_dim, hidden_dim=hidden_dim, num_layers=num_layers, activation=F.relu,)

        self.norm = nn.LayerNorm(weight_dim)

    def forward(self, weight_vec):
        weight_vec = self.norm(weight_vec)
        weight_vec = self.post_mlp(weight_vec)  # [B, token_dim, token_input_dim]

        W = weight_vec
        weight_dict = {
            "weight": W.view(-1, *self.name_shape_dict["weight"]),
        }
        return weight_dict


class OptBlock(nn.Module):
    """
    Learned, chain-rule-inspired forward and backward message passing.

    The predicted update is a learned surrogate, not an autograd derivative.
    """

    def __init__(
        self,
        module,
        ftask_dim,
        out_dim,
        weight_dim,
        hidden_dim,
        num_layers,
        dl_din_way="direct",
        dl_dw_way="direct",
        **kwargs,
    ):
        super().__init__()
        self.module = module

        self.ftask_dim = ftask_dim
        self.in_dim = 0
        self.out_dim = out_dim
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.dl_din_way = dl_din_way
        self.dl_dw_way = dl_dw_way
        self.num_layers = num_layers
        self.next_blocks = []
        self.prev_blocks = []
        self.use_hyper_crossattn = kwargs.get("use_hyper_crossattn")
        self.enable_each_layer_lr = kwargs.get("enable_each_layer_lr")
        self.column_grouping = kwargs.get("column_grouping")
        if self.column_grouping:
            raise NotImplementedError("This release supports row grouping only.")
        self.slice_dl_din = kwargs.get("slice_dl_din")

    def setup(self):
        if self.column_grouping:
            self.num_tokens = int(self.module.in_features)
        else:
            self.num_tokens = int(self.module.out_features)
        self.ftask_zin = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.token_task = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.learn_p_forward = nn.Parameter(torch.randn(1, self.num_tokens, self.weight_dim))
        if self.use_hyper_crossattn:
            self.weightemb_zprime = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
            self.task_grad = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
            self.token_grad = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
            self.dl_dout_prime_net = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
            self.learn_p_backward_z_grad = nn.Parameter(torch.randn(1, self.num_tokens, self.weight_dim))
            self.learn_p_backward_dl_dout = nn.Parameter(torch.randn(1, self.num_tokens, self.weight_dim))

        else:
            self.forward_net = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )

        if self.enable_each_layer_lr:
            self.lr_attn = CrossAttn(
                self.weight_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
            self.lr_pre_mlp = MLP(self.weight_dim, 32, self.weight_dim//2, num_layers=2)  # e.g., D → 32
            self.lr_mlp = MLP(32 * self.num_tokens, 1, 128, num_layers=2)

        self.out_din = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.dout_dw = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.dl_din_net = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.dl_dw_net = CrossAttn(
            self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

    def pseudo_forward(self, ftask, weight_emb):
        z_in = torch.cat([prev.get_zin() for prev in self.prev_blocks], dim=-2)

        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, "1 L D -> n L D", n=z_in.shape[0])

        task_context = self.ftask_zin(ftask, z_in)
        z_prime = self.token_task(self.learn_p_forward.expand(ftask.shape[0], -1, -1), task_context)

        if self.use_hyper_crossattn:
            out = self.weightemb_zprime(weight_emb, z_prime)
        else:
            out = self.forward_net(weight_emb, z_in)

        if self.enable_each_layer_lr:
            lr_attn = self.lr_attn(z_prime, weight_emb)  # [B, L, D]
            lr_reduced = self.lr_pre_mlp(lr_attn)        # [B, L, 32]
            lr_flat = lr_reduced.view(lr_reduced.shape[0], -1)  # [B, L×32]
            self.lr = self.lr_mlp(lr_flat).unsqueeze(-1) 
        else:
            self.lr = 1.0

        self.z_in = z_in
        self.nxt_z_in = out                  # [B, 1, 1]
        return out

    def get_zin(self):
        return self.nxt_z_in

    def get_lr(self):
        return self.lr

    def pseudo_backward(self, ftask, weight_emb):
        z_in = self.z_in
        dl_dout = torch.cat([next.get_dldin(self) for next in self.next_blocks], dim=-2)

        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, "1 L D -> n L D", n=z_in.shape[0])

        if self.use_hyper_crossattn:
            task_context = self.task_grad(ftask, z_in)
            z_grad = self.token_grad(self.learn_p_backward_z_grad.expand(ftask.shape[0], -1, -1), task_context)
            dout_din = self.out_din(weight_emb, z_grad)  # N L D
            dout_dw = self.dout_dw(z_grad, weight_emb)  # N L D

            dl_dout_prime = self.dl_dout_prime_net(self.learn_p_backward_dl_dout.expand(ftask.shape[0], -1, -1), dl_dout)
            dl_din = self.dl_din_net(dl_dout_prime, dout_din)
            dl_dw = self.dl_dw_net(dl_dout_prime, dout_dw)
        else:
            dout_din = self.out_din(z_in, weight_emb)  # N L D
            dout_dw = self.dout_dw(weight_emb, z_in)  # N L D
            
            dl_din = self.dl_din_net(dout_din, dl_dout)
            dl_dw = self.dl_dw_net(dout_dw, dl_dout)

        self.dl_din = dl_din
        return dl_dw

    def get_dldin(self, prev_blk):
        len_prev_blocks = len(self.prev_blocks)
        idx = self.prev_blocks.index(prev_blk)
        context_len = self.dl_din.shape[1]
        part_len = context_len // len_prev_blocks
        if self.slice_dl_din:
            return self.dl_din[
                :,
                part_len * idx : part_len * (idx + 1),
            ]
        return self.dl_din

    """
    connect the output of a module to another module's input
    """

    def link(self, next_block):
        next_block.add2in_dim(self.weight_dim)

        self.next_blocks.append(next_block)
        next_block.prev_blocks.append(self)

    def add2in_dim(self, dim):
        self.in_dim += dim


class OptLayer(nn.Module):
    """
    A OptBlock takes the input and output shape of each sub-block of the target network.
    it first does a peusdo forward mimicking the forward pass of the target network.
    stores the internal results, and mimicking the backward pass.

    target net must implement three functions returning the submodule, input shapes, and output shapes.

    """

    def __init__(
        self,
        target_net: TargetNet,
        ftask_dim,
        weight_dim,
        deriv_hidden_dim,
        driv_num_layers,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.opt_blocks, self.forward_in, self.dloss_dout = (
            target_net.construct_opt_blocks(
                ftask_dim,
                weight_dim,
                deriv_hidden_dim,
                driv_num_layers,
                *args,
                **kwargs,
            )
        )
        for opt_block in self.opt_blocks:
            opt_block.setup()

    def pseudo_forward(self, ftask, weight_embs):
        for fin in self.forward_in:
            fin.pseudo_forward(ftask)

        for idx, (opt_block, weight_emb) in enumerate(
            zip(self.opt_blocks, weight_embs)
        ):
            opt_block.pseudo_forward(ftask, weight_emb)

    def pseudo_backward(self, ftask, weight_embs):
        for fout in self.dloss_dout:
            fout.pseudo_backward(ftask)

        dw_dicts = []
        for opt_block, weight_emb, idx in reversed(
            list(zip(self.opt_blocks, weight_embs, range(len(self.opt_blocks))))
        ):
            dl_dw = opt_block.pseudo_backward(ftask, weight_emb)
            dw_dicts.append(dl_dw)

        dw_dicts = list(reversed(dw_dicts))
        return dw_dicts

    def get_lrs(self):
        return [blk.get_lr() for blk in self.opt_blocks]

    def forward(self, ftask, weight_embs):
        self.pseudo_forward(ftask, weight_embs)
        dw_dicts = self.pseudo_backward(ftask, weight_embs)
        return dw_dicts


class ParamLN(nn.Module):
    def __init__(self, weight_dim):
        super().__init__()
        self.ln = nn.LayerNorm(weight_dim)

    def forward(self, weight_dict):
        return self.ln(weight_dict)


class InitWeightHypernet(nn.Module):
    def __init__(self, target_net, embed_dim, hidden_dim=128, num_layers=4, column_grouping=False):

        super(InitWeightHypernet, self).__init__()
        if column_grouping:
            raise NotImplementedError("This release supports row grouping only.")
        self.embed_dim = embed_dim
        self.num_tokens = []               
        self.token_queries = nn.ParameterList()  
        self.cross_attn_layers = nn.ModuleList() 
        for module in target_net.get_submodules():
            if not hasattr(module, 'in_features') or not hasattr(module, 'out_features'):
                raise ValueError("Target module must have `in_features` and `out_features` attributes")
            in_feat = module.in_features
            out_feat = module.out_features
            if column_grouping:
                L = in_feat
            else:
                L = out_feat
            self.num_tokens.append(L)

            token = nn.Parameter(torch.randn(1, L, embed_dim))
            self.token_queries.append(token)

            cross_attn = CrossAttn(self.embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            self.cross_attn_layers.append(cross_attn)

    def forward(self, ftask):
        B, N, D = ftask.shape
        assert D == self.embed_dim, "Task embedding dimension D must match hypernet embed_dim"

        weight_embeddings = []

        for token, cross_attn in zip(self.token_queries, self.cross_attn_layers):
            attn_output = cross_attn(token.expand(B, -1, -1), ftask)  # attn_output: [B, L, D]
            weight_embeddings.append(attn_output)
        return weight_embeddings
