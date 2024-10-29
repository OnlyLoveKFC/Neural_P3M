import math
from functools import wraps

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torch_scatter import scatter


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"
    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator

class Scalar(nn.Module):
    def __init__(self, hidden_channels):
        super(Scalar, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def forward(self, x):
        return self.output_network(x)
    
class GatedEquivariantBlock(nn.Module):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)
    
    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
class EquivariantScalar(nn.Module):
    def __init__(self, hidden_channels):
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList([
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 
                    1, 
                    scalar_activation=False,
                ),
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()
    
    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
    
class EquivariantVector(nn.Module):
    def __init__(self, hidden_channels):
        super(EquivariantVector, self).__init__()
        self.output_network = nn.ModuleList([
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 
                    1, 
                    scalar_activation=False,
                ),
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = nn.SiLU()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, dim_size=None):
        x = self.conv(x, edge_index, edge_weight, edge_attr, dim_size)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        mlp: nn.Sequential,
        cutoff: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.mlp = mlp
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, dim_size=None):
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.mlp(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x_j = torch.index_select(x, 0, edge_index[0])
        x_j = x_j * W
        x = scatter(x_j, edge_index[1], dim=0, reduce='add', dim_size=dim_size)
        x = self.lin2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        values = torch.matmul(attention_weights, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o
    
class L2MAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)
    
    @property
    def __name__(self):
        return "l2mae_loss"