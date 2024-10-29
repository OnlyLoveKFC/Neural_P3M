import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from ..radius_utils import radius_graph_determinstic, radius_graph_pbc, get_distances_pbc, get_distances


class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 1.0

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class Sphere(nn.Module):
    
    def __init__(self, l=1):
        super(Sphere, self).__init__()
        self.l = l
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)


class Distance_Deterministic(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance_Deterministic, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop
    
    def forward(self, data):
        edge_index, _ = radius_graph_determinstic(
            data, 
            radius=self.cutoff,
            max_num_neighbors_threshold=self.max_num_neighbors,
            symmetrize=False,
            enforce_max_neighbors_strictly=True,
        )
        edge_weight, edge_vec = get_distances(edge_index, data.pos, return_distance_vec=True)
        
        if self.loop:
            edge_index = add_self_loops(edge_index, num_nodes=data.pos.size(0))[0]
            edge_weight = torch.cat([edge_weight, torch.zeros(data.pos.size(0), device=edge_weight.device)])
            edge_vec = torch.cat([edge_vec, torch.zeros((data.pos.size(0), 3), device=edge_weight.device)], dim=0)

        return edge_index, edge_weight, edge_vec
    
class Distance_PBC(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance_PBC, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop
    
    def forward(self, data):
        cell = data.cell
        pos = data.pos
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            data, 
            radius=self.cutoff,
            max_num_neighbors_threshold=self.max_num_neighbors,
            symmetrize=False,
            enforce_max_neighbors_strictly=True,
        )
        edge_weight, edge_vec = get_distances_pbc(
            edge_index, 
            cell, 
            cell_offsets, 
            neighbors, 
            pos, 
            return_distance_vec=True
        )
        
        if self.loop:
            edge_index = add_self_loops(edge_index, num_nodes=data.pos.size(0))[0]
            edge_weight = torch.cat([edge_weight, torch.zeros(data.pos.size(0), device=edge_weight.device)])
            edge_vec = torch.cat([edge_vec, torch.zeros((data.pos.size(0), 3), device=edge_weight.device)], dim=0)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W

    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)
    
    def aggregate(self, features, index):
        # no aggregate
        return features
    
class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-6

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        if norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def none_norm(self, vec):
        return vec

    def max_min_norm(self, vec):
        # vec: (B, N, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=-2, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(vec)

        dist = dist.clamp(min=self.eps)
        direct = vec / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        # delta: (B, N, 1)
        delta = max_val - min_val
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.unsqueeze(-1)) / delta.unsqueeze(-1)

        return dist * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[-2] == 3:
            vec = self.norm(vec)
            return vec * self.weight.view(1, 1, -1)
        elif vec.shape[-2] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=-2)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=-2)
            return vec * self.weight.view(1, 1, -1)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")