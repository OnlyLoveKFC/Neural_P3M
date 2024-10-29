from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from ..fno import FNO3d
from ..radius_utils import (get_distances, get_distances_pbc,
                            radius_determinstic, radius_pbc)
from ..utils import InteractionBlock, MultiheadAttention, Scalar
from .output_modules import EquivariantScalar
from .visnet_utils import (CosineCutoff, Distance_Deterministic, Distance_PBC,
                           EdgeEmbedding, ExpNormalSmearing, NeighborEmbedding,
                           Sphere, VecLayerNorm)


class ViSNet_P3M(nn.Module):

    def __init__(
        self,
        regress_forces=True,
        use_pbc=False,
        num_layers=6,
        num_rbf=32,
        num_filters=64,
        hidden_channels=128,
        max_z=100,
        atom_cutoff=5.0,
        max_a2a_neighbors=32,
        grid_cutoff=4.0,
        max_a2m_neighbors=6,
        num_grids=3,
        long_type='FNO', # 'FNO' or 'MHA'
        lmax=1,
        num_heads=8,
    ):
        super(ViSNet_P3M, self).__init__()
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.num_filters = num_filters
        self.hidden_channels = hidden_channels
        self.max_z = max_z
        self.atom_cutoff = atom_cutoff
        self.max_a2a_neighbors = max_a2a_neighbors
        self.grid_cutoff = grid_cutoff
        self.max_a2m_neighbors = max_a2m_neighbors
        
        if isinstance(num_grids, int):
            self.num_grids = [num_grids, num_grids, num_grids]
        else:
            self.num_grids = num_grids
            
        self.total_num_grids = self.num_grids[0] * self.num_grids[1] * self.num_grids[2]
        self.lmax = lmax
        self.num_heads = num_heads
        
        self.embedding = nn.Embedding(max_z, hidden_channels)
        if not self.use_pbc:
            self.a2a_distance = Distance_Deterministic(atom_cutoff, max_num_neighbors=max_a2a_neighbors, loop=True)
        else:
            self.a2a_distance = Distance_PBC(atom_cutoff, max_num_neighbors=max_a2a_neighbors, loop=True)
        self.a2a_sphere = Sphere(l=lmax)
        self.a2a_distance_expansion = ExpNormalSmearing(atom_cutoff, num_rbf, False)
        self.a2m_distance_expansion = ExpNormalSmearing(grid_cutoff, num_rbf, False)
        self.m2a_distance_expansion = ExpNormalSmearing(grid_cutoff, num_rbf, False)
        
        self.a2a_neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, atom_cutoff, max_z)
        self.a2a_edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        self.sl_block = nn.ModuleList()
        for i in range(num_layers):
            a2m_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            m2a_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            short_mp = ViS_MP(num_heads, hidden_channels, atom_cutoff, last_layer=False if i < num_layers - 1 else True)
            if long_type == 'FNO':
                long_mp = FNO3d(
                    *self.num_grids, 
                    hidden_channels=hidden_channels // 2, 
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels, 
                    n_layers=1,
                    lifting_channels=hidden_channels // 2,
                    projection_channels=hidden_channels // 2,
                    non_linearity=nn.SiLU(),
                )
            
            elif long_type == 'MHA': 
                long_mp = MultiheadAttention(hidden_channels, hidden_channels, 8)
            else:
                raise ValueError(f'Unknown long range interaction type: {long_type}')
            self.sl_block.append(
                ShortLongMixLayer(
                    hidden_channels,
                    self.num_grids,
                    a2m_mp,
                    m2a_mp,
                    short_mp,
                    long_mp,
                )
            )
        self.out_a_norm = nn.LayerNorm(hidden_channels)
        self.out_a_vec_norm = VecLayerNorm(hidden_channels, trainable=False, norm_type='none')
        self.out_m_norm = nn.LayerNorm(hidden_channels)
        self.a_output = EquivariantScalar(hidden_channels)
        self.m_output = Scalar(hidden_channels)
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, (nn.Embedding, nn.LayerNorm, VecLayerNorm, ExpNormalSmearing)):
            module.reset_parameters()
        
    def forward(self, mesh_atom_graph) -> Tuple[Tensor, Tensor]:
        
        if self.regress_forces:
            mesh_atom_graph['atom'].pos.requires_grad_(True)
            
        bs = mesh_atom_graph.num_graphs if hasattr(mesh_atom_graph['atom'], 'batch') else 1
        
        z, a_pos = mesh_atom_graph['atom'].atomic_numbers, mesh_atom_graph['atom'].pos
        m_pos = mesh_atom_graph['mesh'].pos
        
        a2a_edge_index, a2a_edge_weights, a2a_edge_vecs = self.a2a_distance(mesh_atom_graph['atom'])
        
        # Embedding Layers
        if not self.use_pbc:  
            a2m_edge_index = radius_determinstic(
                    mesh_atom_graph['atom'],
                    mesh_atom_graph['mesh'],
                    self.grid_cutoff,
                    self.max_a2m_neighbors,
                    enforce_max_neighbors_strictly=True,
                )
            a2m_edge_weights = get_distances(a2m_edge_index, a_pos, m_pos, return_distance_vec=False)

            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_edge_weights = get_distances(m2a_edge_index, m_pos, a_pos, return_distance_vec=False)
        else: 
            cell = mesh_atom_graph['atom'].cell
            a2m_edge_index, a2m_cell_offset, a2m_neighbors = radius_pbc(
                mesh_atom_graph['atom'],
                mesh_atom_graph['mesh'],
                self.grid_cutoff,
                self.max_a2m_neighbors,
                enforce_max_neighbors_strictly=True,
            )
            
            a2m_edge_weights = get_distances_pbc(
                a2m_edge_index, 
                cell, 
                a2m_cell_offset, 
                a2m_neighbors, 
                a_pos, 
                m_pos,
                return_distance_vec=False
            )
            
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_cell_offset = -1 * a2m_cell_offset
            m2a_neighbors = a2m_neighbors
            
            m2a_edge_weights = get_distances_pbc(
                m2a_edge_index,
                cell,
                m2a_cell_offset,
                m2a_neighbors,
                m_pos,
                a_pos,
                return_distance_vec=False
            )
            
        a2m_edge_attr = self.a2m_distance_expansion(a2m_edge_weights)
        m2a_edge_attr = self.m2a_distance_expansion(m2a_edge_weights)
        
        a_x = self.embedding(z)
        # NonTrainable Message Passing For Initial Mesh Embedding
        a_x_j = torch.index_select(a_x, 0, a2m_edge_index[0])
        m_x = scatter(a_x_j, a2m_edge_index[1], dim=0, reduce='mean', dim_size=self.total_num_grids * bs)
        
        a2a_edge_attr = self.a2a_distance_expansion(a2a_edge_weights)
        a2a_edge_vecs = self.a2a_sphere(a2a_edge_vecs)
        a_x = self.a2a_neighbor_embedding(z, a_x, a2a_edge_index, a2a_edge_weights, a2a_edge_attr)
        a_vec = torch.zeros(a_x.size(0), ((self.lmax + 1) ** 2) - 1, a_x.size(1), device=a_x.device)
        a2a_edge_attr = self.a2a_edge_embedding(a2a_edge_index, a2a_edge_attr, a_x)
        
        for i in range(self.num_layers):
            a_x, m_x, a_vec, a2a_edge_attr = self.sl_block[i](
                a_x,
                a_vec, 
                m_x, 
                a2a_edge_index, 
                a2m_edge_index, 
                m2a_edge_index,
                a2a_edge_weights,
                a2m_edge_weights,
                m2a_edge_weights,
                a2a_edge_attr,
                a2m_edge_attr,
                m2a_edge_attr,
                a2a_edge_vecs,
            )
        
        out_a_x = self.out_a_norm(a_x)
        out_a_vec = self.out_a_vec_norm(a_vec)
        out_m_x = self.out_m_norm(m_x)
        
        output_a_x = self.a_output(out_a_x, out_a_vec)
        energy_a = scatter(
            output_a_x, 
            mesh_atom_graph['atom'].batch if hasattr(mesh_atom_graph['atom'], 'batch') else torch.zeros_like(mesh_atom_graph['atom'].atomic_numbers),
            dim=0, 
            reduce='sum'
        )
        output_m_x = self.m_output(out_m_x)
        energy_m = torch.sum(output_m_x.reshape(bs, -1), dim=-1, keepdim=True)
        
        energy = energy_a + energy_m

        if self.regress_forces:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
            dy = grad(
                [energy],
                [mesh_atom_graph['atom'].pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return energy, -dy
        return energy, None


class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        cutoff,
        last_layer=False,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        
        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()
        
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)
        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None
        
    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):

        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):

        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        
        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1)

        return f1 * w_dot + f2 * t_dot

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ShortLongMixLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_grids: List[int],
        a2m_mp: nn.Module,
        m2a_mp: nn.Module,
        short_mp: nn.Module,
        long_mp: nn.Module,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.a2m_mp = a2m_mp
        self.m2a_mp = m2a_mp
        self.short_mp = short_mp
        self.long_mp = long_mp
        self.num_grids = num_grids
        self.short_layernorm = nn.LayerNorm(hidden_channels)
        self.short_vec_layernorm = VecLayerNorm(hidden_channels, trainable=False, norm_type='none')
        self.short_f_layernorm = nn.LayerNorm(hidden_channels)
        self.long_layernorm = nn.LayerNorm(hidden_channels)
        
    def reset_parameters(self):
        self.a2m_mp.reset_parameters()
        self.m2a_mp.reset_parameters()
        self.short_mp.reset_parameters()
        self.long_mp.reset_parameters()
        self.short_layernorm.reset_parameters()
        self.short_vec_layernorm.reset_parameters()
        self.long_layernorm.reset_parameters()
        self.short_f_layernorm.reset_parameters()
    
    def forward(
        self, 
        a_x,
        a_vec, 
        m_x, 
        a2a_edge_index, 
        a2m_edge_index, 
        m2a_edge_index,
        a2a_edge_weights,
        a2m_edge_weights,
        m2a_edge_weights,
        a2a_edge_attr,
        a2m_edge_attr,
        m2a_edge_attr,
        a2a_edge_vecs,
    ):
        
        delta_a_x, delta_m_x, delta_a_vec = a_x, m_x, a_vec
        delta_a2a_edge_attr = a2a_edge_attr
        
        # N_atoms, F
        a_x = self.short_layernorm(a_x)
        a_vec = self.short_vec_layernorm(a_vec)
        a2a_edge_attr = self.short_f_layernorm(a2a_edge_attr) / self.hidden_channels
        
        s_a_x, s_a_vec, s_a2a_edge_attr = self.short_mp(
            a_x, 
            a_vec, 
            a2a_edge_index, 
            a2a_edge_weights, 
            a2a_edge_attr,
            a2a_edge_vecs,
        )
        if s_a2a_edge_attr is None:
            s_a2a_edge_attr = torch.zeros_like(delta_a2a_edge_attr)
        
        # N_meshs, F
        m_x = self.long_layernorm(m_x)
        if isinstance(self.long_mp, MultiheadAttention):
            l_m_x = m_x.reshape(-1, torch.prod(torch.tensor(self.num_grids)), self.hidden_channels)
            l_m_x = self.long_mp(l_m_x)
            l_m_x = l_m_x.reshape(-1, self.hidden_channels)
        else:
            l_m_x = m_x.reshape(-1, self.num_grids[0], self.num_grids[1], self.num_grids[2], self.hidden_channels).permute(0, 4, 1, 2, 3)
            l_m_x = self.long_mp(l_m_x).permute(0, 2, 3, 4, 1).reshape(-1, self.hidden_channels)
        # N_meshs, F
        a2m_message = self.a2m_mp(a_x, a2m_edge_index, a2m_edge_weights, a2m_edge_attr, dim_size=m_x.shape[0])
        # N_atoms, F
        m2a_message = self.m2a_mp(m_x, m2a_edge_index, m2a_edge_weights, m2a_edge_attr, dim_size=a_x.shape[0])
        return s_a_x + m2a_message + delta_a_x, l_m_x + a2m_message + delta_m_x, s_a_vec + delta_a_vec, s_a2a_edge_attr + delta_a2a_edge_attr