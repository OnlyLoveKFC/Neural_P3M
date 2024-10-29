from typing import List, Union

import torch
from torch import nn
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (BesselBasisLayer,
                                               EmbeddingBlock, ResidualLayer,
                                               SphericalBasisLayer)
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.nn.resolver import activation_resolver
from torch_scatter import scatter
from torch_sparse import SparseTensor

from src.models.components.fno import FNO3d

from .radius_utils import (get_distances, get_distances_pbc,
                           radius_determinstic, radius_graph_determinstic,
                           radius_graph_pbc, radius_pbc)
from .utils import InteractionBlock, MultiheadAttention, Scalar


class DimeNetPlusPlus_P3M(nn.Module):
    def __init__(
        self,
        regress_forces:bool,
        use_pbc: bool,
        num_layers: int,
        num_rbf: int,
        num_filters: int,
        hidden_channels: int,
        max_z: int,
        atom_cutoff: int,
        max_a2a_neighbors: int,
        grid_cutoff: int,
        max_a2m_neighbors: int,
        num_grids: Union[List[int], int],
        long_type: str,
        int_emb_size: int,
        basis_emb_size: int,
        num_radial: int,
        num_spherical: int,
        envelope_exponent: int,
        num_before_skip: int,
        num_after_skip: int,
        act="silu",
    ):
        super().__init__()
        self.regress_forces=regress_forces
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
            
        act = activation_resolver(act)

        self.rbf = BesselBasisLayer(num_radial, atom_cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, atom_cutoff, envelope_exponent)
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)
        
        self.a2m_distance_expansion = GaussianSmearing(0.0, grid_cutoff, num_rbf)
        self.m2a_distance_expansion = GaussianSmearing(0.0, grid_cutoff, num_rbf)
        
        self.sl_block = nn.ModuleList()
        for _ in range(num_layers):
            a2m_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            m2a_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            short_mp = InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act
            )
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
                    num_radial,
                    hidden_channels,
                    self.num_grids,
                    a2m_mp,
                    m2a_mp,
                    short_mp,
                    long_mp,
                )
            )
        self.out_a_norm = nn.LayerNorm(hidden_channels)
        self.out_m_norm = nn.LayerNorm(hidden_channels)
        self.out_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.a_output = Scalar(hidden_channels)
        self.m_output = Scalar(hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for block in self.sl_block:
            block.reset_parameters()
        self.out_a_norm.reset_parameters()
        self.out_m_norm.reset_parameters()
        glorot_orthogonal(self.out_rbf.weight, scale=2.0)
        self.a_output.reset_parameters()
        self.m_output.reset_parameters()
        
    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


    def forward(self, mesh_atom_graph):
        
        if self.regress_forces:
            mesh_atom_graph['atom'].pos.requires_grad_(True)

        bs = mesh_atom_graph.num_graphs if hasattr(mesh_atom_graph['atom'], 'batch') else 1

        a_pos = mesh_atom_graph['atom'].pos
        m_pos = mesh_atom_graph['mesh'].pos
        
        if not self.use_pbc:
            a2a_edge_index, _ = radius_graph_determinstic(
                mesh_atom_graph['atom'], 
                self.atom_cutoff, 
                self.max_a2a_neighbors
            )
            a2a_edge_weights = get_distances(a2a_edge_index, a_pos, return_distance_vec=False)

            a2m_edge_index = radius_determinstic(
                    mesh_atom_graph['atom'],
                    mesh_atom_graph['mesh'],
                    self.grid_cutoff,
                    self.max_a2m_neighbors,
                )
            a2m_edge_weights = get_distances(a2m_edge_index, a_pos, m_pos, return_distance_vec=False)
                
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_edge_weights = get_distances(m2a_edge_index, m_pos, a_pos, return_distance_vec=False)
        else:
            cell = mesh_atom_graph['atom'].cell
            a2a_edge_index, a2a_cell_offsets, a2a_neighbors = radius_graph_pbc(
                mesh_atom_graph['atom'], 
                self.atom_cutoff, 
                self.max_a2a_neighbors,
                symmetrize=True,
            )
            a2a_edge_weights = get_distances_pbc(
                a2a_edge_index, 
                cell, 
                a2a_cell_offsets, 
                a2a_neighbors, 
                a_pos, 
                return_distance_vec=False
            )
            
            a2m_edge_index, a2m_cell_offset, a2m_neighbors = radius_pbc(
                mesh_atom_graph['atom'],
                mesh_atom_graph['mesh'],
                self.grid_cutoff,
                self.max_a2m_neighbors,
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
        
        if not self.use_pbc:
            a2a_cell_offsets = torch.zeros((a2a_edge_index.size(1), 3), device=a2a_edge_index.device, dtype=a2a_edge_index.dtype)
        
        num_nodes = mesh_atom_graph['atom'].atomic_numbers.size(0)
        j, i = a2a_edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            a2a_edge_index,
            a2a_cell_offsets,
            num_nodes=num_nodes,
        )

        # Calculate angles.
        pos_i = a_pos[idx_i].detach()
        pos_j = a_pos[idx_j].detach()
        if self.use_pbc:
            cell_edge = torch.repeat_interleave(cell, a2a_neighbors, dim=0)
            offsets = a2a_cell_offsets.float().view(-1, 1, 3).bmm(cell_edge.float()).view(-1, 3)
            pos_ji, pos_kj = (
                a_pos[idx_j].detach() - pos_i + offsets[idx_ji],
                a_pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                a_pos[idx_j].detach() - pos_i,
                a_pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(a2a_edge_weights)
        sbf = self.sbf(a2a_edge_weights, angle, idx_kj)

        # Embedding block.
        x = self.emb(mesh_atom_graph['atom'].atomic_numbers, rbf, i, j)
        a_x = self.emb.emb(mesh_atom_graph['atom'].atomic_numbers)
        a_x_j = torch.index_select(a_x, 0, a2m_edge_index[0])
        m_x = scatter(a_x_j, a2m_edge_index[1], dim=0, reduce='mean', dim_size=self.total_num_grids * bs)
        
        for num in range(self.num_layers):
            x, m_x = self.sl_block[num](
                x, 
                rbf, 
                sbf, 
                idx_kj, 
                idx_ji,
                m_x, 
                a2a_edge_index,
                a2m_edge_index,
                m2a_edge_index,
                a2m_edge_weights,
                m2a_edge_weights,
                a2m_edge_attr,
                m2a_edge_attr,
                num_nodes
            )
        
        x = self.out_rbf(rbf) * self.out_a_norm(x)
        a_x = scatter(x, i, dim=0, dim_size=num_nodes)
        P_a = self.a_output(a_x)
        P_m = self.m_output(self.out_m_norm(m_x))
        
        energy_a = scatter(
            P_a, 
            mesh_atom_graph['atom'].batch if hasattr(mesh_atom_graph['atom'], 'batch') else torch.zeros_like(mesh_atom_graph['atom'].atomic_numbers),
            dim=0, 
            reduce='sum'
        )
        
        energy_m = torch.sum(P_m.reshape(bs, -1), dim=-1, keepdim=True)
        energy = energy_a + energy_m

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    mesh_atom_graph['atom'].pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy, None
        
        
class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act="silu",
    ):
        act = activation_resolver(act)
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class ShortLongMixLayer(nn.Module):
    def __init__(
        self,
        num_radial: int,
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
        
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        
        self.a2m_layernorm = nn.LayerNorm(hidden_channels)
        self.m2a_layernorm = nn.LayerNorm(hidden_channels)
        self.m2a_combine = nn.Linear(2 * hidden_channels, hidden_channels)
        self.m2a_act = nn.SiLU()
        self.short_layernorm = nn.LayerNorm(hidden_channels)
        self.long_layernorm = nn.LayerNorm(hidden_channels)
    
    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        self.a2m_mp.reset_parameters()
        self.m2a_mp.reset_parameters()
        self.short_mp.reset_parameters()
        self.long_mp.reset_parameters()
        self.a2m_layernorm.reset_parameters()
        self.m2a_layernorm.reset_parameters()
        glorot_orthogonal(self.m2a_combine.weight, scale=2.0)
        self.m2a_combine.bias.data.fill_(0)
        self.short_layernorm.reset_parameters()
        self.long_layernorm.reset_parameters()
    
    def forward(
        self, 
        x, # edge-level embedding
        rbf, 
        sbf, 
        idx_kj, 
        idx_ji,
        m_x, # node-level embedding
        a2a_edge_index,
        a2m_edge_index,
        m2a_edge_index,
        a2m_edge_weights,
        m2a_edge_weights,
        a2m_edge_attr,
        m2a_edge_attr,
        num_nodes
    ):
        
        delta_m_x = m_x
        delta_x = x
        
        # N_edges, F
        x = self.short_layernorm(x)
        x = self.short_mp(x, rbf, sbf, idx_kj, idx_ji)
        
        a_x = self.lin_rbf(rbf) * x
        a_x = scatter(a_x, a2a_edge_index[1], dim=0, dim_size=num_nodes)

        # N_meshs, F
        m_x = self.long_layernorm(m_x)
        if isinstance(self.long_mp, MultiheadAttention):
            m_x = m_x.reshape(-1, torch.prod(torch.tensor(self.num_grids)), self.hidden_channels)
            m_x = self.long_mp(m_x)
            m_x = m_x.reshape(-1, self.hidden_channels)
        else:
            m_x = m_x.reshape(-1, self.num_grids[0], self.num_grids[1], self.num_grids[2], self.hidden_channels).permute(0, 4, 1, 2, 3)
            m_x = self.long_mp(m_x).permute(0, 2, 3, 4, 1).reshape(-1, self.hidden_channels)
        
        # N_meshs, F
        a2m_message = self.a2m_mp(a_x, a2m_edge_index, a2m_edge_weights, a2m_edge_attr, dim_size=m_x.shape[0])
        a2m_message = self.a2m_layernorm(a2m_message)
        
        # N_atoms, F
        m2a_message = self.m2a_mp(m_x, m2a_edge_index, m2a_edge_weights, m2a_edge_attr, dim_size=a_x.shape[0])
        m2a_message_j = m2a_message.index_select(0, a2a_edge_index[0])
        m2a_message_i = m2a_message.index_select(0, a2a_edge_index[1])
        m2a_message = self.m2a_act(self.m2a_combine(torch.cat([m2a_message_j, m2a_message_i], dim=-1)))
        m2a_message = self.m2a_layernorm(m2a_message)
        
        return delta_x + x + m2a_message, m_x + a2m_message + delta_m_x
 
