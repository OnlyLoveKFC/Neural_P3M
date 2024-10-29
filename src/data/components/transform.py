from typing import List, Union

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class NonPBCAddCell(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        # For non-pbc molecules, we need to define a cell that is large enough to contain the molecule and rotate with it
        pos = data.pos
        pos_centered = pos - pos.mean(dim=0)
        if pos_centered.shape[0] > 2:
            _, _, V = torch.svd(pos_centered)
        else:
            raise ValueError("The molecule has less than 3 atoms, cannot define a cell.")
        cell = V.t()
        data.cell = cell
        return data


class NonPBCAddGrid(BaseTransform):
    def __init__(self, expand_size: int, num_grids: Union[List[int], int]) -> None:
        super().__init__()
        if isinstance(num_grids, int):
            num_grids = [num_grids, num_grids, num_grids]
        self.num_grids = num_grids
        self.expand_size = expand_size

    def __call__(self, data):
        pos = data.pos
        pos_centered = pos - pos.mean(dim=0)
        rotated_pos_centered = torch.matmul(pos_centered, data.cell.T)
        cell_lengths = rotated_pos_centered.max(dim=0).values - rotated_pos_centered.min(dim=0).values
        translation = rotated_pos_centered.min(dim=0).values - 1 / 2 * self.expand_size
        translation = torch.matmul(translation, data.cell)
        cell_lengths += self.expand_size
        new_cell = data.cell * cell_lengths.unsqueeze(1)
        new_pos = pos_centered - translation

        x_linespace = torch.linspace(0, 1, self.num_grids[0] + 1, dtype=torch.float32)
        y_linespace = torch.linspace(0, 1, self.num_grids[1] + 1, dtype=torch.float32)
        z_linespace = torch.linspace(0, 1, self.num_grids[2] + 1, dtype=torch.float32)
        # calculate centers of the mesh
        x_centers = (x_linespace[1:] + x_linespace[:-1]) / 2
        y_centers = (y_linespace[1:] + y_linespace[:-1]) / 2
        z_centers = (z_linespace[1:] + z_linespace[:-1]) / 2
        # create mesh: (N, num_x_centers, num_y_centers, num_z_centers, 3)
        mesh = torch.stack(torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij'), dim=-1)
        # (N, num_x_centers, num_y_centers, num_z_centers, 3)
        mesh_coord = torch.einsum("ijkl,lm->ijkm", mesh, new_cell)
        # Use heterogeneous graph to save both mesh and atom in PyG
        mesh_atom_graph = HeteroData()
        mesh_atom_graph['atom'].pos = new_pos.float()
        mesh_atom_graph['atom'].atomic_numbers = data.atomic_numbers.long()
        mesh_atom_graph['atom'].cell = new_cell.reshape(1, 3, 3).float()
        mesh_atom_graph['atom'].natoms = torch.tensor(data.atomic_numbers.shape[0]).long()
        mesh_atom_graph['atom'].fixed = data.fixed.bool() if 'fixed' in data else None
        mesh_atom_graph.y = torch.tensor(data.y_relaxed, dtype=torch.float32).squeeze(0) if 'y_relaxed' in data else data.y.squeeze(0)
        mesh_atom_graph.neg_dy = data.neg_dy.float() if 'neg_dy' in data else None
        mesh_atom_graph['mesh'].pos = mesh_coord.reshape(-1, 3).float()
        mesh_atom_graph['mesh'].nmeshs = torch.tensor(mesh_atom_graph['mesh'].pos.shape[0]).long()
        return mesh_atom_graph

class PBCAddGrid(BaseTransform):
    def __init__(self, num_grids: Union[List[int], int]) -> None:
        super().__init__()
        if isinstance(num_grids, int):
            num_grids = [num_grids, num_grids, num_grids]
        self.num_grids = num_grids

    def __call__(self, data):
        x_linespace = torch.linspace(0, 1, self.num_grids[0] + 1, dtype=torch.float32)
        y_linespace = torch.linspace(0, 1, self.num_grids[1] + 1, dtype=torch.float32)
        z_linespace = torch.linspace(0, 1, self.num_grids[2] + 1, dtype=torch.float32)
        # calculate centers of the mesh
        x_centers = (x_linespace[1:] + x_linespace[:-1]) / 2
        y_centers = (y_linespace[1:] + y_linespace[:-1]) / 2
        z_centers = (z_linespace[1:] + z_linespace[:-1]) / 2
        # create mesh: (N, num_x_centers, num_y_centers, num_z_centers, 3)
        mesh = torch.stack(torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij'), dim=-1)
        # (N, num_x_centers, num_y_centers, num_z_centers, 3)
        mesh_coord = torch.einsum("ijkl,blm->bijkm", mesh, data.cell)
        # Use heterogeneous graph to save both mesh and atom in PyG
        mesh_atom_graph = HeteroData()
        mesh_atom_graph['atom'].pos = data.pos.float()
        mesh_atom_graph['atom'].atomic_numbers = data.atomic_numbers.long()
        mesh_atom_graph['atom'].cell = data.cell.reshape(1, 3, 3).float()
        mesh_atom_graph['atom'].natoms = torch.tensor(data.atomic_numbers.shape[0]).long()
        mesh_atom_graph.fixed = data.fixed.bool() if 'fixed' in data else None
        mesh_atom_graph.y = torch.tensor(data.energy, dtype=torch.float32).squeeze(0) if 'energy' in data else None
        mesh_atom_graph.neg_dy = data.forces.float() if 'forces' in data else None
        mesh_atom_graph['mesh'].pos = mesh_coord.reshape(-1, 3).float()
        mesh_atom_graph['mesh'].nmeshs = torch.tensor(mesh_atom_graph['mesh'].pos.shape[0]).long()
        mesh_atom_graph.sid = data.sid if 'sid' in data else None
        mesh_atom_graph.fid = data.fid if 'fid' in data else None
        return mesh_atom_graph