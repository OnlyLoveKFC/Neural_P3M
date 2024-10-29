import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from ase.io import read
from torch_geometric.data import Data, InMemoryDataset, download_url


class AgDataset(InMemoryDataset):
    def __init__(
        self, 
        root: str ,
        transform:  Optional[Callable] = None ,
        pre_transform:  Optional[Callable] = None , 
        pre_filter:  Optional[Callable] = None 
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'Ag_warm_nospin.xyz'
         
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def download(self):
        url = "https://archive.materialscloud.org/record/file?record_id=1387&filename=Ag_warm_nospin.xyz"
        download_url(url, self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'file'), osp.join(self.raw_dir, 'Ag_warm_nospin.xyz'))

    def process(self):
        samples = []
        atoms = read(osp.join(self.raw_dir, self.raw_file_names), index=':', format='extxyz')
        for atom in atoms:
            pos = torch.from_numpy(atom.get_positions()).to(torch.float32)
            cell = atom.get_cell()
            cell = (torch.eye(3) * cell.diagonal()).to(dtype=torch.float32).reshape(1, 3, 3)
            z = torch.from_numpy(atom.get_atomic_numbers()).to(torch.long)
            energy = torch.tensor(atom.get_potential_energy(), dtype=torch.float32).reshape(1)
            forces = torch.from_numpy(atom.get_forces()).to(torch.float32)
            data = Data(atomic_numbers=z, pos=pos, energy=energy, forces=forces, cell=cell)
            samples.append(data)
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

