import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class MD22(InMemoryDataset):

    gdml_url = 'http://quantum-machine.org/gdml/data/npz'

    file_names = {
        'AT-AT-CG-CG': 'md22_AT-AT-CG-CG.npz',
        'AT-AT': 'md22_AT-AT.npz',
        'Ac-Ala3-NHMe': 'md22_Ac-Ala3-NHMe.npz',
        'DHA': 'md22_DHA.npz',
        'buckyball-catcher': 'md22_buckyball-catcher.npz',
        'dw-nanotube': 'md22_dw_nanotube.npz',
        'stachyose': 'md22_stachyose.npz',        
    }

    def __init__(
        self,
        root: str,
        molecules: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        name = molecules
        if name not in self.file_names:
            raise ValueError(f"Unknown dataset name '{name}'")

        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self):
        url = f'{self.gdml_url}/{self.file_names[self.name]}'
        download_url(url, self.raw_dir)

    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)
            
            z = torch.from_numpy(raw_data['z']).long()
            pos = torch.from_numpy(raw_data['R']).float()
            energy = torch.from_numpy(raw_data['E'])
            force = torch.from_numpy(raw_data['F']).float()

            data_list = []
            for i in range(pos.size(0)):
                data = Data(atomic_numbers=z, pos=pos[i], y=energy[i], neg_dy=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)
    
    @property
    def molecule_splits(self):
        """
            Splits refer to MD22 https://arxiv.org/pdf/2209.14865.pdf
        """
        return {
            'Ac-Ala3-NHMe': 6000,
            'DHA': 8000,
            'stachyose': 8000,
            'AT-AT': 3000,
            'AT-AT-CG-CG': 2000,
            'buckyball-catcher': 600,
            'dw-nanotube': 800
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"