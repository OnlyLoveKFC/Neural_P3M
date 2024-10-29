import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data


class LmdbDataset(Dataset):

    def __init__(self, root, transform=None):
        super(LmdbDataset, self).__init__()
        self.path = Path(root)
        if not self.path.is_file():
            db_paths = sorted(self.path.rglob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = self.pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = self.pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()
            
    def pyg2_data_transform(self, data):
        """
        if we're on the new pyg (2.0 or later) and if the Data stored is in older format
        we need to convert the data to the new format
        """
        if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
            return Data(
                **{k: v for k, v in data.__dict__.items() if v is not None}
            )

        return data