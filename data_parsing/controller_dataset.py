import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class ControllerDataset(Dataset):
    """
    Read dataset for training controller
    """

    def __init__(self, pkl_file: str) -> None:
        super().__init__()
        self.pkl_file = pkl_file
        self.data = pd.read_pickle(pkl_file).to_numpy()
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> list([torch.Tensor, torch.Tensor]):
        latent_z = torch.from_numpy(self.data[index][0]).float()
        latent_w = torch.from_numpy(self.data[index][1]).float()
        age = self.data[index][2]
        gender = self.data[index][3]

        attributes = np.array([age, gender])
        attributes = torch.from_numpy(attributes).float()

        return attributes, latent_z, latent_w