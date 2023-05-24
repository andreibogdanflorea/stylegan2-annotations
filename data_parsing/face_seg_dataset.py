import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2

from data_parsing.parse_face_seg import FaceSegParser

class FaceSegDataset(Dataset):
    """
    Read dataset for training face segmentation annotations
    """

    def __init__(self, db_root: str, return_image: bool = False) -> None:
        super().__init__()
        self.db_root = db_root
        self.return_image = return_image

        self.db_parser = FaceSegParser(db_root)
        self.annotations = self.db_parser.parse()
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> list([torch.Tensor, torch.Tensor]):
        annotation = self.annotations[index]

        latent = np.load(annotation.latent_path)
        mask = np.load(annotation.mask_path)

        latent = torch.from_numpy(latent).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.int64)

        if self.return_image:
            image = cv2.imread(annotation.image_path)
            image = torch.from_numpy(image).to(torch.int8)
            return latent, mask, image
        
        return latent, mask