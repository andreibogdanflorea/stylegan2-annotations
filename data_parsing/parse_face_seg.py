import os
import numpy as np
from tqdm import tqdm
from typing import List

from utils.data_utils import SegmentationAnnotation

class FaceSegParser:
    def __init__(self, db_root: str) -> None:
        self.db_root = db_root
    
    def parse(self) -> List[SegmentationAnnotation]:
        annotations = []

        for sample_dir in os.listdir(self.db_root):
            if not os.path.isdir(os.path.join(self.db_root, sample_dir)):
                continue

            image_path = os.path.join(self.db_root, sample_dir, sample_dir + '.png')
            latent_path = image_path.replace('.png', '.npy')
            mask_path = image_path.replace('.png', '_mask.npy')
            mask_image_path = image_path.replace('.png', ' - all labels.png')
            json_path = image_path.replace('.png', '.json')

            annotation = SegmentationAnnotation(
                image_path=image_path,
                latent_path=latent_path,
                mask_path=mask_path,
                mask_image_path=mask_image_path,
                json_path=json_path)
            
            annotations.append(annotation)
        
        return annotations