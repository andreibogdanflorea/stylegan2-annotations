import os
import numpy as np
import cv2
import sys

sys.path.append('.')
from utils.data_utils import label_mapping

if __name__ == '__main__':
    db_root = '/home/andrei/Documents/licenta/face_seg_dataset/test'
    mask_size = 256

    for sample_dir in os.listdir(db_root):
        sample_path = os.path.join(db_root, sample_dir)

        if not os.path.isdir(sample_path):
            continue

        mask_images = sorted([
            filename
            for filename in os.listdir(sample_path)
            if ('-' in filename) and ('all' not in filename)
        ])

        mask = np.zeros((mask_size, mask_size))

        for filename in mask_images:
            label_name = filename.split('.')[0].split('-')[1][1:]
            label = label_mapping[label_name]
            
            file_path = os.path.join(sample_path, filename)
            label_image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)

            mask[gray_image != 0] = label
        
        mask_filename = os.path.join(sample_path, sample_dir + '_mask.npy')
        with open(mask_filename, 'wb') as f_mask:
            np.save(f_mask, mask)
        
        cv2.imwrite(os.path.join(sample_path, sample_dir + '_mask_dump.png'), mask)

            
