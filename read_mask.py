import numpy as np
import cv2

path = '../face_seg_dataset/000001/000001_mask.npy'
mask = np.load(path)

print(np.unique(mask))
cv2.imwrite('mask_img.png', mask)