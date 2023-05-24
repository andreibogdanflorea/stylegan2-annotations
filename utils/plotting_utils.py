import numpy as np
import cv2

face_parts_palette = [
    (0, 0, 0),
    (0,255,245),
    (11,0,255),
    (0,255,74),
    (64,255,0),
    (96,0,255),
    (234,0,255),
    (0,127,255),
    (202,255,0),
    (0,255,22),
    (255,0,53),
    (255,85,0),
    (255,171,0),
    (0,255,160),
    (0,212,255),
    (149,0,255),
    (255,223,0),
    (255,0,0),
    (255,0,138)
]

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = face_parts_palette[mask[i, j]]
            mask_rgb[i, j, :] = color[0], color[1], color[2]

    return mask_rgb

def blend_image_and_mask(
    img: np.ndarray,
    mask_rgb: np.ndarray
) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    composite_image = cv2.addWeighted(img, 1.0, mask_rgb, 0.5, 0)

    return composite_image