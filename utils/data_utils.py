from dataclasses import dataclass, field

@dataclass
class SegmentationAnnotation:
    """
    Definition of a class that encompasses paths to an image, its latent and its segmentation mask
    """
    
    image_path: str = field(default="")
    latent_path: str = field(default="")
    mask_path: str = field(default="")
    mask_image_path: str = field(default="")
    json_path: str = field(default="")

label_mapping = {
    'background': 0,
    'cloth': 1,
    'earring': 2,
    'eyeglass': 3,
    'hair': 4,
    'hat': 5,
    'left_eye': 6,
    'left_ear': 7,
    'left_eyebrow': 8,
    'lower_lip': 9,
    'mouth': 10,
    'neck': 11,
    'necklace': 12,
    'nose': 13,
    'right_ear': 14,
    'right_eye': 15,
    'right_eyebrow': 16,
    'skin': 17,
    'upper_lip': 18
}


