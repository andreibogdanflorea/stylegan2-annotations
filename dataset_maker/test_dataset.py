import argparse
from yaml import safe_load
import os
import sys

import torch
from torchvision import utils
import numpy as np

sys.path.append('.')
from utils.model_utils import load_generator_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset for labelling")
    parser.add_argument(
        "--config", help="dataset generation configuration file", required=True, type=str
    )

    return parser.parse_args()

def parse_config(config_file: str) -> dict:
    config = {}
    if os.path.isfile(config_file) and config_file.endswith(".yml"):
        with open(config_file, "r") as f_config:
            config = safe_load(f_config) 
    else:
        print("Invalid config path: {}".format(config_file))
    
    return config


def generate(config, G, device):
    output_dir = config["DATASET_INFERENCE"]["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        G.eval()

        latents = np.load("/home/andrei/Documents/licenta/face_seg_dataset/000001_latent.npy")
        latents = torch.tensor(latents, device=device)

        sample, _ = G(
            [latents],
            input_is_latent=True,
            randomize_noise=False
        )

        utils.save_image(
            sample,
            f"{output_dir}/{str(0).zfill(6)}_other.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

if __name__ == '__main__':
    device = "cuda"
    args = parse_args()
    config = parse_config(args.config)

    G = load_generator_model(config)

    generate(config, G, device)