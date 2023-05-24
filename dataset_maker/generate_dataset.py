import argparse
from yaml import safe_load
import os
import sys

import torch
from torchvision import utils
import numpy as np

sys.path.append('.')
from utils.model_utils import load_generator_model
from tqdm import tqdm


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

def generate(config, G, device, mean_latent):
    output_dir = config["DATASET_INFERENCE"]["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    
    images_output_dir = f"{output_dir}/images"
    os.makedirs(images_output_dir, exist_ok=True)
    latents_output_dir = f"{output_dir}/latents"
    os.makedirs(latents_output_dir, exist_ok=True)

    if mean_latent is not None:
        with open(f"{output_dir}/mean_latent.npy", "wb") as f_latent:
            np.save(f_latent, mean_latent.cpu().numpy())

    with torch.no_grad():
        G.eval()
        for i in tqdm(range(config["DATASET_INFERENCE"]["SAMPLES"])):
            sample_z = torch.randn(1, config["MODEL"]["LATENT"], device=device)

            sample, latents = G(
                [sample_z], truncation=config["MODEL"]["TRUNCATION"], truncation_latent=mean_latent,
                return_latents=True,
                randomize_noise=False
            )

            utils.save_image(
                sample,
                f"{images_output_dir}/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            latents = latents.cpu().numpy()
            with open(f"{latents_output_dir}/{str(i).zfill(6)}.npy", "wb") as f_latent:
                np.save(f_latent, latents)

            # sample, _ = G(
            #     [latents],
            #     input_is_latent=True,
            #     randomize_noise=False
            # )


if __name__ == "__main__":
    device = "cuda"
    args = parse_args()
    config = parse_config(args.config)

    G = load_generator_model(config)

    if config["MODEL"]["TRUNCATION"] < 1.0:
        with torch.no_grad():
            mean_latent = G.mean_latent(config["MODEL"]["TRUNCATION_MEAN"])
    else:
        mean_latent = None

    generate(config, G, device, mean_latent)