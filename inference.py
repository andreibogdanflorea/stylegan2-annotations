import argparse
from yaml import safe_load
import os

import torch
from torchvision import utils
from models.stylegan import Generator
from models.annotations_generator import AnnotationsGAN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="inference configuration file", required=True, type=str
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
    output_dir = config["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        G.eval()
        for i in tqdm(range(config["SAMPLES"])):
            sample_z = torch.randn(1, config["LATENT"], device=device)

            sample, _ = G(
                [sample_z], truncation=config["TRUNCATION"], truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                f"{output_dir}/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"
    args = parse_args()
    config = parse_config(args.config)

    # G = Generator(
    #     config["SIZE"], config["LATENT"], config["N_MLP"], channel_multiplier=config["CHANNEL_MULTIPLIER"]
    # ).to(device)
    # checkpoint = torch.load(config["CHECKPOINT_PATH"])

    # G.load_state_dict(checkpoint["g_ema"])

    # if config["TRUNCATION"] < 1.0:
    #     with torch.no_grad():
    #         mean_latent = G.mean_latent(config["TRUNCATION_MEAN"])
    # else:
    #     mean_latent = None

    # generate(config, G, device, mean_latent)

    annot_gan = AnnotationsGAN(config).to(device)

    sample_z = torch.randn(1, config["LATENT"], device=device)
    annot_gan([sample_z])

    import time
    time.sleep(10)