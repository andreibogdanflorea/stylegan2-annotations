import argparse
from yaml import safe_load
import os
import cv2
import torch
from torchvision import utils
from tqdm import tqdm
import numpy as np

from models.stylegan import Generator
from models.annotations_generator import AnnotationsGAN
from utils.plotting_utils import blend_image_and_mask, mask_to_rgb
from utils.model_utils import load_generator_model


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
    output_dir = config["DATASET_INFERENCE"]["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        G.eval()
        for i in tqdm(range(config["DATASET_INFERENCE"]["SAMPLES"])):
            sample_z = torch.randn(1, config["MODEL"]["LATENT"], device=device)

            sample, latent = G(
                [sample_z], truncation=config["MODEL"]["TRUNCATION"], truncation_latent=mean_latent, return_latents=True
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
    #     config["MODEL"]["SIZE"],
    #     config["MODEL"]["LATENT"],
    #     config["MODEL"]["N_MLP"],
    #     channel_multiplier=config["MODEL"]["CHANNEL_MULTIPLIER"]
    # ).to(device)
    # checkpoint = torch.load(config["MODEL"]["PRETRAINED"])

    # G.load_state_dict(checkpoint["g_ema"])

    # if config["MODEL"]["TRUNCATION"] < 1.0:
    #     with torch.no_grad():
    #         mean_latent = G.mean_latent(config["MODEL"]["TRUNCATION_MEAN"])
    # else:
    #     mean_latent = None

    # generate(config, G, device, mean_latent)

    annot_gan = AnnotationsGAN(config, inference=True)
    annot_gan = torch.nn.DataParallel(annot_gan, device_ids=[0]).cuda()

    model_state_file = config["CHECKPOINT"]["PATH"]
    checkpoint = torch.load(model_state_file)
    if "state_dict" in checkpoint.keys():
        state_dict = checkpoint["state_dict"]
        annot_gan.load_state_dict(state_dict)
    else:
        annot_gan.module.load_state_dict(model_state_file)
    
    annot_gan.eval()

    if config["MODEL"]["TRUNCATION"] < 1.0:
        with torch.no_grad():
            mean_latent = annot_gan.module.G.mean_latent(config["MODEL"]["TRUNCATION_MEAN"])
    else:
        mean_latent = None

    output_dir = config["DATASET_INFERENCE"]["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(config["DATASET_INFERENCE"]["SAMPLES"])):
        sample_z = torch.randn(1, config["MODEL"]["LATENT"], device=device)
        
        with torch.no_grad():
            image, logits = annot_gan([sample_z], mean_latent=mean_latent, return_image=True)

        preds = torch.argmax(logits, dim=1)[0]

        utils.save_image(
            image,
            f"{output_dir}/{str(i).zfill(6)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        image = cv2.imread(f"{output_dir}/{str(i).zfill(6)}.png")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mask = preds.detach().cpu().numpy()
        rgb_mask = mask_to_rgb(mask)

        blended_image = blend_image_and_mask(image, rgb_mask)

        cv2.imwrite(f"{output_dir}/{str(i).zfill(6)}.png", cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{str(i).zfill(6)}.png", cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
