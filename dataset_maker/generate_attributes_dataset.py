import argparse
from yaml import safe_load
import os
import sys
from tqdm import tqdm

import torch
from torchvision import utils
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

sys.path.append('.')
from utils.model_utils import load_generator_model
from models.attributes.age_model import AgeModel
from models.attributes.gender_model import GenderModel


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
    save_path = config["DATASET_INFERENCE"]["SAVE_PATH"]
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    age_model = AgeModel().to(device)
    age_model.eval()
    gender_model = GenderModel().to(device)
    gender_model.eval()

    attributes_df = pd.DataFrame(columns=['styles', 'latents', 'age', 'gender'])

    with torch.no_grad():
        G.eval()
        for i in tqdm(range(config["DATASET_INFERENCE"]["SAMPLES"])):
            sample_z = torch.randn(config["DATASET_INFERENCE"]["BATCH_SIZE"], config["MODEL"]["LATENT"], device=device)

            sample, latents = G(
                [sample_z], truncation=config["MODEL"]["TRUNCATION"], truncation_latent=mean_latent,
                return_latents=True
            )

            gender_logits = gender_model(sample)
            gender = gender_model.get_gender(gender_logits)
            age_logits = age_model(sample)
            age = age_model.get_age(age_logits)

            for sample_z_i, latents_i, age_i, gender_i in zip(
                sample_z.cpu().split(1),
                latents.cpu().split(1),
                age.cpu().split(1),
                gender.cpu().split(1),
            ):
                attributes = {
                    'styles': sample_z_i[0].numpy(),
                    'latents': latents_i[0][0].numpy(),
                    'age': age_i[0].numpy(),
                    'gender': gender_i[0].numpy()
                }

                attributes_df = attributes_df.append(attributes, ignore_index=True)
                if len(attributes_df.latents) % 50000 == 0:
                    attributes_df.to_pickle(save_path)

        attributes_df.to_pickle(save_path)

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