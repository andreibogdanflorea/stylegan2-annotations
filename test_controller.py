import argparse
from yaml import safe_load
import os
import cv2
import torch
from torchvision import utils
from tqdm import tqdm
import numpy as np
import math

from utils.model_utils import load_generator_model
from models.attributes.age_model import AgeModel
from models.attributes.gender_model import GenderModel
from models.control_mapping_network import ControlMappingNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="inference configuration file", required=True, type=str
    )

    return parser.parse_args()

class ControllerTester:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self.parse_config(config_file)

        self.device = "cuda"

        self.G = load_generator_model(self.config).to(self.device)
        self.G.eval()
        for param in self.G.parameters():
            param.requires_grad = False
        self.age_model = AgeModel()
        self.age_model.eval()
        for param in self.age_model.parameters():
            param.requires_grad = False
        self.gender_model = GenderModel()
        self.gender_model.eval()
        for param in self.gender_model.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            self.mean_latent = self.G.mean_latent(self.config["MODEL"]["TRUNCATION_MEAN"])
        
        self.mapping_network = ControlMappingNetwork().to(self.device)
        mapping_network_checkpoint = torch.load(self.config["TEST_SETUP"]["MAPPING_NETWORK_CHECKPOINT"])
        self.mapping_network.load_state_dict(mapping_network_checkpoint["state_dict"])
        self.mapping_network.eval()

    def parse_config(self, config_file: str) -> dict:
        config = {}
        if os.path.isfile(config_file) and config_file.endswith(".yml"):
            with open(config_file, "r") as f_config:
                config = safe_load(f_config) 
        else:
            print("Invalid config path: {}".format(config_file))
        
        return config
    
    def test_step(
        self,
        attributes: torch.Tensor
    ) -> list([torch.Tensor, torch.Tensor]):
        pred_latents_w = self.mapping_network(attributes)

        sample_z = torch.randn(1, 512, device=self.device)
        sample_w = self.G(
            [sample_z],
            truncation=self.config["MODEL"]["TRUNCATION"],
            truncation_latent=self.mean_latent, # pred_latents_w,
            return_mapped_style=True
        )

        alpha = 0.2
        coarse_features = pred_latents_w + alpha * (sample_w[0] - pred_latents_w)
        coarse_features = coarse_features.unsqueeze(1).repeat(1, 2, 1)

        alpha = 0.2
        mid_features = pred_latents_w + alpha * (sample_w[0] - pred_latents_w)
        mid_features = mid_features.unsqueeze(1).repeat(1, 6, 1)

        high_features = sample_w[0].unsqueeze(1).repeat(1, 6, 1)
        latent = torch.cat([coarse_features, mid_features, high_features], dim=1)
        sample, _ = self.G([latent], input_is_latent=True, inject_index=4)

        #sample, _ = self.G([sample_w[0], pred_latents_w], input_is_latent=True, inject_index=4)
        sample2, _ = self.G([pred_latents_w], input_is_latent=True)
        sample3, _ = self.G([sample_w[0]], input_is_latent=True)

        sample = sample.to("cpu")
        pred_ages_logits = self.age_model(sample)
        pred_ages = self.age_model.get_age(pred_ages_logits)

        pred_genders = self.gender_model(sample)

        pred_ages = pred_ages.to("cuda")
        pred_genders = pred_genders.to("cuda")

        return sample, sample2, sample3, pred_ages, pred_genders

    def test(self) -> None:
        ages_mean_error = 0
        gender_correct = 0

        if not os.path.exists(self.config["TEST_SETUP"]["SAMPLES_SAVE_DIR"]):
            os.makedirs(self.config["TEST_SETUP"]["SAMPLES_SAVE_DIR"])

        self.mapping_network.eval()
        with torch.no_grad():
            for i in tqdm(range(self.config["TEST_SETUP"]["N_TEST_SAMPLES"])):
                age = np.random.uniform(10, 70)
                gender = np.random.uniform(0, 1)
                gender_binary = torch.from_numpy(np.array([gender > 0.5])).float().cuda(non_blocking=True)

                attributes = np.array([age, gender])
                attributes = torch.from_numpy(attributes).float()
                attributes = attributes.unsqueeze(0)

                attributes = attributes.cuda(non_blocking=True)
                sample, sample2, sample3, pred_ages, pred_genders = self.test_step(attributes)

                ages_mean_error += torch.abs(pred_ages - attributes[:, 0])
                pred_genders = torch.argmax(pred_genders, dim=1)
                gender_correct += (pred_genders == gender_binary).float().sum()

                utils.save_image(
                    sample,
                    os.path.join(
                        self.config["TEST_SETUP"]["SAMPLES_SAVE_DIR"],
                        "sample_{}.png".format(0)
                    ),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    sample2,
                    os.path.join(
                        self.config["TEST_SETUP"]["SAMPLES_SAVE_DIR"],
                        "sample_{}.png".format(1)
                    ),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    sample3,
                    os.path.join(
                        self.config["TEST_SETUP"]["SAMPLES_SAVE_DIR"],
                        "sample_{}.png".format(2)
                    ),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

                print(pred_ages)
                print(pred_genders)
                print(attributes)
                print(pred_ages[0].item(), pred_genders[0].item(), attributes[0, :])

                import time
                time.sleep(5)
        
        age_mean_error = ages_mean_error / self.config["TEST_SETUP"]["N_TEST_SAMPLES"]
        gender_accuracy = 100 * gender_correct / self.config["TEST_SETUP"]["N_TEST_SAMPLES"]
        print("Age mean error: {}".format(age_mean_error.item()))
        print("gender accuracy: {}".format(gender_accuracy))

if __name__ == "__main__":
    args = parse_args()
    tester = ControllerTester(config_file=args.config)
    tester.test()
        