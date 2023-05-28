import argparse
from yaml import safe_load
import os
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from logger import create_logger
from tensorboardX import SummaryWriter
from evaluate import AverageMeter


from utils.model_utils import load_generator_model
from data_parsing.controller_dataset import ControllerDataset
from logger import create_logger
from models.attributes.age_model import AgeModel
from models.attributes.gender_model import GenderModel
from models.control_mapping_network import ControlMappingNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="inference configuration file", required=True, type=str
    )

    return parser.parse_args()

class ControllerTrainer:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self.parse_config(config_file)

        self.batch_size = self.config["TRAIN_SETUP"]["BATCH_SIZE"]

        current_time = datetime.now()
        self.save_location = os.path.join(
            'weights', 'controller_experiments', current_time.strftime("%d_%m_%Y__%H_%M_%S")
        )
        os.makedirs(self.save_location, exist_ok=True)

        self.device = "cuda"

        self.G = load_generator_model(self.config).to(self.device)
        self.G.eval()
        self.age_model = AgeModel().to(self.device)
        self.age_model.eval()
        self.gender_model = GenderModel().to(self.device)
        self.gender_model.eval()

        self.mapping_network = ControlMappingNetwork().to(self.device)

        self.train_dataset = ControllerDataset(self.config["DATASET"]["PATH"])
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config["TRAIN_SETUP"]["WORKERS"],
            shuffle=True
        )

        self.valid_dataset = ControllerDataset(self.config["DATASET"]["PATH"])
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False
        )

        self.rec_loss = torch.nn.L1Loss()
        self.age_loss = torch.nn.MSELoss()
        self.gender_loss = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.config["TRAIN_SETUP"]["LEARNING_RATE"],
            weight_decay=self.config["TRAIN_SETUP"]["WEIGHT_DECAY"]
        )

        self.scaler = GradScaler()


    def parse_config(self, config_file: str) -> dict:
        config = {}
        if os.path.isfile(config_file) and config_file.endswith(".yml"):
            with open(config_file, "r") as f_config:
                config = safe_load(f_config) 
        else:
            print("Invalid config path: {}".format(config_file))
        
        return config
    
    def save_checkpoint(
        self, 
        states: dict,
    ) -> None:
        filename = "model_best.pth"
        torch.save(states, os.path.join(self.save_location, filename))

    def train_step(
        self,
        attributes: torch.Tensor,
        latents_w: torch.Tensor,
    ) -> list([torch.Tensor, torch.Tensor]):
        
        self.optimizer.zero_grad(set_to_none=True)

        pred_latents_w = self.mapping_network(attributes)
        
        rec_loss = self.rec_loss(pred_latents_w, latents_w)

        sample, _ = self.G([pred_latents_w], input_is_latent=True)

        pred_ages_logits = self.age_model(sample)
        pred_ages = self.age_model.get_age(pred_ages_logits)
        age_loss = self.age_loss(pred_ages, attributes[:, 0])

        pred_genders = self.gender_model(sample)
        genders = attributes[:, 1].long()
        gender_loss = self.gender_loss(pred_genders, genders)

        loss = rec_loss + age_loss + gender_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, pred_latents_w

    def train_epoch(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.mapping_network.train()
        for attributes, _, latents_w in tqdm(self.train_loader):
            attributes = attributes.cuda(non_blocking=True)
            latents_w = latents_w.cuda(non_blocking=True)

            loss, pred_latents_w = self.train_step(attributes, latents_w)

            losses.update(loss, pred_latents_w.size(0))

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        
        train_result = 'Train Epoch {} loss:{:.4f}'.format(epoch, losses.avg)
        return train_result

    def valid_step(
        self,
        attributes: torch.Tensor,
        latents_w: torch.Tensor,
    ) -> list([torch.Tensor, torch.Tensor]):
        pred_latents_w = self.mapping_network(attributes)

        rec_loss = self.rec_loss(pred_latents_w, latents_w)

        sample, _ = self.G([pred_latents_w], input_is_latent=True)
        pred_ages = self.age_model(sample)
        age_loss = self.age_loss(pred_ages, attributes[:, 0])

        pred_genders = self.gender_model(sample)
        gender_loss = self.gender_loss(pred_genders, attributes[:, 1])

        loss = rec_loss + age_loss + gender_loss

        return loss, pred_latents_w

    def validate(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.mapping_network.eval()
        with torch.no_grad():
            for attributes, _, latents_w in tqdm(self.valid_loader):
                attributes = attributes.cuda(non_blocking=True)
                latents_w = latents_w.cuda(non_blocking=True)
                loss, pred_latents_w = self.valid_step(attributes, latents_w)

                losses.update(loss.item(), pred_latents_w.size(0))

        valid_result = (
            "Valid Epoch {} loss:{:.4f}\n".format(epoch, losses.avg)
        )

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        writer.flush()

        return valid_result, losses.avg

    def fit(self) -> None:
        torch.backends.cudnn.benchmark = True
        best_loss = 1e10

        logger, tensorboard_log_dir = create_logger(self.save_location)
        writer_dict = {
            'writer': SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=5),
            'train_global_steps': 0,
            'valid_global_steps': 0
        }
        
        for epoch in tqdm(range(self.config["TRAIN_SETUP"]["EPOCHS"])):
            logger.info("Training epoch: {}".format(epoch))
            train_result = self.train_epoch(epoch, writer_dict)
            logger.info(train_result)

            if ((epoch + 1) % self.config["TRAIN_SETUP"]["SAVE_FREQ"]) == 0:
                valid_result, valid_loss = self.validate(epoch, writer_dict)
                logger.info(valid_result)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    self.save_checkpoint(
                        {
                            "state_dict": self.mapping_network.state_dict(),
                            "epoch": epoch + 1,
                            "optimizer": self.optimizer.state_dict()
                        },
                    )
            
            writer_dict['writer'].flush()
            
        writer_dict['writer'].close()

if __name__ == "__main__":
    args = parse_args()
    trainer = ControllerTrainer(config_file=args.config)
    trainer.fit()
