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
        for param in self.G.parameters():
            param.requires_grad = False
        self.age_model = AgeModel().to(self.device)
        self.age_model.eval()
        for param in self.age_model.parameters():
            param.requires_grad = False
        self.gender_model = GenderModel().to(self.device)
        self.gender_model.eval()
        for param in self.gender_model.parameters():
            param.requires_grad = False 

        self.mapping_network = ControlMappingNetwork().to(self.device)

        self.train_dataset = ControllerDataset(self.config["DATASET_TRAIN"]["PATH"])

        random_sampler = torch.utils.data.RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.config["TRAIN_SETUP"]["ITERS_PER_EPOCH"] * self.config["TRAIN_SETUP"]["BATCH_SIZE"]
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config["TRAIN_SETUP"]["WORKERS"],
            sampler=random_sampler
        )

        self.valid_dataset = ControllerDataset(self.config["DATASET_VALID"]["PATH"])
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

        pred_latents_w = self.mapping_network(attributes)
        
        rec_loss = self.rec_loss(pred_latents_w, latents_w)
        
        sample, _ = self.G([pred_latents_w], input_is_latent=True)
        torch.cuda.empty_cache()

        pred_ages_logits = self.age_model(sample)
        pred_ages = self.age_model.get_age(pred_ages_logits)
        age_loss = self.age_loss(pred_ages, attributes[:, 0])

        pred_genders = self.gender_model(sample)
        genders = attributes[:, 1].long()
        gender_loss = self.gender_loss(pred_genders, genders)

        loss = rec_loss + 0.01 * age_loss + gender_loss
        loss.backward()

        return loss, pred_latents_w

    def train_epoch(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.mapping_network.train()
        for i, (attributes, _, latents_w) in tqdm(enumerate(self.train_loader)):
            attributes = attributes.cuda(non_blocking=True)
            latents_w = latents_w.cuda(non_blocking=True)

            loss, pred_latents_w = self.train_step(attributes, latents_w)

            losses.update(loss, pred_latents_w.size(0))

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if i % 64 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        
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
        pred_ages_logits = self.age_model(sample)
        pred_ages = self.age_model.get_age(pred_ages_logits)
        age_loss = self.age_loss(pred_ages, attributes[:, 0])

        pred_genders = self.gender_model(sample)
        genders = attributes[:, 1].long()
        gender_loss = self.gender_loss(pred_genders, genders)

        loss = rec_loss + 0.01 * age_loss + gender_loss

        return loss, pred_latents_w, rec_loss, pred_ages, pred_genders

    def validate(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()
        rec_losses = AverageMeter()
        ages_mean_error = 0
        gender_correct = 0

        self.mapping_network.eval()
        with torch.no_grad():
            for attributes, _, latents_w in tqdm(self.valid_loader):
                attributes = attributes.cuda(non_blocking=True)
                latents_w = latents_w.cuda(non_blocking=True)
                loss, pred_latents_w, rec_loss, pred_ages, pred_genders = self.valid_step(attributes, latents_w)

                losses.update(loss.item(), pred_latents_w.size(0))
                rec_losses.update(rec_loss.item(), pred_latents_w.size(0))
                ages_mean_error += torch.abs(pred_ages - attributes[:, 0])

                pred_genders = torch.argmax(pred_genders, dim=1)
                gender_correct += (pred_genders == attributes[:, 1]).float().sum() 

        valid_result = (
            "Valid Epoch {} loss:{:.4f}\n".format(epoch, losses.avg)
        )
        
        age_mean_error = ages_mean_error / len(self.valid_loader)
        gender_accuracy = 100 * gender_correct / len(self.valid_loader)
        print("Age mean error: {}".format(age_mean_error.item()))
        print("gender accuracy: {}".format(gender_accuracy))

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_rec_loss', rec_losses.avg, global_steps)
        writer.add_scalar('valid_age_mean_error', age_mean_error, global_steps)
        writer.add_scalar('valid_gender_accuracy', gender_accuracy, global_steps)
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
