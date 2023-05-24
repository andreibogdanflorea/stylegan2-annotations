import argparse
from yaml import safe_load
import os
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision import utils
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from logger import create_logger
from tensorboardX import SummaryWriter
from evaluate import AverageMeter


from models.annotations_generator import AnnotationsGAN
from data_parsing.face_seg_dataset import FaceSegDataset
from logger import create_logger
from utils.plotting_utils import blend_image_and_mask, mask_to_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--config", help="inference configuration file", required=True, type=str
    )

    return parser.parse_args()

class Trainer:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self.parse_config(config_file)

        self.batch_size = self.config["TRAIN_SETUP"]["BATCH_SIZE"]

        current_time = datetime.now()
        self.save_location = os.path.join(
            'weights', 'experiments', current_time.strftime("%d_%m_%Y__%H_%M_%S")
        )
        os.makedirs(self.save_location, exist_ok=True)

        self.model = AnnotationsGAN(self.config)
        gpus = self.config["TRAIN_SETUP"]["DEVICES"]
        self.model = torch.nn.DataParallel(self.model, device_ids=gpus).cuda()

        self.train_dataset = FaceSegDataset(self.config["DATASET_TRAIN"]["PATH"])
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config["TRAIN_SETUP"]["WORKERS"],
            shuffle=self.config["DATASET_TRAIN"]["SHUFFLE"]
        )

        self.valid_dataset = FaceSegDataset(self.config["DATASET_VALID"]["PATH"], return_image=True)
        self.valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=self.config["DATASET_VALID"]["SHUFFLE"]
        )

        self.loss = torch.nn.CrossEntropyLoss()

        for name, param in self.model.named_parameters():
            if name.startswith("module.G."):
                param.requires_grad = False
        
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of trainable parameters: {}".format(params))

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
    
    def save_checkpoint(self, states: dict, filename: str) -> None:
        torch.save(states, os.path.join(self.save_location, filename))

    def train_step(
        self,
        latent: torch.Tensor,
        target: torch.Tensor
    ) -> list([torch.Tensor, torch.Tensor]):
        
        self.optimizer.zero_grad(set_to_none=True)
        
        logits = self.model(latent)
        loss = self.loss(logits, target)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, logits

    def train_epoch(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.model.train()
        for latent, mask in tqdm(self.train_loader):
            latent = latent.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            loss, logits = self.train_step(latent, mask)

            losses.update(loss, logits.size(0))

            preds = torch.argmax(logits, dim=1)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        
        train_result = 'Train Epoch {} loss:{:.4f}'.format(epoch, losses.avg)
        return train_result

    def valid_step(
        self,
        latent: torch.Tensor,
        target: torch.Tensor
    ) -> list([torch.Tensor, torch.Tensor]):
        logits = self.model(latent)
        loss = self.loss(logits, target)

        preds = torch.argmax(logits, dim=1)

        return loss, preds

    def validate(self, epoch: int, writer_dict: dict) -> str:
        losses = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for latent, mask, image in tqdm(self.valid_loader):
                latent = latent.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                loss, preds = self.valid_step(latent, mask)

                losses.update(loss.item(), latent.size(0))
        
        valid_result = (
            "Valid Epoch {} loss:{:.4f}".format(epoch, losses.avg)
        )

        image = image[0].numpy().astype(np.uint8)
        mask = preds[0].detach().cpu().numpy()
        rgb_mask = mask_to_rgb(mask)
        blended_image = blend_image_and_mask(image, rgb_mask)

        rgb_mask = torch.from_numpy(rgb_mask)
        rgb_mask = rgb_mask.permute((2, 0, 1))
        blended_image = torch.from_numpy(blended_image)
        blended_image = blended_image.permute((2, 0, 1))

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_image('mask', rgb_mask, global_steps)
        writer.add_image('blended_image_and_mask', blended_image, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
        writer.flush()

        return valid_result

    def fit(self) -> None:
        torch.backends.cudnn.benchmark = True

        logger, tensorboard_log_dir = create_logger(self.save_location)
        writer_dict = {
            'writer': SummaryWriter(log_dir=tensorboard_log_dir, flush_secs=5),
            'train_global_steps': 0,
            'valid_global_steps': 0
        }

        if self.config["TRAIN_SETUP"]["RESUME"]:
            model_state_file = self.config["TRAIN_SETUP"]["RESUME_CHECKPOINT"]
            if os.path.is_file(model_state_file):
                checkpoint = torch.load(model_state_file)
                self.config["TRAIN_SETUP"]["BEGIN_EPOCH"] = checkpoint["epoch"]
                if "state_dict" in checkpoint.keys():
                    state_dict = checkpoint["state_dict"]
                    self.model.load_state_dict(state_dict)
                else:
                    self.model.module.load_state_dict(model_state_file)
                
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print("Error: no checkpoint found")
        
        for epoch in tqdm(range(self.config["TRAIN_SETUP"]["BEGIN_EPOCH"], self.config["TRAIN_SETUP"]["END_EPOCH"])):
            logger.info("Training epoch: {}".format(epoch))
            train_result = self.train_epoch(epoch, writer_dict)
            logger.info(train_result)

            if ((epoch + 1) % self.config["TRAIN_SETUP"]["SAVE_FREQ"]) == 0:
                valid_result = self.validate(epoch, writer_dict)
                logger.info(valid_result)
                
                self.save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "epoch": epoch + 1,
                        "optimizer": self.optimizer.state_dict()
                    },
                    'checkpoint_{}.pth'.format(epoch)
                )
            
            writer_dict['writer'].flush()
            
        writer_dict['writer'].close()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(config_file=args.config)
    trainer.fit()
