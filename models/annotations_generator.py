import torch
from torch import nn
from torch.nn import functional as F

from models.stylegan import Generator
from utils.model_utils import load_generator_model

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels

        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.activation(self.bn1(x))
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        return h

class AnnotationsGAN(nn.Module):
    def __init__(self, config):
        super(AnnotationsGAN, self).__init__()

        self.config = config
        self.G = load_generator_model(self.config)
        self.G.eval()

        if self.config["TRUNCATION"] < 1.0:
            self.mean_latent = self.G.mean_latent(self.config["TRUNCATION_MEAN"])
        else:
            self.mean_latent = None
        
        self.activation = nn.ReLU()

        self.low_feature_size = 64
        self.low_feature_channels = 128
        self.conv_low_features = nn.Conv2d(5 * 512, self.low_feature_channels, kernel_size=1, bias=False)
        
        self.mid_feature_size = 128
        self.mid_feature_channels = 64
        self.conv_mid_features = nn.Conv2d(256, self.mid_feature_channels, kernel_size=1, bias=False)

        self.low_mid_mix = SegBlock(
            self.low_feature_channels + self.mid_feature_channels,
            self.low_feature_channels + self.mid_feature_channels
        )

        self.high_feature_size = 512
        self.high_feature_channels = 32
        self.conv_high_features = nn.Conv2d(192, self.high_feature_channels, kernel_size=1, bias=False)

        self.low_mid_high_mix = SegBlock(
            self.low_feature_channels + self.mid_feature_channels + self.high_feature_channels,
            self.low_feature_channels + self.mid_feature_channels + self.high_feature_channels
        )

        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(self.low_feature_channels + self.mid_feature_channels + self.high_feature_channels),
            self.activation,
            nn.Conv2d(self.low_feature_channels + self.mid_feature_channels + self.high_feature_channels, 1, kernel_size=3, padding=1)
        )


    def forward(self, z, return_image=False):
        image, features = self.G(
            z,
            truncation=self.config["TRUNCATION"],
            truncation_latent=self.mean_latent,
            return_features=True
        )

        print(list(map(lambda x: x.shape, features)))

        low_features = [
            F.interpolate(features[0], size=self.low_feature_size, mode='bilinear'),
            F.interpolate(features[1], size=self.low_feature_size, mode='bilinear'),
            F.interpolate(features[2], size=self.low_feature_size, mode='bilinear'),
            F.interpolate(features[3], size=self.low_feature_size, mode='bilinear'),
            features[4],
        ]
        low_features = torch.cat(low_features, dim=1)
        low_features = self.conv_low_features(low_features)
        low_features = F.interpolate(low_features, size=self.mid_feature_size, mode='bilinear')

        mid_features = [
            features[5],
        ]
        mid_features = torch.cat(mid_features, dim=1)
        mid_features = self.conv_mid_features(mid_features)

        low_mid_features = torch.cat([low_features, mid_features], dim=1)
        low_mid_features = self.low_mid_mix(low_mid_features)
        low_mid_features = F.interpolate(low_mid_features, size=self.high_feature_size, mode='bilinear')

        high_features = [
            F.interpolate(features[6], size=self.high_feature_size, mode='bilinear'),
            features[7],
        ]
        high_features = torch.cat(high_features, dim=1)
        high_features = self.conv_high_features(high_features)

        low_mid_high_features = torch.cat([low_mid_features, high_features], dim=1)
        low_mid_high_features = self.low_mid_high_mix(low_mid_high_features)

        out = self.out_layer(low_mid_high_features)

        if return_image:
            return image, out
        
        return out

        
