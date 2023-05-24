import torch
from models.stylegan import Generator

def load_generator_model(config, device='cuda'):
    G = Generator(
        config["MODEL"]["SIZE"],
        config["MODEL"]["LATENT"],
        config["MODEL"]["N_MLP"],
        channel_multiplier=config["MODEL"]["CHANNEL_MULTIPLIER"]
    ).to(device)
    checkpoint = torch.load(config["MODEL"]["PRETRAINED"])

    G.load_state_dict(checkpoint["g_ema"])

    return G