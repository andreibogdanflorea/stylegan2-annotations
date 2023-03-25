import torch
from models.stylegan import Generator

def load_generator_model(config, device='cuda'):
    G = Generator(
        config["SIZE"], config["LATENT"], config["N_MLP"], channel_multiplier=config["CHANNEL_MULTIPLIER"]
    ).to(device)
    checkpoint = torch.load(config["CHECKPOINT_PATH"])

    G.load_state_dict(checkpoint["g_ema"])

    return G