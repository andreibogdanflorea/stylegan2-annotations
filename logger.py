import os
import logging

def create_logger(save_location: str) -> list([logging.Logger, str]):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = os.path.join(save_location, 'log_dir')
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    return logger, tensorboard_log_dir