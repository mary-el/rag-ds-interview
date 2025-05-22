import logging
import os
import time
from pathlib import Path

from configs import load_config

logger = None


def setup_logger(name: str) -> logging.Logger:
    config = load_config()["app"]
    global logger
    if logger:
        return logger
    logger = logging.getLogger(name)
    log_folder = Path(config["log_folder"])
    os.makedirs(log_folder, exist_ok=True)
    filename = time.strftime("%Y-%m-%d-%H-%M-%S") + ".log"
    log_file = str(log_folder / filename)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.encoding = "utf-8"
        logger.addHandler(file_handler)

        if config["stream_log"]:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    return logger
