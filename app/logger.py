import logging
import os

from configs import load_config


def setup_logger(name: str) -> logging.Logger:
    config = load_config()["app"]
    logger = logging.getLogger(name)
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/" + config["log_file"]

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
