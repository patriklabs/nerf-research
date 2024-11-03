import argparse

import torch

from train.trainer import Trainer
from util.config import load_config

torch.set_float32_matmul_precision("high")


def add_arguments():
    parser = argparse.ArgumentParser(description="nerf trainer")

    parser.add_argument("--config", type=str, default="config/nerf_config.yaml")

    return parser.parse_args()


def main(config):

    Trainer(
        trainer=config["trainer"],
        model=config["model"],
        datamodule=config["datamodule"],
    ).run()


if __name__ == "__main__":

    main(load_config(add_arguments().config))
