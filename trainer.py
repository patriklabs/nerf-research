import argparse

import torch

from train.trainer import Trainer
from util.config import load_config

torch.set_float32_matmul_precision("high")


def add_arguments():
    parser = argparse.ArgumentParser(description="nerf trainer")

    parser.add_argument("--config", type=str)

    return parser.parse_args()


def main(config):

    Trainer(
        **config,
    ).run()


if __name__ == "__main__":

    main(load_config(add_arguments().config))
