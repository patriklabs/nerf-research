import argparse

import torch

from train.trainer import Trainer
from util.config import load_config

torch.set_float32_matmul_precision("high")


def add_arguments():
    parser = argparse.ArgumentParser(description="nerf trainer")

    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str, default=None)

    return parser.parse_args()


def main(config, ckpt_path):

    Trainer(
        **config,
        ckpt_path=ckpt_path,
    ).run()


if __name__ == "__main__":

    args = add_arguments()
    main(load_config(args.config), args.ckpt)
