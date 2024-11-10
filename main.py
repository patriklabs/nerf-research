import argparse

import torch

from train.trainer import Trainer
from util.config import load_config
from visualize import main as visualize_main

torch.set_float32_matmul_precision("high")


def add_arguments():
    parser = argparse.ArgumentParser(description="nerf trainer")

    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--visualize",
        action="store_true",
    )

    return parser.parse_args()


def main(config, ckpt_path):

    Trainer(
        **config,
        ckpt_path=ckpt_path,
    ).run()


if __name__ == "__main__":

    args = add_arguments()

    if args.visualize:
        visualize_main(load_config(args.config), args.ckpt)
    else:
        main(load_config(args.config), args.ckpt)
