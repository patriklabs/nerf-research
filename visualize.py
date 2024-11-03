import argparse

import pytorch_lightning as pl
import torch
from torch import nn

from nerf_visualizer import NerfVisualizer
from util.config import load_config

torch.set_float32_matmul_precision("high")


def add_arguments():
    parser = argparse.ArgumentParser(description="nerf trainer")

    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)

    return parser.parse_args()


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_renderer(model: pl.LightningModule, ckpt) -> nn.Module:
    return model.__class__.load_from_checkpoint(ckpt, model=model.nerf).nerf.render


def main(config, ckpt):

    NerfVisualizer(
        get_device(), load_renderer(config["model"], ckpt), **config["visualizer"]
    ).run()


if __name__ == "__main__":

    args = add_arguments()
    main(load_config(args.config), args.ckpt)
