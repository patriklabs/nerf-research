import argparse
from distutils.util import strtobool

import pytorch_lightning as pl
import torch

from nerf.nerf.nerf_hypothesis import Nerf
from nerf_visualizer import NerfVisualizer
from train.LiNerf import LiNerf
from train.trainer import Trainer

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--visualize", type=strtobool, default=False)

    pl.Trainer.add_argparse_args(parser)

    Trainer.add_model_specific_args(parser)

    LiNerf.add_model_specific_args(parser)

    Nerf.add_model_specific_args(parser)

    NerfVisualizer.add_model_specific_args(parser)

    args = parser.parse_args()

    pl_trainer = pl.Trainer.from_argparse_args(args)

    param = vars(args)

    if not args.visualize:

        li_model = LiNerf(Nerf(**param), **param)

        trainer = Trainer(trainer=pl_trainer, li_model=li_model, **param)

        trainer.run()

    else:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        li_model = LiNerf.load_from_checkpoint(args.model_path,
                                               model=Nerf(**param),
                                               **param).to(device)

        visualize = NerfVisualizer(device,
                                   li_model.nerf.render,
                                   **param)

        visualize.run()
