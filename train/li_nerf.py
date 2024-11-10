from typing import Any
import lightning as pl
import torch
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from nerf.util.util import to_rays, to_image
import matplotlib

matplotlib.use("Agg")


def make_grid(image, output_nbr):
    """
    makes a grid for tensorboard output

    Args:
        image (torch.Tensor): Image.
        output_nbr (int): Output number.

    Returns:
        torch.Tensor: Image grid
    """

    return torchvision.utils.make_grid(image[0:output_nbr], padding=10, pad_value=1.0)


def img2mse(x, y):
    """
    calculates the mean squared error between two images

    Args:
        x (torch.Tensor): Image.
        y (torch.Tensor): Image.

    Returns:
        torch.Tensor: Mean squared error
    """

    return torch.square(x - y).mean()


def mse2psnr(x):
    """
    calculates the peak signal to noise ratio from the mean squared error

    Args:
        x (torch.Tensor): Mean squared error.

    Returns:
        torch.Tensor: Peak signal to noise ratio
    """

    return -10.0 * torch.log(x) / torch.log(10.0 * torch.ones_like(x))


class LiNerf(pl.LightningModule):
    def __init__(self, model, lr=1e-4, reg_weight=1e-4, **kwargs):
        super().__init__()

        self.lr = lr
        self.reg_weight = reg_weight
        self.nerf = model

        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        for param in self.lpips.parameters():
            param.requires_grad = False

        self.psnr = PeakSignalNoiseRatio(1)
        self.ssim = StructuralSimilarityIndexMeasure()

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            image = batch["image"]
            T = batch["T"]
            intrinsics = batch["intrinsics"]
            tn = batch["tn"]
            tf = batch["tf"]

            B, _, H, W = image.shape

            rays, tn, tf = to_rays(T, intrinsics, tn, tf, B, H, W)

            color, depth = self.render_frame(rays, tn, tf)

            color = to_image(color, B, H, W)
            depth = to_image(depth, B, H, W)

            self.logger.experiment.add_image(
                f"val/img_rendered", make_grid(color, 4), self.global_step + batch_idx
            )

            depth_max = torch.max(depth.view(B, -1, H * W), dim=-1, keepdim=True)[
                0
            ].view(-1, 1, 1, 1)
            depth_min = torch.min(depth.view(B, -1, H * W), dim=-1, keepdim=True)[
                0
            ].view(-1, 1, 1, 1)

            depth = (depth - depth_min) / (depth_max - depth_min)

            self.logger.experiment.add_image(
                f"val/depth_rendered", make_grid(depth, 4), self.global_step + batch_idx
            )

            self.logger.experiment.add_image(
                f"val/img_gt", make_grid(image, 4), self.global_step + batch_idx
            )

            self.log("val/lpips", self.lpips(color.clamp(0, 1), image))
            self.log("val/psnr", self.psnr(color, image))
            self.log("val/ssim", self.ssim(color, image))

    def forward(self, ray, tn, tf):

        return self.nerf(ray, tn, tf, self.global_step)

    def render_frame(self, rays, tn, tf, max_chunk=4096):

        with torch.no_grad():

            color_list = []
            depth_list = []

            rays = torch.split(rays, max_chunk)
            tn = torch.split(tn, max_chunk)
            tf = torch.split(tf, max_chunk)

            for idx, (ray, tn, tf) in enumerate(zip(rays, tn, tf)):

                result = self.forward(ray, tn, tf)

                color_list.append(result["color_high_res"])
                depth_list.append(result["depth"])

                print(f"rendering ray batch {idx} of {len(rays)}", end="\r")

        return torch.cat(color_list, dim=0), torch.cat(depth_list, dim=0)

    def training_step(self, batch, batch_idx):

        color_gt = batch["rgb"]

        result = self.forward(batch["ray"], batch["tn"], batch["tf"])

        loss = (color_gt - result["color_high_res"]).abs().mean()

        if "color_low_res" in result:
            loss += (color_gt - result["color_low_res"]).abs().mean()

            loss /= 2.0

        if "reg_val" in result:
            loss += self.reg_weight * result["reg_val"].mean()

        if "eikonal_loss" in result:
            loss += 1e-2 * result["eikonal_loss"].mean()

        if "plot" in result:

            self.logger.experiment.add_image(
                f"train/plot", make_grid(result["plot"], 4), self.global_step
            )

        if "s" in result:
            self.log("train/s", result["s"])

        self.log("train/loss", loss)

        self.log("train/psnr", mse2psnr(img2mse(color_gt, result["color_high_res"])))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
