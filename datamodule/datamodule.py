import lightning as pl
from torch.utils.data import DataLoader

from dataset.colmap_solution import ColmapSolution
from dataset.image_loader import ImageLoader
from dataset.ray_loader import RayLoader


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_path,
        split=False,
        batch=64,
        num_workers=16,
        width=320,
        height=320,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.batch_size = batch
        self.num_workers = num_workers
        self.width = width
        self.height = height
        self.split = split

    def setup(self, stage: str) -> None:

        colmap_solution = ColmapSolution(self.dataset_path, 0)

        colmap_solution.unit_rescale()

        colmap_solution.rescale_image([self.height, self.width])

        if self.split:

            train_sol, test_sol, splits = colmap_solution.split(ratio=0.7)

            train_dataset = RayLoader(train_sol)

            val_dataset = ImageLoader(test_sol)

        else:
            train_dataset = RayLoader(colmap_solution)

            val_dataset = ImageLoader(colmap_solution)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
        )
