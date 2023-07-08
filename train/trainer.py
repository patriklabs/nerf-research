
from distutils.util import strtobool

import mlflow
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from dataset.colmap_solution import ColmapSolution
from dataset.image_loader import ImageLoader
from dataset.ray_loader import RayLoader


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


class Trainer:
    def __init__(self, trainer, li_model, model_path, dataset_path, split, batch, num_workers, width, height, **kwargs) -> None:

        self.model = li_model

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = None

        self.dataset_path = dataset_path
        self.batch = batch
        self.num_workers = num_workers

        self.trainer = trainer
        self.height = height
        self.width = width
        self.split = split

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("Trainer")

        parser.add_argument("--model_path", type=str, default="")
        parser.add_argument("--dataset_path", type=str, default="")
        parser.add_argument("--batch", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--width", type=int, default=320)
        parser.add_argument("--height", type=int, default=320)

        parser.add_argument("--split",
                            type=strtobool,
                            default=False)

        return parent_parser

    def load_datasets(self):

        colmap_solution = ColmapSolution(self.dataset_path, 0)

        colmap_solution.unit_rescale()

        colmap_solution.rescale_image([self.height, self.width])

        if self.split:

            train_sol, test_sol, splits = colmap_solution.split(ratio=0.7)

            train_dataset = RayLoader(train_sol)

            test_dataset = ImageLoader(test_sol)

        else:
            train_dataset = RayLoader(colmap_solution)

            test_dataset = ImageLoader(colmap_solution)

            splits = {}

        return train_dataset, test_dataset, splits

    def run(self):

        train_dataset, test_dataset, splits = self.load_datasets()

        train_loader = DataLoader(train_dataset,
                                  self.batch,
                                  True,
                                  num_workers=self.num_workers)

        test_loader = DataLoader(test_dataset,
                                 1,
                                 False,
                                 num_workers=self.num_workers)

        mlflow.pytorch.autolog()

        with mlflow.start_run() as run:

            mlflow.log_dict(splits, "splits")

            self.trainer.fit(model=self.model,
                             train_dataloaders=train_loader,
                             val_dataloaders=test_loader,
                             ckpt_path=self.model_path)

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
