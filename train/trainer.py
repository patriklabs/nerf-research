import lightning as pl
import mlflow
from mlflow import MlflowClient


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


class Trainer:
    def __init__(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        **kwargs
    ) -> None:

        self.model = model
        self.trainer = trainer
        self.datamodule = datamodule

    def run(self):

        mlflow.pytorch.autolog()

        with mlflow.start_run() as run:

            self.trainer.fit(
                model=self.model,
                datamodule=self.datamodule,
            )

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
