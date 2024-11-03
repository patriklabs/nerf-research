import lightning as pl


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

        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
        )
