import lightning as pl


class Trainer:
    def __init__(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        ckpt_path=None,
        **kwargs
    ) -> None:

        self.model = model
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path

    def run(self):

        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )
