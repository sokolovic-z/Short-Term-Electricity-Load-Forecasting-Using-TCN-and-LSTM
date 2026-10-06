import pytorch_lightning as pl


class LossHistory(pl.Callback):

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")

        if loss is not None:
            self.train_loss.append(
                loss.detach().cpu().item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")

        if loss is not None:
            self.val_loss.append(
                loss.detach().cpu().item()
            )