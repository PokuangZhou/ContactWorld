from __future__ import annotations

import lightning as pl
import torch

from model import SpatialActionConditionedModel


class PredictorTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: SpatialActionConditionedModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        rollout_ctxt_window: int = 4,
        rollout_steps: int | None = None,
        log_rollout_metrics: bool = True,
        val_proprio_rollout_mode: str = "use_ground_truth",
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.rollout_ctxt_window = rollout_ctxt_window
        self.rollout_steps = rollout_steps
        self.log_rollout_metrics = log_rollout_metrics
        self.val_proprio_rollout_mode = val_proprio_rollout_mode

    def training_step(self, batch, batch_idx):
        losses = self.model.compute_losses(batch)
        self._log_dict(losses, prefix="train", on_step=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.model.compute_losses(batch)
        self._log_dict(losses, prefix="val", on_step=False, on_epoch=True)

        if self.log_rollout_metrics:
            metrics = self.model.compute_rollout_metrics(
                batch=batch,
                ctxt_window=self.rollout_ctxt_window,
                rollout_steps=self.rollout_steps,
                proprio_mode=self.val_proprio_rollout_mode,
            )
            self._log_dict(metrics, prefix="val", on_step=False, on_epoch=True)

    def _log_dict(self, values: dict[str, torch.Tensor], prefix: str, on_step: bool, on_epoch: bool) -> None:
        for key, value in values.items():
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=key.endswith("loss"),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
