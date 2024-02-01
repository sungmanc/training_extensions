# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Callback to reschedule the validation interval adaptively."""

from __future__ import annotations

import logging as log
import math
from typing import TYPE_CHECKING

from lightning import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from lightning.pytorch.utilities.types import LRSchedulerConfig


class AdaptiveTrainScheduling(Callback):
    """Adaptive Training Scheduling Hook.

    Depending on the size of iteration per epoch, adaptively update the validation interval and related values.

    Args:
        max_interval: Maximum value of validation interval.
            Defaults to 5.
        decay: Parameter to control the interval. This value is set by manual manner.
            Defaults to -0.025.
    """

    def __init__(self, max_interval: int = 5, decay: float = -0.025):
        self.max_interval = max_interval
        self.decay = decay
        self._saved_check_val_every_n_epoch: int | None = None
        self._saved_log_every_n_steps: int | None = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute this function at starting the train stage."""
        max_interval = min(self.max_interval, trainer.max_epochs)
        iter_per_epoch = len(trainer.train_dataloader)

        adaptive_check_val_every_n_epoch = self._get_adaptive_interval(
            iter_per_epoch=iter_per_epoch,
            max_interval=max_interval,
        )
        self.is_enabled = adaptive_check_val_every_n_epoch > 1

        if adaptive_check_val_every_n_epoch != trainer.check_val_every_n_epoch:
            msg = (
                "You are using AdaptiveTrainScheduling hook. "
                "This hook will temporarily update Trainer.check_val_every_n_epoch adaptively: "
                f"{trainer.check_val_every_n_epoch} => {adaptive_check_val_every_n_epoch}"
            )
            log.warning(msg)

            self._saved_check_val_every_n_epoch = trainer.check_val_every_n_epoch
            trainer.check_val_every_n_epoch = adaptive_check_val_every_n_epoch
            self._change_early_stopping_patience(trainer.callbacks, adaptive_check_val_every_n_epoch)
            self._change_lr_scheduler_frequency(trainer.lr_scheduler_configs, adaptive_check_val_every_n_epoch)

        if iter_per_epoch < trainer.log_every_n_steps:
            msg = (
                "Trainer.log_every_n_steps is higher than the number of iterations in a training epoch. "
                "To ensure logging at the last batch, temporarily update Trainer.log_every_n_steps: "
                f"{trainer.log_every_n_steps} => {iter_per_epoch}"
            )
            log.warning(msg)

            self._saved_log_every_n_steps = trainer.log_every_n_steps
            trainer.log_every_n_steps = iter_per_epoch

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Execute this function at terminating the train stage."""
        if self._saved_check_val_every_n_epoch:
            trainer.check_val_every_n_epoch = self._saved_check_val_every_n_epoch
            self._saved_check_val_every_n_epoch = None

        if self._saved_log_every_n_steps:
            trainer.log_every_n_steps = self._saved_log_every_n_steps
            self._saved_log_every_n_steps = None

    def _get_adaptive_interval(self, iter_per_epoch: int, max_interval: int) -> int:
        """Get adaptive interval."""
        return max(round(math.exp(self.decay * iter_per_epoch) * max_interval), 1)

    def _change_lr_scheduler_frequency(self, lr_configs: list[LRSchedulerConfig], adaptive_interval: int) -> None:
        """Change the frequency of LRscheduler.

        Since adaptive interval changes the validation interval, the frequency of LRscheduler also
        should be changed according to the adaptive interval.
        """
        for config in lr_configs:
            if hasattr(config, "frequency"):
                msg = (
                    "The frequency of LRscheduler will be changed due to the effect of adaptive interval: "
                    f"{config.frequency} --> {adaptive_interval}."
                )
                log.warning(msg)
                config.frequency = adaptive_interval

    def _change_early_stopping_patience(self, callbacks: list[Callback], adaptive_interval: int) -> None:
        """Change the EarlyStopping patience to change the patience.

        Since adaptive interval changes the validation interval, the patience of early stopping also
        should be changed according to the adaptive interval.
        """
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping

        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                adjusted_patience = int(callback.patience / adaptive_interval)
                msg = (
                    "The patience of early stopping will be changed due to the effect of adaptive interval: "
                    f"{callback.patience} --> {adjusted_patience}."
                )
                log.warning(msg)
                callback.patience = adjusted_patience
