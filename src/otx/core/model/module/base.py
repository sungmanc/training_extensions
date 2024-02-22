# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base lightning module used in OTX."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningModule
from torch import Tensor

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchLossEntity,
    OTXBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel
from otx.core.utils.utils import is_ckpt_for_finetuning, is_ckpt_from_otx_v1

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics import Metric

    from otx.core.data.dataset.base import LabelInfo
    from otx.core.metrics import MetricCallable


class OTXLitModule(LightningModule):
    """Base class for the lightning module used in OTX."""

    def __init__(
        self,
        *,
        otx_model: OTXModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda n: Metric(n),
    ):
        super().__init__()

        self.model = otx_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.torch_compile = torch_compile
        self.metric = metric

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["otx_model", "metric"])

    def training_step(self, inputs: OTXBatchDataEntity, batch_idx: int) -> Tensor:
        """Step for model training."""
        train_loss = self.model(inputs)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return train_loss
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            total_train_loss = sum(train_loss.values())
            self.log(
                "train/loss",
                total_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return total_train_loss

        raise TypeError(train_loss)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        if isinstance(self.metric, Metric):
            self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        if isinstance(self.metric, Metric):
            self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.metric, "test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.torch_compile and stage == "fit":
            self.model = torch.compile(self.model)

        self.model.setup_callback(self.trainer)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        optimizers = [
            optimizer(params=self.parameters()) if callable(optimizer) else optimizer
            for optimizer in ensure_list(self.hparams.optimizer)
        ]

        lr_schedulers = []
        for scheduler_config in ensure_list(self.hparams.scheduler):
            scheduler = scheduler_config(optimizers[0]) if callable(scheduler_config) else scheduler_config
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_schedulers.append(lr_scheduler_config)

        return optimizers, lr_schedulers

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register self.model's load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model.register_load_state_dict_pre_hook(model_classes, ckpt_classes)

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary of model entity with meta information.

        Returns:
            A dictionary containing datamodule state.

        """
        state_dict = super().state_dict()
        state_dict["meta_info"] = self.meta_info
        return state_dict

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        It successfully loads the checkpoint from OTX v1.x and for finetune and for resume.

        If checkpoint's meta_info and OTXLitModule's meta_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        if is_ckpt_from_otx_v1(ckpt):
            msg = "The checkpoint comes from OTXv1, checkpoint keys will be updated automatically."
            warnings.warn(msg, stacklevel=2)
            state_dict = self.model.load_from_otx_v1_ckpt(ckpt)
        elif is_ckpt_for_finetuning(ckpt):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        ckpt_meta_info = state_dict.pop("meta_info", None)

        if ckpt_meta_info and self.meta_info is None:
            msg = (
                "`state_dict` to load has `meta_info`, but the current model has no `meta_info`. "
                "It is recommended to set proper `meta_info` for the incremental learning case."
            )
            warnings.warn(msg, stacklevel=2)
        if ckpt_meta_info and self.meta_info and ckpt_meta_info != self.meta_info:
            logger = logging.getLogger()
            logger.info(
                f"Data classes from checkpoint: {ckpt_meta_info.label_names} -> "
                f"Data classes from training data: {self.meta_info.label_names}",
            )
            self.register_load_state_dict_pre_hook(
                self.meta_info.label_names,
                ckpt_meta_info.label_names,
            )
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def label_info(self) -> LabelInfo:
        """Get the member `OTXModel` label information."""
        return self.model.label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo | list[str]) -> None:
        """Set the member `OTXModel` label information."""
        self.model.label_info = label_info  # type: ignore[assignment]

    def forward(self, *args, **kwargs) -> OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward pass."""
        if self.model.explain_mode:
            return self.model.forward_explain(*args, **kwargs)
        return self.model.forward(*args, **kwargs)
