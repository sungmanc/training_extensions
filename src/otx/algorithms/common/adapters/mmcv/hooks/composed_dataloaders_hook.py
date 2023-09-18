"""Composed dataloader hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from typing import List, Sequence, Union

from mmcv.runner import HOOKS, Hook
from mmcv.utils import Config
from torch.utils.data import DataLoader

from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.adapters.torch.dataloaders import ComposedDL
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class ComposedDataLoadersHook(Hook):
    """Composed dataloader hook, which makes a composed dataloader which can combine multiple data loaders.

    Especially used for semi-supervised learning to aggregate a unlabeled dataloader and a labeled dataloader.
    """

    def __init__(
        self,
        cfg: Config,
    ):
        self.data_loaders: List[DataLoader] = []
        self.composed_loader = None

        model_task = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}
        if "unlabeled" in cfg.data:
            task_lib_module = importlib.import_module(f"{model_task[cfg.model_task]}.datasets")
            dataset_builder = getattr(task_lib_module, "build_dataset")
            dataloader_builder = getattr(task_lib_module, "build_dataloader")

            dataset = build_dataset(cfg, "unlabeled", dataset_builder, consume=True)
            unlabeled_dataloader = build_dataloader(
                dataset,
                cfg,
                "unlabeled",
                dataloader_builder,
                distributed=cfg.distributed,
                consume=True,
            )

            self.add_dataloaders(unlabeled_dataloader)

    def add_dataloaders(self, data_loaders: Union[Sequence[DataLoader], DataLoader]):
        """Create data_loaders to be added into composed dataloader."""
        if isinstance(data_loaders, DataLoader):
            data_loaders = [data_loaders]
        else:
            data_loaders = list(data_loaders)

        self.data_loaders.extend(data_loaders)

    def before_epoch(self, runner):
        """Create composedDL before running epoch."""
        if self.composed_loader is None:
            logger.info("ComposedDL is created, now two DataLoader are enabled." )
            self.composed_loader = ComposedDL([runner.data_loader, *self.data_loaders])
        else:
            self.composed_loader = runner.data_loader
        # Per-epoch replacement: train-only loader -> train loader + additional loaders
        # (It's similar to local variable in epoch. Need to update every epoch...)
        runner.data_loader = self.composed_loader
