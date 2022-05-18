"""HPO reporting callback"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Optional

import pytorch_lightning as pl
from ote_sdk.entities.callbacks import HpoCallback
from pytorch_lightning import Callback


class AnomalyHpoCallback(Callback):
    """Callback for reporting score to the ote hpo callback handle."""

    def __init__(self, hpo_callback: Optional[HpoCallback] = None) -> None:
        if hpo_callback is not None:
            self.hpo_callback = hpo_callback
        else:
            self.hpo_callback = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        If score exists in trainer.logged_metrics, report the score.
        """
        if self.hpo_callback is not None:
            score = None
            metric = getattr(self.hpo_callback, "metric", None)
            if metric in trainer.logged_metrics:
                score = float(trainer.logged_metrics[metric])
            self.hpo_callback(score=score)
