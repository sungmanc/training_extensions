"""Implements callback for HPO"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

try:
    import hpopt
except ImportError:
    hpopt = None


class HpoCallback:
    """Callback class to report score to hpopt"""

    def __init__(self, hp_config, metric, hpo_task):
        super().__init__()
        self.hp_config = hp_config
        self.metric = metric
        self.hpo_task = hpo_task

    def __call__(self, score: Optional[float] = None):
        if score is not None:
            if hpopt.report(config=self.hp_config, score=score) == hpopt.Status.STOP:
                self.hpo_task.cancel_training()
