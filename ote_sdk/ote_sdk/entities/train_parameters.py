"""This module implements the TrainingParameters entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from typing import Callable, Optional

from ote_sdk.entities.callbacks import (
    HpoCallback,
    UpdateProgressCallback,
    default_progress_callback,
)


def default_save_model_callback():
    """
    Default save model callback. It is a placeholder (does nothing) and is used in empty TrainParameters.
    """


@dataclass
class TrainParameters:
    """
    Train parameters.

    :var resume: Set to ``True`` if training must be resume with the optimizer state;
        set to ``False`` to discard the optimizer state and start with fresh optimizer
    :var update_progress: Callback which can be used to provide updates about the progress of a task.
    :var save_model: Callback to notify that the model weights have been changed.
        This callback can be used by the task when temporary weights should be saved (for instance, at the
        end of an epoch). If this callback has been used to save temporary weights, those weights will be
        used to resume training if for some reason training was suspended.
    """

    resume: bool = False
    update_progress: UpdateProgressCallback = default_progress_callback
    hpo: Optional[HpoCallback] = None
    save_model: Callable[[], None] = default_save_model_callback
