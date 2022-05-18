"""Implements callback for updating progress"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional, Protocol


# pylint: disable=unused-argument
def default_progress_callback(progress: float, score: Optional[float] = None):
    """
    Default progress callback. It is a placeholder (does nothing) and is used in empty TrainParameters.
    """


class UpdateProgressCallback(Protocol):
    """
    UpdateProgressCallback protocol.
    Used as a replacement of Callable[] type since Callable doesn’t handle default parameters like
    `score: Optional[float] = None`
    """

    def __call__(self, progress: float, score: Optional[float] = None):
        """
        Callback to provide updates about the progress of a task.
        It is recommended to call this function at least once per epoch.
        However, the exact frequency is left to the task implementer.

        An optional `score` can also be passed. If specified, this score can be used by HPO
        to monitor the improvement of the task.

        :param progress: Progress as a percentage
        :param score: Optional validation score
        """
