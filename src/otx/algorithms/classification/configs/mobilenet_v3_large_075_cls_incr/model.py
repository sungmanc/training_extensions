"""MobileNet-V3-large-075 for multi-class config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/mobilenet_v3.py"]

model = dict(
    type="CustomImageClassifier",
    task="classification",
    backbone=dict(
        mode="large",
        width_mult=0.75,
    ),
    head=dict(
        in_channels=720,
        hid_channels=1280,
    ),
)