"""MobileNet-V3-large-1 for multi-class config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/mobilenet_v3.py"]

model = dict(
    type="CustomImageClassifier",
    task="classification",
    backbone=dict(mode="large"),
    head=dict(
        type="CustomNonLinearClsHead",
        in_channels=960,
        hid_channels=1280,
        loss=dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
        ),
    ),
)

fp16 = dict(loss_scale=512.0)

custom_hooks = [dict(type="AdaptiveRepeatDataHook", priority="ABOVE_NORMAL")]
