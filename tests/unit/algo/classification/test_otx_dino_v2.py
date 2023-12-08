# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn
from unittest.mock import patch, MagicMock
from otx.algo.classification import DINOv2


class TestDINOv2:
    def setup(self) -> None:
        mock_backbone = MagicMock()
        mock_backbone.return_value = torch.randn(1, 12)
        
        with patch('torch.hub.load', autospec=True) as mock_load:
            mock_load.return_value = mock_backbone
            
            self.model = DINOv2(
                backbone_name="dinov2_vits14_reg",
                freeze_backbone=True,
                head_in_channels=12,
                num_classes=2,
                training=True,
            )
        

    def test_freeze_backbone(self) -> None:
        for _, v in self.model.backbone.named_parameters():
            assert v.requires_grad is False

    def test_forward(self) -> None:
        rand_img = torch.randn((1, 3, 224, 224), dtype=torch.float32)
        rand_label = torch.ones((1), dtype=torch.int64)
        outputs = self.model(rand_img, rand_label)
        assert isinstance(outputs, torch.Tensor)