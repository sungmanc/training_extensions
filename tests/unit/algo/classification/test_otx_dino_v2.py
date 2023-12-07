# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from torch import nn
from omegaconf import DictConfig
from otx.algo.classification import DINOv2RegisterClassifier
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity

class TestDINOv2RegisterClassifier:
    def test__init(self, fxt_config_mock: DictConfig):
        dino_classifier = DINOv2RegisterClassifier(fxt_config_mock)
        assert isinstance(dino_classifier.get_submodule("backbone"), nn.Module)
        assert isinstance(dino_classifier.get_submodule("head"), nn.Module)
    
    def test__freeze_backbone(self, fxt_config_mock: DictConfig):
        mock_config = fxt_config_mock
        mock_config.backbone.frozen = True
        
        dino_classifier_frozen = DINOv2RegisterClassifier(mock_config)
        for _, v in dino_classifier_frozen.backbone.named_parameters():
            assert v.requires_grad == False
    
    @pytest.mark.parametrize("training", [True, False]) 
    def test_forward(
        self,
        fxt_config_mock: DictConfig, 
        fxt_multiclass_cls_batch_data_entity : MulticlassClsBatchDataEntity,
        training: bool,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        dino_classifier = DINOv2RegisterClassifier(fxt_config_mock)
        dino_classifier.training = training
        
        outputs = dino_classifier(fxt_multiclass_cls_batch_data_entity)
        if training:
            assert isinstance(outputs, OTXBatchLossEntity)
        else:
            assert isinstance(outputs, MulticlassClsBatchPredEntity)