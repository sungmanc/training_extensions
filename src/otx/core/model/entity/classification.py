# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model, get_classification_layers

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from torch import device, nn


class OTXMulticlassClsModel(
    OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
    """Base class for the classification models used in OTX."""


def _create_mmpretrain_model(config: DictConfig, load_from: str) -> tuple[nn.Module, list[str]]:
    from mmpretrain.models.utils import ClsDataPreprocessor as _ClsDataPreprocessor
    from mmpretrain.registry import MODELS

    # NOTE: For the history of this monkey patching, please see
    # https://github.com/openvinotoolkit/training_extensions/issues/2743
    @MODELS.register_module(force=True)
    class ClsDataPreprocessor(_ClsDataPreprocessor):
        @property
        def device(self) -> device:
            try:
                buf = next(self.buffers())
            except StopIteration:
                return super().device
            else:
                return buf.device

    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers


class MMPretrainMulticlassClsModel(OTXMulticlassClsModel):
    """Multi-class Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(
    OTXModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity],
):
    """Multi-label classification models used in OTX."""


class MMPretrainMultilabelClsModel(OTXMultilabelClsModel):
    """Multi-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def _customize_inputs(self, entity: MultilabelClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_score=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )


class OTXHlabelClsModel(OTXModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity]):
    """H-label classification models used in OTX."""


class MMPretrainHlabelClsModel(OTXHlabelClsModel):
    """H-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def _customize_inputs(self, entity: HlabelClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )


class OVClassificationCompatibleModel(OTXMulticlassClsModel):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(self, config: DictConfig) -> None:
        self.model_name = config.pop("model_name")
        self.config = config
        super().__init__()

    def _create_model(self) -> nn.Module:
        from openvino.model_api.models import ClassificationModel

        return ClassificationModel.create_model(self.model_name, model_type="Classification")

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if entity.batch_size > 1:
            msg = "Only sync inference with batch = 1 is supported for now"
            raise RuntimeError(msg)
        # restore original numpy image
        img = np.transpose(entity.images[-1].numpy(), (1, 2, 0))
        return {"inputs": img}

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity:
        # add label index
        labels = [torch.tensor(outputs.top_labels[0][0], dtype=torch.long)]
        # add probability
        scores = [torch.tensor(outputs.top_labels[0][2])]

        return MulticlassClsBatchPredEntity(
            batch_size=1,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )