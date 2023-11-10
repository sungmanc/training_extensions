"""Module for defining semi-supervised classification head for multi-label classification task."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Iterable, List, Optional

import torch
from mmpretrain.models.builder import HEADS, build_loss
from mmpretrain.models.heads import ClsHead
from torch import nn

from otx.v2.adapters.torch.mmengine.mmpretrain.modules.models.heads.custom_multi_label_linear_cls_head import (
    CustomMultiLabelLinearClsHead,
)
from otx.v2.adapters.torch.mmengine.mmpretrain.modules.models.heads.custom_multi_label_non_linear_cls_head import (
    CustomMultiLabelNonLinearClsHead,
)

from .mixin import OTXHeadMixin


def generate_aux_mlp(aux_mlp_cfg: dict, in_channels: int) -> nn.Module:
    """Generate auxiliary MLP."""
    out_channels = aux_mlp_cfg["out_channels"]
    if out_channels <= 0:
        raise ValueError(f"out_channels={out_channels} must be a positive integer")
    if "hid_channels" in aux_mlp_cfg and aux_mlp_cfg["hid_channels"] > 0:
        hid_channels = aux_mlp_cfg["hid_channels"]
        mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hid_channels, out_features=out_channels),
        )
    else:
        mlp = nn.Linear(in_features=in_channels, out_features=out_channels)

    return mlp


class EMAMeter:
    """Exponential Moving Average Meter class."""

    def __init__(self, alpha: float = 0.9) -> None:
        """Initialize the Exponential Moving Average Meter.

        Args:
            alpha (float): Smoothing factor for the exponential moving average. Defaults to 0.9.
        """
        self.alpha = alpha
        self.val = 0.0

    def reset(self) -> None:
        """Reset the Exponential Moving Average Meter."""
        self.val = 0.0

    def update(self, val: float) -> None:
        """Update the Exponential Moving Average Meter with new value.

        Args:
            val (float): New value to update the meter.
        """
        self.val = self.alpha * self.val + (1 - self.alpha) * val


class LossBalancer:
    """Loss Balancer class."""

    def __init__(self, num_losses: int, weights: Optional[List[float]] = None, ema_weight: float = 0.7) -> None:
        """Initialize the Loss Balancer.

        Args:
            num_losses (int): Number of losses to balance.
            weights (list): List of weights to be applied to each loss. If None, equal weights are applied.
            ema_weight (float): Smoothing factor for the exponential moving average meter. Defaults to 0.7.
        """
        self.epsilon = 1e-9
        self.avg_estimators = [EMAMeter(ema_weight) for _ in range(num_losses)]

        if weights is not None:
            self.final_weights = weights
        else:
            self.final_weights = [1.0] * num_losses

    def balance_losses(self, losses: Iterable) -> float:
        """Balance the given losses using the weights and exponential moving average.

        Args:
            losses (list): List of losses to be balanced.

        Returns:
            total_loss (float): Balanced loss value.
        """
        total_loss = 0.0
        for i, loss in enumerate(losses):
            self.avg_estimators[i].update(float(loss))
            total_loss += (
                self.final_weights[i] * loss / (self.avg_estimators[i].val + self.epsilon) * self.avg_estimators[0].val
            )
        return total_loss


class SemiMultilabelClsHead(OTXHeadMixin, ClsHead):
    """Multilabel Classification head for Semi-SL.

    Args:
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0.
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True.
        aux_loss (dict, optional): auxiliary loss function, default is None.
    """

    def __init__(
        self,
        unlabeled_coef: float = 0.1,
        use_dynamic_loss_weighting: bool = True,
        aux_loss: Optional[dict] = None,
    ) -> None:
        aux_loss = (
            aux_loss if aux_loss else {"type": "BarlowTwinsLoss", "off_diag_penality": 1.0 / 128.0, "loss_weight": 1.0}
        )
        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_loss_weighting = use_dynamic_loss_weighting
        self.aux_loss = build_loss(aux_loss)
        self.loss_balancer: Optional[LossBalancer] = None
        if self.use_dynamic_loss_weighting:
            self.loss_balancer = LossBalancer(2, [1.0, unlabeled_coef])
        self.num_pseudo_label = 0

    def loss(
        self,
        logits: torch.Tensor,
        gt_label: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        features: Optional[tuple] = None,
        **kwargs,
    ) -> dict:
        """Loss function in which unlabeled data is considered.

        Args:
            logits (Tensor): Labeled data logits
            gt_label (Tensor): target features for labeled data
            features (tuple): (weak data features, strong data features)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if features is None:
            raise ValueError("features should tuple.")
        num_samples = gt_label.shape[0]
        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        l_labeled = self.loss_module(
            logits,
            _gt_label,
            avg_factor=num_samples,
        )

        features_weak, features_strong = features
        aux_loss = self.aux_loss(features_weak, features_strong)

        losses = {"loss": 0.0}
        if self.use_dynamic_loss_weighting and self.loss_balancer is not None:
            losses["loss"] = self.loss_balancer.balance_losses((l_labeled, aux_loss))
        else:
            losses["loss"] = l_labeled + self.unlabeled_coef * aux_loss
        losses["unlabeled_loss"] = self.unlabeled_coef * aux_loss

        return losses

    def forward_train_with_last_layers(
        self,
        x: torch.Tensor,
        gt_label: torch.Tensor,
        final_cls_layer: nn.Module,
        final_emb_layer: nn.Module,
    ) -> dict:
        """Forwards multilabel semi-sl head and losses.

        Args:
            x (dict): dict(labeled_weak. labeled_strong, unlabeled_weak, unlabeled_strong) or NxC input features.
            gt_label (Tensor): NxC target features.
            final_cls_layer (nn.Linear or nn.Sequential): a final layer forwards feature from backbone.
            final_emb_layer (nn.Linear or nn.Sequential): a final layer forwards embeddings from backbone.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        for key in x:
            x[key] = self.pre_logits(x[key])
        logits = final_cls_layer(x["labeled_weak"])
        features_weak = torch.cat((final_emb_layer(x["labeled_weak"]), final_emb_layer(x["unlabeled_weak"])))
        features_strong = torch.cat((final_emb_layer(x["labeled_strong"]), final_emb_layer(x["unlabeled_strong"])))
        features = (features_weak, features_strong)
        losses = self.loss(logits, gt_label, features)
        return losses


@HEADS.register_module()
class SemiLinearMultilabelClsHead(SemiMultilabelClsHead, CustomMultiLabelLinearClsHead):
    """Linear multilabel classification head for Semi-SL.

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        scale (float): Scale for metric learning loss
        normalized (boolean): flag that enables metric learining in loss,
        aux_mlp (dict): Config for embeddings MLP
        loss (dict): configuration of loss, default is CrossEntropyLoss
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        scale: float = 1.0,
        normalized: bool = False,
        aux_mlp: Optional[dict] = None,
        loss: Optional[dict] = None,
        unlabeled_coef: float = 0.1,
        aux_loss: Optional[dict] = None,
        use_dynamic_loss_weighting: bool = True,
    ) -> None:
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")
        aux_mlp = aux_mlp if aux_mlp else {"hid_channels": 0, "out_channels": 1024}
        loss = loss if loss else {"type": "CrossEntropyLoss", "loss_weight": 1.0}
        aux_loss = (
            aux_loss if aux_loss else {"type": "BarlowTwinsLoss", "off_diag_penality": 1.0 / 128.0, "loss_weight": 1.0}
        )
        CustomMultiLabelLinearClsHead.__init__(self, num_classes, in_channels, normalized, scale, loss)
        SemiMultilabelClsHead.__init__(self, unlabeled_coef, use_dynamic_loss_weighting, aux_loss)

        self.aux_mlp = generate_aux_mlp(aux_mlp, in_channels)

    def loss(
        self,
        logits: torch.Tensor,
        gt_label: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        features: Optional[tuple] = None,
        **kwargs,
    ) -> dict:
        """Calculate loss for given logits/gt_label."""
        return SemiMultilabelClsHead.loss(self, logits, gt_label, features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward fuction of SemiLinearMultilabelClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> dict:
        """Forward_train fuction of SemiLinearMultilabelClsHead class."""
        return self.forward_train_with_last_layers(
            cls_score,
            gt_label,
            final_cls_layer=self.fc,
            final_emb_layer=self.aux_mlp,
        )


@HEADS.register_module()
class SemiNonLinearMultilabelClsHead(SemiMultilabelClsHead, CustomMultiLabelNonLinearClsHead):
    """Non-linear classification head for Semi-SL.

    Args:
        num_classes (int): The number of classes of dataset used for training
        in_channels (int): The channels of input data from classifier
        hid_channels (int): Number of channels of hidden layer.
        scale (float): Scale for metric learning loss
        normalized (boolean): flag that enables metric learining in loss,
        aux_mlp (dict): Config for embeddings MLP
        act_cfg (dict): Config of activation layer
        loss (dict): configuration of loss, default is CrossEntropyLoss
        topk (tuple): evaluation topk score, default is (1, )
        unlabeled_coef (float): unlabeled loss coefficient, default is 0.1
        use_dynamic_loss_weighting (boolean): whether to use dynamic unlabeled loss weighting, default is True
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hid_channels: int = 1280,
        scale: float = 1.0,
        normalized: bool = False,
        aux_mlp: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        loss: Optional[dict] = None,
        aux_loss: Optional[dict] = None,
        dropout: bool = False,
        unlabeled_coef: float = 0.1,
        use_dynamic_loss_weighting: bool = True,
    ) -> None:
        if in_channels <= 0:
            raise ValueError(f"in_channels={in_channels} must be a positive integer")
        if num_classes <= 0:
            raise ValueError("at least one class must be exist num_classes.")
        aux_mlp = aux_mlp if aux_mlp else {"hid_channels": 0, "out_channels": 1024}
        act_cfg = act_cfg if act_cfg else {"type": "ReLU"}
        loss = loss if loss else {"type": "CrossEntropyLoss", "loss_weight": 1.0}
        aux_loss = (
            aux_loss if aux_loss else {"type": "BarlowTwinsLoss", "off_diag_penality": 1.0 / 128.0, "loss_weight": 1.0}
        )
        CustomMultiLabelNonLinearClsHead.__init__(
            self,
            num_classes,
            in_channels,
            hid_channels=hid_channels,
            act_cfg=act_cfg,
            loss=loss,
            dropout=dropout,
            scale=scale,
            normalized=normalized,
        )
        SemiMultilabelClsHead.__init__(self, unlabeled_coef, use_dynamic_loss_weighting, aux_loss)

        self.aux_mlp = generate_aux_mlp(aux_mlp, in_channels)

    def loss(
        self,
        logits: torch.Tensor,
        gt_label: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        features: Optional[tuple] = None,
        **kwargs,
    ) -> dict:
        """Calculate loss for given logits/gt_label."""
        return SemiMultilabelClsHead.loss(self, logits, gt_label, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward fuction of SemiNonLinearMultilabelClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> dict:
        """Forward_train fuction of SemiNonLinearMultilabelClsHead class."""
        return self.forward_train_with_last_layers(
            cls_score,
            gt_label,
            final_cls_layer=self.classifier,
            final_emb_layer=self.aux_mlp,
        )