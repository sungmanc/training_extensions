"""Module for defining AsymmetricLossWithIgnore."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import torch
from mmpretrain.models.builder import LOSSES
from mmpretrain.models.losses.utils import weight_reduce_loss
from torch import nn


def asymmetric_loss_with_ignore(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_label_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    reduction: str = "none",
    avg_factor: Optional[int] = None,
) -> torch.Tensor:
    """Asymmetric loss, please refer to the `paper <https://arxiv.org/abs/2009.14119>`_ for details.

    Args:
        pred (torch.Tensor): The prediction with shape (N, *).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, *).
        valid_label_mask (torch.Tensor, optional): Label mask for consideration of ignored label.
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Dafaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
             is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: Loss.
    """

    eps = 1e-8
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if reduction != "mean":  # we don't use avg factor with other reductions
        avg_factor = None  # if we are not set this to None the exception will be throwed

    if clip and clip > 0:
        pos_target = (1 - pred_sigmoid + clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pos_target = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pos_target).pow(gamma_pos * target + gamma_neg * (1 - target))
    loss = -torch.log(pos_target.clamp(min=eps)) * asymmetric_weight

    if valid_label_mask is not None:
        loss = loss * valid_label_mask

    if weight is not None:
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class AsymmetricLossWithIgnore(nn.Module):
    """Asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "none",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_label_mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward fuction of asymmetric loss."""
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * asymmetric_loss_with_ignore(
            pred,
            target,
            valid_label_mask,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_cls