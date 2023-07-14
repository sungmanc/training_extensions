"""Custom ATSS head for OTX template."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import force_fp32
from mmdet.core import (
    anchor_inside_flags,
    bbox_overlaps,
    multi_apply,
    reduce_mean,
    unmap,
)
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.losses.utils import weight_reduce_loss

from otx.algorithms.detection.adapters.mmdet.models.heads.cross_dataset_detector_head import (
    CrossDatasetDetectorHead,
    TrackingLossDynamicsMixIn,
)
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.algorithms.detection.adapters.mmdet.models.losses.cross_focal_loss import (
    CrossSigmoidFocalLoss,
)

EPS = 1e-12

# pylint: disable=too-many-arguments, too-many-locals,


@HEADS.register_module()
class CustomATSSHead(CrossDatasetDetectorHead, ATSSHead):
    """CustomATSSHead for OTX template."""

    def __init__(self, *args, bg_loss_weight=-1.0, use_qfl=False, qfl_cfg=None, **kwargs):
        if use_qfl:
            kwargs["loss_cls"] = (
                qfl_cfg
                if qfl_cfg
                else dict(
                    type="QualityFocalLoss",
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                )
            )
        self.adaptive_params = kwargs.pop("adaptive_params", None)
        super().__init__(*args, **kwargs)
        self.bg_loss_weight = bg_loss_weight
        self.use_qfl = use_qfl

    @force_fp32(apply_to=("cls_scores", "bbox_preds", "centernesses"))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            _,
            valid_label_mask,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_pos = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_pos = max(num_total_pos, 1.0)

        num_total_neg = reduce_mean(torch.tensor(num_total_neg, dtype=torch.float, device=device)).item()
        num_total_neg = max(num_total_neg, 1.0)

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            valid_label_mask,
            num_pos_samples=num_total_pos,
            num_neg_samples=num_total_neg,
        )
        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        if bbox_avg_factor < EPS:
            bbox_avg_factor = 1
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

    def loss_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        centerness,
        labels,
        label_weights,
        bbox_targets,
        valid_label_mask,
        num_pos_samples,
        num_neg_samples,
    ):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * num_classes, H, W)
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).
            num_pos_samples (int): Number of positive samples that is
                reduced over all GPUs.
            num_neg_samples (int): Number of negative samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        valid_label_mask = valid_label_mask.reshape(-1, self.cls_out_channels)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)

        if self.use_qfl:
            quality = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            if self.reg_decoded_bbox:
                pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            if self.use_qfl:
                quality[pos_inds] = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True).clamp(
                    min=1e-6
                )

            # regression loss
            loss_bbox = self._get_loss_bbox(pos_bbox_targets, pos_bbox_pred, centerness_targets)

            # centerness loss
            loss_centerness = self._get_loss_centerness(num_pos_samples, pos_centerness, centerness_targets)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        # Re-weigting BG loss
        if self.use_qfl:
            labels = (labels, quality)  # For quality focal loss arg spec

        # Reference: MMDetection
        # If number of positive anchors are too small than the negatives and user want to adaptive param,
        # OTX enhances the loss of positive samples by modifying alpha and weight of positive loss.
        if self.adaptive_params:
            num_neg_ratio = num_neg_samples / (num_pos_samples + num_neg_samples)
            use_pos_enhanced_focal_loss = num_neg_ratio > self.adaptive_params.neg_ratio_threshold
            if use_pos_enhanced_focal_loss and self.adaptive_params.is_small_data and self.adaptive_params.enable:
                pred = cls_score.contiguous().sigmoid()
                target = torch.nn.functional.one_hot(labels.contiguous(), num_classes=self.num_classes + 1)
                target = target[:, : self.num_classes]
                target = target.type_as(pred)

                pt = (1 - pred) * target + pred * (1 - target)
                alpha = num_neg_ratio / self.adaptive_params.get("alpha_param", 1.2)
                gamma = self.adaptive_params.get("gamma", 1.0)

                focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
                loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
                focal_loss = loss * focal_weight
                if label_weights is not None:
                    if label_weights.shape != focal_loss.shape:
                        if label_weights.size(0) == focal_loss.size(0):
                            label_weights = label_weights.view(-1, 1)
                        else:
                            assert label_weights.numel() == focal_loss.numel()
                            label_weights = label_weights.view(focal_loss.size(0), -1)
                    assert label_weights.ndim == loss.ndim
                pos_focal_loss = weight_reduce_loss(
                    focal_loss[pos_inds], label_weights[pos_inds], reduction="mean", avg_factor=num_pos_samples
                )
                neg_inds = labels == self.num_classes
                neg_focal_loss = weight_reduce_loss(
                    focal_loss[neg_inds], label_weights[neg_inds], reduction="mean", avg_factor=num_pos_samples
                )
                loss_cls = pos_focal_loss * self.adaptive_params.get("pos_weight", 3.0) + neg_focal_loss
            else:
                loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, num_pos_samples)
        else:
            loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, num_pos_samples)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_pos_inds(self, labels):
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        return pos_inds

    def _get_loss_cls(self, cls_score, labels, label_weights, valid_label_mask, num_total_samples):
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples, valid_label_mask=valid_label_mask
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        return loss_cls

    def _get_loss_centerness(self, num_total_samples, pos_centerness, centerness_targets):
        return self.loss_centerness(pos_centerness, centerness_targets, avg_factor=num_total_samples)

    def _get_loss_bbox(self, pos_bbox_targets, pos_bbox_pred, centerness_targets):
        return self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Get targets for Detection head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        However, if the detector's head loss uses CrossSigmoidFocalLoss,
        the labels_weights_list consists of (binarized label schema * weights) of batch images
        """
        return self.get_atss_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
        )


@HEADS.register_module()
class CustomATSSHeadTrackingLossDynamics(TrackingLossDynamicsMixIn, CustomATSSHead):
    """CustomATSSHead which supports tracking loss dynamics."""

    tracking_loss_types = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.centerness)

    def __init__(self, *args, bg_loss_weight=-1, use_qfl=False, qfl_cfg=None, **kwargs):
        super().__init__(*args, bg_loss_weight=bg_loss_weight, use_qfl=use_qfl, qfl_cfg=qfl_cfg, **kwargs)

    @TrackingLossDynamicsMixIn._wrap_loss
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        return super().loss(cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

    @TrackingLossDynamicsMixIn._wrap_loss_single
    def loss_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        centerness,
        labels,
        label_weights,
        bbox_targets,
        valid_label_mask,
        num_pos_samples,
        num_neg_samples,
    ):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * num_classes, H, W)
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).
            num_pos_samples (int): Number of positive samples that is
                reduced over all GPUs.
            num_neg_samples (int): Number of negative samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        return super().loss_single(
            anchors,
            cls_score,
            bbox_pred,
            centerness,
            labels,
            label_weights,
            bbox_targets,
            valid_label_mask,
            num_pos_samples,
            num_neg_samples
        )

    def _get_loss_cls(self, cls_score, labels, label_weights, valid_label_mask, num_total_samples):
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples,
                valid_label_mask=valid_label_mask,
                reduction_override="none",
            )
        else:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples, reduction_override="none"
            )

        self._store_loss_dyns(loss_cls[self.pos_inds].detach().mean(-1), TrackingLossType.cls)
        return self._postprocess_loss(loss_cls, self.loss_cls.reduction, avg_factor=num_total_samples)

    def _get_loss_centerness(self, num_total_samples, pos_centerness, centerness_targets):
        loss_centerness = self.loss_centerness(
            pos_centerness, centerness_targets, avg_factor=num_total_samples, reduction_override="none"
        )
        self._store_loss_dyns(loss_centerness, TrackingLossType.centerness)
        return self._postprocess_loss(loss_centerness, self.loss_centerness.reduction, avg_factor=num_total_samples)

    def _get_loss_bbox(self, pos_bbox_targets, pos_bbox_pred, centerness_targets):
        loss_bbox = self.loss_bbox(
            pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0, reduction_override="none"
        )
        self._store_loss_dyns(loss_bbox, TrackingLossType.bbox)
        return self._postprocess_loss(loss_bbox, self.loss_centerness.reduction, avg_factor=1.0)

    def _get_target_single(
        self,
        flat_anchors,
        valid_flags,
        num_level_anchors,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Compute regression, classification targets for anchors in a single image."""
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta["img_shape"][:2], self.train_cfg.allowed_border
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        ########## What we changed from the original mmdet code ###############
        # Store all_pos_assigned_gt_inds to member variable
        # to look up training loss dynamics for each gt_bboxes afterwards
        pos_assigned_gt_inds = anchors.new_full((num_valid_anchors,), -1, dtype=torch.long)
        if len(pos_inds) > 0:
            pos_assigned_gt_inds[pos_inds] = (
                self.cur_batch_idx * self.max_gt_bboxes_len + sampling_result.pos_assigned_gt_inds
            )
        if unmap_outputs:
            pos_assigned_gt_inds = unmap(pos_assigned_gt_inds, num_total_anchors, inside_flags, fill=-1)
        self.pos_assigned_gt_inds_list += [pos_assigned_gt_inds]
        self.cur_batch_idx += 1
        ########################################################################

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    @TrackingLossDynamicsMixIn._wrap_get_targets(False)
    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Get targets for Detection head."""
        return super().get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
        )
