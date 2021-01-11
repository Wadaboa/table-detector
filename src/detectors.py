'''
This module is used to instantiate different types of object detection networks
'''


import os
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import backbones
import utils
import region_proposal


class CustomRoIHeads(RoIHeads):
    '''
    Custom subclass of the torchvision RoIHeads module
    '''

    def __init__(self, box_roi_pool, box_head, box_predictor,
                 fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
                 positive_fraction, bbox_reg_weights,
                 score_thresh, nms_thresh, detections_per_img):
        super(CustomRoIHeads, self).__init__(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights,
            score_thresh, nms_thresh, detections_per_img
        )

    def forward(self, features, proposals, image_shapes, targets=None):
        '''
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        '''
        labels, regression_targets = None, None
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )

        # Rescale features with the RoI pooling layer
        if (isinstance(self.box_roi_pool, torchvision.ops.RoIPool) or
                isinstance(self.box_roi_pool, torchvision.ops.RoIAlign)):
            box_features = self.box_roi_pool(features['0'], proposals)
        elif isinstance(self.box_roi_pool, torchvision.ops.MultiScaleRoIAlign):
            box_features = self.box_roi_pool(
                features['0'], proposals, image_shapes
            )
        elif self.box_roi_pool is None:
            box_features = features
        else:
            raise ValueError(
                "The RoI pooling layer in {self.__class__.__name__} "
                "can only be an instance of torchvision.ops.RoIPool, "
                "torchvision.ops.RoIAlign, torchvision.ops.MultiScaleRoIAlign "
                "or None (when the features are already computed from proposals)"
            )

        # Compute class scores and boxes corrections
        flattened_features = torch.flatten(box_features, start_dim=1)
        class_logits = self.box_head(flattened_features)
        box_regression = self.box_predictor(flattened_features)

        result = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses


class RCNN(nn.Module):
    '''
    R-CNN PyTorch module
    '''

    def __init__(self, params, num_classes):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        self.device = params.generic.device

        # Get backbone
        self.backbone = backbones.Backbone(params)

        # Select region proposals model and parameters
        self.rp_model = region_proposal.RegionProposals(
            params.detector.region_proposals.type,
            params.detector.region_proposals
        )

        self.transform = GeneralizedRCNNTransform(
            min_size=params.backbone.input_size.bound.min,
            max_size=params.backbone.input_size.bound.max,
            image_mean=params.backbone.imagenet_params.mean,
            image_std=params.backbone.imagenet_params.std
        )

        # R-CNN prediction head
        self.class_score = nn.Linear(
            self.backbone.out_size, self.num_classes
        )
        self.bbox_correction = nn.Linear(
            self.backbone.out_size, self.num_classes * 4
        )

        # Class scores and bounding box corrections predictor
        self.roi_heads = CustomRoIHeads(
            None, self.class_score, self.bbox_correction,
            fg_iou_thresh=params.detector.box_fg_iou_thresh,
            bg_iou_thresh=params.detector.box_bg_iou_thresh,
            batch_size_per_image=params.detector.box_batch_size_per_image,
            positive_fraction=params.detector.box_positive_fraction,
            bbox_reg_weights=params.detector.box_regression_weights,
            score_thresh=params.detector.box_score_thresh,
            nms_thresh=params.detector.box_nms_thresh,
            detections_per_img=params.detector.box_detections_per_img
        )

    def forward(self, images, targets=None):
        '''
        1. Region proposals for each image
        2. Backbone for each proposal in an image
        3. R-CNN prediction head for each proposal in an image
           (input is the output of backbone)
        '''
        return self._forward(images, targets)
        assert isinstance(images, list) and len(images[0].shape) == 3
        loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0}

        for i, img in enumerate(images):
            # Compute predictions
            num_proposals, proposals, proposals_coords = self.region_proposals(
                img
            )
            normalized_proposals = utils.normalize_image(proposals)
            standardized_proposals = utils.standardize_image(
                normalized_proposals
            )
            features = self.backbone(standardized_proposals)
            flattened_features = torch.flatten(features, start_dim=1)
            class_scores = self.class_score(flattened_features)
            bbox_corrections = self.bbox_correction(flattened_features).reshape(
                self.num_proposals, self.num_classes, 4
            )

            # Compute losses
            loss_classifier, loss_box_reg, _ = self.compute_loss(
                num_proposals, proposals_coords, class_scores,
                bbox_corrections, targets[i]
            )
            loss_dict['loss_classifier'] += loss_classifier
            loss_dict['loss_box_reg'] += loss_box_reg

        return loss_dict

    def _forward(self, images, targets=None):
        results = []
        losses = {}
        for i, img in enumerate(images):
            proposals, proposals_coords = self.rp_model(img)
            print(len(proposals), proposals[0].shape)

            transformed_images, transformed_targets = self.transform(
                proposals, [targets[i]] * len(proposals)
            )
            print(transformed_images.tensors.shape)

            # Call the backbone
            features = self.backbone(transformed_images.tensors)
            print(features.shape)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([('0', features)])

            # Compute classification scores and box regression values
            detections, detector_losses = self.roi_heads(
                features, proposals_coords,
                transformed_images.image_sizes, transformed_targets
            )

            # Update losses when training
            if self.training:
                if len(losses) == 0:
                    losses.update(detector_losses)
                else:
                    for k, v in detector_losses.items():
                        losses[k] += v
            # Update detections when evaluating
            else:
                results.extend(
                    self.transform.postprocess(
                        detections, transformed_images.image_sizes,
                        utils.get_image_sizes(images)
                    )
                )

        # Return detections when evaluating
        if not self.training:
            return results

        # Return losses when training
        return losses


class FastRCNN(RCNN):
    '''
    Fast R-CNN PyTorch module
    '''

    def __init__(self, params, num_classes):
        super(FastRCNN, self).__init__(params, num_classes)

        # Input standardization/resizing
        self.transform = GeneralizedRCNNTransform(
            min_size=params.backbone.input_size.bound.min,
            max_size=params.backbone.input_size.bound.max,
            image_mean=params.backbone.imagenet_params.mean,
            image_std=params.backbone.imagenet_params.std
        )

        # ROI pooling configuration
        self.roi_pool_output_size = (
            self.backbone.out_height, self.backbone.out_width
        )
        self.roi_pool_spatial_scale = (
            self.backbone.out_height / self.backbone.in_height
        )
        self.roi_pool = torchvision.ops.RoIPool(
            self.roi_pool_output_size, self.roi_pool_spatial_scale
        )

        # Class scores and bounding box corrections predictor
        self.roi_heads = CustomRoIHeads(
            self.roi_pool, self.class_score, self.bbox_correction,
            fg_iou_thresh=params.detector.box_fg_iou_thresh,
            bg_iou_thresh=params.detector.box_bg_iou_thresh,
            batch_size_per_image=params.detector.box_batch_size_per_image,
            positive_fraction=params.detector.box_positive_fraction,
            bbox_reg_weights=params.detector.box_regression_weights,
            score_thresh=params.detector.box_score_thresh,
            nms_thresh=params.detector.box_nms_thresh,
            detections_per_img=params.detector.box_detections_per_img
        )

    def forward(self, images, targets=None):
        '''
        1. Region proposals for each image
        2. Backbone for each image
        3. ROI pooling layer to crop and warp backbone features according to proposals
        4. R-CNN prediction head for each proposal in an image
           (input is the output of ROI pool)
        '''
        # Perform the following transformations:
        # - Image standardization
        # - Image/target resizing
        # Returns a tuple (image_list, targets), where image_list
        # is an instance of torchvision.models.detection.image_list.ImageList
        transformed_images, transformed_targets = self.transform(
            images, targets
        )

        # Get static proposals using one of the the implemented methods
        proposals = []
        for img in images:
            _, proposals_coords = self.rp_model(img)
            proposals.append(proposals_coords)

        # Call the backbone
        features = self.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # Perform ROI pooling, compute classification scores
        # and box regression values
        detections, detector_losses = self.roi_heads(
            features, proposals, transformed_images.image_sizes, transformed_targets
        )

        # Return detections when evaluating
        if not self.training:
            detections = self.transform.postprocess(
                detections, transformed_images.image_sizes,
                utils.get_image_sizes(images)
            )
            return detections

        # Return losses when training
        return detector_losses


class FasterRCNN(nn.Module):
    '''
    Faster R-CNN (without FPN) PyTorch module
    '''

    def __init__(self, params, num_classes):
        super(FasterRCNN, self).__init__()
        self.params = params
        self.num_classes = num_classes
        self.device = params.generic.device

        # Get backbone
        self.backbone = backbones.Backbone(params)

        # Define ROI pooling
        self.roi_pool_output_size = (
            self.backbone.out_height, self.backbone.out_width
        )
        self.featmap_names = ['0']
        roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=self.featmap_names,
            output_size=self.roi_pool_output_size,
            sampling_ratio=-1
        )

        # Define the anchors generator:
        # - Sizes `s` and aspect ratios `r` should have the same number of elements, and it should
        #   correspond to the number of feature maps (without FPN we only have 1 feature map)
        # - Sizes and aspect ratios for one feature map can have an arbitrary number of elements,
        #   and the anchors generator will output a set of `s[i] * r[i]` anchors
        #   per spatial location for feature map `i`
        anchor_sizes = tuple([
            tuple(params.detector.faster_rcnn.anchors.sizes) *
            len(self.featmap_names)
        ])
        aspect_ratios = tuple([
            tuple(params.detector.faster_rcnn.anchors.ratios) *
            len(self.featmap_names)
        ])
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        # Define Faster-RCNN model
        self.faster_rcnn = torchvision.models.detection.faster_rcnn.FasterRCNN(
            self.backbone, num_classes, box_roi_pool=roi_pool,
            rpn_anchor_generator=anchor_generator,
            image_mean=params.backbone.imagenet_params.mean,
            image_std=params.backbone.imagenet_params.std,
        )

    def forward(self, images, targets=None):
        '''
        1. Region proposals for the each image (computed by a RPN), based on anchors
        2. Backbone for each image
        3. ROI align layer to crop and warp backbone features according to proposals
        4. R-CNN prediction head for each proposal in an image
           (input is the output of ROI align)
        '''
        # Returns losses when in training mode and
        # detections when in evaluation mode
        return self.faster_rcnn(images, targets=targets)


def _get_rcnn(params, num_classes):
    return RCNN(params, num_classes)


def _get_fast_rcnn(params, num_classes):
    return FastRCNN(params, num_classes)


def _get_faster_rcnn(params, num_classes):
    return FasterRCNN(params, num_classes)


def get_detector(params, num_classes):
    detector_func_name = f'_get_{params.detector.type}'
    assert detector_func_name in globals(), (
        f'The detector type `{params.detector.type}` is not yet supported'
    )
    return globals()[detector_func_name](params, num_classes)
