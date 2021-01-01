'''
This module is used to instantiate different types of object detection networks
'''

from functools import partial

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN

import backbones
import utils


def cnn_output_size(model, input_size):
    '''
    Return the spatial output size of a CNN model
    '''
    height, width = input_size
    dummy_image = torch.zeros((1, 3, height, width))
    return model(dummy_image).shape[2:]


class RCNN(nn.Module):
    '''
    R-CNN PyTorch module
    '''

    SS = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def __init__(self, backbone, num_classes, device="cpu",
                 num_proposals=2000, ss_type="fast"):
        super(RCNN, self).__init__()
        self.backbone = backbone
        self.class_score = nn.Linear(self.backbone.out_channels, num_classes)
        self.bbox_correction = nn.Linear(
            self.backbone.out_channels, num_classes * 4
        )
        self.device = device
        self.num_proposals = num_proposals
        self.ss_type = ss_type

    def selective_search(self, images):
        # Fast but low recall
        if self.ss_type == "fast":
            self.SS.switchToSelectiveSearchFast()
        # High recall but slow
        elif self.ss_type == "quality":
            self.SS.switchToSelectiveSearchQuality()
        else:
            raise Exception(
                "Argument `ss_type` could only be `fast` or `quality`"
            )

        # Process the input images and return the desired
        # number of region proposals for each one
        proposals = torch.zeros(
            (images.shape[0], self.num_proposals, 4),
            dtype=torch.float32, device=self.device
        )
        for i, img in enumerate(images):
            self.SS.setBaseImage(utils.to_numpy(img))
            proposals[i] = torch.tensor(
                self.SS.process()[:self.num_proposals],
                dtype=torch.float32, device=self.device
            )
        return proposals

    def forward(self, images):
        assert len(images.shape) == 4
        print(images.shape)
        proposals = self.selective_search(images)
        print(proposals.shape)
        features = self.backbone(proposals)
        print(features.shape)
        flattened_features = torch.flatten(features, start_dim=1)
        class_scores = self.class_score(flattened_features)
        bbox_corrections = self.bbox_correction(flattened_features)
        return class_scores, bbox_corrections


class FastRCNN(RCNN):
    '''
    Fast R-CNN PyTorch module
    '''

    def __init__(self, backbone, num_classes, roi_pool_output_size, roi_pool_spatial_scale,
                 device="cpu", num_proposals=2000, ss_type="fast"):
        super(FastRCNN, self).__init__(
            backbone, num_classes, device=device,
            num_proposals=num_proposals, ss_type=ss_type
        )
        self.roi_pool = torchvision.ops.RoIPool(
            roi_pool_output_size, roi_pool_spatial_scale
        )

    def forward(self, images):
        assert len(images.shape) == 4
        features = self.backbone(images)
        proposals = self.selective_search(images)
        rescaled_features = self.roi_pool(features, proposals)
        flattened_features = torch.flatten(rescaled_features)
        class_scores = self.class_score(flattened_features)
        bbox_corrections = self.bbox_correction(flattened_features)
        return class_scores, bbox_corrections


def _get_rcnn(params, num_classes):
    return RCNN(
        backbones.get_backbone(params), num_classes, device=params.device,
        num_proposals=params.detector.rcnn.num_proposals,
        ss_type=params.detector.rcnn.ss_type
    )


def _get_fast_rcnn(params, num_classes):
    backbone = backbones.get_backbone(params)
    roi_pool_output_size = (
        params.detector.fast_rcnn.roi_pool.output_size.height,
        params.detector.fast_rcnn.roi_pool.output_size.width
    )
    backbone_input_size = (
        params.input_size.height, params.input_size.width
    )
    backbone_output_size = cnn_output_size(
        backbone, backbone_input_size
    )
    roi_pool_spatial_scale = backbone_output_size[0] / backbone_input_size[0]
    return FastRCNN(
        backbone, num_classes, roi_pool_output_size,
        roi_pool_spatial_scale, device=params.device,
        num_proposals=params.detector.fast_rcnn.num_proposals,
        ss_type=params.detector.fast_rcnn.ss_type
    )


def _get_faster_rcnn(params, num_classes):
    return FasterRCNN(backbones.get_backbone(params), num_classes)


def get_detector(params, num_classes):
    detector_func_name = f'_get_{params.detector.type}'
    assert detector_func_name in globals(), (
        f'The detector type `{params.detector.type}` is not yet supported'
    )
    return globals()[detector_func_name](params, num_classes)
