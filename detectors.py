'''
This module is used to instantiate different types of object detection networks
'''


import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
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

    def __init__(self, params, backbone, num_classes, device="cpu",
                 num_proposals=2000, ss_type="fast"):
        super(RCNN, self).__init__()
        self.params = params
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device
        self.num_proposals = num_proposals
        self.ss_type = ss_type

        # Save backbone input/output sizes
        self.backbone_input_size = (
            params.input_size.height, params.input_size.width
        )
        self.backbone_output_size = cnn_output_size(
            backbone, self.backbone_input_size
        )

        self.class_score = nn.Linear(self.backbone.out_channels, num_classes)
        self.bbox_correction = nn.Linear(
            self.backbone.out_channels, num_classes * 4
        )

    def selective_search(self, img):
        assert img.sum().item() != 0, (
            "Found a totally black image when running selective search"
        )
        self.SS.setBaseImage(utils.to_numpy(img))

        # Set selective search type
        # Fast but low recall
        if self.ss_type == "fast":
            self.SS.switchToSelectiveSearchFast()
        # High recall but slow
        elif self.ss_type == "quality":
            self.SS.switchToSelectiveSearchQuality()
        # Only fast or quality types are allowed
        else:
            raise Exception(
                "Argument `ss_type` could only be 'fast' or 'quality'"
            )

        # Compute boxes and extract the corresponding proposals
        boxes = self.SS.process()[:self.num_proposals]
        proposals = []
        for box in boxes:
            proposals.append(
                img[:, box[0]:box[0] + box[2], box[1]:box[1] + box[3]]
            )

        return proposals

    def forward(self, images):
        assert isinstance(images, list) and len(images[0].shape) == 3
        class_scores = torch.zeros(
            (len(images), self.num_proposals, self.num_classes)
        )
        bbox_corrections = torch.zeros(
            (len(images), self.num_proposals, 4)
        )
        for i, img in enumerate(images):
            proposals = self.selective_search(img)
            for proposal in proposals:
                warped_proposal = TF.resize(
                    proposal, self.backbone_input_size
                )
                features = self.backbone(warped_proposal.float().unsqueeze(0))
                flattened_features = torch.flatten(features, start_dim=1)
                class_scores[i, :len(proposals), :] = self.class_score(
                    flattened_features
                )
                bbox_corrections[i, :len(proposals), :] = self.bbox_correction(
                    flattened_features
                )
        return class_scores, bbox_corrections


class FastRCNN(RCNN):
    '''
    Fast R-CNN PyTorch module
    '''

    def __init__(self, params, backbone, num_classes,
                 device="cpu", num_proposals=2000, ss_type="fast"):
        super(FastRCNN, self).__init__(
            params, backbone, num_classes, device=device,
            num_proposals=num_proposals, ss_type=ss_type
        )
        self.roi_pool_output_size = (
            params.detector.fast_rcnn.roi_pool.output_size.height,
            params.detector.fast_rcnn.roi_pool.output_size.width
        )
        self.roi_pool_spatial_scale = (
            self.backbone_output_size[0] /
            self.backbone_input_size[0]
        )
        self.roi_pool = torchvision.ops.RoIPool(
            self.roi_pool_output_size, self.roi_pool_spatial_scale
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
        params, backbones.get_backbone(params), num_classes,
        device=params.device, num_proposals=params.detector.rcnn.num_proposals,
        ss_type=params.detector.rcnn.ss_type
    )


def _get_fast_rcnn(params, num_classes):
    backbone = backbones.get_backbone(params)

    return FastRCNN(
        params, backbone, num_classes, roi_pool_output_size,
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
