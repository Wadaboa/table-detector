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


class RCNN(nn.Module):
    '''
    R-CNN PyTorch module
    '''

    SS = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def __init__(self, params, num_classes, device="cpu",
                 num_proposals=2000, ss_type="fast"):
        super(RCNN, self).__init__()
        self.params = params
        self.num_classes = num_classes
        self.device = device
        self.num_proposals = num_proposals
        self.ss_type = ss_type

        # Get backbone
        self.backbone = backbones.get_backbone(params)

        # Save backbone input/output sizes
        self.backbone_input_size = (
            self.params.input_size.height,
            self.params.input_size.width
        )
        self.backbone_output_height, self.backbone_output_width = utils.cnn_output_size(
            self.backbone, self.backbone_input_size
        )
        self.backbone_output_size = (
            self.backbone_output_height *
            self.backbone_output_width *
            self.backbone.out_channels
        )

        # R-CNN prediction head
        self.class_score = nn.Linear(
            self.backbone_output_size, self.num_classes
        )
        self.bbox_correction = nn.Linear(
            self.backbone_output_size, self.num_classes * 4
        )

    def selective_search(self, img):
        '''
        Use OpenCV to obtain a list of bounding box proposals
        to pass to the backbone
        '''
        assert len(img.shape) == 3, (
            "Selective search can be performed on one image at a time"
        )
        assert img.sum().item() != 0, (
            "Found a totally black image when running selective search"
        )
        cv2.setUseOptimized(True)

        # Set base image
        np_img = utils.to_numpy(img)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        self.SS.setBaseImage(rgb_img)

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

        # Compute boxes, extract the corresponding proposals
        # and warp them to be compatible with the chosen backbone
        boxes = self.SS.process()[:self.num_proposals]
        proposals = torch.zeros(
            (self.num_proposals, 3, *self.backbone_input_size),
            dtype=torch.uint8, device=self.device
        )
        boxes_coords = torch.zeros(
            (self.num_proposals, 4), dtype=torch.uint8, device=self.device
        )
        for i, box in enumerate(boxes):
            proposal = img[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            assert all(proposal.shape) != 0, (
                "A selective search box has wrong shape"
            )
            proposals[i] = TF.resize(proposal, self.backbone_input_size)
            boxes_coords[i] = torch.tensor(
                [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            )

        return proposals, boxes_coords

    def forward(self, images):
        '''
        1. Selective search for each image
        2. Backbone for each proposal in an image
        3. R-CNN prediction head for each proposal in an image
           (input is the output of backbone)
        '''
        assert isinstance(images, list) and len(images[0].shape) == 3
        class_scores = torch.zeros(
            (len(images), self.num_proposals, self.num_classes),
            dtype=torch.float, device=self.device
        )
        bbox_corrections = torch.zeros(
            (len(images), self.num_proposals, self.num_classes * 4),
            dtype=torch.float, device=self.device
        )
        for i, img in enumerate(images):
            proposals, _ = self.selective_search(img)
            features = self.backbone(proposals)
            flattened_features = torch.flatten(features, start_dim=1)
            class_scores[i] = self.class_score(flattened_features)
            bbox_corrections[i] = self.bbox_correction(flattened_features)
        return class_scores, bbox_corrections


class FastRCNN(RCNN):
    '''
    Fast R-CNN PyTorch module
    '''

    def __init__(self, params, num_classes, device="cpu",
                 num_proposals=2000, ss_type="fast"):
        super(FastRCNN, self).__init__(
            params, num_classes, device=device,
            num_proposals=num_proposals, ss_type=ss_type
        )
        self.roi_pool_output_size = (
            params.detector.fast_rcnn.roi_pool.output_size.height,
            params.detector.fast_rcnn.roi_pool.output_size.width
        )
        self.roi_pool_spatial_scale = (
            self.backbone_output_height / self.backbone_input_size[0]
        )
        self.roi_pool = torchvision.ops.RoIPool(
            self.roi_pool_output_size, self.roi_pool_spatial_scale
        )

    def forward(self, images):
        '''
        1. Selective search for the each image
        2. Backbone for each image
        3. ROI pooling layer to crop and warp backbone features according to proposals
        4. R-CNN prediction head for each proposal in an image 
           (input is the output of ROI pool)
        '''
        assert isinstance(images, list) and len(images[0].shape) == 3
        class_scores = torch.zeros(
            (len(images), self.num_proposals, self.num_classes),
            dtype=torch.float, device=self.device
        )
        bbox_corrections = torch.zeros(
            (len(images), self.num_proposals, self.num_classes * 4),
            dtype=torch.float, device=self.device
        )
        for i, img in enumerate(images):
            warped_image = TF.resize(img, self.backbone_input_size)
            features = self.backbone(warped_image.float().unsqueeze(0))
            _, boxes_coords = self.selective_search(warped_image)
            rescaled_features = self.roi_pool(features, [boxes_coords.float()])
            flattened_features = torch.flatten(rescaled_features, start_dim=1)
            class_scores[i] = self.class_score(flattened_features)
            bbox_corrections[i] = self.bbox_correction(flattened_features)
        return class_scores, bbox_corrections


def _get_rcnn(params, num_classes):
    return RCNN(
        params, num_classes, device=params.device,
        num_proposals=params.detector.rcnn.num_proposals,
        ss_type=params.detector.rcnn.ss_type
    )


def _get_fast_rcnn(params, num_classes):
    return FastRCNN(
        params, num_classes, device=params.device,
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
