'''
This module is used to instantiate different types of object detection networks
'''


import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.roi_heads import fastrcnn_loss

import backbones
import utils


class RCNN(nn.Module):
    '''
    R-CNN PyTorch module
    '''

    def __init__(self, params, num_classes):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        self.device = params.generic.device
        self.num_proposals = params.detector.region_proposals.num_proposals
        self.proposals_type = params.detector.region_proposals.type

        # Select region proposals model
        assert self.proposals_type in ("selective_search", "edge_boxes"), (
            "Invalid region proposals type"
        )
        if self.proposals_type == "selective_search":
            self.selective_search_strategy = params.detector.region_proposals.selective_search.strategy
            assert self.selective_search_strategy in ("fast", "quality"), (
                "The given selective search strategy is not supported"
            )
            self.proposals_model = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        elif self.proposals_type == "edge_boxes":
            edge_boxes_model_path = params.detector.region_proposals.edge_boxes.model_path
            assert os.path.exists(edge_boxes_model_path), (
                "The given Edge Boxes model path does not exist"
            )
            self.proposals_model = cv2.ximgproc.createStructuredEdgeDetection(
                edge_boxes_model_path
            )

        # Get backbone
        self.backbone = backbones.Backbone(params)

        # R-CNN prediction head
        self.class_score = nn.Linear(
            self.backbone.out_size, self.num_classes
        )
        self.bbox_correction = nn.Linear(
            self.backbone.out_size, self.num_classes * 4
        )

        # Loss criterions
        self.class_score_criterion = nn.CrossEntropyLoss()
        self.bbox_correction_criterion = nn.MSELoss()

    def edge_boxes(self, img):
        '''
        C. L. Zitnick and P. Dollar,
        “Edge boxes: Locating object proposals from edges”,
        European Conference on Computer Vision (ECCV), 2014.
        '''
        edges = self.proposals_model.detectEdges(np.float32(img) / 255.0)
        orientation_map = self.proposals_model.computeOrientation(edges)
        edges = self.proposals_model.edgesNms(edges, orientation_map)
        cv_edge_boxes = cv2.ximgproc.createEdgeBoxes()
        cv_edge_boxes.setMaxBoxes(self.num_proposals)
        boxes, scores = cv_edge_boxes.getBoundingBoxes(edges, orientation_map)
        return boxes

    def selective_search(self, img):
        '''
        J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders,
        “Selective search for object recognition,”
        International Journal of Computer Vision (IJCV), 2013
        '''
        # Set base image
        self.proposals_model.setBaseImage(img)

        # Set selective search type
        # Fast but low recall
        if self.selective_search_strategy == "fast":
            self.proposals_model.switchToSelectiveSearchFast()
        # High recall but slow
        elif self.selective_search_strategy == "quality":
            self.proposals_model.switchToSelectiveSearchQuality()

        # Compute proposals and clip them to the maximum
        # number of regions
        return self.proposals_model.process()[:self.num_proposals]

    def region_proposals(self, img):
        '''
        Use OpenCV to obtain a list of bounding box proposals
        to pass to the backbone
        '''
        assert isinstance(img, torch.Tensor), (
            "The input image for region proposals should be a PyTorch tensor"
        )
        assert len(img.shape) == 3, (
            "Proposals can only be computed on one image at a time"
        )
        assert img.sum().item() != 0, (
            "Found a totally black image when running region proposals"
        )

        # Set the optimized flag for OpenCV
        cv2.setUseOptimized(True)

        # Convert the image to numpy RGB format
        norm_img = utils.denormalize_image(img)
        np_img = utils.to_numpy(norm_img)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Call the specified region proposals model
        boxes = getattr(self, self.proposals_type)(rgb_img)

        # Compute boxes, extract the corresponding proposals
        # and warp them to be compatible with the chosen backbone
        proposals = torch.zeros(
            (self.num_proposals, 3, self.backbone.in_height, self.backbone.in_width),
            dtype=torch.float32, device=self.device
        )
        boxes_coords = torch.zeros(
            (self.num_proposals, 4), dtype=torch.float32, device=self.device
        )
        for i, box in enumerate(boxes):
            proposal = img[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            assert all(proposal.shape) != 0, (
                "A region proposal box has wrong shape"
            )
            proposals[i] = TF.resize(
                proposal, (self.backbone.in_height, self.backbone.in_width)
            )
            boxes_coords[i] = torch.tensor(
                [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            )

        return len(boxes), proposals, boxes_coords

    def compute_loss(self, num_proposals, boxes_coords, class_scores, bbox_corrections, targets):
        '''
        Compute one loss for the classification head and one
        for the bounding box regression head
        '''
        class_labels, bbox_predictions, bbox_ground_truths = [], [], []
        boxes = targets["boxes"].cpu().numpy()

        # Iterate only over the number of computed proposals
        for i in range(num_proposals):

            # Get the actual proposal coordinates
            box = boxes_coords[i].cpu().numpy()

            # Set the default value for the classification label
            # as the background label
            class_label = 0

            # Compute the most overlapping ground truth bounding box
            best_match = utils.most_overlapping_box(box, boxes, 0.5)

            # If not background
            if best_match is not None:

                # Store the actual classification label
                # associated to the ground truth box
                gt_index, gt, _ = best_match
                class_label = targets["labels"][gt_index]

                # Compute proposal and ground truth width and height
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                gt_width = gt[2] - gt[0]
                gt_height = gt[3] - gt[1]

                # Compute the output box, considering the network corrections
                output_box = torch.tensor([
                    box[0] + box_width * bbox_corrections[i, 0],
                    box[1] + box_height * bbox_corrections[i, 1],
                    box_width * torch.exp(bbox_corrections[i, 2]),
                    box_height * torch.exp(bbox_corrections[i, 3])
                ])

                # Compute the regression head target
                regression_target = torch.tensor([
                    (gt[0] - box[0]) / box_width,
                    (gt[1] - box[1]) / box_height,
                    torch.log(gt_width / box_width),
                    torch.log(gt_height / box_height)
                ])

                print(output_box, regression_target)

                # Store output box vs regression head target
                bbox_predictions.append(output_box)
                bbox_ground_truths.append(regression_target)

            # Store classification head target
            class_labels.append(class_label)

        print(len(class_labels), class_scores.shape)
        print(len(bbox_predictions), len(bbox_ground_truths))

        # Return one loss for the classification head and one
        # for the bounding box regression head
        return (
            self.class_score_criterion(
                torch.argmax(class_scores[:num_proposals, :], axis=1),
                torch.tensor(class_labels)
            ),
            self.bbox_correction_criterion(
                torch.tensor(bbox_predictions),
                torch.tensor(bbox_ground_truths)
            )
        )

    def forward(self, images, targets=None):
        '''
        1. Region proposals for each image
        2. Backbone for each proposal in an image
        3. R-CNN prediction head for each proposal in an image
           (input is the output of backbone)
        '''
        assert isinstance(images, list) and len(images[0].shape) == 3
        loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0}

        for i, img in enumerate(images):
            # Compute predictions
            num_proposals, proposals, boxes_coords = self.region_proposals(img)
            normalized_proposals = utils.normalize_image(proposals)
            standardized_proposals = utils.standardize_image(
                normalized_proposals
            )
            features = self.backbone(standardized_proposals)
            flattened_features = torch.flatten(features, start_dim=1)
            class_scores = self.class_score(flattened_features)
            bbox_corrections = self.bbox_correction(flattened_features)

            # Compute losses
            loss_classifier, loss_box_reg = self.compute_loss(
                num_proposals, boxes_coords, class_scores, bbox_corrections, targets[i]
            )
            loss_dict['loss_classifier'] += loss_classifier
            loss_dict['loss_box_reg'] += loss_box_reg

        return loss_dict


class FastRCNN(RCNN):
    '''
    Fast R-CNN PyTorch module
    '''

    def __init__(self, params, num_classes):
        super(FastRCNN, self).__init__(params, num_classes)

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

        # Loss criterions
        self.bbox_correction_criterion = nn.SmoothL1Loss()

    def forward(self, images, targets=None):
        '''
        1. Region proposals for the each image
        2. Backbone for each image
        3. ROI pooling layer to crop and warp backbone features according to proposals
        4. R-CNN prediction head for each proposal in an image
           (input is the output of ROI pool)
        '''
        assert isinstance(images, list) and len(images[0].shape) == 3
        loss_dict = {'loss_classifier': 0, 'loss_box_reg': 0}

        for i, img in enumerate(images):
            # Compute predictions
            warped_image = TF.resize(
                img, (self.backbone.in_height, self.backbone.in_width)
            )
            standardized_image = utils.standardize_image(warped_image)
            features = self.backbone(standardized_image.unsqueeze(0))
            num_proposals, _, boxes_coords = self.region_proposals(
                warped_image
            )
            rescaled_features = self.roi_pool(features, [boxes_coords])
            flattened_features = torch.flatten(rescaled_features, start_dim=1)
            class_scores = self.class_score(flattened_features)
            bbox_corrections = self.bbox_correction(flattened_features)

            # Compute losses
            loss_classifier, loss_box_reg = self.compute_loss(
                num_proposals, boxes_coords, class_scores, bbox_corrections, targets[i]
            )
            loss_dict['loss_classifier'] += loss_classifier
            loss_dict['loss_box_reg'] += loss_box_reg

        return loss_dict


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
        return self.faster_rcnn(
            images, targets=targets
        )


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
