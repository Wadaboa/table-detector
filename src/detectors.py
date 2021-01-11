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
from torchvision.models.detection.transform import resize_boxes
from torchviz import make_dot

import backbones
import utils


class RCNN(nn.Module):
    '''
    R-CNN PyTorch module
    '''

    _SS_STRATEGIES = {
        "color": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor(),
        "fill": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill(),
        "size": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize(),
        "texture": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture(),
    }

    def __init__(self, params, num_classes):
        super(RCNN, self).__init__()
        self.num_classes = num_classes
        self.device = params.generic.device
        self.num_proposals = params.detector.region_proposals.num_proposals
        self.proposals_type = params.detector.region_proposals.type
        self.box_regression_weight = params.detector.box_regression_weight

        # Select region proposals model and parameters
        assert self.proposals_type in ("selective_search", "edge_boxes"), (
            "Invalid region proposals type"
        )
        self.proposals_params = params.detector.region_proposals.__dict__[
            self.proposals_type
        ]
        if self.proposals_type == "selective_search":
            self.proposals_model = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            strategies = utils.get_dict_values(
                self._SS_STRATEGIES, self.proposals_params.strategies.get_true_keys()
            )
            strategies = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
                *strategies
            )
            self.proposals_model.addStrategy(strategies)
        elif self.proposals_type == "edge_boxes":
            assert os.path.exists(self.proposals_params.model_path), (
                "The given Edge Boxes model path does not exist"
            )
            self.proposals_model = cv2.ximgproc.createStructuredEdgeDetection(
                self.proposals_params.model_path
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
        # Get the edges
        edges = self.proposals_model.detectEdges(np.float32(img) / 255.0)

        # Create an orientation map
        orientation_map = self.proposals_model.computeOrientation(edges)

        # Suppress edges
        edges = self.proposals_model.edgesNms(edges, orientation_map)

        # Create edge boxes
        cv_edge_boxes = cv2.ximgproc.createEdgeBoxes()
        cv_edge_boxes.setMaxBoxes(self.num_proposals)
        cv_edge_boxes.setAlpha(self.proposals_params.alpha)
        cv_edge_boxes.setBeta(self.proposals_params.beta)

        # Compute boxes
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
        if self.proposals_params.type == "fast":
            self.proposals_model.switchToSelectiveSearchFast()
        # High recall but slow
        elif self.proposals_params.type == "quality":
            self.proposals_model.switchToSelectiveSearchQuality()
        else:
            raise ValueError(
                "The given selective search type is not supported"
            )

        # Compute proposals and clip them to the maximum
        # number of regions
        return self.proposals_model.process()[:self.num_proposals]

    def show_proposals(self, img, proposals, max_proposals=50):
        '''
        Draw proposals bounding boxes on the of the given image
        '''
        img_out = img.copy()
        for x, y, w, h in proposals[:max_proposals]:
            cv2.rectangle(
                img_out, (x, y), (x + w, y + h),
                (0, 255, 0), 1, cv2.LINE_AA
            )
        utils.show_image(img_out, f"{max_proposals} region proposals")

    def region_proposals(self, img, optim=True, show=False):
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
        cv2.setUseOptimized(optim)

        # Convert the image to numpy RGB format
        norm_img = utils.denormalize_image(img)
        np_img = utils.to_numpy(norm_img)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Call the specified region proposals model
        boxes = getattr(self, self.proposals_type)(rgb_img)

        # Show proposals
        if show:
            self.show_proposals(rgb_img, boxes)

        # Compute boxes, extract the corresponding proposals
        # and warp them to be compatible with the chosen backbone
        proposals = torch.zeros(
            (self.num_proposals, 3, self.backbone.in_height, self.backbone.in_width),
            dtype=torch.float32, device=self.device
        )
        coords = torch.zeros(
            (self.num_proposals, 4), dtype=torch.float32, device=self.device
        )
        for i, box in enumerate(boxes):
            box_coords = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            warped_proposal, x_scale, y_scale = utils.warp_with_context(
                img, box_coords,
                (self.backbone.in_height, self.backbone.in_width)
            )
            proposals[i] = utils.to_tensor(warped_proposal)
            coords[i] = torch.tensor(
                utils.scale_box(box_coords, x_scale=x_scale, y_scale=y_scale)
            )

        return len(boxes), proposals, coords

    def compute_loss(self, num_proposals, boxes_coords,
                     class_scores, bbox_corrections, targets):
        '''
        Compute one loss for the classification head and one
        for the bounding box regression head and also return outputs
        '''
        class_labels, bbox_predictions, bbox_ground_truths = [], [], []
        boxes = resize_boxes(
            targets["boxes"], targets["image_shape"][:-1],
            (self.backbone.in_height, self.backbone.in_width)
        )
        outputs = []

        # Iterate only over the number of computed proposals
        for i in range(num_proposals):

            # Get the actual proposal coordinates
            box = boxes_coords[i]

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
                    box[0] + box_width * bbox_corrections[i, class_label, 0],
                    box[1] + box_height * bbox_corrections[i, class_label, 1],
                    box_width * torch.exp(bbox_corrections[i, class_label, 2]),
                    box_height * torch.exp(bbox_corrections[i, class_label, 3])
                ])

                # Compute the regression head target
                regression_target = torch.tensor([
                    (gt[0] - box[0]) / box_width,
                    (gt[1] - box[1]) / box_height,
                    torch.log(gt_width / box_width),
                    torch.log(gt_height / box_height)
                ])

                # Store predicted box correction vs regression head target
                bbox_predictions.append(bbox_corrections[i, class_label])
                bbox_ground_truths.append(regression_target)

                # Store outputs
                predicted_label = torch.argmax(class_scores[i])
                outputs.append((predicted_label, output_box))

            # Store classification head target
            class_labels.append(class_label)

        # Compute one loss for the classification head and one
        # for the bounding box regression head
        loss_classifier = self.class_score_criterion(
            class_scores[:num_proposals, :],
            torch.tensor(class_labels)
        )
        loss_box_reg = 0
        if len(bbox_predictions) > 0:
            loss_box_reg = self.bbox_correction_criterion(
                torch.stack(bbox_predictions),
                torch.stack(bbox_ground_truths)
            ) * self.box_regression_weight

        #make_dot(loss_classifier).render("attached", format="png")

        return loss_classifier, loss_box_reg, outputs

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
            num_proposals, _, proposals_coords = self.region_proposals(
                warped_image
            )
            rescaled_features = self.roi_pool(features, [proposals_coords])
            flattened_features = torch.flatten(rescaled_features, start_dim=1)
            class_scores = self.class_score(flattened_features)
            bbox_corrections = self.bbox_correction(flattened_features).reshape(
                self.num_proposals, self.num_classes, 4
            )

            # Compute losses
            loss_classifier, loss_box_reg, _ = self.compute_loss(
                num_proposals, proposals_coords,
                class_scores, bbox_corrections, targets[i]
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
