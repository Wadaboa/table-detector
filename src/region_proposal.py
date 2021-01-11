import os

import numpy as np
import cv2
import torch
import torch.nn as nn

import utils


class SelectiveSearch():
    '''
    J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders,
    “Selective search for object recognition,”
    International Journal of Computer Vision (IJCV), 2013
    '''

    _SS_STRATEGIES = {
        "color": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor(),
        "fill": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill(),
        "size": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize(),
        "texture": cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture(),
    }

    def __init__(self, max_proposals, strategies=["color", "fill", "size", "texture"], ss_type="fast"):
        self.ss_model = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.max_proposals = max_proposals
        strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
            *utils.get_dict_values(self._SS_STRATEGIES, strategies)
        )
        self.ss_model.addStrategy(strategy)
        self.ss_type = ss_type

    def _switch_type(self):
        '''
        Set selective search type 
        (to be done after setting the base image)
        '''
        # Fast but low recall
        if self.ss_type == "fast":
            self.ss_model.switchToSelectiveSearchFast()
        # High recall but slow
        elif self.ss_type == "quality":
            self.ss_model.switchToSelectiveSearchQuality()
        # No other types supported
        else:
            raise ValueError(
                "The given selective search type is not supported"
            )

    def __call__(self, img):
        # Set base image
        self.proposals_model.setBaseImage(img)

        # Set selective search type
        self._switch_type()

        # Compute at most the specified number of proposals
        return self.proposals_model.process()[:self.max_proposals]


class EdgeBoxes():
    '''
    C. L. Zitnick and P. Dollar,
    “Edge boxes: Locating object proposals from edges”,
    European Conference on Computer Vision (ECCV), 2014.
    '''

    def __init__(self, model_path, max_proposals, alpha=0.65, beta=0.75):
        assert os.path.exists(model_path), (
            "The given Edge Boxes model path does not exist"
        )
        self.eb_model = cv2.ximgproc.createStructuredEdgeDetection(model_path)
        self.max_proposals = max_proposals
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img, normalized=False, return_scores=False):
        # Convert to float and normalize
        float_img = np.float32(img)
        if not normalized:
            norm_img = float_img / 255.0
        else:
            norm_img = float_img
            assert np.all((float_img >= 0.0) & (float_img <= 1.0)), (
                "If the `normalized` flag is False, then the input image "
                "should already be normalized"
            )

        # Get the edges
        edges = self.eb_model.detectEdges(norm_img)

        # Create an orientation map
        orientation_map = self.eb_model.computeOrientation(edges)

        # Suppress edges
        edges = self.eb_model.edgesNms(edges, orientation_map)

        # Create edge boxes
        cv_edge_boxes = cv2.ximgproc.createEdgeBoxes()
        cv_edge_boxes.setMaxBoxes(self.max_proposals)
        cv_edge_boxes.setAlpha(self.alpha)
        cv_edge_boxes.setBeta(self.beta)

        # Compute boxes
        boxes, scores = cv_edge_boxes.getBoundingBoxes(edges, orientation_map)
        if return_scores:
            return boxes, scores
        return boxes


class RegionProposals():
    '''
    Region proposals class, which wraps a
    SelectiveSearch or EdgeBoxes instance
    '''

    def __init__(self, rp_type, rp_params,
                 min_size=None, max_size=None, optim=True):
        # Select the right region proposals model
        if rp_type == 'selective_search':
            self.rp_model = SelectiveSearch(
                rp_params.max_proposals,
                strategies=rp_params.selective_search.strategies,
                ss_type=rp_params.selective_search.type
            )
        elif rp_type == 'edge_boxes':
            self.rp_model = EdgeBoxes(
                rp_params.edge_boxes.model_path,
                rp_params.max_proposals,
                alpha=rp_params.edge_boxes.alpha,
                beta=rp_params.edge_boxes.beta
            )
        else:
            raise ValueError(
                "Invalid region proposals type"
            )

        # Minimum and maximum sizes used to warp
        # and resize proposals, if given
        self.min_size = min_size
        self.max_size = max_size

        # Set the optimized flag for OpenCV
        cv2.setUseOptimized(optim)

    @classmethod
    def show_proposals(cls, img, proposals, max_proposals=50):
        '''
        Draw proposals bounding boxes on top of the given image
        '''
        img_out = img.copy()
        for x, y, w, h in proposals[:max_proposals]:
            cv2.rectangle(
                img_out, (x, y), (x + w, y + h),
                (0, 255, 0), 1, cv2.LINE_AA
            )
        utils.show_image(img_out, f"{max_proposals} region proposals")

    def _forward(self, img, show=0):
        '''
        Use OpenCV to obtain a list of bounding box proposals
        to pass to the backbone
        (it expects to receive a normalized image in [0, 1])
        '''
        # Sanity checks
        assert isinstance(img, torch.Tensor), (
            "The input image for region proposals should be a PyTorch tensor"
        )
        assert len(img.shape) == 3, (
            "Proposals can only be computed on one image at a time"
        )
        assert img.sum().item() != 0, (
            "Found a totally black image when running region proposals"
        )

        # Convert the image to numpy RGB format
        norm_img = utils.denormalize_image(img)
        np_img = utils.to_numpy(norm_img)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Call the specified region proposals model
        boxes = self.rp_model(rgb_img)

        # Show proposals
        if show > 0:
            self.show_proposals(rgb_img, boxes, max_proposals=show)

        # Extract proposal and corresponding coordinates
        proposals = []
        coords = torch.zeros((len(boxes), 4))
        for i, box in enumerate(boxes):
            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            box_coords = torch.tensor(
                [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            )

            # Extract proposal as-is
            proposal = img[:, box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

            # Warp proposals with context
            if self.min_size is not None and self.max_size is not None:
                proposal, box_coords = utils.warp_with_context(
                    img, box_coords, self.min_size, self.max_size
                )

            # Store proposals and corresponding coordinates
            proposals.append(proposal)
            coords[i] = box_coords

        return proposals, coords

    def __call__(self, images, show=0):
        '''
        Compute region proposals for each PyTorch tensor in the
        input batch and return a list of tuples `(p, c)`,
        where tuple `i` represents proposals `p` and corresponding
        coordinates `c` for input image `i`
        '''
        if images.shape[0] == 1:
            return self._forward(images[0], show=show)

        proposals, coords = [], []
        for i in range(images.shape[0]):
            p, c = self._forward(images[i], show=show)
            proposals.append(p)
            coords.append(c)
        return proposals, coords
