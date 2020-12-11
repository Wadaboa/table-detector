'''
This module is used to instantiate different types of object detection networks
'''

from functools import partial

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

import backbones


def _get_fast_rcnn(params, num_classes):
    partial_classifier = partial(FastRCNNPredictor, num_classes=num_classes)
    return backbones.get_backbone(params, classifier=partial_classifier)


def _get_faster_rcnn(params, num_classes):
    return FasterRCNN(backbones.get_backbone(params), num_classes)


def get_detector(params, num_classes):
    detector_func_name = f'_get_{params.detector.type}'
    assert detector_func_name in globals(), (
        f'The detector type `{params.detector.type}` is not yet supported'
    )
    return globals()[detector_func_name](params, num_classes)
