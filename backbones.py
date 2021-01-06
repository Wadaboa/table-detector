'''
This module deals with patching various types of pre-trained networks,
to be used as backbones for classification, object detection or segmentation tasks

If a classifier is passed to the patching function, it must be a subclass of torch.nn.Module
and it must have only the number of input features as required input parameters
'''


import torch.nn as nn
import torchvision

import utils


class Backbone(nn.Module):
    '''
    Generic backbone class based on PyTorch's models zoo, 
    to be used with classification, object detection
    and segmentation tasks
    '''

    AVAILABLE_MODELS = torchvision.models.__dict__

    def __init__(self, params, classifier=None):
        super(Backbone, self).__init__()
        self.backbone_type = params.backbone.type
        self.backbone_family = params.backbone.family
        self.pretrained = params.backbone.pretrained
        self.in_height, self.in_width = (
            params.backbone.input_size.height,
            params.backbone.input_size.height,
        )

        # Get the backbone model and eventually freeze it
        self.model = self.get_backbone(self.backbone_type, self.pretrained)
        if self.pretrained:
            freeze_module(self.model)

        # Extract only the features part from the backbone
        # or attach the given classifier to the end of it
        self.model = self.patch_backbone(
            self.model, self.backbone_family, classifier=classifier
        )

        # Save backbone output shape and size
        # If a classifier is not provided, then the output of the backbone
        # is a feature map with shape (c, h, w)
        if classifier is None:
            self.out_channels = self.model.out_channels
            self.out_height, self.out_width = utils.cnn_output_size(
                self.model, (self.in_height, self.in_width)
            )
            self.out_size = (
                self.out_height * self.out_width * self.out_channels
            )
        # If a classifier is provided, then the final size is given
        # by the number of features of the last FC layer
        # (ideally, the number of classes)
        else:
            self.out_size = classifier.out_features

    @classmethod
    def get_backbone(cls, backbone_type, pretrained=True):
        '''
        Get the backbone type specified in the given parameters
        '''
        assert backbone_type in cls.AVAILABLE_MODELS, (
            f"The model `{backbone_type}` is not in PyTorch's model zoo"
        )
        return cls.AVAILABLE_MODELS[backbone_type](
            pretrained=pretrained
        )

    @classmethod
    def patch_backbone(cls, backbone, backbone_family, classifier=None):
        '''
        Patch the given backbone model, by calling the appropriate patching function
        '''
        patching_func_name = f'_patch_{backbone_family}'
        assert patching_func_name in globals(), (
            f'The backbone family `{backbone_family}` is not yet supported'
        )
        return globals()[patching_func_name](backbone, classifier=classifier)

    def forward(self, inputs):
        return self.model.forward(inputs)


def freeze_module(module):
    '''
    Remove gradient information from the given module's parameters
    '''
    for param in module.parameters():
        param.requires_grad = False


def _patch_alexnet(backbone, classifier=None):
    '''
    Patch an AlexNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 224x224, with 3 channels
    '''
    if classifier is not None:
        backbone.classifier[6] = classifier(backbone.classifier[6].in_features)
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[1].in_features
    return features


def _patch_densenet(backbone, classifier=None):
    '''
    Patch a DenseNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if classifier is not None:
        backbone.classifier = classifier(backbone.classifier.in_features)
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier.in_features
    return features


def _patch_mobilenet(backbone, classifier=None):
    '''
    Patch a MobileNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if classifier is not None:
        backbone.classifier[1] = classifier(backbone.classifier[1].in_features)
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[1].in_features
    return features


def _patch_vgg(backbone, classifier=None):
    '''
    Patch a VGG backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if classifier is not None:
        backbone.classifier[6] = classifier(backbone.classifier[6].in_features)
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[0].in_features
    return features
