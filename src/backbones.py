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
        self.in_channels = 3
        self.in_height, self.in_width = (
            params.backbone.input_size.exact.height,
            params.backbone.input_size.exact.height,
        )

        # Get the backbone model
        self.model = self.get_backbone(self.backbone_type, self.pretrained)

        # Extract only the features part from the backbone
        # or attach the given classifier to the end of it
        self.model = self.patch_backbone(
            self.model, self.backbone_family,
            pretrained=self.pretrained, classifier=classifier
        )

        # Save backbone output shape and size
        # If a classifier is not provided, then the output of the backbone
        # is a feature map with shape (c, h, w)
        if classifier is None:
            self.out_channels, self.out_height, self.out_width = utils.cnn_output_size(
                self.model, (self.in_height, self.in_width)
            )
        # If a classifier is provided, then the final size is given
        # by the number of features of the last FC layer
        # (ideally, the number of classes)
        else:
            self.out_channels = classifier.out_features
            self.out_height, self.out_width = 1, 1

        # Store whole output size
        self.out_size = (
            self.out_height * self.out_width * self.out_channels
        )

        # Store ImageNet mean/std
        self.image_mean = params.backbone.imagenet_params.mean
        self.image_std = params.backbone.imagenet_params.std

    @classmethod
    def get_backbone(cls, backbone_type, pretrained=True):
        '''
        Get the backbone type specified in the given parameters
        '''
        error_msg = f"The model `{backbone_type}` is not in PyTorch's model zoo"
        assert backbone_type in cls.AVAILABLE_MODELS, error_msg
        try:
            return cls.AVAILABLE_MODELS[backbone_type](
                pretrained=pretrained
            )
        except:
            raise ValueError(error_msg)

    @classmethod
    def patch_backbone(cls, backbone, backbone_family, pretrained=True, classifier=None):
        '''
        Patch the given backbone model, by calling the appropriate patching function
        '''
        error_msg = f'The backbone family `{backbone_family}` is not yet supported'
        patching_func_name = f'_patch_{backbone_family}'
        assert patching_func_name in globals(), error_msg
        try:
            return globals()[patching_func_name](
                backbone, pretrained=pretrained, classifier=classifier
            )
        except:
            raise ValueError(error_msg)

    def __str__(self):
        return (
            f"Backbone(type={self.backbone_type}, family={self.backbone_family}, "
            f"pretrained={self.pretrained}, "
            f"input_shape={self.in_channels}x{self.in_height}x{self.in_width}, "
            f"output_shape={self.out_channels}x{self.out_height}x{self.out_width})"
        )

    def __repr__(self):
        return self.__str__()

    def forward(self, inputs):
        return self.model.forward(inputs)


def _patch_alexnet(backbone, pretrained=True, classifier=None):
    '''
    Patch an AlexNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 224x224, with 3 channels
    '''
    if pretrained:
        utils.freeze_module(backbone.features)
    if classifier is not None:
        backbone.classifier[6] = classifier(backbone.classifier[6].in_features)
        return backbone
    return backbone.features


def _patch_densenet(backbone, pretrained=True, classifier=None):
    '''
    Patch a DenseNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if pretrained:
        utils.freeze_module(backbone.features)
    if classifier is not None:
        backbone.classifier = classifier(backbone.classifier.in_features)
        return backbone
    return backbone.features


def _patch_mobilenet(backbone, pretrained=True, classifier=None):
    '''
    Patch a MobileNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if pretrained:
        utils.freeze_module(backbone.features)
    if classifier is not None:
        backbone.classifier[1] = classifier(backbone.classifier[1].in_features)
        return backbone
    return backbone.features


def _patch_resnet(backbone, pretrained=True, classifier=None):
    '''
    Patch a ResNet backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    features = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    if pretrained:
        utils.freeze_module(features)
    if classifier is not None:
        return nn.Sequential(
            features,
            backbone.avgpool,
            nn.Flatten(start_dim=1),
            classifier(backbone.fc.in_features)
        )
    return features


def _patch_vgg(backbone, pretrained=True, classifier=None):
    '''
    Patch a VGG backbone, by removing the final classifier,
    or substituting it if one is given

    Expected input shape: At least 32x32, with 3 channels
    '''
    if pretrained:
        utils.freeze_module(backbone.features)
    if classifier is not None:
        backbone.classifier[6] = classifier(backbone.classifier[6].in_features)
        return backbone
    return backbone.features
