import torch.nn as nn
import torchvision


def freeze_backbone(backbone, freeze):
    '''
    Remove gradient information from the given backbone's parameters
    '''
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False


def _patch_alexnet(backbone, num_classes):
    '''
    Patch an AlexNet backbone, by substituting the final classifier
    '''
    backbone.classifier[6] = nn.Linear(
        backbone.classifier[6].in_features, num_classes
    )
    return backbone


def _patch_densenet(backbone, num_classes):
    '''
    Patch a DenseNet backbone, by substituting the final classifier
    '''
    backbone.classifier = nn.Linear(
        backbone.classifier.in_features, num_classes
    )
    return backbone


def _patch_inception(backbone, num_classes):
    '''
    Patch an Inception backbone, by substituting the final classifier
    '''
    backbone.AuxLogits.fc = nn.Linear(
        backbone.AuxLogits.fc.in_features, num_classes
    )
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


def _patch_resnet(backbone, num_classes):
    '''
    Patch a ResNet backbone, by substituting the final classifier
    '''
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


def _patch_squeezenet(backbone, num_classes):
    '''
    Patch a SqueezeNet backbone, by substituting the final classifier
    '''
    backbone.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    backbone.num_classes = num_classes
    return backbone


def _patch_vgg(backbone, num_classes):
    '''
    Patch a VGG backbone, by substituting the final classifier
    '''
    backbone.classifier[6] = nn.Linear(
        backbone.classifier[6].in_features, num_classes
    )
    return backbone


def _patch_backbone(backbone, backbone_family, num_classes, freeze):
    '''
    Patch the given backbone model, by calling the appropriate patching function
    '''
    freeze_backbone(backbone, freeze)
    patching_func_name = f'_patch_{backbone_family}'
    assert patching_func_name in globals(), (
        f'The backbone family `{backbone_family}` is not yet supported'
    )
    return globals()[patching_func_name](backbone, num_classes)


def get_backbone(params, num_classes, pretrained=True):
    '''
    Get the patched backbone specified in the given parameters
    '''
    assert params.backbone.type in torchvision.models.__dict__, (
        f"The model {params.backbone.type} is not in PyTorch's model zoo"
    )
    backbone = torchvision.models.__dict__[
        params.backbone.type
    ](pretrained=pretrained)
    return _patch_backbone(
        backbone, params.backbone.family, num_classes, freeze=pretrained
    )
