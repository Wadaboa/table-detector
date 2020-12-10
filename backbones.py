'''
This module deals with patching various types of pre-trained networks,
to be used as backbones for object detection or segmentation tasks
'''


import torchvision


def freeze_backbone(backbone):
    '''
    Remove gradient information from the given backbone's parameters
    '''
    for param in backbone.parameters():
        param.requires_grad = False


def _patch_alexnet(backbone, classifier=None):
    '''
    Patch an AlexNet backbone, by removing the final classifier,
    or substituting it if one is given
    '''
    if classifier is not None:
        backbone.classifier[6] = classifier
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[1].in_features
    return features


def _patch_densenet(backbone, classifier=None):
    '''
    Patch a DenseNet backbone, by removing the final classifier,
    or substituting it if one is given
    '''
    if classifier is not None:
        backbone.classifier = classifier
        return backbone
    features = backbone.features
    features.output_channels = backbone.classifier.in_features
    return features


def _patch_mobilenet(backbone, classifier=None):
    '''
    Patch a MobileNet backbone, by removing the final classifier,
    or substituting it if one is given
    '''
    if classifier is not None:
        backbone.classifier[1] = classifier
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[1].in_features
    return features


def _patch_vgg(backbone, classifier=None):
    '''
    Patch a VGG backbone, by removing the final classifier,
    or substituting it if one is given
    '''
    if classifier is not None:
        backbone.classifier[6] = classifier
        return backbone
    features = backbone.features
    features.out_channels = backbone.classifier[0].in_features
    return features


def _patch_backbone(backbone, backbone_family, classifier=None):
    '''
    Patch the given backbone model, by calling the appropriate patching function
    '''
    freeze_backbone(backbone)
    patching_func_name = f'_patch_{backbone_family}'
    assert patching_func_name in globals(), (
        f'The backbone family `{backbone_family}` is not yet supported'
    )
    return globals()[patching_func_name](backbone, classifier=classifier)


def get_backbone(params, classifier=None):
    '''
    Get the patched backbone specified in the given parameters
    '''
    assert params.backbone.type in torchvision.models.__dict__, (
        f"The model `{params.backbone.type}` is not in PyTorch's model zoo"
    )
    backbone = torchvision.models.__dict__[
        params.backbone.type
    ](pretrained=True)
    return _patch_backbone(backbone, params.backbone.family, classifier=classifier)
