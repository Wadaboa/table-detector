'''
This module deals with patching various types of pre-trained networks,
to be used as backbones for classification, object detection or segmentation tasks

If a classifier is passed to the patching function, it must be a subclass of torch.nn.Module
and it must have only the number of input features as required input parameters
'''


import torchvision


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


def _patch_backbone(backbone, backbone_family, classifier=None, freeze=True):
    '''
    Patch the given backbone model, by calling the appropriate patching function
    '''
    if freeze:
        freeze_module(backbone)
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
    ](pretrained=params.backbone.pretrained)
    return _patch_backbone(
        backbone, params.backbone.family,
        classifier=classifier, freeze=params.backbone.pretrained
    )
