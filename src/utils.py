import random
import traceback
import warnings
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import PIL


class Struct:
    '''
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    '''

    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        '''
        Return the only key in the Struct s.t. its value is True
        '''
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def deterministic_torch(seed):
    '''
    Set a deterministic behaviour in PyTorch
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    version = torch.__version__.split('+')[0]
    major, minor, patch = version.split('.')
    major, minor, patch = int(major), int(minor), int(patch)
    if major >= 1 and minor >= 7:
        torch.set_deterministic(True)


def fix_random(seed):
    '''
    Fix all the possible sources of randomness 
    '''
    random.seed(seed)
    np.random.seed(seed)
    deterministic_torch(seed)


def to_numpy(img):
    '''
    Convert the given image to the numpy format
    '''
    if isinstance(img, np.ndarray):
        return img
    # Convert to channel-last
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(img, PIL.Image.Image):
        return np.asarray(img).copy()
    return None


def to_tensor(img):
    '''
    Convert the given image to the PyTorch format
    '''
    if isinstance(img, torch.Tensor):
        return img
    # Convert to channel-first
    if isinstance(img, np.ndarray):
        return torch.from_numpy(img).permute(2, 0, 1)
    if isinstance(img, PIL.Image.Image):
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    return None


def get_image_size(img):
    '''
    Return the size of an image in format (width, height)
    '''
    if isinstance(img, torch.Tensor):
        return img.shape[2], img.shape[1]
    if isinstance(img, np.ndarray):
        return img.shape[1], img.shape[0]
    if isinstance(img, PIL.Image.Image):
        return img.size
    return None


def box_to_mask(img, box, mask_value=1):
    '''
    Convert a bounding box to a mask image,
    where the box is a list or tuple like (x1, y1, x2, y2)
    '''
    mask = np.zeros(img.shape, np.dtype('uint8'))
    mask[box[1]:box[3], box[0]:box[2], :] = mask_value
    return np.array(mask, dtype=np.float32)


def freeze_module(module):
    '''
    Remove gradient information from the given module's parameters
    '''
    for param in module.parameters():
        param.requires_grad = False


def show_image(img, window_name):
    '''
    Show the given image in a OpenCV window
    '''
    cv2.imshow(window_name, to_numpy(img))
    cv2.waitKey(0)


def cnn_output_size(model, input_size):
    '''
    Return the output size of a CNN model as a tuple (c, h, w)
    '''
    height, width = input_size
    dummy_image = torch.zeros((1, 3, height, width))
    dummy_output = model(dummy_image).squeeze(0)
    return dummy_output.shape


def normalize_image(img, max_value=255):
    '''
    Normalize the given image (or batch of images) to [0, 1]
    and return a float PyTorch tensor
    '''
    assert isinstance(img, torch.Tensor), (
        "Normalization can only be applied to PyTorch tensors"
    )
    return (img.float() / float(max_value))


def standardize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Standardize the given image (or batch of images) in each channel
    (default mean/std given by ImageNet parameters)
    '''
    assert isinstance(img, torch.Tensor), (
        "Standardization can only be applied to PyTorch tensors"
    )
    return TF.normalize(img, mean, std)


def denormalize_image(img, max_value=255):
    '''
    Denormalize the given image (or batch of images) to [0, 255]
    and return a uint8 PyTorch tensor
    '''
    assert isinstance(img, torch.Tensor), (
        "Denormalization can only be applied to PyTorch tensors"
    )
    assert img.max() <= 1, (
        "The input image is not normalized, cannot denormalize"
    )
    return (img.float() / float(max_value)).uint8()


def flatten(a):
    '''
    Given a multidimensional list/set/range, returns its flattened version
    '''
    if isinstance(a, (list, set, range)):
        for s in a:
            yield from flatten(s)
    else:
        yield a


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    '''
    Print detailed traceback of log warnings
    '''
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(
        warnings.formatwarning(
            message, category, filename, lineno, line
        )
    )


warnings.showwarning = warn_with_traceback
