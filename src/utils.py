import random
import traceback
import datetime
import warnings
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import PIL


EXPLODE_WARNINGS = False


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

    def get_true_keys(self):
        '''
        Return all the keys in the Struct s.t. its value is True
        '''
        return [k for k, v in self.__dict__.items() if v == True]

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


def get_image_sizes(imgs):
    '''
    Return a list of (h, w) sizes (one for each input image)
    '''
    sizes = []
    for img in imgs:
        width, height = get_image_size(img)
        sizes.append((height, width))
    return sizes


def check_box_coords(box):
    '''
    Check that the coordinates of the given box are in
    the format (x1, y1, x2, y2) and that it is not degenerate
    '''
    assert box[0] <= box[2] and box[1] <= box[3], (
        "Found invalid box coordinates: "
        f"{box} not in (x1, y1, x2, y2) format"
    )


def box_area(box):
    '''
    Return the area of a bounding box, where the box
    should have the (x1, y1, x2, y2) format
    '''
    check_box_coords(box)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_to_mask(img, box, mask_value=1):
    '''
    Convert a bounding box to a mask image,
    where the box is a list or tuple like (x1, y1, x2, y2)
    '''
    check_box_coords(box)
    mask = np.zeros(img.shape, np.dtype('uint8'))
    mask[box[1]:box[3], box[0]:box[2], :] = mask_value
    return np.array(mask, dtype=np.float32)


def overlap_area(first_box, second_box):
    '''
    Return the area of overlap between two bounding boxes,
    with coords (x1, y1, x2, y2)
    '''
    check_box_coords(first_box)
    check_box_coords(second_box)

    # Get keypoints
    x_left = max(first_box[0], second_box[0])
    y_top = max(first_box[1], second_box[1])
    x_right = min(first_box[2], second_box[2])
    y_bottom = min(first_box[3], second_box[3])

    # No intersection found
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Found intersection
    return box_area([x_left, y_top, x_right, y_bottom])


def get_iou(first_box, second_box):
    '''
    Calculate the Intersection over Union (IoU) of two bounding boxes,
    with coords (x1, y1, x2, y2)
    '''
    # Compute intersection and areas
    intersection_area = overlap_area(first_box, second_box)
    first_box_area = box_area(first_box)
    second_box_area = box_area(second_box)

    # Compute IoU by taking the intersection area and dividing it
    # by the sum of the boxes areas minus their intersection area
    iou = (
        intersection_area /
        float(first_box_area + second_box_area - intersection_area)
    )
    assert iou >= 0.0 and iou <= 1.0, (
        f"Invalid IoU value of {iou}"
    )

    return iou


def most_overlapping_box(box, boxes, min_iou):
    '''
    Return the rectangle that has the most overlap with
    the ones in boxes
    '''
    results = []
    for i, other_box in enumerate(boxes):
        iou = get_iou(box, other_box)
        if iou > min_iou:
            results.append((i, other_box, iou))

    # At least one good match
    if len(results) > 0:
        return max(results, key=lambda t: t[1])

    # No good match
    return None


def extract_patch(img, center_index, patch_shape, pad=0):
    '''
    Extract a patch of the given shape from the given PyTorch image, centered around
    the specified position and pad external values with the given fill value
    '''
    _, m, n = img.shape
    wm, wn = patch_shape
    y_pad, x_pad = wm // 2, wn // 2
    yl, yu = center_index[0] - y_pad, center_index[0] + y_pad
    xl, xu = center_index[1] - x_pad, center_index[1] + x_pad
    if xl >= 0 and xu < n and yl >= 0 and yu < m:
        return img[:, yl:yu + 1, xl:xu + 1]

    padded_img = F.pad(
        img, (x_pad, x_pad, y_pad, y_pad),
        mode='constant', value=pad
    )
    oy = (padded_img.shape[1] // 2) - y_pad
    ox = (padded_img.shape[2] // 2) - x_pad
    return padded_img[:, oy:oy + wm, ox:ox + wn]


def warp_with_context(img, box, min_size, max_size, context=16, pad=0):
    '''
    Given a patch from the image, warp it anisotropically to the desired
    output shape (before warping, the patch is expanded to a new size
    that will result in the given amount of context in the warped frame) 
    '''
    check_box_coords(box)

    # Compute the output shape of the given image
    scale_factor = resize_factor(img, min_size, max_size)
    in_width, in_height = get_image_size(img)
    out_height, out_width = (
        in_height * scale_factor,
        in_width * scale_factor
    )

    # Compute the expanded patch shape
    box_height, box_width = (
        box[3] - box[1],
        box[2] - box[0]
    )
    height_offset = (box_height / (out_height - context)) * (context * 2)
    width_offset = (box_width / (out_width - context)) * (context * 2)
    new_box_shape = (
        int(box_height + height_offset),
        int(box_width + width_offset)
    )

    # Extract the expanded box from the image,
    # by padding external values
    center_index = (
        (box[1] + box[3]) // 2,
        (box[0] + box[2]) // 2
    )
    expanded_box = extract_patch(
        img, center_index, new_box_shape, pad=pad
    )

    # Compute resizing scales
    y_scale = new_box_shape[0] / out_height
    x_scale = new_box_shape[1] / out_width

    # Resize the extracted box to the desired shape
    scaled_img = F.interpolate(
        expanded_box[None], scale_factor=scale_factor,
        mode='bilinear', recompute_scale_factor=True,
        align_corners=False
    )[0]
    scaled_box = torch.tensor(
        scale_box(box, x_scale=x_scale, y_scale=y_scale)
    )

    return scaled_img, scaled_box


def resize_factor(img, min_size, max_size):
    '''
    Compute the scaling factor to be applied to
    the given image, in order to have dimensions 
    between `(min_size, min_size)` and `(max_size, max_size)`
    '''
    shape = torch.tensor(img.shape[-2:])
    min_shape = float(torch.min(shape))
    max_shape = float(torch.max(shape))
    scale_factor = min_size / min_shape
    if max_size * scale_factor > max_size:
        scale_factor = max_size / max_shape
    return scale_factor


def scale_box(box, x_scale, y_scale):
    '''
    Scale the given box with the given x/y scales
    '''
    check_box_coords(box)
    return [
        box[0] * x_scale, box[1] * y_scale,
        box[2] * x_scale, box[3] * y_scale
    ]


def get_dict_values(dictionary, keys):
    '''
    Return a list composed by the values corresponding to
    the given keys in the given dictionary
    '''
    return [dictionary[key] for key in keys]


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
    cv2.destroyAllWindows()


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
    assert int(img.min()) >= 0 and int(img.max()) <= 1, (
        f"The input image is not normalized, cannot denormalize "
        f"(image in range [{img.min()}, {img.max()}])"
    )
    return (img.float() * float(max_value)).type(torch.uint8)


def destandardize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Denstandardize the given image (or batch of images) in each channel
    (default mean/std given by ImageNet parameters)
    '''
    assert isinstance(img, torch.Tensor), (
        "Destandardization can only be applied to PyTorch tensors"
    )
    mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std, dtype=img.dtype, device=img.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    img.mul_(std).add_(mean)
    return img


def flatten(a):
    '''
    Given a multidimensional list/set/range, returns its flattened version
    '''
    if isinstance(a, (list, set, range)):
        for s in a:
            yield from flatten(s)
    else:
        yield a


def now():
    '''
    Returns the current date and time
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


if EXPLODE_WARNINGS:
    warnings.showwarning = warn_with_traceback
