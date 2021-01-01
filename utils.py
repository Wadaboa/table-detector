import random

import numpy as np
import torch
import PIL


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
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    if isinstance(img, PIL.Image.Image):
        return np.asarray(img).copy()
    return None


def flatten(a):
    """
    Given a multidimensional list/set/range, returns its flattened version
    """
    if isinstance(a, (list, set, range)):
        for s in a:
            yield from flatten(s)
    else:
        yield a


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
