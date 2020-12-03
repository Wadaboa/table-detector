import random

import numpy as np
import torch


def deterministic_torch():
    '''
    Set a deterministic behaviour in PyTorch
    '''
    version = torch.__version__.split('+')[0]
    major, minor, patch = version.split('.')
    major, minor, patch = int(major), int(minor), int(patch)

    if major >= 1 and minor >= 7:
        torch.set_deterministic(True)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fix_random(seed):
    '''
    Fix all the possible sources of randomness 
    '''
    random.seed(seed)
    np.random.seed(seed)
    deterministic_torch()
