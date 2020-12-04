'''
Define some PyTorch transformations to be applied on document images
(some of them assume a white background and black text)

See https://github.com/AyanGadpal/Document-Image-Augmentation
'''


import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF

import utils


class DocumentBrightnessTransform(nn.Module):
    '''
    Change the brightness of the given image,
    based on the magnitude and sign of the `alpha` value
    '''

    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, img):
        tmp_img = utils.to_numpy(img)
        transformed_img = cv2.add(tmp_img, np.array([self.alpha]))
        return TF.to_pil_image(transformed_img)

    def __repr__(self):
        return (
            self.__class__.__name__ +
            '(alpha={0})'.format(self.alpha)
        )


class DocumentDilationTransform(nn.Module):
    '''
    Apply a simple dilation morphological operation,
    so as to obtain thicker characters in text
    '''

    def __init__(self, kernel, pixel_range):
        super().__init__()
        self.kernel = kernel
        self.pixel_range = pixel_range

    def forward(self, img):
        tmp_img = utils.to_numpy(img)
        _, mask = cv2.threshold(
            tmp_img, self.pixel_range[0], self.pixel_range[1], cv2.THRESH_BINARY_INV
        )
        dilated = cv2.dilate(mask, self.kernel, iterations=1)
        transformed_img = cv2.bitwise_not(dilated)
        return TF.to_pil_image(transformed_img)

    def __repr__(self):
        return (
            self.__class__.__name__ +
            '(kernel={0}, pixel_range={1})'.format(
                self.kernel, self.pixel_range
            )
        )


class DocumentBackgroundColorTranform(nn.Module):
    '''
    Change the background color of the given document image,
    assuming a white background
    '''

    def __init__(self, color):
        super().__init__()
        self.color = color

    def forward(self, img):
        tmp_img = utils.to_numpy(img)
        if len(tmp_img.shape) < 3:
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        transformed_img = tmp_img.copy()
        transformed_img[mask == 0] = self.color
        return TF.to_pil_image(transformed_img)

    def __repr__(self):
        return (
            self.__class__.__name__ +
            '(color={0})'.format(self.color)
        )


class DocumentSmudgeTransform(nn.Module):
    '''
    In the smudge transform, we transform the original image to
    spread the black pixel regions and make it look like a kind
    of smeary blurred black pixel region
    (See https://arxiv.org/pdf/2004.12629.pdf)
    '''

    def __init__(self, pixel_range):
        super().__init__()
        self.pixel_range = pixel_range

    def basic_transform(self, img):
        _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_not(mask)
        return img

    def forward(self, img):
        tmp_img = utils.to_numpy(img)
        b, g, r = cv2.split(tmp_img)
        b, g, r = (
            self.basic_transform(b),
            self.basic_transform(g),
            self.basic_transform(r)
        )
        b, g, r = (
            cv2.distanceTransform(b, cv2.DIST_L2, 5),
            cv2.distanceTransform(g, cv2.DIST_L1, 5),
            cv2.distanceTransform(r, cv2.DIST_C, 5)
        )
        b, g, r = (
            cv2.normalize(b, b, 0, 1.0, cv2.NORM_MINMAX),
            cv2.normalize(g, g, 0, 1.0, cv2.NORM_MINMAX),
            cv2.normalize(r, r, 0, 1.0, cv2.NORM_MINMAX)
        )
        dist = cv2.merge((b, g, r))
        dist = cv2.normalize(dist, dist, 0, 4.0, cv2.NORM_MINMAX)
        dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)

        data = dist.astype(np.float64) / 4.0
        data = 1800 * data

        transformed_img = data.astype(np.uint8)
        return TF.to_pil_image(transformed_img)

    def __repr__(self):
        return (
            self.__class__.__name__ +
            '(pixel_range={0})'.format(self.pixel_range)
        )
