'''
This module contains the implementation of various table detection
and structure recognition datasets, by leveraging PyTorch Dataset objects

When referring to x1, y1, x2, y2 in this module we mean to
represent a bounding box like the following:
x1,y1 ------
|          |
|          |
|          |
--------x2,y2
'''


import glob
import struct
import binascii
import re
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from pdf2image import convert_from_path

import utils


POUND_TO_INCH_FACTOR = 72


class MarmotTableRecognitionDataset(Dataset):
    '''
    Marmot dataset for table recognition.
    See https://www.icst.pku.edu.cn/cpdp/sjzy/index.htm,
    section `Dataset for table recognition`
    '''

    DPI = 96
    TABLE_LABEL = 'Label="TableBody"'
    BBOX_START = 'BBox="'
    BBOX_END = '" CLIDs'

    def __init__(self, roots, transforms=None):
        if not isinstance(roots, list):
            roots = [roots]
        self.xml_files = []
        self.image_files = []
        for root in roots:
            assert isinstance(root, str)
            assert os.path.exists(os.path.abspath(root))
            xml_files = glob.glob(os.path.join(root, "labeled", "*.xml"))
            self.xml_files.extend(xml_files)
            for xml_file in xml_files:
                image_file = os.path.join(
                    root, "raw", f"{Path(xml_file).stem}.bmp"
                )
                self.image_files.append(image_file)
        self.transforms = transforms
        self.images = dict()

    def get_image(self, index):
        '''
        Return the image associated with the given index
        (load it from the pdf file if not already loaded)
        '''
        if index not in self.images:
            self.images[index] = Image.open(
                self.image_files[index]
            ).convert("RGB")

        return self.images[index]

    def find_tables(self, index):
        '''
        Extract table bounding boxes from the XML
        associated with the given index
        '''
        # Read the XML file associated with the given index
        xml_file = open(self.xml_files[index], "r")
        xml_content = xml_file.read()

        # Find starting index of all the table occurrences
        portions_indexes = [
            l.start()
            for l in re.finditer(self.TABLE_LABEL, xml_content)
        ]

        # Extract coordinates for each table
        boxes = []
        for i in portions_indexes[: -1]:

            # Extract string containing only points
            sub_content = xml_content[i:]
            start, end = (
                sub_content.find(self.BBOX_START),
                sub_content.find(self.BBOX_END)
            )
            points = sub_content[start + len(self.BBOX_START): end]

            # From little endian to "pounds" unit
            coords = []
            for coord in points.split(' '):
                if not coord:
                    break
                coords.append(
                    struct.unpack('>d', binascii.unhexlify(coord))[0]
                )

            # Discard "wrong" bounding boxes
            if len(coords) != 4:
                continue

            # From "pound" to inch to pixel values
            for c in range(len(coords)):
                coords[c] = int(
                    (coords[c] / POUND_TO_INCH_FACTOR) * self.DPI
                )

            # Translate origin from bottom-left corner
            # to top-left corner of the image
            img_size = utils.get_image_size(self.get_image(index))
            coords[1] = img_size[1] - coords[1]
            coords[3] = img_size[1] - coords[3]

            # Add coordinates to the list of bounding boxes
            boxes.append(coords)

        return boxes

    def show_image(self, index, original=False):
        '''
        Open a window to show the image at the given index
        '''
        img = self.get_image(index)
        if self.transforms is not None and not original:
            img = self.transforms(img)
        cv2.imshow(
            f"Marmot dataset ~ Image #{index}",
            utils.to_numpy(img)
        )
        cv2.waitKey(0)

    def show_labeled_image(self, index, original=False):
        '''
        Open a window to show the image at the given index,
        along with the associated labels
        '''
        img = self.get_image(index)
        if self.transforms is not None and not original:
            img = self.transforms(img)
        img = utils.to_numpy(img)
        boxes = self.find_tables(index)
        for box in boxes:
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[2], box[3]),
                (255, 0, 0), 2
            )
        cv2.imshow(f"Marmot dataset ~ Labeled image #{index}", img)
        cv2.waitKey(0)

    def __getitem__(self, index):
        img = self.get_image(index)
        boxes = self.find_tables(index)
        t_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Compute masks from boxes
        masks = []
        cv_img = utils.to_numpy(img)
        for box in boxes:
            mask = utils.box_to_mask(cv_img, box)
            masks.append(mask)

        # Build the target dict
        target = {}
        target["boxes"] = t_boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        target["masks"] = torch.tensor(masks, dtype=torch.float32)
        target["image_id"] = torch.tensor([index])
        target["image_shape"] = torch.tensor(cv_img.shape, dtype=torch.int64)
        target["area"] = (
            torchvision.ops.box_area(target["boxes"]) if len(boxes) > 0
            else torch.zeros((len(boxes),), dtype=torch.int64)
        )
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # Apply the given transformations
        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        # Ensure that the image is a normalized PyTorch tensor
        img = utils.normalize_image(utils.to_tensor(img))
        return [(img, target)]

    def __len__(self):
        return len(self.image_files)


class ICDAR13TableRecognitionDataset(Dataset):
    '''
    ICDAR 2013 dataset for table recognition.
    See https://roundtrippdf.com/en/data-extraction/pdf-table-recognition-dataset/
    '''

    DPI = 96
    TABLE_LABEL = 'table'
    REGION_LABEL = 'region'
    PAGE_LABEL = 'page'
    BOX_LABEL = 'bounding-box'
    BOX_X1_LABEL, BOX_Y1_LABEL, BOX_X2_LABEL, BOX_Y2_LABEL = (
        'x1', 'y1', 'x2', 'y2'
    )

    def __init__(self, roots, transforms=None):
        if not isinstance(roots, list):
            roots = [roots]
        self.xml_files = []
        self.image_files = []
        for root in roots:
            assert isinstance(root, str)
            assert os.path.exists(os.path.abspath(root))
            xml_files = glob.glob(os.path.join(root, "*-reg.xml"))
            self.xml_files.extend(xml_files)
            for xml_file in xml_files:
                image_file = os.path.join(
                    root, f"{Path(xml_file).stem.replace('-reg', '')}.pdf"
                )
                self.image_files.append(image_file)

        self.transforms = transforms
        self.images = dict()

    def get_images(self, index):
        '''
        Return the images associated with the given index
        (load the pages from the pdf file if not already loaded)
        '''
        if index not in self.images:
            self.images[index] = convert_from_path(
                self.image_files[index], dpi=self.DPI
            )

        return self.images[index]

    def find_tables(self, index):
        '''
        Extract table bounding boxes from the XML
        associated with the given index
        '''
        # Read the XML file associated with the given index
        # and extract bounding boxes keys
        xml_content = ET.parse(self.xml_files[index])
        xml_root = xml_content.getroot()
        boxes = dict()

        # Iterate over tables
        imgs = self.get_images(index)
        for table in xml_root.findall(self.TABLE_LABEL):
            # A table could span multiple pages
            # (so it could have multiple regions)
            for region in table.findall(self.REGION_LABEL):
                page = region.get(self.PAGE_LABEL)
                box = region.find(self.BOX_LABEL)

                # Extract the bounding box coordinates
                x1 = float(box.get(self.BOX_X1_LABEL))
                y1 = float(box.get(self.BOX_Y1_LABEL))
                x2 = float(box.get(self.BOX_X2_LABEL))
                y2 = float(box.get(self.BOX_Y2_LABEL))
                coords = [x1, y1, x2, y2]

                # From "pound" to inch to pixel values
                for c in range(len(coords)):
                    coords[c] = int(
                        (coords[c] / POUND_TO_INCH_FACTOR) * self.DPI
                    )

                # Translate origin from bottom-left corner
                # to top-left corner of the image
                img_size = utils.get_image_size(imgs[int(page) - 1])
                coords[1] = img_size[1] - coords[1]
                coords[3] = img_size[1] - coords[3]
                coords[1], coords[3] = coords[3], coords[1]

                # Add coordinates to the list of bounding boxes
                boxes.setdefault(int(page) - 1, []).append(coords)

        return boxes

    def show_images(self, index, original=False):
        '''
        Open a window to show all the pages of the PDF file
        at the given index
        '''
        imgs = self.get_images(index)
        for i, img in enumerate(imgs):
            actual_img = img.copy()
            if self.transforms is not None and not original:
                actual_img = self.transforms(actual_img)
            cv2.imshow(
                f"ICDAR 2013 dataset ~ Image #{index}, page #{i}",
                utils.to_numpy(actual_img)
            )
            cv2.waitKey(0)

    def show_labeled_images(self, index, original=False):
        '''
        Open a window to show all the pages of the PDF file
        at the given index, along with the associated labels
        '''
        imgs = self.get_images(index)
        boxes = self.find_tables(index)
        for page, boxes_in_page in boxes.items():
            actual_img = imgs[page].copy()
            if self.transforms is not None and not original:
                actual_img = self.transforms(actual_img)
            actual_img = utils.to_numpy(actual_img)
            for box in boxes_in_page:
                cv2.rectangle(
                    actual_img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 0, 0), 2
                )
            cv2.imshow(
                f"ICDAR 2013 dataset ~ Labeled image #{index}, page #{page}",
                actual_img
            )
            cv2.waitKey(0)

    def __getitem__(self, index):
        imgs = self.get_images(index)
        boxes = self.find_tables(index)
        res = []
        for page, boxes_in_page in boxes.items():
            img = imgs[page]
            t_boxes = torch.tensor(
                boxes_in_page, dtype=torch.float32
            ).reshape(-1, 4)

            # Compute masks from boxes
            masks = []
            cv_img = utils.to_numpy(img)
            for box in boxes_in_page:
                mask = utils.box_to_mask(cv_img, box)
                masks.append(mask)

            # Build the target dict
            target = {}
            target["boxes"] = t_boxes
            target["labels"] = torch.ones(
                (len(boxes_in_page),), dtype=torch.int64
            )
            target["masks"] = torch.tensor(masks, dtype=torch.float32)
            target["image_id"] = torch.tensor([index])
            target["image_shape"] = torch.tensor(
                cv_img.shape, dtype=torch.int64
            )
            target["area"] = (
                torchvision.ops.box_area(target["boxes"]) if len(boxes_in_page) > 0
                else torch.zeros((len(boxes_in_page),), dtype=torch.int64)
            )
            target["iscrowd"] = torch.zeros(
                (len(boxes_in_page),), dtype=torch.int64
            )

            # Apply the given transformations
            if self.transforms is not None:
                img = self.transforms(img)
                target = self.transforms(target)

            # Ensure that the image is a normalized PyTorch tensor
            img = utils.normalize_image(utils.to_tensor(img))
            res.append((img, target))

        return res

    def __len__(self):
        return len(self.image_files)


DATASETS = {
    'marmot': MarmotTableRecognitionDataset,
    'icdar13': ICDAR13TableRecognitionDataset
}
