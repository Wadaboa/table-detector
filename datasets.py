'''
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
from xml.dom import minidom

import cv2
import torch
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from pdf2image import convert_from_path


POUND_TO_INCH_FACTOR = 72


def pil_to_opencv(img):
    '''
    Convert the given RGB PIL image to the OpenCV format
    '''
    img = np.asarray(img)
    return img[:, :, ::-1].copy()


def get_image_size(img):
    '''
    Return the size of an image in format (width, height)
    '''
    if isinstance(img, np.ndarray):
        return img.shape[1], img.shape[0]
    elif isinstance(img, PIL.Image.Image):
        return img.size
    return None


def box_to_mask(img, box):
    '''
    Convert a bounding box to a mask image,
    where the box is a list or tuple like (x1, y1, x2, y2)
    '''
    mask = np.zeros(img.shape, np.dtype('uint8'))
    mask[box[1]:box[3], box[0]:box[2], :] = 255
    return mask


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

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.xml_files = sorted(
            glob.glob(os.path.join(self.root, "labeled", "*.xml"))
        )
        self.image_files = [
            os.path.join(self.root, "raw", f"{Path(f).stem}.bmp")
            for f in self.xml_files
        ]
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
        for i in portions_indexes[:-1]:

            # Extract string containing only points
            sub_content = xml_content[i:]
            start, end = (
                sub_content.find(self.BBOX_START),
                sub_content.find(self.BBOX_END)
            )
            points = sub_content[start + len(self.BBOX_START):end]

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
            img_size = get_image_size(self.get_image(index))
            coords[1] = img_size[1] - coords[1]
            coords[3] = img_size[1] - coords[3]

            # Add coordinates to the list of bounding boxes
            boxes.append(coords)

        return boxes

    def show_image(self, index):
        '''
        Open a window to show the image at the given index
        '''
        img = pil_to_opencv(self.get_image(index))
        cv2.imshow(f"Marmot dataset ~ Image #{index}", img)
        cv2.waitKey(0)

    def show_labeled_image(self, index):
        '''
        Open a window to show the image at the given index,
        alogn with the associated labels
        '''
        img = pil_to_opencv(self.get_image(index))
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
        t_boxes = torch.tensor(boxes, dtype=torch.float32)

        masks = []
        cv_img = pil_to_opencv(img)
        for box in boxes:
            mask = box_to_mask(cv_img, box)
            masks.append(mask)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        target["masks"] = torch.tensor(masks, dtype=torch.int64)
        target["image_id"] = torch.tensor([index])
        target["area"] = (
            (t_boxes[:, 3] - t_boxes[:, 1]) *
            (t_boxes[:, 2] - t_boxes[:, 0])
        )
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

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

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.xml_files = sorted(
            glob.glob(os.path.join(self.root, "*-reg.xml"))
        )
        self.image_files = [
            os.path.join(self.root, f"{Path(f).stem.replace('-reg', '')}.pdf")
            for f in self.xml_files
        ]
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
                img_size = get_image_size(imgs[int(page) - 1])
                coords[1] = img_size[1] - coords[1]
                coords[3] = img_size[1] - coords[3]
                coords[1], coords[3] = coords[3], coords[1]

            # Add coordinates to the list of bounding boxes
            boxes.setdefault(int(page) - 1, []).append(coords)

        return boxes

    def show_images(self, index):
        '''
        Open a window to show all the pages of the PDF file
        at the given index
        '''
        imgs = self.get_images(index)
        for i, img in enumerate(images):
            cv2.imshow(
                f"ICDAR 2013 dataset ~ Image #{index}, page #{i}",
                pil_to_opencv(img)
            )
            cv2.waitKey(0)

    def show_labeled_images(self, index):
        '''
        Open a window to show all the pages of the PDF file
        at the given index, along with the associated labels
        '''
        imgs = self.get_images(index)
        boxes = self.find_tables(index)
        for page, boxes_in_page in boxes.items():
            img = pil_to_opencv(imgs[page])
            for box in boxes_in_page:
                cv2.rectangle(
                    img,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 0, 0), 2
                )
            cv2.imshow(
                f"ICDAR 2013 dataset ~ Labeled image #{index}, page #{page}", img
            )
            cv2.waitKey(0)

    def __getitem__(self, index):
        imgs = self.get_images(index)
        boxes = self.find_tables(index)
        res = []
        for page, boxes_in_page in boxes.items():
            img = imgs[page]
            t_boxes = torch.tensor(boxes_in_page, dtype=torch.float32)

            masks = []
            cv_img = pil_to_opencv(img)
            for box in boxes_in_page:
                mask = box_to_mask(cv_img, box)
                masks.append(mask)

            target = {}
            target["boxes"] = t_boxes
            target["labels"] = torch.ones(
                (len(boxes_in_page),), dtype=torch.int64
            )
            target["masks"] = torch.tensor(masks, dtype=torch.int64)
            target["image_id"] = torch.tensor([index])
            target["area"] = (
                abs(t_boxes[:, 3] - t_boxes[:, 1]) *
                abs(t_boxes[:, 2] - t_boxes[:, 0])
            )
            target["iscrowd"] = torch.zeros(
                (len(boxes_in_page),), dtype=torch.int64
            )

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            res.append((img, target))

        return res

    def __len__(self):
        return len(self.image_files)


marmot = MarmotTableRecognitionDataset(
    root="datasets/marmot/table_recognition/data/english/positive"
)
# marmot.show_labeled_image(2)
print(marmot[2])

icdar = ICDAR13TableRecognitionDataset(
    root="datasets/icdar13/eu-dataset"
)
# for i in range(34):
# icdar.show_labeled_images(29)
# print(icdar[29])
