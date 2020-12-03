import glob
import struct
import binascii
import re
import os
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
        self.xml_files = glob.glob(os.path.join(self.root, "labeled", "*.xml"))
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
                coords[c] = (coords[c] / POUND_TO_INCH_FACTOR) * self.DPI

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
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0), 2
            )
        cv2.imshow(f"Marmot dataset ~ Labeled image #{index}", img)
        cv2.waitKey(0)

    def __getitem__(self, index):
        img = self.get_image(index)
        boxes = self.find_tables(index)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = None
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)


class ICDAR13TableRecognitionDataset(Dataset):
    '''
    ICDAR 2013 dataset for table recognition.
    See https://roundtrippdf.com/en/data-extraction/pdf-table-recognition-dataset/
    '''

    DPI = 96
    BOX_LABEL = 'bounding-box'
    BOX_X1_LABEL, BOX_Y1_LABEL, BOX_X2_LABEL, BOX_Y2_LABEL = (
        'x1', 'y1', 'x2', 'y2'
    )

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.xml_files = glob.glob(os.path.join(self.root, "*-reg.xml"))
        self.image_files = [
            os.path.join(self.root, f"{Path(f).stem.replace('-reg', '')}.pdf")
            for f in self.xml_files
        ]
        self.images = dict()

    def get_image(self, index):
        '''
        Return the image associated with the given index
        (load it from the pdf file if not already loaded)
        '''
        if index not in self.images:
            self.images[index] = convert_from_path(
                self.image_files[index], dpi=self.DPI
            )[0]

        return self.images[index]

    def find_tables(self, index):
        '''
        Extract table bounding boxes from the XML
        associated with the given index
        '''
        # Read the XML file associated with the given index
        # and extract bounding boxes keys
        xml_content = minidom.parse(self.xml_files[index])
        items = xml_content.getElementsByTagName(BOX_LABEL)

        # Extract coordinates for each bounding box
        boxes = []
        for item in items:

            # Extract the bounding box coordinates
            x1 = float(item.attributes[self.BOX_X1_LABEL].value)
            y1 = float(item.attributes[self.BOX_Y1_LABEL].value)
            x2 = float(item.attributes[self.BOX_X2_LABEL].value)
            y2 = float(item.attributes[self.BOX_Y2_LABEL].value)
            coords = [x1, y1, x2, y2]

            # From "pound" to inch to pixel values
            for c in range(len(coords)):
                coords[c] = (coords[c] / POUND_TO_INCH_FACTOR) * self.DPI

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
        cv2.imshow(f"ICDAR 2013 dataset ~ Image #{index}", img)
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
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0), 2
            )
        cv2.imshow(f"ICDAR 2013 dataset ~ Labeled image #{index}", img)
        cv2.waitKey(0)

    def __getitem__(self, index):
        img = self.get_image(index)
        boxes = self.find_tables(index)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = None
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)


marmot = MarmotTableRecognitionDataset(
    root="datasets/marmot/table_recognition/data/english/positive"


)
marmot.show_labeled_image(2)
print(marmot[2])

# icdar = ICDAR13TableRecognitionDataset(
#    root="datasets/icdar13/eu-dataset"
# )
# icdar.show_labeled_image(31)
# print(icdar[31])
