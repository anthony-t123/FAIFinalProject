import json
import os
import sys

import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        # this should be the path to the data, so either training, or validation dataset
        self.root = root
        self.transforms = transforms
        # Load the images
        self.imgs = list(sorted(os.listdir(root)))
        if 'train' in root:
            self.boxes = json.load(open('../Data/raw/det_train.json'))
        elif 'val' in root:
            self.boxes = json.load(open('../Data/raw/det_val.json'))
        else:
            raise AttributeError(f'Data load for {root} not supported! Use train or val datasets.')
        self.boxes = {item['name']: item for item in self.boxes}

        # Remove bounding boxes that have no labels
        copy = self.boxes.copy()
        for name, box in self.boxes.items():
            if box.get('labels') is None:
                print(f'Removing {name} in {root}')
                del copy[name]

        self.boxes = copy

        for image_name in self.imgs:
            if image_name not in self.boxes.keys():
                self.imgs.remove(image_name)
                print(f'Removing {image_name} in {root}')

        print(f'Loaded {len(self.boxes.keys())} images with labels for {root}')

        # this will be used to convert the labels from their string identifier into an int representation
        self.labelConverter = {
            'pedestrian': 0,
            'rider': 1,
            'car': 2,
            'truck': 3,
            'bus': 4,
            'train': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic light': 8,
            'traffic sign': 9
        }

    def __getitem__(self, idx):
        # Name of the image file
        image_name = self.imgs[idx]
        image_path = os.path.join(self.root, image_name)
        # Get the image, and convert to RGB
        img = Image.open(image_path).convert('RGB')

        # print(f'Image size: {img.size}')

        if self.boxes.get(image_name) is None or self.boxes.get(image_name).get('labels') is None:
            print(f'Image {image_name} has no labels!')
            print(f'in images: {image_name in self.imgs}, in boxes: {image_name in self.boxes.keys()}')
            idx += 1
            image_name = self.imgs[idx]
            image_path = os.path.join(self.root, image_name)
            # Get the image, and convert to RGB
            img = Image.open(image_path).convert('RGB')
            print(f'Skipped to {image_name}')

        labels = self.boxes.get(image_name).get('labels')
        labels = {item['id']: item for item in labels}
        # print(f'Index: {idx}, Image name: {image_name}')

        final_boxes = []
        final_labels = []
        for id, label in labels.items():
            box = label.get('box2d')
            xmin = box.get('x1')
            xmax = box.get('x2')
            ymin = box.get('y1')
            ymax = box.get('y2')
            final_boxes.append([xmin, ymin, xmax, ymax])
            # use the integer representation of the label
            # print(f'Category: {label.get("category")}')
            # Default to truck for other vehicles
            final_labels.append(self.labelConverter.get(label.get('category'), 3))

        final_boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
        # print(f'Image: {image_name}, Type: {type(final_labels)}, Labels: {final_labels}')
        final_labels = torch.as_tensor(final_labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = final_boxes
        target["labels"] = final_labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
