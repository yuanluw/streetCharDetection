# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 12:18, matt '

import sys
sys.path.append('..')

import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tfs

import config


def get_path(index="train"):
    if index == "train":
        path = os.path.join(config.dataset_path, "mchar_train")
        label_path = os.path.join(config.dataset_path, "mchar_train.json")
    elif index == "val":
        path = os.path.join(config.dataset_path, "mchar_val")
        label_path = os.path.join(config.dataset_path, "mchar_val.json")
    elif index == "test":
        path = os.path.join(config.dataset_path, "mchar_test_a")

    images = os.listdir(path)
    sorted(images)
    labels = None
    if index != "test":
        _json = json.load(open(label_path))
        labels = [_json[x]["label"] for x in images]

    images_path = [os.path.join(path, x) for x in images]

    return images_path, labels


def train_transform(img):

    img_aug = tfs.Compose([
        tfs.Resize((72, 140)),
        tfs.RandomCrop((64, 128)),
        tfs.ColorJitter(0.3, 0.3, 0.2),
        tfs.RandomRotation(10),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_aug(img)


def test_transform(img):
    img_aug = tfs.Compose([
        tfs.Resize((64, 128)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_aug(img)


class SVHNDataset(Dataset):
    def __init__(self, image_path, labels, transform=None):
        self.image_path = image_path
        self.labels = labels
        self.transform = transform

        print("read %d images"%(len(self.image_path)))

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        img = self.transform(img)

        # 进行label填充 固定6个标签
        if self.labels is not None:
            label = np.array(self.labels[index], dtype=np.int)
            label = list(label) + (6-len(label))*[10]
            return img, torch.from_numpy(np.array(label[:6]))
        else:
            return img

    def __len__(self):
        return len(self.image_path)


def get_normal_dataset(batch_size=16, index="train"):

    if index != "test":
        image_path, labels = get_path(index)
        normal_dataset = SVHNDataset(image_path, labels, train_transform if index == "train" else test_transform)
        dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=config.num_workers)
        return dataloader


if __name__ == "__main__":
    train_data = get_normal_dataset(index="val")

    for img, label in train_data:
        print(label)