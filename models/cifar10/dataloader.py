import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from numpy import asarray
from torch.utils.data import Dataset

IMAGE_SIZE = 32
IMAGES_DIR = os.path.join('..', 'data', 'cifar10', 'data', 'raw', 'img')

class ClientDataset(Dataset):
    """ CIFAR100 Dataset """

    def __init__(self, data, train=True, loading='training_time', cutout=None):
        """
        Args:
            data: dictionary in the form {'x': list of imgs ids, 'y': list of correspondings labels}
            train (bool, optional): boolean for distinguishing between client's train and test data
        """
        self.root_dir = IMAGES_DIR
        self.imgs = []
        self.labels = []
        self.loading = loading

        if data is None:
            return

        for img_name, label in zip(data['x'], data['y']):
            if loading == 'training_time':
                self.imgs.append(img_name)
            else: # loading == 'init'
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                image.load()
                self.imgs.append(image)
            self.labels.append(label)

        if train:
            self.train_transform = transforms.Compose([
                                        transforms.RandomCrop(IMAGE_SIZE, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
            self.test_transform = None
            if cutout is not None:
                self.train_transform.transforms.append(cutout(n_holes=1, length=16))
        else:
            self.train_transform = None
            self.test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.loading == 'training_time':
            img_name = os.path.join(self.root_dir, self.imgs[idx])
            image = Image.open(img_name).convert('RGB')
        else:
            image = self.imgs[idx]
        label = self.labels[idx]

        if self.train_transform:
            image = self.train_transform(image)
        elif self.test_transform:
            image = self.test_transform(image)
        return image, label