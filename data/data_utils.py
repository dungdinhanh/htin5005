import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim

import torchvision.transforms as transforms
import torchvision
import pandas as pd
from PIL import Image
import os


# data = pd.read_csv("../input/vietai-advanced-final-project-00/train.csv")
# data.head()
IMAGE_SIZE = 224                              # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
BATCH_SIZE = 10
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate
MAX_EPOCHS = 100


class ChestXrayDataset(Dataset):

    def __init__(self, folder_dir, dataframe, image_size, normalization):
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels

        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]

        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

        self.image_transformation = transforms.Compose(image_transformation)

        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row.Path)
            self.image_paths.append(image_path)
            if len(row) < 14:
                labels = [0] * 14
            else:
                labels = []
                for col in row[5:]:
                    if col == 1:
                        labels.append(1)
                    else:
                        labels.append(0)
            self.image_labels.append(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB")  # Convert image to RGB channels

        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data)

        return image_data, torch.FloatTensor(self.image_labels[index])


class ChestXrayDatasetStandardAugmentation(Dataset):

    def __init__(self, folder_dir, dataframe, image_size, normalization):
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels

        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ]

        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

        self.image_transformation = transforms.Compose(image_transformation)

        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row.Path)
            self.image_paths.append(image_path)
            if len(row) < 14:
                labels = [0] * 14
            else:
                labels = []
                for col in row[5:]:
                    if col == 1:
                        labels.append(1)
                    else:
                        labels.append(0)
            self.image_labels.append(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB")  # Convert image to RGB channels

        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data)

        return image_data, torch.FloatTensor(self.image_labels[index])


def get_chexpert(test_size=0.1, train_csv="./datasets/CheXpert-v1.0-small/train.csv",
                 path_train="./datasets"):
    data = pd.read_csv(train_csv)
    LABELS = data.columns[5:]
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=2019)
    train_dataset = ChestXrayDataset(path_train, train_data, IMAGE_SIZE,
                                     True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = ChestXrayDataset(path_train, val_data, IMAGE_SIZE, True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, LABELS

def get_chexpert_standard(test_size=0.1, train_csv="./datasets/CheXpert-v1.0-small/train.csv",
                 path_train="./datasets"):
    data = pd.read_csv(train_csv)
    LABELS = data.columns[5:]
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=2019)
    train_dataset = ChestXrayDatasetStandardAugmentation(path_train, train_data, IMAGE_SIZE,
                                     True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = ChestXrayDataset(path_train, val_data, IMAGE_SIZE, True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, LABELS


def get_chexpert_gan(test_size=0.1, train_csv="./datasets/CheXpert-v1.0-small/train.csv",
                 path_train="./datasets"):
    data = pd.read_csv(train_csv)
    LABELS = data.columns[5:]
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=2019)
    train_dataset = ChestXrayDatasetStandardAugmentation(path_train, train_data, IMAGE_SIZE,
                                     True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = ChestXrayDataset(path_train, val_data, IMAGE_SIZE, True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, LABELS

