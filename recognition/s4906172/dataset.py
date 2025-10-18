import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split



def get_data(metadata_path: str, image_dir: str) -> tuple[list]:
    """
    Load ISIC 2020 image paths and labels from metadata and an image folder

    Args:
        metadata_path: path to train-metadata.csv containing isic_id and target columns
        image_dir: directory containing image files named <isic_id>.jpg

    Returns:
        images: NumPy array of absolute image paths found in image_dir
        labels: NumPy array of integer targets aligned with images
    """
    metadata = pd.read_csv(metadata_path)

    # Build expected filename for each image in metadata
    metadata['image_file'] = metadata['isic_id'] + '.jpg'

    # Map filename to label for quick lookup
    image_to_label = dict(zip(metadata['image_file'], metadata['target']))

    # Keep only files that both exist in the folder and appear in metadata
    image_paths = [
        os.path.join(image_dir, img)
        for img in os.listdir(image_dir)
        if img in image_to_label
    ]

    # Create labels aligned with the collected image paths
    labels = [image_to_label[os.path.basename(p)] for p in image_paths]

    return np.array(image_paths), np.array(labels)



class TripletDataGenerator(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, train=True, transform=None):
        self.is_train = train
        self.transform = transform

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def _read_rgb01(self, path: str) -> np.ndarray:
        return cv2.imread(path) / 255.0

    def __getitem__(self, anchor_index):
        anchor_img = self._read_rgb01(self.images[anchor_index])
        anchor_label = self.labels[anchor_index]

        if self.is_train:
            positive_list = [idx for idx, label in enumerate(self.labels) if label == anchor_label and idx != anchor_index]
            positive_index = random.choice(positive_list)
            positive_img = self._read_rgb01(self.images[positive_index])

            negative_list = [idx for idx, label in enumerate(self.labels) if label != anchor_label and idx != anchor_index]
            negative_index = random.choice(negative_list)
            negative_img = self._read_rgb01(self.images[negative_index])

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img, torch.empty(1), torch.empty(1), anchor_label