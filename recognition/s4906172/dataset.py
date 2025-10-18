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

def split_train_val_test(images: list, labels: list):
    """
    Preform a 0.8 train, 0.1 validation, 0.1 test split on the given data set.

    Returns: train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    train_images, other_images, train_labels, other_labels = train_test_split(
        images, labels, test_size=0.2, stratify=labels
    )
    test_images, val_images, test_labels, val_labels = train_test_split(
        other_images, other_labels, test_size=0.5, stratify=other_labels
    )
    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def oversample_training(train_images, train_labels):
    """
    Oversampling will be performed on the training set to ensure equal number of
    samples for each class.

    Returns: balanced_train_images, balanced_train_labels
    """
    # Preform oversampling of training data
    # Split the images into class 0 and class 1
    class_0_images = train_images[train_labels == 0]
    class_1_images = train_images[train_labels == 1]
    class_0_labels = train_labels[train_labels == 0]
    class_1_labels = train_labels[train_labels == 1]

    # Number of samples to match class 0
    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)

    # If class 1 (the minority class) has fewer samples, oversample it
    if num_class_1 < num_class_0:
        # Randomly choose from class_1_images to oversample it
        oversample_indices = np.random.choice(
            np.arange(num_class_1), size=num_class_0 - num_class_1, replace=True
        )
        oversampled_class_1_images = class_1_images[oversample_indices]
        oversampled_class_1_labels = class_1_labels[oversample_indices]

        # Combine original class 1 images with the oversampled ones
        class_1_images = np.concatenate([class_1_images, oversampled_class_1_images], axis=0)
        class_1_labels = np.concatenate([class_1_labels, oversampled_class_1_labels], axis=0)

    # Concatenate class 0 and the new class 1 images to get the final balanced dataset
    balanced_train_images = np.concatenate([class_0_images, class_1_images], axis=0)
    balanced_train_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    return balanced_train_images, balanced_train_labels

def get_data_loaders(images, labels, train_batch_size=32, test_val_batch_size=64):
    """
    Returns train, validation and testing dataloaders for the ISIC 2020 data set. Given
    the images and labels for the ISIC 2020 data set.

    All three sets will be normalised and converted to tensors and augmentation is applied to the train set.

    Returns: train_loader, val_loader, test_loader
    """
    # Perform the data split
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_train_val_test(images, labels)
    # Perform the oversampling for train dataset
    train_images, train_labels = oversample_training(train_images, train_labels)

    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10, fill=(255, 255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Train dataset
    train_ds = TripletDataGenerator(
        images=train_images,
        labels=train_labels,
        train=True,
        transform=train_transform 
    )

    # Validation dataset
    val_ds = TripletDataGenerator(
        images=val_images,
        labels=val_labels,
        train=True,
        transform=val_transform
    )

    # Testing dataset
    test_ds = TripletDataGenerator(
        images=test_images,
        labels=test_labels,
        train=False,
        transform=test_transform
        
    )
    
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=test_val_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=test_val_batch_size, shuffle=True, num_workers=4)
    return train_loader, val_loader, test_loader



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