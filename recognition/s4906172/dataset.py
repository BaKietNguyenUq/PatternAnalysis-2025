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