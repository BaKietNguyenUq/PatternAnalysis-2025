"""
modules.py

Defines the Siamese network, Triplet loss, and config for data paths and training hyperparameters
for skin lesion classification with ISIC 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    Siamese network with a ResNet-50 backbone and MLP head to produce fixed-dimensional embeddings
    Also exposes a linear classifier on top of the embedding for supervised training or evaluation
    Args: embedding_dim (int): size of the output embedding vector
    """
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()

        # Pretrained ResNet-50 as feature extractor (all conv layers up to global pooling)
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        # ResNet-50 produces 2048-D pooled features
        resnet_output_features = 2048
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(resnet_output_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_dim)
        )
        
        # Classifier head for 2 classes
        self.classifier = nn.Linear(embedding_dim, 2)
        
    def forward(self, x1, x2, x3):
        """
        Forward pass for a triplet: anchor, positive, negative
        Args:
            x1 (torch.Tensor): anchor batch tensor of shape (B, C, H, W)
            x2 (torch.Tensor): positive batch tensor of shape (B, C, H, W)
            x3 (torch.Tensor): negative batch tensor of shape (B, C, H, W)
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            embeddings for (anchor, positive, negative), each of shape (B, embedding_dim)
        """
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        out3 = self.get_embedding(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        """
        Compute an embedding for a batch of images
        Args: x (torch.Tensor): input tensor of shape (B, C, H, W)
        Returns: torch.Tensor: embedding tensor of shape (B, embedding_dim)
        """
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
    
    def classify(self, x):
        """
        Predict class logits from raw images 
        Args: x (torch.Tensor): input tensor of shape (B, C, H, W)
        Returns: torch.Tensor: logits of shape (B, num_classes)
        """
        embedding = self.get_embedding(x)
        return self.classifier(embedding)
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        """
        Compute squared Euclidean distance per sample
        Args:
            x1 (torch.Tensor): tensor of shape (B, D)
            x2 (torch.Tensor): tensor of shape (B, D)
        Returns:
            torch.Tensor: distances of shape (B,) with squared L2 values
        """
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculate triplet margin loss for three input embeddings
        Args: anchor, positive, negative embeddings (B, D)
        Returns: mean loss over batch as a scalar tensor
        """
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
    
def get_config() -> dict:
    """
    Return config for data paths and training hyperparameters
    """
    config = {
        "metadata_path": "./data/train-metadata.csv",
        "image_path": "./data/train-image/image/",
        "train_batch_size": 32,
        "test_val_batch_size": 64,
        "embedding_dims": 300,
        "learning_rate": 0.001,
        "epochs": 25,
    }
    return config