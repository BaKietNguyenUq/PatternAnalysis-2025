import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        resnet50 = models.resnet50()
        
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        
        # ResNet50 output features
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
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, 2)
        
    def forward(self, x1, x2, x3):
        """
        Forward pass for triplet input (anchor, positive, negative).

        Args:
            x1 (torch.Tensor): Anchor image tensor.
            x2 (torch.Tensor): Positive image tensor.
            x3 (torch.Tensor): Negative image tensor.

        Returns:
            tuple: Tuple containing embeddings for anchor, positive, and negative images.
        """
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        out3 = self.get_embedding(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        """
        Get the embedding for a single input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
    
    def classify(self, x):
        """
        Perform classification on the input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Classification output tensor of shape (batch_size, num_classes).
        """
        embedding = self.get_embedding(x)
        return self.classifier(embedding)
    
    
def get_config() -> dict:
    config = {
        "metadata_path": "./data/train-metadata.csv",
        "image_path": "./data/train-image/image/",
        "train_batch_size": 32,
        "test_val_batch_size": 64,
        'embedding_dims': 128,
        'learning_rate': 0.0001,
        'epochs': 20,
    }
    return config