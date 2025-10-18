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