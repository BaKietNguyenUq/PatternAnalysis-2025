import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_data_loaders
from modules import get_model, get_loss
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader

from dataset import get_data, get_data_loaders
from modules import TripletLoss, SiameseNetwork, get_config
from predict import results_siamese_network


def train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, epoch_idx):
    model.train()
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    for batch_idx, (anchor, positive, negative, labels) in enumerate(train_loader):
        anchor, positive, negative, labels = (
            anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
        )

        optimizer.zero_grad()
        anchor_out, positive_out, negative_out = model(anchor, positive, negative)
        t_loss = triplet_loss(anchor_out, positive_out, negative_out)

        logits = model.classify(anchor)
        c_loss = classifier_loss(logits, labels)

        loss = t_loss + c_loss
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        
        running_acc = accuracy_score(all_labels, all_preds)
        if (batch_idx + 1) % 10 == 0:   # adjust frequency as you like
            print(f"Epoch [{epoch_idx}] Step [{batch_idx+1}/{len(train_loader)}] Acc [{running_acc}]")

        

    avg_loss = float(np.mean(running_loss))
    final_acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc


def validate(model, val_loader, triplet_loss, classifier_loss, device, epoch_idx):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The Siamese Network model.
        val_loader (DataLoader): DataLoader for the validation data.
        triplet_loss (nn.Module): The triplet loss function.
        classifier_loss (nn.Module): The classifier loss function.
        device (torch.device): The device to validate on (CPU or GPU).

    Returns:
        tuple: A tuple containing average loss, final accuracy, and AUC-ROC for the validation set.
    """
    model.eval() 
    running_loss, all_labels, all_probs, all_preds = [], [], [], []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative, labels) in enumerate(val_loader):
            # Move data to the specified device for faster computation
            anchor, positive, negative, labels = (
                anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
            )

            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            
            loss = triplet_loss_val + classifier_loss_val

            running_loss.append(loss.item())
            
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            _, preds = classifier_out.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            running_acc = accuracy_score(all_labels, all_preds)
            if (batch_idx + 1) % 1 == 0:   # adjust frequency as you like
                print(f"Epoch [{epoch_idx}] Step [{batch_idx+1}/{len(val_loader)}] Acc [{running_acc}]")

    avg_loss = float(np.mean(running_loss)) if running_loss else 0.0
    final_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    auc_roc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, final_acc, auc_roc