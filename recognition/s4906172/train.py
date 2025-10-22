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