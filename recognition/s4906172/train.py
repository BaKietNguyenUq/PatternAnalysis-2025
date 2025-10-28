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

from dataset import get_data_path, get_data_loaders
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
            
            t_loss = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            c_loss = classifier_loss(classifier_out, labels)
            
            loss = t_loss + c_loss

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

def train_siamese_network(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SiameseNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    triplet_loss: TripletLoss,
    classifier_loss: nn.Module,
    epochs: int,
    device: str
):
    
    train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch = [], [], []
    val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch = [], [], []
    
    for epoch in range(epochs):

        # Train the model for one epoch
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, epoch)
        
        # Validate the model
        val_loss, val_acc, val_auc = validate(model, val_loader, triplet_loss, classifier_loss, device, epoch)
        
        # Store metrics
        train_loss_per_epoch.append(train_loss)
        train_acc_per_epoch.append(train_acc)
        train_auc_per_epoch.append(train_auc)
        
        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_acc)
        val_auc_per_epoch.append(val_auc)
        

        # Adjust learning rate based on validation AUC-ROC
        scheduler.step(val_auc)
        
    return train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch

def plot_training_graphs(
    train_loss_per_epoch,
    val_loss_per_epoch,
    train_acc_per_epoch,
    val_acc_per_epoch,
    train_aucroc_per_epoch,
    val_aucroc_per_epoch,
    epochs: int,
    out_prefix: str = "train_val_progress"
) -> dict:
    """
    Plot train and validation metrics across epochs and save separate figures
    Returns dict of saved file paths
    """
    saved = {}

    def _plot_and_save(y_tr, y_val, title, ylabel, fname):
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(y_tr)+1), y_tr, label="Train")
        plt.plot(range(1, len(y_val)+1), y_val, label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close()
        saved[title] = fname

    _plot_and_save(train_loss_per_epoch, val_loss_per_epoch, "Loss over Epochs", "Loss", f"{out_prefix}_loss.png")
    _plot_and_save(train_acc_per_epoch,  val_acc_per_epoch,  "Accuracy over Epochs", "Accuracy", f"{out_prefix}_accuracy.png")
    _plot_and_save(train_aucroc_per_epoch, val_aucroc_per_epoch, "AUROC over Epochs", "AUROC", f"{out_prefix}_auroc.png")

    return saved

def main():
    # Determine device that we are training on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the current config
    config = get_config()
    
    # Extract the data from the given locations
    images, labels = get_data_path(
        metadata_path=config["metadata_path"],
        image_dir=config["image_path"]
    )
    
    # Get the data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        images=images,
        labels=labels,
        train_batch_size=config["train_batch_size"],
        test_val_batch_size=config["test_val_batch_size"]
    )
    
    # Initalise Model
    model = SiameseNetwork(config["embedding_dims"]).to(device)

    # Initialise loss functions
    triplet_loss = TripletLoss().to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Initialise Optimiser and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    train_loss_per_epoch, train_acc_per_epoch, train_auc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_auc_per_epoch = train_siamese_network(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        triplet_loss,
        classifier_loss,
        config["epochs"],
        device
    )
    
    plot_training_graphs(
        train_loss_per_epoch=train_loss_per_epoch,
        val_loss_per_epoch=val_loss_per_epoch,
        train_acc_per_epoch=train_acc_per_epoch,
        val_acc_per_epoch=val_acc_per_epoch,
        train_aucroc_per_epoch=train_auc_per_epoch,
        val_aucroc_per_epoch=val_auc_per_epoch,
        epochs=config["epochs"],
    )
    
    results_siamese_network(test_loader, model, device)