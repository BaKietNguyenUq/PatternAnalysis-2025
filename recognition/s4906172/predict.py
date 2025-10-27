import os
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train import validate
from modules import TripletLoss, SiameseNetwork, get_config

## Functions
def produce_evaluation_metrics(
    test_y_pred: list,
    test_y_probs: list,
    test_y_true: list
) -> None:
    
    test_accuracy = accuracy_score(test_y_true, test_y_pred)
    test_auc_roc = roc_auc_score(test_y_true, test_y_probs)
    print(f"Testing Accuracy: {test_accuracy}")
    print(f"Testing AUR ROC: {test_auc_roc}")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn) # Calculate Sensitivity (True Positive Rate)
    specificity = tn / (tn + fp) # Calculate Specificity (True Negative Rate)
    print(f"Testing Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Testing Specificity: {specificity:.3f}")


def plot_tsne_from_embeddings(
    embeddings, 
    out_path="testing_tsne_embeddings.png", 
    title="t-SNE visualization of embeddings"
):
    """
    Compute 2D t-SNE from embeddings and save a scatter plot
    Only embeddings are required
    Returns (embeddings_2d, out_path)
    """

    tsne = TSNE(n_components=2, init="pca", random_state=42, n_iter=2000, learning_rate=200, perplexity=30)
    X2 = tsne.fit_transform(embeddings)

    # plot with a single color since no labels provided
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.8, c=range(len(X2)), cmap="cividis")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()