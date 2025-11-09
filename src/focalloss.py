import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Args:
            alpha (Tensor, optional): Weight factor, one weight per class. Shape should be [num_classes].
            gamma (float): Modulation factor to reduce loss for easy-to-classify samples.
            reduction (str): {'none', 'mean', 'sum'}, specifies how to aggregate the loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model's unnormalized output, shape [batch_size, num_classes].
            targets (Tensor): True labels, shape [batch_size].
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt is the model's predicted probability for the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 1. Calculate class distribution in the training set
def get_class_distribution(dataset):
    """
    Calculate class distribution in the training set
    """
    labels = []
    for _, label in dataset:
        # Assume each task has corresponding label columns
        # Concatenate the label columns, using 'multi_task' as an example here
        labels.extend(label['multi_task'])
    labels = np.array(labels)
    
    # Count the number of samples for each class
    class_counts = Counter(labels.flatten())  # Flatten to a one-dimensional array for counting
    return class_counts

# 2. Calculate alpha (class weights)
def compute_alpha(class_counts, num_classes):
    """
    Calculate alpha (class weights) based on class distribution
    """
    total_samples = sum(class_counts.values())
    alpha = torch.zeros(num_classes)
    
    for class_idx in range(num_classes):
        # alpha_i = 1 / (number of samples / total samples) = total samples / number of samples in class i
        alpha[class_idx] = total_samples / (len(class_counts) * class_counts.get(class_idx, 0) + 1e-6)
        
    return alpha

# 3. Calculate gamma (modulation factor)
def compute_gamma(class_counts):
    """
    Automatically adjust gamma based on the degree of class imbalance
    """
    # Assume that when classes are imbalanced, larger gamma focuses more on minority classes
    # Here, dynamically adjust gamma by calculating the imbalance degree of class distribution
    total_samples = sum(class_counts.values())
    max_class_samples = max(class_counts.values())
    imbalance_ratio = max_class_samples / total_samples

    if imbalance_ratio > 0.9:
        gamma = 3.0  # Very imbalanced classes, increase gamma
    elif imbalance_ratio > 0.5:
        gamma = 2.0  # Moderately imbalanced
    else:
        gamma = 1.0  # Relatively balanced
    return gamma

# 4. Integrate into the training process
def train_with_dynamic_alpha_gamma(dataset, num_classes):
    # 1. Get class distribution
    class_counts = get_class_distribution(dataset)

    # 2. Calculate alpha and gamma
    alpha = compute_alpha(class_counts, num_classes)
    gamma = compute_gamma(class_counts)
    
    # Output alpha and gamma
    print(f"Calculated alpha: {alpha}")
    print(f"Calculated gamma: {gamma}")
    
    # Use alpha and gamma in FocalLoss
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
