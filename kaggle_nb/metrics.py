import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BinarySegmentationMetrics:
    """
    Metrics for binary segmentation tasks
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics counters"""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.tn = 0  # True negatives
        self.fn = 0  # False negatives
        self.n_samples = 0
    
    def update(self, outputs, targets):
        """
        Update metrics based on a batch of outputs and targets
        
        Args:
            outputs: Model outputs, shape (B, 1, H, W) or (B, H, W)
            targets: Ground truth, shape (B, 1, H, W) or (B, H, W)
        """
        # Ensure inputs are in the right format
        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        
        # Apply threshold to get binary predictions
        preds = (outputs > self.threshold).float()
        
        # Move tensors to CPU for computation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        # Update counters
        self.tp += torch.sum((preds == 1) & (targets == 1)).item()
        self.fp += torch.sum((preds == 1) & (targets == 0)).item()
        self.tn += torch.sum((preds == 0) & (targets == 0)).item()
        self.fn += torch.sum((preds == 0) & (targets == 1)).item()
        self.n_samples += targets.size(0)
    
    def get_iou(self):
        """Calculate Intersection over Union (IoU)"""
        if self.tp + self.fp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fp + self.fn)
    
    def get_dice(self):
        """Calculate Dice Coefficient (F1 Score)"""
        if 2 * self.tp + self.fp + self.fn == 0:
            return 0.0
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)
    
    def get_accuracy(self):
        """Calculate pixel-wise accuracy"""
        if self.tp + self.tn + self.fp + self.fn == 0:
            return 0.0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def get_sensitivity(self):
        """Calculate sensitivity (recall)"""
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def get_specificity(self):
        """Calculate specificity"""
        if self.tn + self.fp == 0:
            return 0.0
        return self.tn / (self.tn + self.fp)
    
    def get_metrics(self):
        """Get all metrics as a dictionary"""
        return {
            'iou': self.get_iou(),
            'dice': self.get_dice(),
            'accuracy': self.get_accuracy(),
            'sensitivity': self.get_sensitivity(),
            'specificity': self.get_specificity()
        }


def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice loss for binary segmentation
    
    Args:
        pred: Model predictions
        target: Ground truth
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value
    """
    pred = pred.contiguous()
    target = target.contiguous()    
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def bce_dice_loss(pred, target, bce_weight=0.5, smooth=1.0):
    """
    Combined BCE and Dice loss for binary segmentation
    
    Args:
        pred: Model predictions
        target: Ground truth
        bce_weight: Weight for BCE loss component
        smooth: Smoothing factor for Dice loss
        
    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target, smooth)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    return loss 

def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculate Dice coefficient
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    """
    Calculate IoU (Intersection over Union) coefficient
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_score_tensor(y_true, y_pred, threshold=0.5, epsilon=1e-7):
    """
    Calculate Dice score for PyTorch tensors
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        threshold: Threshold for binarizing predictions
        epsilon: Small value to avoid division by zero
        
    Returns:
        Dice score
    """
    # Threshold predictions
    y_pred = (y_pred > threshold).float()
    
    # Flatten tensors
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    # Calculate intersection and union
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum()
    
    # Calculate Dice score
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

def iou_score_tensor(y_true, y_pred, threshold=0.5, epsilon=1e-7):
    """
    Calculate IoU score for PyTorch tensors
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        threshold: Threshold for binarizing predictions
        epsilon: Small value to avoid division by zero
        
    Returns:
        IoU score
    """
    # Threshold predictions
    y_pred = (y_pred > threshold).float()
    
    # Flatten tensors
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    # Calculate intersection and union
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    
    # Calculate IoU score
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou

def accuracy_tensor(y_true, y_pred, threshold=0.5):
    """
    Calculate accuracy for PyTorch tensors
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        threshold: Threshold for binarizing predictions
        
    Returns:
        Accuracy
    """
    # Threshold predictions
    y_pred = (y_pred > threshold).float()
    
    # Flatten tensors
    y_true_f = y_true.view(-1).cpu().numpy()
    y_pred_f = y_pred.view(-1).cpu().numpy()
    
    # Calculate accuracy
    return accuracy_score(y_true_f, y_pred_f)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate multiple metrics for evaluation
    
    Args:
        y_true: Ground truth tensor/array
        y_pred: Predicted tensor/array
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary with metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Apply threshold
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    # Flatten arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_true_f, y_pred_f)
    prec = precision_score(y_true_f, y_pred_f, zero_division=1)
    rec = recall_score(y_true_f, y_pred_f, zero_division=1)
    f1 = f1_score(y_true_f, y_pred_f, zero_division=1)
    dice = dice_coef(y_true, y_pred_bin)
    iou = iou_coef(y_true, y_pred_bin)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'dice': dice,
        'iou': iou
    } 