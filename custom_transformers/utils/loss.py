from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Calculate the binary cross entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-bce_loss)  # Probabilities of the correct class
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Mean focal loss across all samples
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, class_weights=None, focal_weight = 0.95, bce_weight = 0.05):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)  # Binary cross-entropy loss

        self.focal_weight = focal_weight  # Weight for focal loss contribution
        self.bce_weight = bce_weight  # Weight for BCE loss contribution

    def forward(self, inputs, targets):
        # Apply focal loss to focus on the minority labels in each sample
        focal_loss = self.focal_loss(inputs, targets)
        
        # Apply class weights in BCE loss (to address class imbalance)
        bce_loss = self.bce_loss(inputs, targets)
        
        # Combine both losses (can be weighted if needed)
        total_loss = self.focal_weight * focal_loss + self.bce_weight * bce_loss
        return total_loss
    
class FocalLossWithWeight(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, class_weights=None):
        super(FocalLossWithWeight, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weights)

    def forward(self, inputs, targets):
        # Calculate the binary cross entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-bce_loss)  # Probabilities of the correct class
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Mean focal loss across all samples
        return focal_loss.mean()