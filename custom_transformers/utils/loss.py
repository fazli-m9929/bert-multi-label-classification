from torch import nn
import torch

class BCEFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, class_weights=None, reduction="mean"):
        super(BCEFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weights)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the binary cross entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-bce_loss)  # Probabilities of the correct class
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == "none":
            loss = focal_loss
        elif self.reduction == "mean":
            loss = torch.mean(focal_loss)
        elif self.reduction == "sum":
            loss = torch.sum(focal_loss)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, class_weights=None, focal_weight = 0.95, bce_weight = 0.05):
        super(CombinedLoss, self).__init__()
        self.focal_loss = BCEFocalLossWithLogits(gamma=gamma, alpha=alpha)
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