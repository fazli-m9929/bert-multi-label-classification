from torch import nn
import torch

class BCEFocalLossWithLogits(nn.Module):
    """
    Binary Cross Entropy Focal Loss with Logits.

    Combines BCEWithLogitsLoss and focal loss to focus training on hard-to-classify
    examples, useful for imbalanced multi-label classification.

    Args:
        gamma (float): Focusing parameter. Higher gamma means more focus on hard examples.
        alpha (float): Balancing factor between classes.
        class_weights (torch.Tensor, optional): Tensor of positive class weights to handle class imbalance.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    def __init__(self, gamma=2, alpha=0.25, class_weights=None, reduction="mean"):
        super(BCEFocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        # BCEWithLogitsLoss with pos_weight for class imbalance, no reduction here because we handle it manually
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weights)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model (before sigmoid), shape (batch_size, num_classes).
            targets (torch.Tensor): Multi-label ground truth tensor with same shape as inputs.

        Returns:
            torch.Tensor: Computed loss scalar or tensor depending on reduction.
        """
        # Calculate the binary cross entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-bce_loss)  # Probabilities of the correct class
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
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
    """
    Combined loss of BCE Focal Loss and standard BCEWithLogitsLoss.

    This loss mixes focal loss (to focus on hard examples) with BCE loss (to stabilize
    training) with optional class weights for imbalance.

    Args:
        gamma (float): Focusing parameter for focal loss.
        alpha (float): Balancing factor for focal loss.
        class_weights (torch.Tensor, optional): Tensor of positive class weights for BCE.
        focal_weight (float): Weight for focal loss in final loss combination.
        bce_weight (float): Weight for BCE loss in final loss combination.
    """
    def __init__(self, gamma=2, alpha=0.25, class_weights=None, focal_weight = 0.95, bce_weight = 0.05):
        super(CombinedLoss, self).__init__()
        
        # Initialize focal loss with class weights for pos_weight
        self.focal_loss = BCEFocalLossWithLogits(gamma=gamma, alpha=alpha)

        # BCEWithLogitsLoss with class weights for imbalance
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)  # Binary cross-entropy loss

        self.focal_weight = focal_weight  # Weight for focal loss contribution
        self.bce_weight = bce_weight  # Weight for BCE loss contribution

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the combined loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model.
            targets (torch.Tensor): Multi-label ground truth tensor.

        Returns:
            torch.Tensor: Scalar loss combining focal and BCE losses.
        """
        # Apply focal loss to focus on the minority labels in each sample
        focal_loss = self.focal_loss(inputs, targets)
        
        # Apply class weights in BCE loss (to address class imbalance)
        bce_loss = self.bce_loss(inputs, targets)
        
        # Combine both losses (can be weighted if needed)
        total_loss = self.focal_weight * focal_loss + self.bce_weight * bce_loss
        return total_loss