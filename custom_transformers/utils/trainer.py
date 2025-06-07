from transformers import PreTrainedModel
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch
from torch.utils.data import DataLoader
try:
    # Check if running in Jupyter notebook
    shell = get_ipython().__class__.__name__ # type: ignore
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook
    else:
        from tqdm import tqdm  # Other shells
except NameError:
    # Probably standard Python interpreter
    from tqdm import tqdm


class Trainer:
    """
    Trainer class for managing training and validation loops of a PyTorch model.

    Args:
        model (PreTrainedModel): Huggingface Transformers model.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LambdaLR): Learning rate scheduler.
        loss_fn (nn.Module): Loss function to optimize.
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset, can be None.
    """
    def __init__(
            self,
            model: PreTrainedModel,
            optimizer: Optimizer,
            scheduler: LambdaLR,
            loss_fn: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = model.device

        self.losses = []
        self.running_loss = []

        self.validation_loss = []
        self.learning_rates = []

    def train_epoch(self):
        """
        Run one training epoch over train_loader with backpropagation.
        Tracks batch loss and learning rate for monitoring.
        """
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False, position=1)

        self.running_loss = []

        for batch in progress_bar:
            batch = batch.to(self.device)

            # Unpack inputs
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["labels"]
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = self.loss_fn(logits, label)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            self.running_loss.append(loss.item())

            self.learning_rates.append([param_group['lr'] for param_group in self.optimizer.param_groups])

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), next_lr=self.learning_rates[-1])

    def validate_epoch(self):
        """
        Run validation over val_loader without gradient updates.
        Returns average validation loss for the epoch.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = []

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False, position=1)

        with torch.no_grad():
            for batch in progress_bar:
                batch = batch.to(self.device)
                input_ids, attention_mask, label = batch["input_ids"], batch["attention_mask"], batch["labels"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = self.loss_fn(logits, label)
                val_loss.append(loss.item())

                progress_bar.set_postfix(val_loss=val_loss[-1])

        # Track validation loss
        avg_val_loss = torch.tensor(val_loss).mean().item()
        self.validation_loss.append(avg_val_loss)
        return avg_val_loss
    
    def fit(self, num_epochs):
        """
        Run full training and validation for specified number of epochs.
        Prints epoch summary with training and validation losses.
        """
        epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", position=0)
        for epoch in epoch_bar:
            self.train_epoch()
            self.losses.append(self.running_loss)

            train_loss = torch.tensor(self.running_loss).mean().item()
            val_loss = self.validate_epoch()

            epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss if val_loss is not None else "-")