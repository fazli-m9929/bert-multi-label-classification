from transformers import PreTrainedModel
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
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
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        self.running_loss = []

        for batch in progress_bar:
            batch = batch.to(self.device)

            input_ids, attention_mask, label = batch["input_ids"], batch["attention_mask"], batch["labels"]

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

            self.learning_rates.append(self.optimizer.param_groups[0]["lr"])

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), next_lr=self.learning_rates[-1])

    def validate_epoch(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = []

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)

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
        for epoch in range(num_epochs):
            epoch +=1
            tqdm.write(f"Epoch {epoch: 3d}/{num_epochs}")
            self.train_epoch()
            self.losses.append(self.running_loss)

            train_loss = torch.tensor(self.running_loss).mean().item()
            val_loss = self.validate_epoch()

            tqdm.write(f"Training Loss: {train_loss:.4f}")
            if val_loss is not None:
                tqdm.write(f"Validation Loss: {val_loss:.4f}")
            tqdm.write('')