# Custom BERT Model with Focal Loss for Multi-Label Classification

This repository contains a custom implementation for training a BERT-based model for multi-label classification, using a combination of Binary Cross Entropy (BCE) loss with Focal Loss and a custom SQL dataset loader. It also includes the following features:

- **Custom Focal Loss**: Combines BCE with Focal Loss for better handling of class imbalance.
- **SQL Dataset Loader**: Fetches data from a PostgreSQL database and prepares it for training, including tokenization and multi-label encoding.
- **Trainer Class**: Implements the training loop with logging and learning rate scheduling.

## Features

- **Custom Loss Function**: `BCEFocalLossWithLogits` - A custom loss function combining Binary Cross Entropy with Focal Loss, designed for multi-label classification tasks.
- **SQLDataset**: Custom dataset class that loads text and labels from a PostgreSQL database, tokenizes the text, and creates multi-hot encoded labels for multi-label classification.
- **Trainer Class**: Handles training and validation loops with metric tracking, supports dynamic learning rate scheduling, and can be easily extended for different models.
- **Modified BERT Model**: A subclass of `BertForSequenceClassification` that allows for potential device map optimizations.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/fazli-m9929/bert-multi-label-classification.git
cd bert-multi-label-classification
```

Make sure to have a compatible Python version (Python 3.7 or later).

## Dependencies

The project requires the following libraries:

- `torch`
- `transformers`
- `psycopg2-binary` (for PostgreSQL database access)
- `tqdm` (for progress bars)
- `torchvision` (for image processing if applicable)
- `matplotlib`
- `python-dotenv` (for environment variable management)
- `scikit-learn` (for splitting data)

You can install all the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Set up the PostgreSQL database

Make sure your PostgreSQL database is set up and contains the required tables and columns. The SQLDataset is designed to fetch data from the `dbo.activity` table where the `labels2` column contains a list of labels.

### 2. Create a `SQLDataset` Instance

To initialize the dataset loader, provide the tokenizer and connection parameters:

```python
from transformers import AutoTokenizer
from dataset import SQLDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

connection_params = {
    'dbname': 'your_db_name',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'port': '5432'
}

dataset = SQLDataset(
    tokenizer=tokenizer,
    connection_params=connection_params,
    num_classes=10  # Example: number of possible classes for multi-label classification
)
```

### 3. Training the Model

First, import the necessary components and set up the dataset:

~~~python
from custom_transformers import (
    ModifiedBertForSequenceClassification,
    SQLDataset,
    BCEFocalLossWithLogits,
    Trainer,
    AutoTokenizer,
    get_scheduler,
)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch

# Load and split the dataset
train_indices, test_indices = train_test_split(list(dataset.id_map.keys()), test_size=0.1)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, 32, shuffle=True, collate_fn=dataset.collate_fn)
test_loader = DataLoader(test_dataset, 32, shuffle=True, collate_fn=dataset.collate_fn)
~~~

Next, set up the optimizer with frozen and unfrozen layers, and define the custom loss function:

~~~python
# Freeze certain layers and set learning rates for others
frozen_params, unfrozen_params = [], []
for name, param in model.named_parameters():
    if name in ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']:
        unfrozen_params.append(param)
    else:
        frozen_params.append(param)

optimizer = torch.optim.AdamW(
    [{'params': frozen_params, 'lr': 5e-6},  # Frozen layers with smaller learning rate
     {'params': unfrozen_params, 'lr': 5e-5}], # Unfrozen layers with larger learning rate
)

# Define the custom loss function
loss_fn = BCEFocalLossWithLogits(gamma=2, alpha=0.957, class_weights=dataset.scaled_class_weights.to(model.device))

# Set up the scheduler
num_epochs = 20
num_training_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * num_training_steps)

scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)
~~~

Finally, initialize the trainer and start training:

~~~python
# Set up and train the model
trainer = Trainer(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    test_loader
)
~~~

This setup will train the model for 20 epochs, using a custom focal loss and learning rate scheduling with warm-up.

## Contributing

If you find any issues or want to contribute improvements to the project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
