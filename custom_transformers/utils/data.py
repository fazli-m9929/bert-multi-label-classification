import psycopg2
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from ast import literal_eval
from transformers import AutoTokenizer
from collections import Counter


class SQLDataset(Dataset):
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            connection_params: Dict[str, str], 
            num_classes: int
        ):
        """
        Initializes the SQLDataset by establishing a connection to the database and setting up the tokenizer.
        
        Args:
            tokenizer (AutoTokenizer): the pretrained model's tokenizer.
            connection_params (Dict[str, str]): Dictionary containing the database connection parameters.
                The user should provide a dictionary like the following:
                
                {
                    'dbname': 'your_db_name',
                    'user': 'your_username',
                    'password': 'your_password',
                    'host': 'your_host',
                    'port': '5432'
                }
            num_classes (int): The number of possible classes for multi-label classification.
        """
        self.num_classes = num_classes

        # Initialize the tokenizer using a pretrained model
        self.tokenizer = tokenizer

        # Store the connection parameters
        self.connection_params = connection_params
        
        # Establish a connection to the database using psycopg2
        self.connection: psycopg2.extensions.connection = psycopg2.connect(**self.connection_params)
        self.cursor = self.connection.cursor()

        # Define and execute the id_query to fetch all valid IDs
        self.id_query = """
            SELECT id, labels2
            FROM dbo.activity 
            WHERE labels2 IS NOT NULL AND labels2 NOT LIKE '%[]%'
        """
        # Execute the query and store the results in a dictionary mapping index to db_id
        self.cursor.execute(self.id_query)
        rows = self.cursor.fetchall()

         # Map IDs and parse labels
        self.id_map = {i: row[0] for i, row in enumerate(rows)}
        labels = [literal_eval(row[1]) for row in rows]

        # Compute class weights
        self.scaled_class_weights = self._compute_class_weights(labels)

    @staticmethod
    def scale_to_range(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Scales a tensor from the range [0, 1] to a specified range [a, b].

        Args:
            x (torch.Tensor): Input tensor with values in the range [0, 1].
            a (float): The lower bound of the target range.
            b (float): The upper bound of the target range.

        Returns:
            torch.Tensor: Tensor scaled to the range [a, b].
        """
        return x * (b - a) + a

    def _compute_class_weights(self, labels: list) -> torch.Tensor:
        """
        Computes scaled class weights based on label frequency for multi-label classification.

        Args:
            labels (list): A list of lists, where each inner list contains labels for a sample.

        Returns:
            torch.Tensor: A tensor containing the scaled class weights for each label.
        """
        # Flatten the list to count occurrences of each label
        flat_labels = [label for sublist in labels for label in sublist]
        label_counts = Counter(flat_labels)

        # Total number of samples
        total_samples = sum(label_counts.values())

        # Compute class weights as the inverse of label frequency
        class_weights = {label: total_samples / count for label, count in label_counts.items()}

        # Normalize weights
        max_weight = max(class_weights.values())
        class_weights = {label: weight / max_weight for label, weight in class_weights.items()}

        # Initialize class weights tensor
        class_weights_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        for label, weight in class_weights.items():
            class_weights_tensor[label] = weight

        # Apply log transformation and scaling
        log_scaled_weights = torch.log(class_weights_tensor + 1e-10)  # Avoid log(0)
        log_scaled_weights = (log_scaled_weights - log_scaled_weights.min()) / (log_scaled_weights.max() - log_scaled_weights.min())

        # Scale weights from [0, 1] to [1, 10]
        return self.scale_to_range(log_scaled_weights, 1, 10)


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: The number of samples (i.e., number of IDs).
        """
        return len(self.id_map)
    
    def __getitem__(self, index: int):
        """
        Retrieves the database ID for a given index.
        
        Args:
            index (int): The index of the sample.
        
        Returns:
            int: The database ID corresponding to the given index.
        """
        db_id = self.id_map[index]
        return db_id
    
    def fetch_batch_data(self, ids: List[int]):
        """
        Fetches data from the database for a batch of given IDs.
        
        Args:
            ids (List[int]): List of database IDs to fetch data for.
        
        Returns:
            list: Data rows fetched from the database. Each row contains the 'companyactivity' text and 'labels2'.
        """
        # Formulate the SQL query to fetch the required data for the given batch of IDs
        query = f"SELECT companyactivity, labels2 FROM dbo.activity WHERE id IN ({', '.join(['%s'] * len(ids))})"
        # Execute the query
        self.cursor.execute(query, tuple(ids))
        return self.cursor.fetchall()

    def _multi_label_one_hot(self, labels: List[List[int]] | Tuple[List[int]]):
        """
        Converts multi-labels (list of class indices) into a multi-hot encoding.

        Args:
            labels (List[List[int]] | Tuple[List[int]]): List or tuple of label sets, where each set is a list of class indices.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_classes), where each row is a multi-hot vector.
        """
        num_classes = self.num_classes

        # Create a tensor of zeros with shape (batch_size, num_classes)
        multi_hot = torch.zeros(len(labels), num_classes, dtype=torch.float32)
        
        # Set the corresponding indices to 1 for each label set
        for i, label_set in enumerate(labels):
            if not isinstance(label_set, (list, tuple)):
                raise ValueError(f"Each label set must be a list or tuple of integers. Found: {type(label_set)}")
            
            # Check if the indices are valid
            for label in label_set:
                if label < 0 or label >= num_classes:
                    raise ValueError(f"Label index {label} is out of bounds. Must be between 0 and {num_classes-1}.")

            # Set the multi-hot vector for this sample
            multi_hot[i, label_set] = 1
        
        return multi_hot

    def collate_fn(self, batch: List[int]):
        """
        Custom collate function for batching. It retrieves the batch data from the database, tokenizes
        the texts, and one-hot encodes the labels.

        Args:
            batch (List[int]): A list of database IDs (integers) retrieved from the SQLDataset.

        Returns:
            dict: A dictionary containing the tokenized inputs (texts) and multi-hot encoded labels.
        """
        
        # Fetch the data for the batch from the SQLDataset
        batch_data = self.fetch_batch_data(batch)

        # Extract texts and labels into separate tuples, using literal_eval to parse the labels
        texts, labels = zip(*[(text, literal_eval(label)) for text, label in batch_data])

        # Tokenize the texts using the pretrained tokenizer
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Convert the labels into multi-hot encoding
        labels = self._multi_label_one_hot(labels)

        # Add the multi-hot encoded labels to the tokenized inputs
        inputs['labels'] = labels

        return inputs