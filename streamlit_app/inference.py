from custom_transformers import AutoTokenizer, ModifiedBertForSequenceClassification
import torch
from .config import MODEL_PATH, THRESHOLD_DEFAULT, LABEL_MAP_PATH
from typing import Tuple, List
import os
import pandas as pd
from math import prod

# Fix torch runtime error with streamlit
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Load mapping once
group_map = pd.read_csv(LABEL_MAP_PATH)[['Id', 'Section', 'Division', 'Group']]

section_map = group_map.groupby("Section").first()['Id'].apply(lambda s: s[:1]).reset_index()
section_map = section_map.sort_values(by='Id').reset_index(drop=True)
section_map = section_map[['Id', 'Section']]

division_map = group_map.groupby("Division").aggregate('first').reset_index()
division_map["Id"] = division_map["Id"].apply(lambda x: x[:3])
division_map = division_map[["Id", "Section", "Division"]].sort_values(by='Id').reset_index(drop=True)

label_dict = group_map['Id'].to_dict()
label_dict_inv = {v: k for k, v in label_dict.items()}
section_dict = section_map.set_index('Id').to_dict()['Section']
division_dict = division_map.set_index('Id').to_dict()['Division']
group_dict = group_map.set_index('Id').to_dict()['Group']

@torch.no_grad()
def load_model() -> Tuple[AutoTokenizer, ModifiedBertForSequenceClassification]:
    """
    Load the tokenizer and custom BERT model from the saved model directory.

    Returns:
        tuple: A tuple containing:
            - tokenizer (PreTrainedTokenizer): Loaded tokenizer.
            - model (ModifiedBertForSequenceClassification): Loaded custom BERT model in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ModifiedBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def predict_sorted(
    text: str, tokenizer: AutoTokenizer,
    model: ModifiedBertForSequenceClassification,
    threshold=THRESHOLD_DEFAULT
) -> List[Tuple[int, float]]:
    """
    Run inference on input text and return sorted label-probability pairs.

    Args:
        text (str): Input text for classification.
        tokenizer (PreTrainedTokenizer): Tokenizer instance loaded with the model.
        model (ModifiedBertForSequenceClassification): Trained multi-label classification model.
        threshold (float): Threshold to consider a label as active (default: 0.5).

    Returns:
        list[tuple[int, float]]: List of (label_id, prob) tuples, sorted by probability descending.
    """
    tokens = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    tokens = tokens.to(model.device)

    outputs = model(**tokens)
    logits = outputs.logits.squeeze()
    
    probs = torch.sigmoid(logits).cpu() # - 0.2 # A little offset to avoid false positives
    labels = torch.argwhere(probs >= threshold).view(-1)

    # Create sorted list of (label, prob, binary prediction)
    sorted_output = sorted(
        {label_dict[label.item()]: prob.item() for label, prob in zip(labels, probs[labels])}.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_output

def noisy_or(probs: List[float]) -> float:
    return 1 - prod([1 - p for p in probs]) if probs else 0.0

def compute_hierarchy_probs(predictions: List[Tuple[str, float]], return_all = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute hierarchical Noisy-OR probabilities for Section, Division, and Group levels
    based on prediction probabilities for individual groups.

    Args:
        predictions (List[Tuple[str, float]]): List of tuples containing (Group ID, probability).
        return_all (bool): A flag to generate detailed df.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - section_probs: DataFrame with columns ["Section", "Prob"]
            - division_probs: DataFrame with columns ["Division", "Prob"]
            - group_probs: DataFrame with columns ["Group", "Prob"]
            - pred_df: the pred_df itself if needed
    """

    # Convert input list to DataFrame
    pred_df = pd.DataFrame(predictions, columns=["Group", "Prob"])

    # Derive hierarchy levels from Group ID
    pred_df["Section"] = pred_df["Group"].str.slice(0, 1)
    pred_df["Division"] = pred_df["Group"].str.slice(0, 3)
    
    pred_df = pred_df[['Section', "Division", 'Group', 'Prob']]

    if return_all:
        # Apply Noisy-OR per hierarchy
        section_probs = (
            pred_df.groupby("Section")["Prob"]
            .apply(lambda x: noisy_or(x.tolist()))
            .reset_index(name="Prob")
            .sort_values(by="Prob", ascending=False)
            .reset_index(drop=True)
        )

        division_probs = (
            pred_df.groupby("Division")["Prob"]
            .apply(lambda x: noisy_or(x.tolist()))
            .reset_index(name="Prob")
            .sort_values(by="Prob", ascending=False)
            .reset_index(drop=True)
        )

        group_probs = pred_df[["Group", "Prob"]].sort_values(by="Prob", ascending=False).reset_index(drop=True)

        return (
            section_probs.rename(columns={"Section": "Id"}),
            division_probs.rename(columns={"Division": "Id"}),
            group_probs.rename(columns={"Group": "Id"}),
            pred_df
        )
    else:
        return None, None, None, pred_df
