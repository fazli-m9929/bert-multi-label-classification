from pathlib import Path
import os

# Get the absolute path to the project's base directory (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Full path to the saved BERT model directory and sigmoid output cutoff
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models/myBERT-Base-700k/')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'files/ISIC-FA-3.csv')
THRESHOLD_DEFAULT = 0.5