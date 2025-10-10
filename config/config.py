"""
Configuration file for the Graph Generation project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset configuration
DATASET_NAME = "itsanmolgupta/mimic-cxr-dataset"
MAX_SAMPLES = 10000  # Limit for faster processing during development

# Model configurations
BIOCLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
RADGRAPH_MODEL_PATH = "models/radgraph"

# Graph configuration
MAX_GRAPH_NODES = 50
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
GAT_HEADS = 8
GAT_LAYERS = 2

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Evaluation configuration
EVAL_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
VISUALIZATION_FORMATS = ["png", "pdf"]

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)
