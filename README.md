# Graph Generation from X-ray Reports

This project implements a comprehensive pipeline for generating attributed abnormality graphs from X-ray reports using the MIMIC-CXR dataset. The pipeline follows the architecture:

```
[Text Reports] → RadGraph + RadLex → [Attributed Abnormality Graph] → BioClinicalBERT → GAT → Graph Embeddings → Visualization/Clustering
```

## Project Structure

```
Graph-Generation/
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py     # MIMIC-CXR dataset loader
│   │   └── radgraph_processor.py  # RadGraph + RadLex processing
│   ├── models/
│   │   ├── embeddings.py      # BioClinicalBERT embeddings
│   │   └── gat_model.py       # Graph Attention Network
│   ├── training/
│   │   └── train_pipeline.py  # Training pipeline
│   └── visualization/
│       └── graph_viz.py       # Visualization and clustering
├── data/                      # Processed data storage
├── results/                   # Results and visualizations
├── models/                    # Trained model checkpoints
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Quick Start (Preprocessing Only - for Evaluation)

For your evaluation tomorrow, run the preprocessing pipeline:

```bash
python main.py --mode preprocess --max-samples 1000
```

This will:
- Download and preprocess the MIMIC-CXR dataset
- Generate abnormality graphs using RadGraph + RadLex
- Create BioClinicalBERT embeddings
- Save all processed data for evaluation

### Full Pipeline

To run the complete pipeline:

```bash
python main.py --mode full --max-samples 5000
```

### Individual Components

Run specific parts of the pipeline:

```bash
# Only preprocessing
python main.py --mode preprocess

# Only training (requires preprocessed data)
python main.py --mode train

# Only evaluation and visualization (requires trained model)
python main.py --mode evaluate
```

## Pipeline Components

### 1. Data Preprocessing (`src/preprocessing/`)

- **data_loader.py**: Loads MIMIC-CXR dataset from HuggingFace, extracts findings and impressions
- **radgraph_processor.py**: Processes text to create attributed abnormality graphs using RadGraph concepts and RadLex terminology

### 2. Model Components (`src/models/`)

- **embeddings.py**: Generates term embeddings using BioClinicalBERT
- **gat_model.py**: Implements Graph Attention Network for learning graph representations

### 3. Training (`src/training/`)

- **train_pipeline.py**: Complete training pipeline with evaluation metrics

### 4. Visualization (`src/visualization/`)

- **graph_viz.py**: Graph visualization and clustering analysis

## Key Features

### Graph Generation
- Extracts anatomical terms, abnormalities, and observations from radiology reports
- Creates attributed graphs with semantic relationships
- Supports both static and interactive visualizations

### Embeddings
- Uses BioClinicalBERT for medical domain-specific embeddings
- Caches embeddings for efficient processing
- Supports both node-level and graph-level representations

### Graph Attention Network
- Multi-head attention mechanism for learning graph representations
- Supports both node and graph-level tasks
- Includes graph pooling for graph-level predictions

### Evaluation and Visualization
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, AUC)
- 2D embedding visualizations using t-SNE and PCA
- Clustering analysis with K-means and DBSCAN
- Interactive graph visualizations

## Output Files

### Data Directory (`data/`)
- `processed_mimic_cxr.json/csv`: Preprocessed text data
- `abnormality_graphs.json`: Generated graphs
- `embedded_graphs.json`: Graphs with embeddings
- `graph_statistics.json`: Graph generation statistics

### Results Directory (`results/`)
- `training_summary.json`: Training results and metrics
- `evaluation_metrics.csv`: Detailed evaluation metrics
- `training_curves.png`: Training and validation curves
- `sample_graph.png`: Sample graph visualization
- `multiple_graphs_visualization.png`: Multiple graphs overview
- `embeddings_2d_tsne.png`: t-SNE visualization of embeddings
- `embeddings_2d_pca.png`: PCA visualization of embeddings
- `clustering_analysis.png`: Clustering analysis results
- `interactive_graph_*.html`: Interactive graph visualizations
- `gat_embeddings.npz`: GAT-generated embeddings

### Models Directory (`models/`)
- `best_gat_model.pth`: Best trained GAT model checkpoint

## Configuration

Modify `config/config.py` to adjust:
- Dataset size limits
- Model hyperparameters
- Training parameters
- File paths

## Dataset

Uses the MIMIC-CXR dataset from HuggingFace:
- **Source**: `itsanmolgupta/mimic-cxr-dataset`
- **Fields**: image, findings, impression
- **Focus**: Text data (findings and impressions)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers (HuggingFace)
- NetworkX
- Matplotlib/Seaborn/Plotly
- scikit-learn
- spaCy

## Notes for Evaluation

The preprocessing pipeline is designed to be fast and efficient for evaluation purposes:
1. Limits dataset size for quicker processing
2. Caches embeddings to avoid recomputation
3. Saves intermediate results for inspection
4. Generates comprehensive statistics and visualizations

For your evaluation tomorrow, focus on the preprocessing results in the `data/` and `results/` directories, which will demonstrate the complete graph generation process from raw text to attributed abnormality graphs.
