#!/bin/bash

# Graph Generation Project Setup and Execution Script
# This script sets up the environment and runs the preprocessing pipeline

echo "=========================================="
echo "Graph Generation Project Setup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if not already installed
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model if not already downloaded
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "=========================================="
echo "Running Preprocessing Pipeline"
echo "=========================================="

# Run preprocessing
python main.py --mode preprocess --max-samples 1000

echo "=========================================="
echo "Setup and Preprocessing Complete!"
echo "=========================================="

echo "To evaluate results, run:"
echo "python evaluate_results.py"

echo ""
echo "To run the full pipeline (including training), run:"
echo "python main.py --mode full --max-samples 1000"

echo ""
echo "Project structure:"
echo "- data/: Processed datasets and graphs"
echo "- results/: Visualizations and evaluation metrics"
echo "- models/: Trained model checkpoints"
