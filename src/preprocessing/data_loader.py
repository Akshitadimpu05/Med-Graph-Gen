"""
Data loading and preprocessing module for MIMIC-CXR dataset
"""
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm

from config.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIMICCXRDataLoader:
    """Data loader for MIMIC-CXR dataset from HuggingFace"""
    
    def __init__(self, dataset_name: str = DATASET_NAME, max_samples: int = MAX_SAMPLES):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.dataset = None
        self.processed_data = []
        
    def load_dataset(self) -> None:
        """Load the MIMIC-CXR dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        try:
            self.dataset = load_dataset(self.dataset_name, split="train")
            logger.info(f"Dataset loaded successfully. Total samples: {len(self.dataset)}")
            
            # Limit samples for faster processing
            if self.max_samples and len(self.dataset) > self.max_samples:
                self.dataset = self.dataset.select(range(self.max_samples))
                logger.info(f"Limited to {self.max_samples} samples for processing")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text or pd.isna(text):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def extract_text_data(self) -> List[Dict]:
        """Extract and preprocess text data from the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Extracting and preprocessing text data...")
        processed_data = []
        
        for idx, sample in enumerate(tqdm(self.dataset, desc="Processing samples")):
            try:
                # Extract findings and impression
                findings = self.preprocess_text(sample.get('findings', ''))
                impression = self.preprocess_text(sample.get('impression', ''))
                
                # Skip samples with no text content
                if not findings and not impression:
                    continue
                
                # Combine findings and impression
                combined_text = f"{findings} {impression}".strip()
                
                processed_sample = {
                    'id': idx,
                    'findings': findings,
                    'impression': impression,
                    'combined_text': combined_text,
                    'text_length': len(combined_text.split())
                }
                
                processed_data.append(processed_sample)
                
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        self.processed_data = processed_data
        logger.info(f"Successfully processed {len(processed_data)} samples")
        return processed_data
    
    def save_processed_data(self, output_path: Optional[Path] = None) -> None:
        """Save processed data to JSON file"""
        if not self.processed_data:
            raise ValueError("No processed data to save. Call extract_text_data() first.")
        
        if output_path is None:
            output_path = DATA_DIR / "processed_mimic_cxr.json"
        
        logger.info(f"Saving processed data to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(self.processed_data, f, indent=2)
        
        # Also save as CSV for easier inspection
        df = pd.DataFrame(self.processed_data)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Also saved as CSV: {csv_path}")
    
    def get_statistics(self) -> Dict:
        """Get basic statistics about the processed data"""
        if not self.processed_data:
            return {}
        
        df = pd.DataFrame(self.processed_data)
        
        stats = {
            'total_samples': len(df),
            'avg_text_length': df['text_length'].mean(),
            'median_text_length': df['text_length'].median(),
            'max_text_length': df['text_length'].max(),
            'min_text_length': df['text_length'].min(),
            'samples_with_findings': len(df[df['findings'] != '']),
            'samples_with_impression': len(df[df['impression'] != '']),
            'samples_with_both': len(df[(df['findings'] != '') & (df['impression'] != '')])
        }
        
        return stats

def main():
    """Main function to run data preprocessing"""
    logger.info("Starting MIMIC-CXR data preprocessing...")
    
    # Initialize data loader
    data_loader = MIMICCXRDataLoader()
    
    # Load and process dataset
    data_loader.load_dataset()
    processed_data = data_loader.extract_text_data()
    
    # Save processed data
    data_loader.save_processed_data()
    
    # Print statistics
    stats = data_loader.get_statistics()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
