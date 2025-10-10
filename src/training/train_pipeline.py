"""
Training pipeline for the graph generation project
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.config import *
from src.models.gat_model import GraphAttentionNetwork, GraphDataProcessor
from src.models.embeddings import BioClinicalBERTEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphDataset(Dataset):
    """Dataset class for graph data"""
    
    def __init__(self, graph_data: List[Dict], processor: GraphDataProcessor):
        self.graph_data = graph_data
        self.processor = processor
        
    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        sample = self.graph_data[idx]
        data = self.processor.create_pyg_data(sample)
        
        # Create dummy labels for demonstration (can be modified for specific tasks)
        # For example, binary classification based on presence of abnormalities
        has_abnormality = any(node['type'] == 'abnormality' for node in sample['graph']['nodes'])
        label = torch.tensor(1 if has_abnormality else 0, dtype=torch.long)
        
        return data, label

class GraphTrainer:
    """Main training class for graph models"""
    
    def __init__(self, model: GraphAttentionNetwork, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.processor = GraphDataProcessor(device)
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def setup_training(self, learning_rate: float = LEARNING_RATE,
                      weight_decay: float = 1e-5):
        """Setup training components"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=PATIENCE//2, factor=0.5, verbose=True
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Training")):
            try:
                # Move data to device
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get graph embeddings
                _, graph_embeddings = self.model(data.x, data.edge_index, data.batch)
                
                # Classification head
                logits = self.model.classifier(graph_embeddings)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Validation")):
                try:
                    # Move data to device
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    _, graph_embeddings = self.model(data.x, data.edge_index, data.batch)
                    logits = self.model.classifier(graph_embeddings)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    
                    # Collect predictions and probabilities
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        metrics = self._calculate_metrics(all_predictions, all_labels, all_probabilities)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: List, labels: List, 
                          probabilities: Optional[List] = None) -> Dict:
        """Calculate evaluation metrics"""
        if not predictions or not labels:
            return {}
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Add AUC if probabilities are available
        if probabilities is not None and len(set(labels)) == 2:
            try:
                # For binary classification, use probabilities of positive class
                pos_probs = [prob[1] for prob in probabilities]
                auc = roc_auc_score(labels, pos_probs)
                metrics['auc'] = auc
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = NUM_EPOCHS, patience: int = PATIENCE) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Train Acc: {train_metrics.get('accuracy', 0):.4f}, "
                       f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(MODELS_DIR / "best_gat_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Training summary
        training_summary = {
            'num_epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        return training_summary
    
    def save_model(self, path: Path) -> None:
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, graph_data: List[Dict], 
                    test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Split data
        train_data, test_data = train_test_split(
            graph_data, test_size=test_size, random_state=42
        )
        train_data, val_data = train_test_split(
            train_data, test_size=val_size/(1-test_size), random_state=42
        )
        
        logger.info(f"Data split - Train: {len(train_data)}, "
                   f"Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create datasets
        processor = GraphDataProcessor(self.device)
        train_dataset = GraphDataset(train_data, processor)
        val_dataset = GraphDataset(val_data, processor)
        test_dataset = GraphDataset(test_data, processor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                              shuffle=False, collate_fn=self._collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                               shuffle=False, collate_fn=self._collate_fn)
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """Custom collate function for graph data"""
        from torch_geometric.data import Batch
        
        data_list = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        
        batched_data = Batch.from_data_list(data_list)
        return batched_data, labels
    
    def run_training(self, graph_data: List[Dict]) -> Dict:
        """Run complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(graph_data)
        
        # Initialize model
        model = GraphAttentionNetwork()
        trainer = GraphTrainer(model, self.device)
        trainer.setup_training()
        
        # Train model
        training_summary = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        trainer.load_model(MODELS_DIR / "best_gat_model.pth")
        test_loss, test_metrics = trainer.validate_epoch(test_loader)
        
        training_summary['test_loss'] = test_loss
        training_summary['test_metrics'] = test_metrics
        
        logger.info("Training completed!")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Metrics: {test_metrics}")
        
        return training_summary
    
    def save_training_results(self, training_summary: Dict) -> None:
        """Save training results and create plots"""
        # Save summary
        results_path = RESULTS_DIR / "training_summary.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            summary_copy = {}
            for key, value in training_summary.items():
                if isinstance(value, np.ndarray):
                    summary_copy[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Handle list of metric dictionaries
                    summary_copy[key] = [{k: float(v) if isinstance(v, np.number) else v 
                                        for k, v in item.items()} for item in value]
                else:
                    summary_copy[key] = value
            
            json.dump(summary_copy, f, indent=2)
        
        logger.info(f"Training summary saved to {results_path}")
        
        # Create training plots
        self._plot_training_curves(training_summary)
        
        # Save metrics to CSV
        self._save_metrics_csv(training_summary)
    
    def _plot_training_curves(self, training_summary: Dict) -> None:
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(training_summary['train_losses']) + 1)
        
        # Loss curves
        ax1.plot(epochs, training_summary['train_losses'], 'b-', label='Train Loss')
        ax1.plot(epochs, training_summary['val_losses'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        train_acc = [m.get('accuracy', 0) for m in training_summary['train_metrics']]
        val_acc = [m.get('accuracy', 0) for m in training_summary['val_metrics']]
        ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
        ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score curves
        train_f1 = [m.get('f1', 0) for m in training_summary['train_metrics']]
        val_f1 = [m.get('f1', 0) for m in training_summary['val_metrics']]
        ax3.plot(epochs, train_f1, 'b-', label='Train F1')
        ax3.plot(epochs, val_f1, 'r-', label='Val F1')
        ax3.set_title('Training and Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Final metrics comparison
        final_metrics = ['accuracy', 'precision', 'recall', 'f1']
        train_final = [training_summary['final_train_metrics'].get(m, 0) for m in final_metrics]
        val_final = [training_summary['final_val_metrics'].get(m, 0) for m in final_metrics]
        test_final = [training_summary['test_metrics'].get(m, 0) for m in final_metrics]
        
        x = np.arange(len(final_metrics))
        width = 0.25
        
        ax4.bar(x - width, train_final, width, label='Train', alpha=0.8)
        ax4.bar(x, val_final, width, label='Validation', alpha=0.8)
        ax4.bar(x + width, test_final, width, label='Test', alpha=0.8)
        
        ax4.set_title('Final Metrics Comparison')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(final_metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = RESULTS_DIR / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
        plt.close()
    
    def _save_metrics_csv(self, training_summary: Dict) -> None:
        """Save evaluation metrics to CSV"""
        metrics_data = []
        
        # Add training metrics
        for epoch, metrics in enumerate(training_summary['train_metrics']):
            row = {'epoch': epoch + 1, 'split': 'train'}
            row.update(metrics)
            metrics_data.append(row)
        
        # Add validation metrics
        for epoch, metrics in enumerate(training_summary['val_metrics']):
            row = {'epoch': epoch + 1, 'split': 'validation'}
            row.update(metrics)
            metrics_data.append(row)
        
        # Add test metrics
        test_row = {'epoch': 'final', 'split': 'test'}
        test_row.update(training_summary['test_metrics'])
        metrics_data.append(test_row)
        
        # Save to CSV
        df = pd.DataFrame(metrics_data)
        csv_path = RESULTS_DIR / "evaluation_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Evaluation metrics saved to {csv_path}")

def main():
    """Main function to run training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Load embedded graph data
    embedded_path = DATA_DIR / "embedded_graphs.json"
    if not embedded_path.exists():
        logger.error(f"Embedded graph data not found at {embedded_path}")
        logger.info("Please run the preprocessing pipeline first")
        return
    
    with open(embedded_path, 'r') as f:
        graph_data = json.load(f)
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline()
    training_summary = pipeline.run_training(graph_data)
    
    # Save results
    pipeline.save_training_results(training_summary)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
