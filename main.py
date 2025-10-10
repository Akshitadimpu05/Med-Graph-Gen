"""
Main execution script for the Graph Generation project
"""
import sys
import logging
from pathlib import Path
import argparse
import time
from typing import Optional
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import *
from src.preprocessing.data_loader import MIMICCXRDataLoader
from src.preprocessing.radgraph_processor import RadGraphProcessor
from src.models.embeddings import BioClinicalBERTEmbedder
from src.models.gat_model import GraphAttentionNetwork, GATTrainer
from src.training.train_pipeline import TrainingPipeline
from src.visualization.graph_viz import GraphVisualizer, GraphClustering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphGenerationPipeline:
    """Complete pipeline for graph generation from X-ray reports"""
    
    def __init__(self, max_samples=None):
        self.max_samples = max_samples or MAX_SAMPLES
        self.data_loader = None
        self.radgraph_processor = None
        self.embedder = None
        self.trainer = None
        self.visualizer = None
        self.clustering = None
        
        # Pipeline state
        self.processed_data = None
        self.graph_data = None
        self.embedded_data = None
        self.training_results = None
        
    def run_preprocessing(self) -> None:
        """Run data preprocessing pipeline"""
        logger.info("="*50)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*50)
        
        # Step 1: Load and preprocess MIMIC-CXR dataset
        logger.info("Step 1: Loading MIMIC-CXR dataset...")
        self.data_loader = MIMICCXRDataLoader(max_samples=self.max_samples)
        self.data_loader.load_dataset()
        self.processed_data = self.data_loader.extract_text_data()
        self.data_loader.save_processed_data()
        
        # Print statistics
        stats = self.data_loader.get_statistics()
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Step 2: Generate abnormality graphs using RadGraph
        logger.info("Step 2: Generating abnormality graphs...")
        self.radgraph_processor = RadGraphProcessor()
        self.graph_data = self.radgraph_processor.process_dataset(self.processed_data)
        self.radgraph_processor.save_graph_data(self.graph_data)
        
        # Print graph statistics
        graph_stats = self.radgraph_processor.get_graph_statistics(self.graph_data)
        logger.info("Graph Generation Statistics:")
        for key, value in graph_stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Step 3: Generate BioClinicalBERT embeddings
        logger.info("Step 3: Generating BioClinicalBERT embeddings...")
        self.embedder = BioClinicalBERTEmbedder()
        self.embedder.load_embedding_cache()
        self.embedded_data = self.embedder.embed_graph_nodes(self.graph_data)
        self.embedder.save_embeddings(self.embedded_data)
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Total samples processed: {len(self.embedded_data)}")
        
    def run_training(self) -> None:
        """Run model training pipeline"""
        logger.info("="*50)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*50)
        
        if self.embedded_data is None:
            logger.error("No embedded data available. Run preprocessing first.")
            return
        
        # Initialize training pipeline
        self.trainer = TrainingPipeline()
        
        # Run training
        self.training_results = self.trainer.run_training(self.embedded_data)
        
        # Save training results
        self.trainer.save_training_results(self.training_results)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {self.training_results['test_metrics'].get('accuracy', 0):.4f}")
        
    def run_evaluation_and_visualization(self) -> None:
        """Run evaluation and visualization pipeline"""
        logger.info("="*50)
        logger.info("STARTING EVALUATION AND VISUALIZATION")
        logger.info("="*50)
        
        if self.embedded_data is None:
            logger.error("No embedded data available. Run preprocessing first.")
            return
        
        # Initialize visualization and clustering
        self.visualizer = GraphVisualizer()
        self.clustering = GraphClustering()
        
        # Generate GAT embeddings for visualization
        logger.info("Generating GAT embeddings for visualization...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = GraphAttentionNetwork()
        gat_trainer = GATTrainer(model, device)
        
        # Load trained model if available
        model_path = MODELS_DIR / "best_gat_model.pth"
        if model_path.exists():
            gat_trainer.model.load_state_dict(
                torch.load(model_path, map_location=device)['model_state_dict']
            )
            logger.info("Loaded trained GAT model")
        
        # Generate embeddings
        embeddings = gat_trainer.generate_embeddings(self.embedded_data)
        
        # Visualizations
        logger.info("Creating visualizations...")
        
        # 1. Sample graph visualizations
        if self.embedded_data:
            self.visualizer.visualize_single_graph(
                self.embedded_data[0], 
                save_path=RESULTS_DIR / "sample_graph.png"
            )
            self.visualizer.visualize_multiple_graphs(self.embedded_data)
            self.visualizer.create_interactive_graph(self.embedded_data[0])
        
        # 2. Embedding visualizations
        if embeddings['graph_embeddings'].shape[0] > 0:
            self.visualizer.visualize_embeddings_2d(
                embeddings['graph_embeddings'], method='tsne'
            )
            self.visualizer.visualize_embeddings_2d(
                embeddings['graph_embeddings'], method='pca'
            )
            
            # 3. Clustering analysis
            if embeddings['graph_embeddings'].shape[0] > 2:
                kmeans_results = self.clustering.perform_kmeans_clustering(
                    embeddings['graph_embeddings']
                )
                self.clustering.visualize_clustering_results(
                    embeddings['graph_embeddings'], kmeans_results
                )
                self.clustering.save_clustering_results(kmeans_results, 'kmeans')
        
        logger.info("Evaluation and visualization completed successfully!")
        
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline"""
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("STARTING COMPLETE GRAPH GENERATION PIPELINE")
        logger.info("="*60)
        
        try:
            # Run preprocessing
            self.run_preprocessing()
            
            # Run training
            self.run_training()
            
            # Run evaluation and visualization
            self.run_evaluation_and_visualization()
            
            # Final summary
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"Results saved in: {RESULTS_DIR}")
            
            # Print final statistics
            if self.training_results:
                logger.info("Final Results Summary:")
                logger.info(f"  Samples processed: {len(self.embedded_data)}")
                logger.info(f"  Test accuracy: {self.training_results['test_metrics'].get('accuracy', 0):.4f}")
                logger.info(f"  Test F1 score: {self.training_results['test_metrics'].get('f1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
    
    def run_preprocessing_only(self) -> None:
        """Run only the preprocessing pipeline (for evaluation preparation)"""
        logger.info("="*50)
        logger.info("RUNNING PREPROCESSING ONLY")
        logger.info("="*50)
        
        self.run_preprocessing()
        
        logger.info("Preprocessing completed! Ready for evaluation.")
        logger.info(f"Processed data saved in: {DATA_DIR}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Graph Generation from X-ray Reports")
    parser.add_argument(
        '--mode', 
        choices=['full', 'preprocess', 'train', 'evaluate'], 
        default='preprocess',
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=MAX_SAMPLES,
        help='Maximum number of samples to process'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline with max_samples
    pipeline = GraphGenerationPipeline(max_samples=args.max_samples)
    
    # Run based on mode
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.mode == 'preprocess':
        pipeline.run_preprocessing_only()
    elif args.mode == 'train':
        pipeline.run_training()
    elif args.mode == 'evaluate':
        pipeline.run_evaluation_and_visualization()
    
    logger.info("Execution completed!")

if __name__ == "__main__":
    main()
