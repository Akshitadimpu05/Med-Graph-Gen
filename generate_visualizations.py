"""
Generate comprehensive visualizations and clustering analysis
This script creates all enhanced visualizations with medical interpretation
"""
import sys
import logging
from pathlib import Path
import numpy as np
import json
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import *
from src.models.gat_model import GraphAttentionNetwork, GATTrainer
from src.visualization.graph_viz import GraphVisualizer, GraphClustering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate all visualizations with medical interpretation"""
    
    logger.info("="*70)
    logger.info("MEDICAL GRAPH VISUALIZATION & CLUSTERING ANALYSIS")
    logger.info("="*70)
    
    # Load embedded graph data
    embedded_path = DATA_DIR / "embedded_graphs.json"
    if not embedded_path.exists():
        logger.error(f"Embedded data not found at {embedded_path}")
        logger.error("Please run: python main.py --mode preprocess")
        return
    
    logger.info(f"Loading embedded graphs from {embedded_path}...")
    with open(embedded_path, 'r') as f:
        graph_data = json.load(f)
    logger.info(f"Loaded {len(graph_data)} graphs")
    
    # Initialize visualization and clustering
    visualizer = GraphVisualizer()
    clustering = GraphClustering()
    
    # ========== STEP 1: Graph Visualizations ==========
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Graph Structure Visualizations")
    logger.info("="*70)
    
    if graph_data:
        # Individual graph samples
        logger.info("Creating 10 individual graph visualizations...")
        visualizer.visualize_individual_graphs(graph_data, num_graphs=10)
        
        # Grid of multiple graphs
        logger.info("Creating grid visualization of 6 sample graphs...")
        visualizer.visualize_multiple_graphs(graph_data)
        
        # Interactive graph
        logger.info("Creating interactive graph visualization...")
        visualizer.create_interactive_graph(graph_data[0])
    
    # ========== STEP 2: Generate Graph Embeddings ==========
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Generate GAT Embeddings (256-dim vectors)")
    logger.info("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = GraphAttentionNetwork()
    gat_trainer = GATTrainer(model, device)
    
    # Load trained model if available
    model_path = MODELS_DIR / "best_gat_model.pth"
    if model_path.exists():
        gat_trainer.model.load_state_dict(
            torch.load(model_path, map_location=device)['model_state_dict']
        )
        logger.info("✓ Loaded trained GAT model")
    else:
        logger.warning("⚠ No trained model found, using untrained GAT")
    
    # Generate embeddings
    logger.info("Generating graph embeddings...")
    embeddings = gat_trainer.generate_embeddings(graph_data)
    graph_embeddings = embeddings['graph_embeddings']
    
    logger.info(f"✓ Generated {graph_embeddings.shape[0]} graph embeddings")
    logger.info(f"  Embedding dimension: {graph_embeddings.shape[1]}")
    
    # Save embeddings (only graph embeddings which have uniform shape)
    embeddings_path = RESULTS_DIR / "gat_embeddings.npz"
    np.savez(embeddings_path, graph_embeddings=graph_embeddings)
    logger.info(f"✓ Saved embeddings to {embeddings_path}")
    
    if graph_embeddings.shape[0] == 0:
        logger.error("No embeddings generated. Exiting.")
        return
    
    # ========== STEP 3: 2D Projections with Disease Categories ==========
    logger.info("\n" + "="*70)
    logger.info("STEP 3: 2D Projections (t-SNE & PCA) with Disease Categories")
    logger.info("="*70)
    
    logger.info("Creating t-SNE visualization...")
    logger.info("  → Shows: Pneumonia, Effusion, Normal, Mixed Pathologies, etc.")
    tsne_embeddings_2d = visualizer.visualize_embeddings_2d(
        graph_embeddings, 
        method='tsne',
        graph_data=graph_data
    )
    logger.info("✓ t-SNE plot with disease categories saved")
    
    logger.info("Creating PCA visualization with density heatmap...")
    pca_embeddings_2d = visualizer.visualize_embeddings_2d(
        graph_embeddings, 
        method='pca',
        graph_data=graph_data
    )
    logger.info("✓ PCA plot with variance explained saved")
    
    # ========== STEP 4: K-means Clustering Analysis ==========
    logger.info("\n" + "="*70)
    logger.info("STEP 4: K-means Clustering (Unsupervised Disease Categories)")
    logger.info("="*70)
    
    if graph_embeddings.shape[0] > 2:
        logger.info("Running K-means with k=2 to k=10...")
        kmeans_results = clustering.perform_kmeans_clustering(
            graph_embeddings,
            n_clusters_range=(2, 10)
        )
        
        optimal_k = kmeans_results.get('optimal_k', 2)
        logger.info(f"✓ Optimal number of clusters: {optimal_k}")
        logger.info(f"  Silhouette score: {kmeans_results[optimal_k]['silhouette_score']:.3f}")
        
        # Create comprehensive visualization
        logger.info("Creating comprehensive K-means analysis plots...")
        logger.info("  → Silhouette analysis")
        logger.info("  → Elbow method")
        logger.info("  → Cluster size distribution")
        logger.info("  → 2D cluster visualization with centroids")
        logger.info("  → Disease composition per cluster")
        clustering.visualize_clustering_results(
            graph_embeddings,
            kmeans_results,
            embeddings_2d=tsne_embeddings_2d,
            graph_data=graph_data
        )
        logger.info("✓ Comprehensive K-means analysis saved")
        
        clustering.save_clustering_results(kmeans_results, 'kmeans')
        logger.info("✓ K-means results saved to pickle file")
        
        # ========== STEP 5: DBSCAN Clustering (Outlier Detection) ==========
        logger.info("\n" + "="*70)
        logger.info("STEP 5: DBSCAN Clustering (Outlier & Rare Pattern Detection)")
        logger.info("="*70)
        
        logger.info("Running DBSCAN with multiple epsilon values...")
        dbscan_results = clustering.perform_dbscan_clustering(
            graph_embeddings,
            eps_range=(0.5, 3.0),
            n_eps=15
        )
        
        if dbscan_results:
            # Find best result
            best_eps = None
            best_score = -1
            for eps, result in dbscan_results.items():
                if result['n_clusters'] > 1 and result['silhouette_score'] > best_score:
                    best_score = result['silhouette_score']
                    best_eps = eps
            
            if best_eps:
                result = dbscan_results[best_eps]
                logger.info(f"✓ Best epsilon: {best_eps:.2f}")
                logger.info(f"  Number of clusters: {result['n_clusters']}")
                logger.info(f"  Number of outliers: {result['n_noise']}")
                logger.info(f"  Silhouette score: {result['silhouette_score']:.3f}")
                
                logger.info("Creating DBSCAN analysis plots...")
                logger.info("  → Cluster visualization with outliers")
                logger.info("  → Parameter sensitivity analysis")
                logger.info("  → Outlier disease distribution")
                logger.info("  → Quality metrics")
                clustering.visualize_dbscan_results(
                    graph_embeddings,
                    tsne_embeddings_2d,
                    dbscan_results,
                    graph_data=graph_data
                )
                logger.info("✓ DBSCAN analysis saved")
                
                clustering.save_clustering_results(dbscan_results, 'dbscan')
                logger.info("✓ DBSCAN results saved to pickle file")
            else:
                logger.warning("⚠ Could not find optimal DBSCAN parameters")
        else:
            logger.warning("⚠ DBSCAN clustering failed")
    else:
        logger.warning("⚠ Not enough samples for clustering analysis")
    
    # ========== SUMMARY ==========
    logger.info("\n" + "="*70)
    logger.info("✅ VISUALIZATION & CLUSTERING COMPLETE!")
    logger.info("="*70)
    logger.info("\nGenerated Files:")
    logger.info("─" * 70)
    
    results = [
        ("Graph Visualizations", [
            "graph_sample_0.png to graph_sample_9.png",
            "multiple_graphs_visualization.png",
            "interactive_graph_sample_0.html"
        ]),
        ("Embedding Visualizations", [
            "embeddings_2d_tsne_enhanced.png (with disease categories)",
            "embeddings_2d_pca_enhanced.png (with density heatmap)"
        ]),
        ("K-means Analysis", [
            "clustering_analysis_comprehensive.png (9 subplots)",
            "kmeans_clustering_results.pkl"
        ]),
        ("DBSCAN Analysis", [
            "dbscan_clustering_analysis.png (4 subplots)",
            "dbscan_clustering_results.pkl"
        ]),
        ("Embeddings", [
            "gat_embeddings.npz"
        ])
    ]
    
    for category, files in results:
        logger.info(f"\n📊 {category}:")
        for file in files:
            logger.info(f"   • {file}")
    
    logger.info(f"\n📁 All results saved in: {RESULTS_DIR}")
    
    # Medical Interpretation
    logger.info("\n" + "="*70)
    logger.info("🔬 MEDICAL INTERPRETATION GUIDE")
    logger.info("="*70)
    logger.info("""
What to look for in the visualizations:

1. Disease Category Clustering (t-SNE/PCA plots):
   • Do pneumonia cases cluster together? (Red points)
   • Are normal cases separate? (Green points)
   • Do effusion cases form their own cluster? (Blue points)
   • Are mixed pathologies in between? (Gray points)

2. K-means Cluster Composition:
   • Which diseases dominate each cluster?
   • Are clusters medically meaningful?
   • Do similar abnormalities group together?

3. DBSCAN Outliers:
   • What are the rare pathological patterns?
   • Which disease types appear as outliers?
   • These could be unusual/complex cases

4. Embedding Quality:
   • High silhouette scores = good separation
   • Clear clusters = model learned meaningful representations
   • Overlap = structurally similar diseases
    """)
    
    logger.info("="*70)
    logger.info("Done! Open the PNG files to see the results.")
    logger.info("="*70)

if __name__ == "__main__":
    main()
