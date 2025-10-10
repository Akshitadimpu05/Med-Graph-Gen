"""
Evaluation script to demonstrate the graph generation results
Run this script to show the preprocessing and graph generation results
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_display_results():
    """Load and display all preprocessing results"""
    
    # Paths
    data_dir = Path("data")
    results_dir = Path("results")
    
    print("="*60)
    print("GRAPH GENERATION PROJECT - EVALUATION RESULTS")
    print("="*60)
    
    # 1. Dataset Statistics
    print("\n1. DATASET PREPROCESSING RESULTS")
    print("-" * 40)
    
    processed_data_path = data_dir / "processed_mimic_cxr.csv"
    if processed_data_path.exists():
        df = pd.read_csv(processed_data_path)
        print(f"✓ Total samples processed: {len(df)}")
        print(f"✓ Average text length: {df['text_length'].mean():.1f} words")
        print(f"✓ Samples with findings: {len(df[df['findings'] != ''])}")
        print(f"✓ Samples with impressions: {len(df[df['impression'] != ''])}")
        
        # Show sample texts
        print(f"\nSample processed text:")
        print(f"Findings: {df.iloc[0]['findings'][:200]}...")
        print(f"Impression: {df.iloc[0]['impression'][:200]}...")
    else:
        print("✗ Processed data not found")
    
    # 2. Graph Generation Statistics
    print(f"\n2. GRAPH GENERATION RESULTS")
    print("-" * 40)
    
    graph_stats_path = data_dir / "graph_statistics.json"
    if graph_stats_path.exists():
        with open(graph_stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"✓ Total graphs generated: {stats['total_graphs']}")
        print(f"✓ Average nodes per graph: {stats['avg_nodes_per_graph']:.2f}")
        print(f"✓ Average edges per graph: {stats['avg_edges_per_graph']:.2f}")
        print(f"✓ Average graph density: {stats['avg_graph_density']:.3f}")
        print(f"✓ Graphs with edges: {stats['graphs_with_edges']}")
        
        if 'entity_statistics' in stats:
            print(f"\nEntity extraction statistics:")
            for entity_type, count in stats['entity_statistics'].items():
                print(f"  - {entity_type}: {count}")
    else:
        print("✗ Graph statistics not found")
    
    # 3. Sample Graph Analysis
    print(f"\n3. SAMPLE GRAPH ANALYSIS")
    print("-" * 40)
    
    graphs_path = data_dir / "abnormality_graphs.json"
    if graphs_path.exists():
        with open(graphs_path, 'r') as f:
            graphs = json.load(f)
        
        # Analyze first few graphs
        for i, graph in enumerate(graphs[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Original text: {graph['original_text'][:150]}...")
            print(f"  Nodes: {graph['graph_stats']['num_nodes']}")
            print(f"  Edges: {graph['graph_stats']['num_edges']}")
            print(f"  Entities found:")
            for entity_type, entities in graph['entities'].items():
                if entities:
                    print(f"    - {entity_type}: {', '.join(entities[:3])}{'...' if len(entities) > 3 else ''}")
    else:
        print("✗ Graph data not found")
    
    # 4. Embedding Results
    print(f"\n4. EMBEDDING GENERATION RESULTS")
    print("-" * 40)
    
    embedded_path = data_dir / "embedded_graphs.json"
    if embedded_path.exists():
        with open(embedded_path, 'r') as f:
            embedded_data = json.load(f)
        
        print(f"✓ Graphs with embeddings: {len(embedded_data)}")
        
        # Check embedding dimensions
        if embedded_data and 'text_embedding' in embedded_data[0]:
            embedding_dim = len(embedded_data[0]['text_embedding'])
            print(f"✓ Embedding dimension: {embedding_dim}")
        
        # Check node embeddings
        if embedded_data and embedded_data[0]['graph']['nodes']:
            node_with_embedding = None
            for node in embedded_data[0]['graph']['nodes']:
                if 'embedding' in node:
                    node_with_embedding = node
                    break
            
            if node_with_embedding:
                print(f"✓ Node embeddings generated (dim: {len(node_with_embedding['embedding'])})")
            else:
                print("✗ Node embeddings not found")
    else:
        print("✗ Embedded data not found")
    
    # 5. Visualization Files
    print(f"\n5. GENERATED VISUALIZATIONS")
    print("-" * 40)
    
    viz_files = [
        "sample_graph.png",
        "multiple_graphs_visualization.png",
        "embeddings_2d_tsne.png",
        "embeddings_2d_pca.png",
        "clustering_analysis.png"
    ]
    
    for viz_file in viz_files:
        viz_path = results_dir / viz_file
        if viz_path.exists():
            print(f"✓ {viz_file}")
        else:
            print(f"✗ {viz_file} (not generated yet)")
    
    # 6. Training Results (if available)
    print(f"\n6. TRAINING RESULTS")
    print("-" * 40)
    
    training_path = results_dir / "training_summary.json"
    metrics_path = results_dir / "evaluation_metrics.csv"
    
    if training_path.exists():
        with open(training_path, 'r') as f:
            training_results = json.load(f)
        
        print(f"✓ Training completed")
        print(f"✓ Number of epochs: {training_results.get('num_epochs', 'N/A')}")
        
        if 'test_metrics' in training_results:
            test_metrics = training_results['test_metrics']
            print(f"✓ Test accuracy: {test_metrics.get('accuracy', 0):.4f}")
            print(f"✓ Test F1 score: {test_metrics.get('f1', 0):.4f}")
            print(f"✓ Test precision: {test_metrics.get('precision', 0):.4f}")
            print(f"✓ Test recall: {test_metrics.get('recall', 0):.4f}")
    else:
        print("✗ Training not completed yet")
    
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        print(f"✓ Detailed metrics saved ({len(metrics_df)} records)")
    else:
        print("✗ Detailed metrics not available")

def create_summary_visualization():
    """Create a summary visualization of the results"""
    
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    graphs_path = data_dir / "abnormality_graphs.json"
    if not graphs_path.exists():
        print("No graph data available for visualization")
        return
    
    with open(graphs_path, 'r') as f:
        graphs = json.load(f)
    
    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribution of nodes per graph
    node_counts = [g['graph_stats']['num_nodes'] for g in graphs]
    ax1.hist(node_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Nodes per Graph')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of edges per graph
    edge_counts = [g['graph_stats']['num_edges'] for g in graphs]
    ax2.hist(edge_counts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Edges per Graph')
    ax2.grid(True, alpha=0.3)
    
    # 3. Entity type distribution
    entity_counts = {'anatomy': 0, 'abnormality': 0, 'observation': 0}
    for graph in graphs:
        for entity_type, entities in graph['entities'].items():
            entity_counts[entity_type] += len(entities)
    
    ax3.bar(entity_counts.keys(), entity_counts.values(), 
            color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax3.set_xlabel('Entity Type')
    ax3.set_ylabel('Total Count')
    ax3.set_title('Distribution of Entity Types')
    ax3.grid(True, alpha=0.3)
    
    # 4. Graph density distribution
    densities = [g['graph_stats']['density'] for g in graphs if g['graph_stats']['density'] > 0]
    if densities:
        ax4.hist(densities, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('Graph Density')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Graph Density')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No connected graphs', ha='center', va='center')
        ax4.set_title('Graph Density (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = results_dir / "evaluation_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Summary visualization saved to {save_path}")
    plt.close()

def main():
    """Main evaluation function"""
    print("Starting evaluation of graph generation results...\n")
    
    # Display results
    load_and_display_results()
    
    # Create summary visualization
    create_summary_visualization()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    print("\nFor your evaluation tomorrow, you can:")
    print("1. Show the preprocessing statistics above")
    print("2. Display the generated graph visualizations in results/")
    print("3. Explain the pipeline architecture and components")
    print("4. Demonstrate the entity extraction and graph creation process")
    print("\nAll results are saved in the 'data/' and 'results/' directories.")

if __name__ == "__main__":
    main()
