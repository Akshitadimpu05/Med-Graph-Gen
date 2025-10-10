"""
Graph visualization and clustering module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import pickle

from config.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Visualize graphs and embeddings"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_single_graph(self, graph_sample: Dict, 
                              save_path: Optional[Path] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualize a single graph"""
        nodes = graph_sample['graph']['nodes']
        edges = graph_sample['graph']['edges']
        
        if not nodes:
            logger.warning("Empty graph, skipping visualization")
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_colors = {'anatomy': 'lightblue', 'abnormality': 'lightcoral', 'observation': 'lightgreen'}
        colors = []
        labels = {}
        
        for node in nodes:
            node_id = node['id']
            G.add_node(node_id)
            labels[node_id] = node['name']
            colors.append(node_colors.get(node['type'], 'gray'))
        
        # Add edges
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge.get('weight', 1.0))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        if len(G.nodes()) > 1:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = {list(G.nodes())[0]: (0, 0)}
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=colors, 
                              node_size=1000, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Anatomy'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Abnormality'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='Observation')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f"Abnormality Graph - Sample {graph_sample.get('sample_id', 'Unknown')}")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {save_path}")
        
        plt.close()
    
    def visualize_multiple_graphs(self, graph_data: List[Dict], 
                                 num_samples: int = 6) -> None:
        """Visualize multiple graphs in a grid"""
        num_samples = min(num_samples, len(graph_data))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_samples):
            sample = graph_data[i]
            nodes = sample['graph']['nodes']
            edges = sample['graph']['edges']
            
            if not nodes:
                axes[i].text(0.5, 0.5, 'Empty Graph', ha='center', va='center')
                axes[i].set_title(f"Sample {sample.get('sample_id', i)}")
                continue
            
            # Create NetworkX graph
            G = nx.Graph()
            node_colors = {'anatomy': 'lightblue', 'abnormality': 'lightcoral', 'observation': 'lightgreen'}
            colors = []
            
            for node in nodes:
                G.add_node(node['id'])
                colors.append(node_colors.get(node['type'], 'gray'))
            
            for edge in edges:
                G.add_edge(edge['source'], edge['target'])
            
            # Layout and draw
            if len(G.nodes()) > 1:
                pos = nx.spring_layout(G, k=0.8, iterations=30)
            else:
                pos = {list(G.nodes())[0]: (0, 0)}
            
            nx.draw_networkx_nodes(G, pos, node_color=colors, 
                                  node_size=300, alpha=0.8, ax=axes[i])
            nx.draw_networkx_edges(G, pos, alpha=0.6, ax=axes[i])
            
            axes[i].set_title(f"Sample {sample.get('sample_id', i)}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = self.results_dir / "multiple_graphs_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Multiple graphs visualization saved to {save_path}")
        plt.close()
    
    def visualize_embeddings_2d(self, embeddings: np.ndarray, 
                               labels: Optional[np.ndarray] = None,
                               method: str = 'tsne') -> None:
        """Visualize embeddings in 2D using t-SNE or PCA"""
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to visualize")
            return
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        ax.set_title(f'Graph Embeddings Visualization ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        
        save_path = self.results_dir / f"embeddings_2d_{method}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"2D embeddings visualization saved to {save_path}")
        plt.close()
    
    def create_interactive_graph(self, graph_sample: Dict) -> None:
        """Create interactive graph visualization using Plotly"""
        nodes = graph_sample['graph']['nodes']
        edges = graph_sample['graph']['edges']
        
        if not nodes:
            logger.warning("Empty graph, skipping interactive visualization")
            return
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], **node)
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Get layout
        if len(G.nodes()) > 1:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = {list(G.nodes())[0]: (0, 0)}
        
        # Prepare node traces
        node_trace = go.Scatter(
            x=[pos[node['id']][0] for node in nodes],
            y=[pos[node['id']][1] for node in nodes],
            mode='markers+text',
            text=[node['name'] for node in nodes],
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"Name: {node['name']}<br>Type: {node['type']}" for node in nodes],
            marker=dict(
                size=20,
                color=[{'anatomy': 'lightblue', 'abnormality': 'lightcoral', 
                       'observation': 'lightgreen'}.get(node['type'], 'gray') for node in nodes],
                line=dict(width=2, color='black')
            )
        )
        
        # Prepare edge traces
        edge_traces = []
        for edge in edges:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        fig.update_layout(
            title=f"Interactive Abnormality Graph - Sample {graph_sample.get('sample_id', 'Unknown')}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Save interactive plot
        save_path = self.results_dir / f"interactive_graph_sample_{graph_sample.get('sample_id', 'unknown')}.html"
        fig.write_html(save_path)
        logger.info(f"Interactive graph saved to {save_path}")

class GraphClustering:
    """Clustering analysis for graph embeddings"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.clustering_results = {}
    
    def perform_kmeans_clustering(self, embeddings: np.ndarray, 
                                 n_clusters_range: Tuple[int, int] = (2, 10)) -> Dict:
        """Perform K-means clustering with different number of clusters"""
        if embeddings.shape[0] == 0:
            return {}
        
        results = {}
        silhouette_scores = []
        inertias = []
        
        for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
            if n_clusters >= embeddings.shape[0]:
                break
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate metrics
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            inertia = kmeans.inertia_
            
            results[n_clusters] = {
                'labels': cluster_labels,
                'centroids': kmeans.cluster_centers_,
                'silhouette_score': silhouette_avg,
                'inertia': inertia
            }
            
            silhouette_scores.append(silhouette_avg)
            inertias.append(inertia)
        
        # Find optimal number of clusters
        best_k = n_clusters_range[0] + np.argmax(silhouette_scores)
        
        results['optimal_k'] = best_k
        results['silhouette_scores'] = silhouette_scores
        results['inertias'] = inertias
        
        return results
    
    def perform_dbscan_clustering(self, embeddings: np.ndarray,
                                 eps_range: Tuple[float, float] = (0.1, 2.0),
                                 n_eps: int = 10) -> Dict:
        """Perform DBSCAN clustering with different epsilon values"""
        if embeddings.shape[0] == 0:
            return {}
        
        results = {}
        eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            cluster_labels = dbscan.fit_predict(embeddings)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
            else:
                silhouette_avg = -1
            
            results[eps] = {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette_avg
            }
        
        return results
    
    def visualize_clustering_results(self, embeddings: np.ndarray, 
                                   clustering_results: Dict) -> None:
        """Visualize clustering results"""
        # K-means elbow curve
        if 'silhouette_scores' in clustering_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            k_values = range(2, len(clustering_results['silhouette_scores']) + 2)
            
            # Silhouette scores
            ax1.plot(k_values, clustering_results['silhouette_scores'], 'bo-')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Silhouette Score vs Number of Clusters')
            ax1.grid(True)
            
            # Inertia (elbow method)
            ax2.plot(k_values, clustering_results['inertias'], 'ro-')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Inertia')
            ax2.set_title('Elbow Method for Optimal k')
            ax2.grid(True)
            
            plt.tight_layout()
            save_path = self.results_dir / "clustering_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Clustering analysis saved to {save_path}")
            plt.close()
    
    def save_clustering_results(self, clustering_results: Dict, 
                               method: str = 'kmeans') -> None:
        """Save clustering results to file"""
        save_path = self.results_dir / f"{method}_clustering_results.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(clustering_results, f)
        logger.info(f"Clustering results saved to {save_path}")

def main():
    """Main function to run visualization and clustering"""
    logger.info("Starting graph visualization and clustering...")
    
    # Load graph data
    graph_path = DATA_DIR / "embedded_graphs.json"
    if not graph_path.exists():
        logger.error(f"Embedded graph data not found at {graph_path}")
        return
    
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Load GAT embeddings
    embeddings_path = RESULTS_DIR / "gat_embeddings.npz"
    if embeddings_path.exists():
        embeddings_data = np.load(embeddings_path)
        graph_embeddings = embeddings_data['graph_embeddings']
    else:
        logger.warning("GAT embeddings not found, skipping embedding analysis")
        graph_embeddings = np.array([])
    
    # Initialize visualizer and clustering
    visualizer = GraphVisualizer()
    clustering = GraphClustering()
    
    # Visualize sample graphs
    if graph_data:
        # Single graph visualization
        visualizer.visualize_single_graph(
            graph_data[0], 
            save_path=RESULTS_DIR / "sample_graph.png"
        )
        
        # Multiple graphs visualization
        visualizer.visualize_multiple_graphs(graph_data)
        
        # Interactive graph
        visualizer.create_interactive_graph(graph_data[0])
    
    # Embedding analysis
    if graph_embeddings.shape[0] > 0:
        # 2D visualization
        visualizer.visualize_embeddings_2d(graph_embeddings, method='tsne')
        visualizer.visualize_embeddings_2d(graph_embeddings, method='pca')
        
        # Clustering analysis
        if graph_embeddings.shape[0] > 2:
            kmeans_results = clustering.perform_kmeans_clustering(graph_embeddings)
            clustering.visualize_clustering_results(graph_embeddings, kmeans_results)
            clustering.save_clustering_results(kmeans_results, 'kmeans')
    
    logger.info("Graph visualization and clustering completed successfully!")

if __name__ == "__main__":
    main()
