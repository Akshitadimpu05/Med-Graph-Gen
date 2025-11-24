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
from scipy.stats import gaussian_kde
from collections import Counter
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
    
    def visualize_individual_graphs(self, graph_data: List[Dict], 
                                    num_graphs: int = 10) -> None:
        """Generate individual visualizations for multiple graphs"""
        num_graphs = min(num_graphs, len(graph_data))
        logger.info(f"Generating {num_graphs} individual graph visualizations...")
        
        for i in range(num_graphs):
            save_path = self.results_dir / f"graph_sample_{i}.png"
            self.visualize_single_graph(graph_data[i], save_path=save_path)
        
        logger.info(f"Generated {num_graphs} individual graph visualizations")
    
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
                               method: str = 'tsne',
                               graph_data: Optional[List[Dict]] = None) -> np.ndarray:
        """Visualize embeddings in 2D using t-SNE or PCA with disease type coloring"""
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings to visualize")
            return None
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0]-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Extract disease categories from graph data
        if graph_data is not None:
            disease_labels, disease_colors = self._extract_disease_labels(graph_data)
            
            # Create comprehensive visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot 1: Colored by disease type
            for disease, color in disease_colors.items():
                mask = np.array([label == disease for label in disease_labels])
                if mask.any():
                    ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=color, label=disease, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            
            ax1.set_title(f'Medical Graph Embeddings by Disease Type ({method.upper()})', fontsize=14, fontweight='bold')
            ax1.set_xlabel(f'{method.upper()} Component 1' + 
                          (f' ({explained_var[0]:.1%} var)' if method == 'pca' else ''), fontsize=12)
            ax1.set_ylabel(f'{method.upper()} Component 2' + 
                          (f' ({explained_var[1]:.1%} var)' if method == 'pca' else ''), fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Density/heatmap
            if embeddings_2d.shape[0] > 2:
                try:
                    xy = embeddings_2d.T
                    density = gaussian_kde(xy)(xy)
                    scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                        c=density, cmap='YlOrRd', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
                    plt.colorbar(scatter, ax=ax2, label='Density')
                    ax2.set_title(f'Embedding Density Heatmap ({method.upper()})', fontsize=14, fontweight='bold')
                except:
                    # Fallback if density estimation fails
                    ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
                    ax2.set_title(f'All Embeddings ({method.upper()})', fontsize=14, fontweight='bold')
            else:
                ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
                ax2.set_title(f'All Embeddings ({method.upper()})', fontsize=14, fontweight='bold')
            
            ax2.set_xlabel(f'{method.upper()} Component 1' + 
                          (f' ({explained_var[0]:.1%} var)' if method == 'pca' else ''), fontsize=12)
            ax2.set_ylabel(f'{method.upper()} Component 2' + 
                          (f' ({explained_var[1]:.1%} var)' if method == 'pca' else ''), fontsize=12)
            ax2.grid(True, alpha=0.3)
            
        else:
            # Fallback: simple visualization
            fig, ax1 = plt.subplots(figsize=(10, 8))
            if labels is not None:
                scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=labels, cmap='viridis', alpha=0.7, s=100)
                plt.colorbar(scatter, ax=ax1)
            else:
                ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
            
            ax1.set_title(f'Graph Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
            ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
            ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / f"embeddings_2d_{method}_enhanced.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced 2D embeddings visualization saved to {save_path}")
        plt.close()
        
        return embeddings_2d
    
    def _extract_disease_labels(self, graph_data: List[Dict]) -> Tuple[List[str], Dict[str, str]]:
        """Extract disease categories from graph data"""
        disease_labels = []
        
        for sample in graph_data:
            abnormalities = [node['name'] for node in sample['graph']['nodes'] 
                           if node['type'] == 'abnormality']
            
            # Categorize based on primary abnormality
            if not abnormalities:
                disease_labels.append('Normal')
            elif any('pneumonia' in abn.lower() for abn in abnormalities):
                disease_labels.append('Pneumonia')
            elif any('effusion' in abn.lower() for abn in abnormalities):
                disease_labels.append('Effusion')
            elif any('pneumothorax' in abn.lower() for abn in abnormalities):
                disease_labels.append('Pneumothorax')
            elif any('edema' in abn.lower() for abn in abnormalities):
                disease_labels.append('Edema')
            elif any('mass' in abn.lower() or 'nodule' in abn.lower() for abn in abnormalities):
                disease_labels.append('Mass/Nodule')
            elif any('cardiomegaly' in abn.lower() or 'enlargement' in abn.lower() for abn in abnormalities):
                disease_labels.append('Cardiomegaly')
            elif len(abnormalities) > 2:
                disease_labels.append('Mixed Pathologies')
            else:
                disease_labels.append('Other Abnormality')
        
        # Define colors for each disease type
        disease_colors = {
            'Normal': '#2ecc71',  # Green
            'Pneumonia': '#e74c3c',  # Red
            'Effusion': '#3498db',  # Blue
            'Pneumothorax': '#f39c12',  # Orange
            'Edema': '#9b59b6',  # Purple
            'Mass/Nodule': '#e67e22',  # Dark orange
            'Cardiomegaly': '#1abc9c',  # Turquoise
            'Mixed Pathologies': '#95a5a6',  # Gray
            'Other Abnormality': '#34495e'  # Dark gray
        }
        
        return disease_labels, disease_colors
    
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
    
    def _extract_disease_labels(self, graph_data: List[Dict]) -> Tuple[List[str], Dict[str, str]]:
        """Extract disease categories from graph data"""
        disease_labels = []
        
        for sample in graph_data:
            abnormalities = [node['name'] for node in sample['graph']['nodes'] 
                           if node['type'] == 'abnormality']
            
            # Categorize based on primary abnormality
            if not abnormalities:
                disease_labels.append('Normal')
            elif any('pneumonia' in abn.lower() for abn in abnormalities):
                disease_labels.append('Pneumonia')
            elif any('effusion' in abn.lower() for abn in abnormalities):
                disease_labels.append('Effusion')
            elif any('pneumothorax' in abn.lower() for abn in abnormalities):
                disease_labels.append('Pneumothorax')
            elif any('edema' in abn.lower() for abn in abnormalities):
                disease_labels.append('Edema')
            elif any('mass' in abn.lower() or 'nodule' in abn.lower() for abn in abnormalities):
                disease_labels.append('Mass/Nodule')
            elif any('cardiomegaly' in abn.lower() or 'enlargement' in abn.lower() for abn in abnormalities):
                disease_labels.append('Cardiomegaly')
            elif len(abnormalities) > 2:
                disease_labels.append('Mixed Pathologies')
            else:
                disease_labels.append('Other Abnormality')
        
        # Define colors for each disease type
        disease_colors = {
            'Normal': '#2ecc71',  # Green
            'Pneumonia': '#e74c3c',  # Red
            'Effusion': '#3498db',  # Blue
            'Pneumothorax': '#f39c12',  # Orange
            'Edema': '#9b59b6',  # Purple
            'Mass/Nodule': '#e67e22',  # Dark orange
            'Cardiomegaly': '#1abc9c',  # Turquoise
            'Mixed Pathologies': '#95a5a6',  # Gray
            'Other Abnormality': '#34495e'  # Dark gray
        }
        
        return disease_labels, disease_colors
    
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
                                   clustering_results: Dict,
                                   embeddings_2d: Optional[np.ndarray] = None,
                                   graph_data: Optional[List[Dict]] = None) -> None:
        """Visualize comprehensive clustering results with medical interpretation"""
        # K-means elbow curve and metrics
        if 'silhouette_scores' in clustering_results:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            k_values = range(2, len(clustering_results['silhouette_scores']) + 2)
            
            # Plot 1: Silhouette scores
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(k_values, clustering_results['silhouette_scores'], 'bo-', linewidth=2, markersize=8)
            optimal_k = clustering_results.get('optimal_k', k_values[np.argmax(clustering_results['silhouette_scores'])])
            ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax1.set_ylabel('Silhouette Score', fontsize=12)
            ax1.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Inertia (elbow method)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(k_values, clustering_results['inertias'], 'ro-', linewidth=2, markersize=8)
            ax2.axvline(x=optimal_k, color='b', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax2.set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
            ax2.set_title('Elbow Method', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Cluster size distribution
            ax3 = fig.add_subplot(gs[0, 2])
            if optimal_k in clustering_results:
                cluster_labels = clustering_results[optimal_k]['labels']
                unique, counts = np.unique(cluster_labels, return_counts=True)
                ax3.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Cluster ID', fontsize=12)
                ax3.set_ylabel('Number of Samples', fontsize=12)
                ax3.set_title(f'Cluster Size Distribution (k={optimal_k})', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add counts on top of bars
                for i, (cluster_id, count) in enumerate(zip(unique, counts)):
                    ax3.text(cluster_id, count, str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot 4-6: 2D clustering visualization if embeddings_2d provided
            if embeddings_2d is not None and optimal_k in clustering_results:
                cluster_labels = clustering_results[optimal_k]['labels']
                
                # Plot 4: K-means clusters
                ax4 = fig.add_subplot(gs[1, :])
                scatter = ax4.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                    c=cluster_labels, cmap='tab10', alpha=0.7, s=100, 
                                    edgecolors='black', linewidth=0.5)
                
                # Plot centroids if available
                if 'centroids' in clustering_results[optimal_k]:
                    # Project centroids to 2D if needed
                    centroids = clustering_results[optimal_k]['centroids']
                    if centroids.shape[1] != 2:
                        # Use PCA to project centroids
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        pca.fit(embeddings)
                        centroids_2d = pca.transform(centroids)
                    else:
                        centroids_2d = centroids
                    
                    ax4.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                              c='red', marker='X', s=500, edgecolors='black', 
                              linewidth=2, label='Centroids', zorder=5)
                
                ax4.set_title(f'K-means Clustering (k={optimal_k})', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Component 1', fontsize=12)
                ax4.set_ylabel('Component 2', fontsize=12)
                ax4.grid(True, alpha=0.3)
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Cluster ID', fontsize=12)
                if 'centroids' in clustering_results[optimal_k]:
                    ax4.legend()
                
                # Plot 5-6: Disease type comparison (if graph_data available)
                if graph_data is not None:
                    disease_labels, disease_colors = self._extract_disease_labels(graph_data)
                    
                    # Plot 5: True disease types
                    ax5 = fig.add_subplot(gs[2, 0])
                    for disease, color in disease_colors.items():
                        mask = np.array([label == disease for label in disease_labels])
                        if mask.any():
                            ax5.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                      c=color, label=disease, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
                    ax5.set_title('True Disease Categories', fontsize=12, fontweight='bold')
                    ax5.set_xlabel('Component 1', fontsize=10)
                    ax5.set_ylabel('Component 2', fontsize=10)
                    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax5.grid(True, alpha=0.3)
                    
                    # Plot 6: Cluster interpretation
                    ax6 = fig.add_subplot(gs[2, 1:])
                    cluster_disease_dist = self._analyze_cluster_composition(cluster_labels, disease_labels)
                    self._plot_cluster_composition(ax6, cluster_disease_dist, disease_colors)
            
            plt.tight_layout()
            save_path = self.results_dir / "clustering_analysis_comprehensive.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive clustering analysis saved to {save_path}")
            plt.close()
    
    def _analyze_cluster_composition(self, cluster_labels: np.ndarray, 
                                    disease_labels: List[str]) -> Dict:
        """Analyze what disease types are in each cluster"""
        cluster_composition = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            diseases_in_cluster = [disease_labels[i] for i in range(len(disease_labels)) if mask[i]]
            
            disease_counts = Counter(diseases_in_cluster)
            total = len(diseases_in_cluster)
            
            cluster_composition[cluster_id] = {
                'total': total,
                'diseases': {disease: count/total for disease, count in disease_counts.items()}
            }
        
        return cluster_composition
    
    def _plot_cluster_composition(self, ax, cluster_composition: Dict, 
                                 disease_colors: Dict) -> None:
        """Plot stacked bar chart showing disease composition of each cluster"""
        clusters = sorted(cluster_composition.keys())
        
        # Get all disease types
        all_diseases = set()
        for comp in cluster_composition.values():
            all_diseases.update(comp['diseases'].keys())
        all_diseases = sorted(all_diseases)
        
        # Create stacked bar data
        bottom = np.zeros(len(clusters))
        
        for disease in all_diseases:
            heights = [cluster_composition[c]['diseases'].get(disease, 0) 
                      for c in clusters]
            ax.bar(clusters, heights, bottom=bottom, 
                  label=disease, color=disease_colors.get(disease, 'gray'),
                  edgecolor='black', linewidth=0.5)
            bottom += heights
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_title('Disease Composition per Cluster', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(clusters)
    
    def visualize_dbscan_results(self, embeddings: np.ndarray,
                                embeddings_2d: np.ndarray,
                                dbscan_results: Dict,
                                graph_data: Optional[List[Dict]] = None) -> None:
        """Visualize DBSCAN clustering results"""
        if not dbscan_results:
            return
        
        # Find best eps (highest silhouette score with reasonable clusters)
        best_eps = None
        best_score = -1
        
        for eps, result in dbscan_results.items():
            if result['n_clusters'] > 1 and result['silhouette_score'] > best_score:
                best_score = result['silhouette_score']
                best_eps = eps
        
        if best_eps is None:
            logger.warning("No good DBSCAN clustering found")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: DBSCAN clusters with best eps
        result = dbscan_results[best_eps]
        cluster_labels = result['labels']
        
        # Separate outliers
        outliers = cluster_labels == -1
        clusters = cluster_labels != -1
        
        ax1 = axes[0, 0]
        if clusters.any():
            scatter = ax1.scatter(embeddings_2d[clusters, 0], embeddings_2d[clusters, 1],
                                c=cluster_labels[clusters], cmap='tab10', alpha=0.7, s=100,
                                edgecolors='black', linewidth=0.5, label='Clusters')
            plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        if outliers.any():
            ax1.scatter(embeddings_2d[outliers, 0], embeddings_2d[outliers, 1],
                       c='red', marker='x', s=100, alpha=0.8, linewidth=2, label='Outliers')
        
        ax1.set_title(f'DBSCAN Clustering (eps={best_eps:.2f})\n'
                     f'{result["n_clusters"]} clusters, {result["n_noise"]} outliers',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Component 1', fontsize=11)
        ax1.set_ylabel('Component 2', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter sensitivity
        ax2 = axes[0, 1]
        eps_vals = sorted(dbscan_results.keys())
        n_clusters = [dbscan_results[eps]['n_clusters'] for eps in eps_vals]
        n_noise = [dbscan_results[eps]['n_noise'] for eps in eps_vals]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(eps_vals, n_clusters, 'b-o', label='# Clusters', linewidth=2)
        line2 = ax2_twin.plot(eps_vals, n_noise, 'r-s', label='# Outliers', linewidth=2)
        ax2.axvline(x=best_eps, color='g', linestyle='--', linewidth=2, label=f'Best eps={best_eps:.2f}')
        
        ax2.set_xlabel('Epsilon (eps)', fontsize=11)
        ax2.set_ylabel('Number of Clusters', color='b', fontsize=11)
        ax2_twin.set_ylabel('Number of Outliers', color='r', fontsize=11)
        ax2.set_title('DBSCAN Parameter Sensitivity', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        lines = line1 + line2 + [ax2.get_lines()[-1]]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Outlier analysis
        if outliers.any() and graph_data is not None:
            ax3 = axes[1, 0]
            disease_labels, _ = self._extract_disease_labels(graph_data)
            outlier_diseases = [disease_labels[i] for i in range(len(disease_labels)) if outliers[i]]
            
            disease_counts = Counter(outlier_diseases)
            
            diseases = list(disease_counts.keys())
            counts = list(disease_counts.values())
            
            ax3.barh(diseases, counts, color='coral', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Count', fontsize=11)
            ax3.set_title(f'Outlier Analysis: Rare Pathological Patterns\n'
                         f'Total Outliers: {result["n_noise"]}',
                         fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            for i, (disease, count) in enumerate(zip(diseases, counts)):
                ax3.text(count, i, f' {count}', va='center', fontweight='bold')
        
        # Plot 4: Silhouette scores
        ax4 = axes[1, 1]
        silhouette_scores = [dbscan_results[eps]['silhouette_score'] 
                           for eps in eps_vals 
                           if dbscan_results[eps]['silhouette_score'] > -1]
        valid_eps = [eps for eps in eps_vals if dbscan_results[eps]['silhouette_score'] > -1]
        
        if silhouette_scores:
            ax4.plot(valid_eps, silhouette_scores, 'go-', linewidth=2, markersize=8)
            ax4.axvline(x=best_eps, color='r', linestyle='--', linewidth=2, 
                       label=f'Best eps={best_eps:.2f}, score={best_score:.3f}')
            ax4.set_xlabel('Epsilon (eps)', fontsize=11)
            ax4.set_ylabel('Silhouette Score', fontsize=11)
            ax4.set_title('DBSCAN Quality Metric', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / "dbscan_clustering_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"DBSCAN clustering analysis saved to {save_path}")
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
