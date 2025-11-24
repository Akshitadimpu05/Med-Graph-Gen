"""
Graph Attention Network (GAT) implementation for graph embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm

from config.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for learning graph embeddings"""
    
    def __init__(self, 
                 input_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = HIDDEN_DIM,
                 num_heads: int = GAT_HEADS,
                 num_layers: int = GAT_LAYERS,
                 dropout: float = 0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, 
                       heads=num_heads, dropout=dropout)
            )
        
        # Output layer
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, output_dim, 
                       heads=1, dropout=dropout)
            )
        
        # Graph-level pooling
        self.graph_pooling = 'mean'  # Can be 'mean', 'max', or 'attention'
        
        # Optional graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification example
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GAT
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices for graph-level pooling [num_nodes]
        
        Returns:
            node_embeddings: Node-level embeddings [num_nodes, output_dim]
            graph_embeddings: Graph-level embeddings [batch_size, output_dim]
        """
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        node_embeddings = x
        
        # Graph-level pooling
        if batch is not None:
            if self.graph_pooling == 'mean':
                graph_embeddings = global_mean_pool(x, batch)
            elif self.graph_pooling == 'max':
                graph_embeddings = global_max_pool(x, batch)
            else:
                graph_embeddings = global_mean_pool(x, batch)  # Default
        else:
            # Single graph case
            graph_embeddings = torch.mean(x, dim=0, keepdim=True)
        
        return node_embeddings, graph_embeddings
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights from GAT layers"""
        attention_weights = []
        
        for gat_layer in self.gat_layers:
            # Get attention weights (this requires modifying GATConv to return attention)
            # For now, we'll return empty list
            pass
        
        return attention_weights

class GraphDataProcessor:
    """Process graph data for GAT training"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def create_pyg_data(self, graph_sample: Dict) -> Data:
        """Convert graph sample to PyTorch Geometric Data object"""
        nodes = graph_sample['graph']['nodes']
        edges = graph_sample['graph']['edges']
        
        if not nodes:
            # Create dummy graph with single node
            x = torch.zeros((1, EMBEDDING_DIM))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # Node features
        node_features = []
        node_id_map = {}
        
        for i, node in enumerate(nodes):
            node_id_map[node['id']] = i
            if 'embedding' in node:
                node_features.append(node['embedding'])
            else:
                # Use zero embedding if not available
                node_features.append([0.0] * EMBEDDING_DIM)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Edge indices
        edge_indices = []
        edge_weights = []
        
        for edge in edges:
            src_id = edge['source']
            tgt_id = edge['target']
            
            if src_id in node_id_map and tgt_id in node_id_map:
                src_idx = node_id_map[src_id]
                tgt_idx = node_id_map[tgt_id]
                
                edge_indices.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])  # Undirected
                weight = edge.get('weight', 1.0)
                edge_weights.extend([weight, weight])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def create_batch(self, graph_samples: List[Dict]) -> Batch:
        """Create batch of PyTorch Geometric Data objects"""
        data_list = []
        
        for sample in graph_samples:
            data = self.create_pyg_data(sample)
            data_list.append(data)
        
        return Batch.from_data_list(data_list)

class GATTrainer:
    """Trainer class for GAT model"""
    
    def __init__(self, model: GraphAttentionNetwork, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.processor = GraphDataProcessor(device)
        
        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
    
    def setup_training(self, learning_rate: float = LEARNING_RATE):
        """Setup training components"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # For reconstruction task
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_step(self, batch_data: Batch) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        node_embeddings, graph_embeddings = self.model(
            batch_data.x, batch_data.edge_index, batch_data.batch
        )
        
        # Simple reconstruction loss (can be modified for specific tasks)
        # For now, we'll use a dummy loss
        loss = torch.mean(torch.sum(graph_embeddings ** 2, dim=1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def generate_embeddings(self, graph_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all graphs"""
        self.model.eval()
        
        all_node_embeddings = []
        all_graph_embeddings = []
        
        logger.info("Generating GAT embeddings...")
        
        with torch.no_grad():
            for sample in tqdm(graph_data, desc="Processing graphs"):
                data = self.processor.create_pyg_data(sample)
                data = data.to(self.device)
                
                if data.x.size(0) == 0:  # Skip empty graphs
                    continue
                
                node_emb, graph_emb = self.model(data.x, data.edge_index)
                
                all_node_embeddings.append(node_emb.cpu().numpy())
                all_graph_embeddings.append(graph_emb.cpu().numpy())
        
        return {
            'node_embeddings': all_node_embeddings,
            'graph_embeddings': np.vstack(all_graph_embeddings) if all_graph_embeddings else np.array([])
        }

def main():
    """Main function to run GAT model"""
    logger.info("Starting GAT model processing...")
    
    # Load embedded graph data
    embedded_path = DATA_DIR / "embedded_graphs.json"
    if not embedded_path.exists():
        logger.error(f"Embedded graph data not found at {embedded_path}")
        logger.info("Please run embeddings.py first to generate embeddings")
        return
    
    with open(embedded_path, 'r') as f:
        graph_data = json.load(f)
    
    # Initialize GAT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = GraphAttentionNetwork()
    trainer = GATTrainer(model, device)
    trainer.setup_training()
    
    # Generate embeddings (without training for now)
    embeddings = trainer.generate_embeddings(graph_data)
    
    # Save embeddings
    output_path = RESULTS_DIR / "gat_embeddings.npz"
    np.savez(output_path, **embeddings)
    logger.info(f"GAT embeddings saved to {output_path}")
    
    # Print statistics
    logger.info("GAT Processing Statistics:")
    logger.info(f"  Total graphs processed: {len(graph_data)}")
    logger.info(f"  Graph embeddings shape: {embeddings['graph_embeddings'].shape}")
    logger.info(f"  Device used: {device}")
    
    logger.info("GAT model processing completed successfully!")

if __name__ == "__main__":
    main()
