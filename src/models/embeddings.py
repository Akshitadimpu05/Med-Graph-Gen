"""
BioClinicalBERT embeddings module for generating term embeddings
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pickle

from config.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioClinicalBERTEmbedder:
    """Generate embeddings using BioClinicalBERT for medical terms and text"""
    
    def __init__(self, model_name: str = BIOCLINICAL_BERT_MODEL, 
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.embedding_cache = {}
        
        logger.info(f"Initializing BioClinicalBERT embedder on {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load BioClinicalBERT model and tokenizer"""
        try:
            logger.info(f"Loading BioClinicalBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("BioClinicalBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BioClinicalBERT model: {e}")
            raise
    
    def get_text_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        """Get embedding for a text string"""
        if not text or not text.strip():
            return np.zeros(EMBEDDING_DIM)
        
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embedding = embedding.squeeze()
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Error getting embedding for text: {e}")
            return np.zeros(EMBEDDING_DIM)
    
    def get_term_embeddings(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of terms"""
        embeddings = {}
        
        logger.info(f"Generating embeddings for {len(terms)} terms...")
        for term in tqdm(terms, desc="Generating term embeddings"):
            embeddings[term] = self.get_text_embedding(term)
        
        return embeddings
    
    def embed_graph_nodes(self, graph_data: List[Dict]) -> List[Dict]:
        """Add embeddings to graph nodes"""
        logger.info("Adding embeddings to graph nodes...")
        
        # Collect all unique terms
        all_terms = set()
        for sample in graph_data:
            for node in sample['graph']['nodes']:
                all_terms.add(node['name'])
        
        # Generate embeddings for all terms
        term_embeddings = self.get_term_embeddings(list(all_terms))
        
        # Add embeddings to graph data
        embedded_data = []
        for sample in tqdm(graph_data, desc="Embedding graph nodes"):
            embedded_sample = sample.copy()
            
            # Add embeddings to nodes
            for node in embedded_sample['graph']['nodes']:
                node_name = node['name']
                node['embedding'] = term_embeddings[node_name].tolist()
            
            # Add text embedding for the entire sample
            text_embedding = self.get_text_embedding(sample['original_text'])
            embedded_sample['text_embedding'] = text_embedding.tolist()
            
            embedded_data.append(embedded_sample)
        
        logger.info(f"Successfully embedded {len(embedded_data)} graph samples")
        return embedded_data
    
    def create_node_feature_matrix(self, graph_sample: Dict) -> torch.Tensor:
        """Create node feature matrix for a single graph"""
        nodes = graph_sample['graph']['nodes']
        
        if not nodes:
            return torch.zeros((1, EMBEDDING_DIM))
        
        # Stack node embeddings
        node_features = []
        for node in nodes:
            if 'embedding' in node:
                node_features.append(node['embedding'])
            else:
                # Generate embedding if not present
                embedding = self.get_text_embedding(node['name'])
                node_features.append(embedding.tolist())
        
        return torch.tensor(node_features, dtype=torch.float32)
    
    def create_adjacency_matrix(self, graph_sample: Dict) -> torch.Tensor:
        """Create adjacency matrix for a single graph"""
        nodes = graph_sample['graph']['nodes']
        edges = graph_sample['graph']['edges']
        
        num_nodes = len(nodes)
        if num_nodes == 0:
            return torch.zeros((1, 1))
        
        # Create node id mapping
        node_id_map = {node['id']: i for i, node in enumerate(nodes)}
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        
        # Fill adjacency matrix
        for edge in edges:
            src_idx = node_id_map.get(edge['source'])
            tgt_idx = node_id_map.get(edge['target'])
            
            if src_idx is not None and tgt_idx is not None:
                weight = edge.get('weight', 1.0)
                adj_matrix[src_idx, tgt_idx] = weight
                adj_matrix[tgt_idx, src_idx] = weight  # Undirected graph
        
        return adj_matrix
    
    def save_embeddings(self, embedded_data: List[Dict], 
                       output_path: Optional[Path] = None) -> None:
        """Save embedded graph data"""
        if output_path is None:
            output_path = DATA_DIR / "embedded_graphs.json"
        
        logger.info(f"Saving embedded graph data to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(embedded_data, f, indent=2)
        
        # Save embedding cache
        cache_path = output_path.parent / "embedding_cache.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        
        logger.info(f"Embedding cache saved to {cache_path}")
    
    def load_embedding_cache(self, cache_path: Optional[Path] = None) -> None:
        """Load embedding cache from file"""
        if cache_path is None:
            cache_path = DATA_DIR / "embedding_cache.pkl"
        
        if cache_path.exists():
            logger.info(f"Loading embedding cache from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                logger.info("Starting with empty cache")
                self.embedding_cache = {}
        else:
            logger.info("No embedding cache found")

def main():
    """Main function to run embedding generation"""
    logger.info("Starting BioClinicalBERT embedding generation...")
    
    # Load graph data
    graph_path = DATA_DIR / "abnormality_graphs.json"
    if not graph_path.exists():
        logger.error(f"Graph data not found at {graph_path}")
        logger.info("Please run radgraph_processor.py first to generate graphs")
        return
    
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Initialize embedder
    embedder = BioClinicalBERTEmbedder()
    
    # Load existing cache if available
    embedder.load_embedding_cache()
    
    # Generate embeddings
    embedded_data = embedder.embed_graph_nodes(graph_data)
    
    # Save embedded data
    embedder.save_embeddings(embedded_data)
    
    # Print statistics
    logger.info("Embedding Generation Statistics:")
    logger.info(f"  Total samples embedded: {len(embedded_data)}")
    logger.info(f"  Embedding dimension: {EMBEDDING_DIM}")
    logger.info(f"  Cache size: {len(embedder.embedding_cache)}")
    
    logger.info("BioClinicalBERT embedding generation completed successfully!")

if __name__ == "__main__":
    main()
