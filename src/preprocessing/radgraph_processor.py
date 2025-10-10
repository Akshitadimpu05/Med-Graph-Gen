"""
RadGraph processing module for generating attributed abnormality graphs
"""
import json
import re
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
import logging
from pathlib import Path
import spacy
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from config.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadGraphProcessor:
    """Process radiology reports to generate attributed abnormality graphs using RadGraph concepts"""
    
    def __init__(self):
        # RadLex anatomical terms (simplified subset)
        self.anatomical_terms = {
            'lung', 'heart', 'chest', 'thorax', 'pleura', 'mediastinum',
            'diaphragm', 'rib', 'spine', 'vertebra', 'clavicle', 'sternum',
            'aorta', 'pulmonary', 'cardiac', 'pericardium', 'bronchus',
            'trachea', 'esophagus', 'shoulder', 'neck', 'abdomen'
        }
        
        # Common abnormality terms
        self.abnormality_terms = {
            'pneumonia', 'consolidation', 'opacity', 'infiltrate', 'effusion',
            'pneumothorax', 'atelectasis', 'edema', 'mass', 'nodule',
            'fracture', 'dislocation', 'enlargement', 'cardiomegaly',
            'hyperinflation', 'congestion', 'fibrosis', 'scarring',
            'calcification', 'thickening', 'deviation', 'displacement'
        }
        
        # Observation terms
        self.observation_terms = {
            'normal', 'abnormal', 'clear', 'unremarkable', 'stable',
            'improved', 'worsened', 'new', 'old', 'chronic', 'acute',
            'mild', 'moderate', 'severe', 'bilateral', 'unilateral',
            'left', 'right', 'upper', 'lower', 'middle', 'central',
            'peripheral', 'diffuse', 'focal', 'patchy', 'extensive'
        }
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic processing.")
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract anatomical, abnormality, and observation entities from text"""
        text_lower = text.lower()
        
        entities = {
            'anatomy': set(),
            'abnormality': set(),
            'observation': set()
        }
        
        # Extract anatomical terms
        for term in self.anatomical_terms:
            if term in text_lower:
                entities['anatomy'].add(term)
        
        # Extract abnormality terms
        for term in self.abnormality_terms:
            if term in text_lower:
                entities['abnormality'].add(term)
        
        # Extract observation terms
        for term in self.observation_terms:
            if term in text_lower:
                entities['observation'].add(term)
        
        return entities
    
    def create_graph_from_entities(self, entities: Dict[str, Set[str]], 
                                 text: str, sample_id: int) -> nx.Graph:
        """Create a graph from extracted entities"""
        G = nx.Graph()
        
        # Add nodes for each entity type
        node_id = 0
        node_mapping = {}
        
        for entity_type, entity_set in entities.items():
            for entity in entity_set:
                G.add_node(node_id, 
                          name=entity, 
                          type=entity_type,
                          sample_id=sample_id)
                node_mapping[entity] = node_id
                node_id += 1
        
        # Create edges based on co-occurrence and semantic relationships
        self._add_semantic_edges(G, entities, node_mapping, text)
        
        return G
    
    def _add_semantic_edges(self, G: nx.Graph, entities: Dict[str, Set[str]], 
                          node_mapping: Dict[str, int], text: str) -> None:
        """Add edges based on semantic relationships and co-occurrence"""
        text_lower = text.lower()
        
        # Connect anatomical terms with abnormalities if they appear close together
        for anatomy in entities['anatomy']:
            for abnormality in entities['abnormality']:
                # Check if they appear in the same sentence or close proximity
                anatomy_pos = text_lower.find(anatomy)
                abnormality_pos = text_lower.find(abnormality)
                
                if anatomy_pos != -1 and abnormality_pos != -1:
                    # If within 50 characters, create an edge
                    if abs(anatomy_pos - abnormality_pos) < 50:
                        if anatomy in node_mapping and abnormality in node_mapping:
                            G.add_edge(node_mapping[anatomy], 
                                     node_mapping[abnormality],
                                     relation='affects',
                                     weight=1.0)
        
        # Connect observations with abnormalities
        for observation in entities['observation']:
            for abnormality in entities['abnormality']:
                obs_pos = text_lower.find(observation)
                abnorm_pos = text_lower.find(abnormality)
                
                if obs_pos != -1 and abnorm_pos != -1:
                    if abs(obs_pos - abnorm_pos) < 30:
                        if observation in node_mapping and abnormality in node_mapping:
                            G.add_edge(node_mapping[observation], 
                                     node_mapping[abnormality],
                                     relation='describes',
                                     weight=0.8)
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample to create an abnormality graph"""
        text = sample.get('combined_text', '')
        sample_id = sample.get('id', 0)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Create graph
        graph = self.create_graph_from_entities(entities, text, sample_id)
        
        # Convert graph to serializable format
        graph_data = {
            'nodes': [],
            'edges': [],
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }
        
        # Add node data
        for node_id, node_data in graph.nodes(data=True):
            graph_data['nodes'].append({
                'id': node_id,
                'name': node_data.get('name', ''),
                'type': node_data.get('type', ''),
                'sample_id': node_data.get('sample_id', sample_id)
            })
        
        # Add edge data
        for source, target, edge_data in graph.edges(data=True):
            graph_data['edges'].append({
                'source': source,
                'target': target,
                'relation': edge_data.get('relation', ''),
                'weight': edge_data.get('weight', 1.0)
            })
        
        return {
            'sample_id': sample_id,
            'original_text': text,
            'entities': {k: list(v) for k, v in entities.items()},
            'graph': graph_data,
            'graph_stats': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0
            }
        }
    
    def process_dataset(self, processed_data: List[Dict]) -> List[Dict]:
        """Process entire dataset to create abnormality graphs"""
        logger.info("Processing dataset to create abnormality graphs...")
        
        graph_data = []
        
        for sample in tqdm(processed_data, desc="Creating graphs"):
            try:
                graph_sample = self.process_sample(sample)
                graph_data.append(graph_sample)
            except Exception as e:
                logger.warning(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully created graphs for {len(graph_data)} samples")
        return graph_data
    
    def save_graph_data(self, graph_data: List[Dict], 
                       output_path: Optional[Path] = None) -> None:
        """Save graph data to JSON file"""
        if output_path is None:
            output_path = DATA_DIR / "abnormality_graphs.json"
        
        logger.info(f"Saving graph data to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Save summary statistics
        stats = self.get_graph_statistics(graph_data)
        stats_path = output_path.parent / "graph_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Graph statistics saved to {stats_path}")
    
    def get_graph_statistics(self, graph_data: List[Dict]) -> Dict:
        """Calculate statistics about the generated graphs"""
        if not graph_data:
            return {}
        
        node_counts = [g['graph_stats']['num_nodes'] for g in graph_data]
        edge_counts = [g['graph_stats']['num_edges'] for g in graph_data]
        densities = [g['graph_stats']['density'] for g in graph_data]
        
        # Entity type statistics
        entity_stats = defaultdict(int)
        for sample in graph_data:
            for entity_type, entities in sample['entities'].items():
                entity_stats[f'{entity_type}_count'] += len(entities)
        
        stats = {
            'total_graphs': len(graph_data),
            'avg_nodes_per_graph': np.mean(node_counts),
            'avg_edges_per_graph': np.mean(edge_counts),
            'avg_graph_density': np.mean(densities),
            'max_nodes': max(node_counts) if node_counts else 0,
            'max_edges': max(edge_counts) if edge_counts else 0,
            'graphs_with_edges': sum(1 for count in edge_counts if count > 0),
            'entity_statistics': dict(entity_stats)
        }
        
        return stats

def main():
    """Main function to run RadGraph processing"""
    logger.info("Starting RadGraph processing...")
    
    # Load processed data
    data_path = DATA_DIR / "processed_mimic_cxr.json"
    if not data_path.exists():
        logger.error(f"Processed data not found at {data_path}")
        logger.info("Please run data_loader.py first to preprocess the dataset")
        return
    
    with open(data_path, 'r') as f:
        processed_data = json.load(f)
    
    # Initialize RadGraph processor
    processor = RadGraphProcessor()
    
    # Process dataset to create graphs
    graph_data = processor.process_dataset(processed_data)
    
    # Save graph data
    processor.save_graph_data(graph_data)
    
    # Print statistics
    stats = processor.get_graph_statistics(graph_data)
    logger.info("Graph Generation Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("RadGraph processing completed successfully!")

if __name__ == "__main__":
    main()
