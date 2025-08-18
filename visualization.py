"""
Knowledge Graph Network Visualizer
Generates a network visualization of your SQLite knowledge graph.
"""

import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional
import numpy as np

class KnowledgeGraphVisualizer:
    """Visualize knowledge graph as a network."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.G = nx.Graph()
        
    def load_graph_data(self, 
                       limit_entities: int = 100,
                       min_confidence: float = 0.5,
                       entity_types: Optional[List[str]] = None) -> Dict:
        """Load entities and relationships from the knowledge graph database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build entity query
        entity_query = "SELECT id, type, text, confidence, properties FROM entities WHERE confidence >= ?"
        params = [min_confidence]
        
        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            entity_query += f" AND type IN ({placeholders})"
            params.extend(entity_types)
            
        entity_query += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit_entities)
        
        # Get entities
        cursor.execute(entity_query, params)
        entities = []
        entity_ids = set()
        
        for row in cursor.fetchall():
            # Handle confidence as bytes, string, or numeric
            confidence = row[3]
            try:
                if isinstance(confidence, bytes):
                    # Try to interpret as float bytes first
                    import struct
                    if len(confidence) == 8:
                        # Assume it's a double (8 bytes)
                        confidence = struct.unpack('d', confidence)[0]
                    elif len(confidence) == 4:
                        # Assume it's a float (4 bytes)
                        confidence = struct.unpack('f', confidence)[0]
                    else:
                        # Try to decode as string
                        try:
                            confidence = float(confidence.decode('utf-8'))
                        except:
                            confidence = 0.5  # Default value
                elif isinstance(confidence, str):
                    confidence = float(confidence)
                elif confidence is None:
                    confidence = 0.5
                else:
                    confidence = float(confidence)
            except (ValueError, struct.error, UnicodeDecodeError):
                confidence = 0.5  # Default fallback value
                
            # Ensure confidence is between 0 and 1
            if confidence > 1.0:
                confidence = confidence / 100.0  # Convert percentage to decimal
            if confidence < 0:
                confidence = 0.0
                
            entity = {
                'id': row[0],
                'type': row[1], 
                'text': row[2],
                'confidence': confidence,
                'properties': json.loads(row[4]) if row[4] else {}
            }
            entities.append(entity)
            entity_ids.add(row[0])
        
        # Get relationships between these entities
        if entity_ids:
            placeholders = ','.join('?' * len(entity_ids))
            rel_query = f'''
                SELECT source_entity_id, target_entity_id, type, confidence
                FROM relationships 
                WHERE source_entity_id IN ({placeholders}) 
                AND target_entity_id IN ({placeholders})
                AND confidence >= ?
            '''
            cursor.execute(rel_query, list(entity_ids) * 2 + [min_confidence])
            
            relationships = []
            for row in cursor.fetchall():
                # Handle confidence as bytes, string, or numeric
                confidence = row[3]
                try:
                    if isinstance(confidence, bytes):
                        # Try to interpret as float bytes first
                        import struct
                        if len(confidence) == 8:
                            confidence = struct.unpack('d', confidence)[0]
                        elif len(confidence) == 4:
                            confidence = struct.unpack('f', confidence)[0]
                        else:
                            try:
                                confidence = float(confidence.decode('utf-8'))
                            except:
                                confidence = 0.5
                    elif isinstance(confidence, str):
                        confidence = float(confidence)
                    elif confidence is None:
                        confidence = 0.5
                    else:
                        confidence = float(confidence)
                except (ValueError, struct.error, UnicodeDecodeError):
                    confidence = 0.5
                
                # Ensure confidence is between 0 and 1
                if confidence > 1.0:
                    confidence = confidence / 100.0
                if confidence < 0:
                    confidence = 0.0
                    
                relationships.append({
                    'source': row[0],
                    'target': row[1],
                    'type': row[2],
                    'confidence': confidence
                })
        else:
            relationships = []
        
        conn.close()
        
        return {
            'entities': entities,
            'relationships': relationships
        }
    
    def build_network(self, data: Dict):
        """Build NetworkX graph from entities and relationships."""
        
        self.G.clear()
        
        # Add nodes (entities)
        for entity in data['entities']:
            self.G.add_node(
                entity['id'],
                label=entity['text'],
                type=entity['type'],
                confidence=entity['confidence'],
                properties=entity['properties']
            )
        
        # Add edges (relationships)
        for rel in data['relationships']:
            if rel['source'] in self.G and rel['target'] in self.G:
                self.G.add_edge(
                    rel['source'],
                    rel['target'],
                    relation_type=rel['type'],
                    confidence=rel['confidence']
                )
    
    def get_node_colors(self) -> Dict[str, str]:
        """Define colors for different entity types."""
        return {
            'PERSON': '#e74c3c',      # Red
            'ORG': '#3498db',         # Blue  
            'COMPANY': '#3498db',     # Blue (matching your config mapping)
            'MONEY': '#f39c12',       # Orange
            'DATE': '#9b59b6',        # Purple
            'LOCATION': '#27ae60',    # Green
            'GPE': '#27ae60',         # Green (Geo-political entity)
            'EVENT': '#e67e22',       # Dark orange
            'PRODUCT': '#34495e',     # Dark blue-gray
            'WORK_OF_ART': '#8e44ad', # Dark purple
            'LAW': '#2c3e50',         # Very dark blue
            'LANGUAGE': '#16a085',    # Teal
            'NORP': '#d35400',        # Dark orange (Nationalities/groups)
            'FACILITY': '#7f8c8d',    # Gray
            'MISC': '#95a5a6',        # Light gray
            'PERCENTAGE': '#e67e22',  # Dark orange
            'ACCOUNT_NUMBER': '#2c3e50',  # Dark blue
            'ROUTING_NUMBER': '#8e44ad',  # Purple
        }
    
    def visualize_network(self, 
                         figsize: tuple = (15, 10),
                         layout: str = 'spring',
                         node_size_by_confidence: bool = True,
                         show_labels: bool = True,
                         save_path: Optional[str] = None,
                         title: str = "Knowledge Graph Network"):
        """
        Create network visualization.
        
        Args:
            figsize: Figure size (width, height)
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell')
            node_size_by_confidence: Scale node size by confidence
            show_labels: Whether to show node labels
            save_path: Path to save the image (optional)
            title: Plot title
        """
        
        if len(self.G.nodes()) == 0:
            print("No nodes to visualize. Check your data loading.")
            return
        
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        elif layout == 'random':
            pos = nx.random_layout(self.G)
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            pos = nx.spring_layout(self.G)
        
        # Get node attributes
        node_colors = self.get_node_colors()
        colors = []
        sizes = []
        labels = {}
        
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            
            # Color by type
            node_type = node_data.get('type', 'DEFAULT')
            colors.append(node_colors.get(node_type, '#95a5a6'))
            
            # Size by confidence
            if node_size_by_confidence:
                confidence = node_data.get('confidence', 0.5)
                size = 300 + (confidence * 500)  # Scale between 300-800
            else:
                size = 500
            sizes.append(size)
            
            # Labels
            if show_labels:
                label = node_data.get('label', node_id)
                # Truncate long labels
                if len(label) > 15:
                    label = label[:15] + '...'
                labels[node_id] = label
        
        # Draw edges with varying thickness based on confidence
        edge_widths = []
        edge_colors = []
        
        for edge in self.G.edges():
            edge_data = self.G.edges[edge]
            confidence = edge_data.get('confidence', 0.5)
            edge_widths.append(confidence * 3)  # Scale edge width
            edge_colors.append('#bdc3c7')
        
        # Draw the network
        nx.draw_networkx_edges(self.G, pos, 
                             width=edge_widths,
                             edge_color=edge_colors,
                             alpha=0.6)
        
        nx.draw_networkx_nodes(self.G, pos,
                             node_color=colors,
                             node_size=sizes,
                             alpha=0.8)
        
        if show_labels:
            nx.draw_networkx_labels(self.G, pos, labels,
                                  font_size=8,
                                  font_weight='bold')
        
        # Create legend for entity types
        legend_elements = []
        entity_types_in_graph = set(self.G.nodes[node]['type'] for node in self.G.nodes())
        
        for entity_type in sorted(entity_types_in_graph):
            color = node_colors.get(entity_type, '#95a5a6')
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            label=entity_type))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Add statistics as text
        stats_text = f"Nodes: {len(self.G.nodes())}\nEdges: {len(self.G.edges())}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_network_stats(self):
        """Print basic statistics about the network."""
        if len(self.G.nodes()) == 0:
            print("No network loaded.")
            return
            
        print(f"\n=== Network Statistics ===")
        print(f"Nodes: {len(self.G.nodes())}")
        print(f"Edges: {len(self.G.edges())}")
        print(f"Average degree: {sum(dict(self.G.degree()).values()) / len(self.G.nodes()):.2f}")
        print(f"Connected components: {nx.number_connected_components(self.G)}")
        
        # Entity type distribution
        type_counts = {}
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', 'UNKNOWN')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        print("\n=== Entity Types ===")
        for entity_type, count in sorted(type_counts.items()):
            print(f"{entity_type}: {count}")
        
        # Most connected nodes
        degrees = dict(self.G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\n=== Most Connected Entities ===")
        for node_id, degree in top_nodes:
            label = self.G.nodes[node_id].get('label', node_id)
            node_type = self.G.nodes[node_id].get('type', 'UNKNOWN')
            print(f"{label} ({node_type}): {degree} connections")


# Example usage
def visualize_knowledge_graph(db_path: str, 
                            limit_entities: int = 50,
                            min_confidence: float = 0.7,
                            entity_types: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Convenience function to visualize a knowledge graph.
    
    Args:
        db_path: Path to SQLite database
        limit_entities: Maximum number of entities to include
        min_confidence: Minimum confidence threshold
        entity_types: List of entity types to include (None for all)
        save_path: Path to save image (None to just display)
    """
    
    visualizer = KnowledgeGraphVisualizer(db_path)
    
    print("Loading knowledge graph data...")
    data = visualizer.load_graph_data(
        limit_entities=limit_entities,
        min_confidence=min_confidence, 
        entity_types=entity_types
    )
    
    print(f"Loaded {len(data['entities'])} entities and {len(data['relationships'])} relationships")
    
    print("Building network...")
    visualizer.build_network(data)
    
    # Print statistics
    visualizer.print_network_stats()
    
    # Create visualization
    print("Creating visualization...")
    visualizer.visualize_network(
        figsize=(16, 12),
        layout='spring',
        node_size_by_confidence=True,
        show_labels=True,
        save_path=save_path,
        title="Knowledge Graph Network Visualization"
    )


if __name__ == "__main__":
    # Try to import your config
    import sys
    import os
    
    DB_PATH = None
    
    # Try different ways to find the config
    try:
        # Try importing from current package
        from .config import get_config
        config = get_config()
        DB_PATH = config.DATABASE_PATH
        print(f"Using database from config: {DB_PATH}")
    except ImportError:
        try:
            # Try importing from parent directory
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import get_config
            config = get_config()
            DB_PATH = config.DATABASE_PATH
            print(f"Using database from config: {DB_PATH}")
        except ImportError:
            try:
                # Try the direct path approach
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
                if os.path.exists(config_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config = config_module.get_config()
                    DB_PATH = config.DATABASE_PATH
                    print(f"Using database from config: {DB_PATH}")
                else:
                    raise ImportError("Config not found")
            except:
                print("Could not import config. Using default path.")
                DB_PATH = "financial_kg.db"
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        print("Please make sure your knowledge graph database exists.")
        
        # Look for database files in current directory
        db_files = [f for f in os.listdir('.') if f.endswith('.db')]
        if db_files:
            print(f"Found these database files: {db_files}")
            DB_PATH = db_files[0]
            print(f"Using: {DB_PATH}")
        else:
            print("No database files found.")
            sys.exit(1)
    
    # Basic visualization
    print("Creating main visualization...")
    visualize_knowledge_graph(
        db_path=DB_PATH,
        limit_entities=100,
        min_confidence=0.3,  # Lower threshold since you have data
        save_path="knowledge_graph.png"
    )
    
    # Since you have no relationships, create a simpler entity-only visualization
    print("\nCreating entity-only visualization...")
    visualizer = KnowledgeGraphVisualizer(DB_PATH)
    data = visualizer.load_graph_data(limit_entities=50, min_confidence=0.3)
    
    if data['entities']:
        print(f"Creating visualization with {len(data['entities'])} entities...")
        visualizer.build_network(data)
        
        # Create a simple layout for disconnected nodes
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(16, 12))
        plt.title("Financial Knowledge Graph - Entity Distribution", fontsize=16, fontweight='bold')
        
        # Group entities by type
        entity_types = {}
        for entity in data['entities']:
            entity_type = entity['type']
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        colors = visualizer.get_node_colors()
        
        # Create a grid layout for entities by type
        y_offset = 0
        for entity_type, entities in entity_types.items():
            x_positions = np.linspace(0, 10, len(entities))
            
            for i, entity in enumerate(entities):
                plt.scatter(x_positions[i], y_offset, 
                          s=300 + (entity['confidence'] * 500),
                          c=colors.get(entity_type, '#95a5a6'),
                          alpha=0.7,
                          edgecolors='black')
                
                # Add labels
                label = entity['text']
                if len(label) > 15:
                    label = label[:15] + '...'
                plt.text(x_positions[i], y_offset - 0.3, label, 
                        ha='center', va='top', fontsize=8, fontweight='bold')
            
            # Add type label
            plt.text(-0.5, y_offset, entity_type, 
                    ha='right', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=colors.get(entity_type, '#95a5a6'), alpha=0.3))
            
            y_offset -= 2
        
        plt.xlim(-2, 12)
        plt.ylim(y_offset - 1, 1)
        plt.axis('off')
        
        # Add statistics
        stats_text = f"Total Entities: {len(data['entities'])}\n"
        for entity_type, entities in entity_types.items():
            stats_text += f"{entity_type}: {len(entities)}\n"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig("entities_by_type.png", dpi=300, bbox_inches='tight')
        print("Entity visualization saved to: entities_by_type.png")
        plt.show()