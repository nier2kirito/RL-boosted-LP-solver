"""
Bipartite graph builder for LP problems.

Constructs bipartite graph representations from LP problem matrices.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import networkx as nx


class BipartiteGraphBuilder:
    """
    Builder for bipartite graph representations of LP problems.
    
    Creates graphs where variables and constraints are different node types
    connected by edges representing non-zero coefficients.
    """
    
    def __init__(self, min_coefficient_threshold: float = 1e-8):
        """
        Initialize graph builder.
        
        Args:
            min_coefficient_threshold: Minimum absolute coefficient value to create edge
        """
        self.min_coefficient_threshold = min_coefficient_threshold
    
    def build_graph(self, lp_problem: Dict) -> Dict:
        """
        Build bipartite graph from LP problem.
        
        Args:
            lp_problem: Dictionary containing 'A', 'b', 'c' matrices
            
        Returns:
            graph_data: Dictionary containing graph representation
        """
        A = np.array(lp_problem['A'])
        b = np.array(lp_problem['b'])
        c = np.array(lp_problem['c'])
        
        num_constraints, num_variables = A.shape
        
        # Build node features
        constraint_features = self._build_constraint_features(A, b)
        variable_features = self._build_variable_features(A, c)
        
        # Build edge connectivity
        edge_index, edge_attr = self._build_edges(A)
        
        graph_data = {
            'constraint_features': constraint_features,
            'variable_features': variable_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_constraints': num_constraints,
            'num_variables': num_variables,
            'constraint_node_ids': list(range(num_constraints)),
            'variable_node_ids': list(range(num_constraints, num_constraints + num_variables))
        }
        
        return graph_data
    
    def _build_constraint_features(self, A: np.ndarray, b: np.ndarray) -> torch.Tensor:
        """Build features for constraint nodes."""
        num_constraints = len(b)
        features = torch.zeros(num_constraints, 5)
        
        # Feature 0: RHS value
        features[:, 0] = torch.tensor(b)
        
        # Feature 1: Upper bound (for <= constraints, same as RHS)
        features[:, 1] = torch.tensor(b).clamp(min=0)
        
        # Feature 2: Lower bound (for >= constraints, would be negative)
        features[:, 2] = torch.tensor(b).clamp(max=0)
        
        # Feature 3: Normalized RHS
        b_tensor = torch.tensor(b)
        b_max = b_tensor.abs().max()
        if b_max > 0:
            features[:, 3] = b_tensor / b_max
        
        # Feature 4: Bias term
        features[:, 4] = 1.0
        
        return features
    
    def _build_variable_features(self, A: np.ndarray, c: np.ndarray) -> torch.Tensor:
        """Build features for variable nodes."""
        num_variables = len(c)
        features = torch.zeros(num_variables, 3)
        
        # Feature 0: Objective coefficient
        features[:, 0] = torch.tensor(c)
        
        # Feature 1: Upper bound (default: large value for unbounded)
        features[:, 1] = 1e6
        
        # Feature 2: Lower bound (default: 0 for non-negativity)
        features[:, 2] = 0.0
        
        return features
    
    def _build_edges(self, A: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge connectivity and attributes."""
        num_constraints, num_variables = A.shape
        
        edge_list = []
        edge_weights = []
        
        for i in range(num_constraints):
            for j in range(num_variables):
                if abs(A[i, j]) > self.min_coefficient_threshold:
                    # Edge from variable j to constraint i
                    edge_list.append([j, i])
                    edge_weights.append(A[i, j])
        
        if len(edge_list) == 0:
            # Handle empty graph case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).T
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def to_networkx(self, graph_data: Dict) -> nx.Graph:
        """
        Convert to NetworkX graph for visualization and analysis.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            G: NetworkX bipartite graph
        """
        G = nx.Graph()
        
        # Add constraint nodes
        for i in range(graph_data['num_constraints']):
            G.add_node(f'c_{i}', 
                      bipartite=0, 
                      node_type='constraint',
                      features=graph_data['constraint_features'][i].numpy())
        
        # Add variable nodes
        for j in range(graph_data['num_variables']):
            G.add_node(f'v_{j}', 
                      bipartite=1, 
                      node_type='variable',
                      features=graph_data['variable_features'][j].numpy())
        
        # Add edges
        edge_index = graph_data['edge_index']
        edge_attr = graph_data['edge_attr']
        
        for k in range(edge_index.shape[1]):
            var_idx = edge_index[0, k].item()
            cons_idx = edge_index[1, k].item()
            weight = edge_attr[k].item()
            
            G.add_edge(f'v_{var_idx}', f'c_{cons_idx}', weight=weight)
        
        return G
    
    def get_graph_statistics(self, graph_data: Dict) -> Dict:
        """Get statistics about the graph."""
        num_edges = graph_data['edge_index'].shape[1]
        num_nodes = graph_data['num_constraints'] + graph_data['num_variables']
        
        # Compute density
        max_possible_edges = graph_data['num_constraints'] * graph_data['num_variables']
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Degree statistics
        edge_index = graph_data['edge_index']
        
        # Variable degrees (out-degree in bipartite graph)
        var_degrees = torch.zeros(graph_data['num_variables'])
        for j in range(graph_data['num_variables']):
            var_degrees[j] = (edge_index[0] == j).sum()
        
        # Constraint degrees (in-degree in bipartite graph)
        cons_degrees = torch.zeros(graph_data['num_constraints'])
        for i in range(graph_data['num_constraints']):
            cons_degrees[i] = (edge_index[1] == i).sum()
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_constraints': graph_data['num_constraints'],
            'num_variables': graph_data['num_variables'],
            'density': density,
            'avg_variable_degree': var_degrees.float().mean().item(),
            'avg_constraint_degree': cons_degrees.float().mean().item(),
            'max_variable_degree': var_degrees.max().item(),
            'max_constraint_degree': cons_degrees.max().item()
        }
        
        return stats


def visualize_bipartite_graph(graph_data: Dict, save_path: str = None):
    """
    Visualize bipartite graph using matplotlib.
    
    Args:
        graph_data: Graph data dictionary
        save_path: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        builder = BipartiteGraphBuilder()
        G = builder.to_networkx(graph_data)
        
        # Create layout
        pos = {}
        
        # Position constraint nodes on the left
        constraint_nodes = [n for n in G.nodes() if n.startswith('c_')]
        for i, node in enumerate(constraint_nodes):
            pos[node] = (0, i)
        
        # Position variable nodes on the right
        variable_nodes = [n for n in G.nodes() if n.startswith('v_')]
        for i, node in enumerate(variable_nodes):
            pos[node] = (1, i)
        
        # Draw graph
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=constraint_nodes,
                              node_color='lightblue',
                              node_size=300,
                              label='Constraints')
        
        nx.draw_networkx_nodes(G, pos,
                              nodelist=variable_nodes, 
                              node_color='lightcoral',
                              node_size=300,
                              label='Variables')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Bipartite Graph Representation of LP Problem")
        plt.legend()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for visualization") 