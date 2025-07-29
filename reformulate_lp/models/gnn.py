"""
Graph Neural Network for learning embeddings from bipartite LP representations.

This module implements the two-interleaved graph convolutional neural network
as described in the paper for processing bipartite graphs of LP problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class BipartiteGCNLayer(MessagePassing):
    """
    A single layer of the bipartite graph convolutional network.
    Implements message passing between variable and constraint nodes.
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(BipartiteGCNLayer, self).__init__(aggr='mean')
        
        # MLPs for constraint and variable updates
        self.constraint_mlp = nn.Sequential(
            nn.Linear(in_dim + in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        self.variable_mlp = nn.Sequential(
            nn.Linear(in_dim + in_dim, out_dim),
            nn.LayerNorm(out_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        # Message computation MLPs
        self.constraint_message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim),  # +1 for edge feature
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.variable_message_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim),  # +1 for edge feature  
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, constraint_features, variable_features, edge_index, edge_attr):
        """
        Forward pass implementing two-stage message passing:
        1. Variable -> Constraint
        2. Constraint -> Variable
        """
        # Stage 1: Variable to Constraint message passing
        constraint_messages = self.propagate(
            edge_index, 
            x=(variable_features, constraint_features),
            edge_attr=edge_attr,
            message_mlp=self.constraint_message_mlp,
            target_type='constraint'
        )
        
        # Update constraint features
        constraint_input = torch.cat([constraint_features, constraint_messages], dim=1)
        new_constraint_features = self.constraint_mlp(constraint_input)
        
        # Stage 2: Constraint to Variable message passing  
        variable_messages = self.propagate(
            edge_index.flip(0),  # Flip to go constraint -> variable
            x=(new_constraint_features, variable_features),
            edge_attr=edge_attr,
            message_mlp=self.variable_message_mlp,
            target_type='variable'
        )
        
        # Update variable features
        variable_input = torch.cat([variable_features, variable_messages], dim=1)
        new_variable_features = self.variable_mlp(variable_input)
        
        return new_constraint_features, new_variable_features
    
    def message(self, x_i, x_j, edge_attr, message_mlp, target_type):
        """Compute messages between nodes."""
        # x_i: target node features, x_j: source node features
        message_input = torch.cat([x_i, x_j, edge_attr.unsqueeze(-1)], dim=1)
        return message_mlp(message_input)


class BipartiteGNN(nn.Module):
    """
    Bipartite Graph Neural Network for LP reformulation.
    
    Takes bipartite graph representation of LP and learns embeddings
    for variables and constraints.
    """
    
    def __init__(self, 
                 constraint_feature_dim=5,
                 variable_feature_dim=3, 
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.1):
        super(BipartiteGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projections
        self.constraint_input_proj = nn.Linear(constraint_feature_dim, hidden_dim)
        self.variable_input_proj = nn.Linear(variable_feature_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            BipartiteGCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projections
        self.constraint_output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.variable_output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, constraint_features, variable_features, edge_index, edge_attr):
        """
        Forward pass through the bipartite GNN.
        
        Args:
            constraint_features: Tensor of shape (num_constraints, constraint_feature_dim)
            variable_features: Tensor of shape (num_variables, variable_feature_dim)  
            edge_index: Tensor of shape (2, num_edges) with variable->constraint edges
            edge_attr: Tensor of shape (num_edges,) with edge weights (A_ij values)
            
        Returns:
            constraint_embeddings: Tensor of shape (num_constraints, hidden_dim)
            variable_embeddings: Tensor of shape (num_variables, hidden_dim)
        """
        # Project input features to hidden dimension
        constraint_h = self.constraint_input_proj(constraint_features)
        variable_h = self.variable_input_proj(variable_features)
        
        # Apply GCN layers
        for layer in self.gcn_layers:
            constraint_h, variable_h = layer(constraint_h, variable_h, edge_index, edge_attr)
            
        # Final output projections
        constraint_embeddings = self.constraint_output_proj(constraint_h)
        variable_embeddings = self.variable_output_proj(variable_h)
        
        return constraint_embeddings, variable_embeddings


def build_bipartite_features(lp_problem):
    """
    Build node features for the bipartite graph representation.
    
    Args:
        lp_problem: LP problem instance
        
    Returns:
        constraint_features: Features for constraint nodes
        variable_features: Features for variable nodes
        edge_index: Edge connectivity  
        edge_attr: Edge attributes (coefficient values)
    """
    A = lp_problem['A']  # Constraint matrix
    b = lp_problem['b']  # RHS values
    c = lp_problem['c']  # Objective coefficients
    
    num_constraints, num_variables = A.shape
    
    # Constraint features: [rhs, ub_cons, lb_cons, normalized_rhs, bias]
    constraint_features = torch.zeros(num_constraints, 5)
    constraint_features[:, 0] = torch.tensor(b)  # RHS values
    constraint_features[:, 1] = torch.tensor(b).clamp(min=0)  # Upper bounds  
    constraint_features[:, 2] = torch.tensor(b).clamp(max=0)  # Lower bounds
    constraint_features[:, 3] = torch.tensor(b) / (torch.tensor(b).abs().max() + 1e-8)  # Normalized
    constraint_features[:, 4] = 1.0  # Bias term
    
    # Variable features: [obj_coeff, ub_var, lb_var]  
    variable_features = torch.zeros(num_variables, 3)
    variable_features[:, 0] = torch.tensor(c)  # Objective coefficients
    variable_features[:, 1] = 1e6  # Default upper bound (assume no explicit bounds)
    variable_features[:, 2] = 0.0  # Default lower bound (non-negativity)
    
    # Build edge connectivity (variable -> constraint)
    edge_list = []
    edge_weights = []
    
    for i in range(num_constraints):
        for j in range(num_variables):
            if abs(A[i, j]) > 1e-8:  # Non-zero coefficient
                edge_list.append([j, i])  # variable j -> constraint i
                edge_weights.append(A[i, j])
                
    edge_index = torch.tensor(edge_list).T.long()
    edge_attr = torch.tensor(edge_weights).float()
    
    return constraint_features, variable_features, edge_index, edge_attr 