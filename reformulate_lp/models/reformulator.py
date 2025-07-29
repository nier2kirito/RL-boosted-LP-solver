"""
Main Reformulation System that combines all components.

This module implements the complete reformulation system that uses GNN to learn
embeddings, clusters variables, and uses pointer network to generate permutations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .gnn import BipartiteGNN, build_bipartite_features
from .pointer_net import PointerNetwork


class VariableClusterer:
    """
    Clusters variables into groups for aggregation.
    """
    
    def __init__(self, num_clusters=20, clustering_method='kmeans'):
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        
    def cluster_variables(self, variable_embeddings, lp_problem=None):
        """
        Cluster variables based on their embeddings or problem structure.
        
        Args:
            variable_embeddings: Variable embeddings from GNN
            lp_problem: Original LP problem (optional, for structure-based clustering)
            
        Returns:
            cluster_assignments: Cluster assignment for each variable
            cluster_embeddings: Aggregated embeddings for each cluster
        """
        num_variables = variable_embeddings.shape[0]
        
        if self.clustering_method == 'kmeans':
            return self._kmeans_clustering(variable_embeddings)
        elif self.clustering_method == 'random':
            return self._random_clustering(variable_embeddings, num_variables)
        elif self.clustering_method == 'sequential':
            return self._sequential_clustering(variable_embeddings, num_variables)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def _kmeans_clustering(self, variable_embeddings):
        """K-means clustering of variable embeddings."""
        from sklearn.cluster import KMeans
        
        embeddings_np = variable_embeddings.detach().cpu().numpy()
        
        # Adjust number of clusters if we have fewer variables
        actual_clusters = min(self.num_clusters, embeddings_np.shape[0])
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(embeddings_np)
        
        # Aggregate embeddings by cluster
        cluster_embeddings = []
        for i in range(actual_clusters):
            cluster_mask = cluster_assignments == i
            if cluster_mask.sum() > 0:
                cluster_emb = variable_embeddings[cluster_mask].mean(dim=0)
            else:
                cluster_emb = torch.zeros_like(variable_embeddings[0])
            cluster_embeddings.append(cluster_emb)
            
        cluster_embeddings = torch.stack(cluster_embeddings)
        
        return torch.tensor(cluster_assignments), cluster_embeddings
    
    def _random_clustering(self, variable_embeddings, num_variables):
        """Random clustering of variables."""
        actual_clusters = min(self.num_clusters, num_variables)
        cluster_assignments = torch.randint(0, actual_clusters, (num_variables,))
        
        # Aggregate embeddings by cluster
        cluster_embeddings = []
        for i in range(actual_clusters):
            cluster_mask = cluster_assignments == i
            if cluster_mask.sum() > 0:
                cluster_emb = variable_embeddings[cluster_mask].mean(dim=0)
            else:
                cluster_emb = torch.zeros_like(variable_embeddings[0])
            cluster_embeddings.append(cluster_emb)
            
        cluster_embeddings = torch.stack(cluster_embeddings)
        
        return cluster_assignments, cluster_embeddings
    
    def _sequential_clustering(self, variable_embeddings, num_variables):
        """Sequential clustering - divide variables into sequential groups."""
        actual_clusters = min(self.num_clusters, num_variables)
        cluster_size = num_variables // actual_clusters
        
        cluster_assignments = torch.zeros(num_variables, dtype=torch.long)
        for i in range(num_variables):
            cluster_assignments[i] = min(i // cluster_size, actual_clusters - 1)
            
        # Aggregate embeddings by cluster
        cluster_embeddings = []
        for i in range(actual_clusters):
            cluster_mask = cluster_assignments == i
            cluster_emb = variable_embeddings[cluster_mask].mean(dim=0)
            cluster_embeddings.append(cluster_emb)
            
        cluster_embeddings = torch.stack(cluster_embeddings)
        
        return cluster_assignments, cluster_embeddings


class ReformulationSystem(nn.Module):
    """
    Complete system for learning to reformulate LP problems.
    
    Combines bipartite GNN, variable clustering, and pointer network
    to generate reformulated LP problems.
    """
    
    def __init__(self, 
                 constraint_feature_dim=5,
                 variable_feature_dim=3,
                 gnn_hidden_dim=64,
                 gnn_num_layers=2,
                 pointer_hidden_dim=128,
                 num_clusters=20,
                 clustering_method='kmeans',
                 dropout=0.1):
        super(ReformulationSystem, self).__init__()
        
        self.num_clusters = num_clusters
        self.gnn_hidden_dim = gnn_hidden_dim
        self.pointer_hidden_dim = pointer_hidden_dim
        
        # Bipartite GNN for learning embeddings
        self.gnn = BipartiteGNN(
            constraint_feature_dim=constraint_feature_dim,
            variable_feature_dim=variable_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout
        )
        
        # Variable clusterer
        self.clusterer = VariableClusterer(num_clusters, clustering_method)
        
        # Pointer network for generating permutations
        self.pointer_net = PointerNetwork(
            input_dim=gnn_hidden_dim,
            hidden_dim=pointer_hidden_dim
        )
        
        # Critic network for baseline estimation (used in REINFORCE)
        self.critic = nn.Sequential(
            nn.Linear(gnn_hidden_dim * num_clusters, pointer_hidden_dim),
            nn.ReLU(),
            nn.Linear(pointer_hidden_dim, pointer_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(pointer_hidden_dim // 2, 1)
        )
        
    def forward(self, lp_problem, target_permutation=None, teacher_forcing_ratio=0.0):
        """
        Forward pass through the complete reformulation system.
        
        Args:
            lp_problem: LP problem instance
            target_permutation: Target permutation for supervised training
            teacher_forcing_ratio: Ratio for teacher forcing
            
        Returns:
            outputs: Dictionary containing all intermediate and final outputs
        """
        # Build bipartite graph features
        constraint_features, variable_features, edge_index, edge_attr = build_bipartite_features(lp_problem)
        
        # Get embeddings from GNN
        constraint_embeddings, variable_embeddings = self.gnn(
            constraint_features, variable_features, edge_index, edge_attr
        )
        
        # Cluster variables and aggregate embeddings
        cluster_assignments, cluster_embeddings = self.clusterer.cluster_variables(
            variable_embeddings, lp_problem
        )
        
        # Add batch dimension for pointer network
        cluster_embeddings_batch = cluster_embeddings.unsqueeze(0)
        
        # Generate permutation using pointer network
        pointer_probs, permutation = self.pointer_net(
            cluster_embeddings_batch, 
            target_permutation.unsqueeze(0) if target_permutation is not None else None,
            teacher_forcing_ratio
        )
        
        # Compute baseline for REINFORCE
        cluster_state = cluster_embeddings.flatten().unsqueeze(0)
        baseline = self.critic(cluster_state)
        
        outputs = {
            'constraint_embeddings': constraint_embeddings,
            'variable_embeddings': variable_embeddings,
            'cluster_assignments': cluster_assignments,
            'cluster_embeddings': cluster_embeddings,
            'pointer_probs': pointer_probs,
            'permutation': permutation.squeeze(0),
            'baseline': baseline.squeeze(0),
            'graph_features': {
                'constraint_features': constraint_features,
                'variable_features': variable_features,
                'edge_index': edge_index,
                'edge_attr': edge_attr
            }
        }
        
        return outputs
    
    def sample_permutation(self, lp_problem, temperature=1.0):
        """
        Sample a permutation for the given LP problem.
        
        Args:
            lp_problem: LP problem instance
            temperature: Sampling temperature
            
        Returns:
            permutation: Sampled permutation
            log_probs: Log probabilities of actions
            cluster_assignments: Cluster assignments
        """
        # Get embeddings
        constraint_features, variable_features, edge_index, edge_attr = build_bipartite_features(lp_problem)
        constraint_embeddings, variable_embeddings = self.gnn(
            constraint_features, variable_features, edge_index, edge_attr
        )
        
        # Cluster variables
        cluster_assignments, cluster_embeddings = self.clusterer.cluster_variables(
            variable_embeddings, lp_problem
        )
        
        # Sample permutation - pass training flag based on model's training mode
        cluster_embeddings_batch = cluster_embeddings.unsqueeze(0)
        permutation, log_probs = self.pointer_net.sample_permutation(
            cluster_embeddings_batch, temperature, training=self.training
        )
        
        return {
            'permutation': permutation.squeeze(0),
            'log_probs': log_probs.squeeze(0),
            'cluster_assignments': cluster_assignments
        }
    
    def apply_permutation(self, lp_problem, permutation, cluster_assignments):
        """
        Apply the generated permutation to reformulate the LP problem.
        
        Args:
            lp_problem: Original LP problem
            permutation: Generated permutation of clusters
            cluster_assignments: Assignment of variables to clusters
            
        Returns:
            reformulated_lp: Reformulated LP problem
        """
        A = lp_problem['A'].copy()
        b = lp_problem['b'].copy()  
        c = lp_problem['c'].copy()
        
        num_variables = len(c)
        
        # Create variable reordering based on cluster permutation
        new_variable_order = []
        
        for cluster_idx in permutation:
            # Find all variables in this cluster
            cluster_variables = torch.where(cluster_assignments == cluster_idx)[0].tolist()
            new_variable_order.extend(cluster_variables)
        
        # Apply reordering to the LP problem
        reformulated_A = A[:, new_variable_order]
        reformulated_c = c[new_variable_order]
        
        reformulated_lp = {
            'A': reformulated_A,
            'b': b,
            'c': reformulated_c,
            'variable_order': new_variable_order,
            'original_order': list(range(num_variables))
        }
        
        return reformulated_lp
    
    def reformulate(self, lp_problem, temperature=1.0):
        """
        Complete reformulation pipeline.
        
        Args:
            lp_problem: Original LP problem
            temperature: Sampling temperature
            
        Returns:
            reformulated_lp: Reformulated LP problem
            info: Additional information about the reformulation
        """
        # Sample permutation
        sample_output = self.sample_permutation(lp_problem, temperature)
        
        # Apply permutation
        reformulated_lp = self.apply_permutation(
            lp_problem,
            sample_output['permutation'],
            sample_output['cluster_assignments']
        )
        
        info = {
            'permutation': sample_output['permutation'],
            'log_probs': sample_output['log_probs'],
            'cluster_assignments': sample_output['cluster_assignments']
        }
        
        return reformulated_lp, info
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 