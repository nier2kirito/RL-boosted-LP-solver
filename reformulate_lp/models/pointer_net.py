"""
Pointer Network for generating variable cluster permutations.

This module implements the pointer network that takes aggregated variable 
embeddings and generates permutations to reformulate the LP problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointerAttention(nn.Module):
    """
    Attention mechanism for the pointer network.
    """
    
    def __init__(self, hidden_dim):
        super(PointerAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Attention parameters
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For decoder state
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # For encoder states
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # Final attention vector
        
    def forward(self, decoder_state, encoder_states, mask=None):
        """
        Compute attention scores.
        
        Args:
            decoder_state: Current decoder hidden state (batch_size, hidden_dim)
            encoder_states: All encoder states (batch_size, seq_len, hidden_dim)  
            mask: Mask for already selected items (batch_size, seq_len)
            
        Returns:
            attention_scores: Probability distribution over encoder states
        """
        batch_size, seq_len, _ = encoder_states.shape
        
        # Project decoder state and encoder states
        decoder_proj = self.W_s(decoder_state).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        encoder_proj = self.W_h(encoder_states)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention energies
        energies = self.v(torch.tanh(decoder_proj + encoder_proj)).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask to prevent selecting already chosen items
        if mask is not None:
            energies = energies.masked_fill(mask, -float('inf'))
            
        # Compute attention probabilities
        attention_scores = F.softmax(energies, dim=1)
        
        return attention_scores, energies


class PointerNetworkEncoder(nn.Module):
    """
    Encoder for the pointer network using LSTM.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(PointerNetworkEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, inputs):
        """
        Encode input sequence.
        
        Args:
            inputs: Input embeddings (batch_size, seq_len, input_dim)
            
        Returns:
            encoder_outputs: All hidden states (batch_size, seq_len, hidden_dim)
            encoder_state: Final hidden and cell states
        """
        encoder_outputs, encoder_state = self.lstm(inputs)
        return encoder_outputs, encoder_state


class PointerNetworkDecoder(nn.Module):
    """
    Decoder for the pointer network using attention mechanism.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(PointerNetworkDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.attention = PointerAttention(hidden_dim)
        
    def forward(self, encoder_outputs, encoder_state, max_length, teacher_forcing_ratio=0.0, target_sequence=None):
        """
        Decode sequence using pointer mechanism.
        
        Args:
            encoder_outputs: Encoder outputs (batch_size, seq_len, hidden_dim)
            encoder_state: Initial decoder state
            max_length: Maximum sequence length to decode
            teacher_forcing_ratio: Probability of using teacher forcing
            target_sequence: Target sequence for teacher forcing
            
        Returns:
            outputs: Pointer probabilities for each step
            selected_indices: Selected indices at each step
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        device = encoder_outputs.device
        
        # Initialize decoder state
        hidden, cell = encoder_state
        hidden = hidden[-1]  # Take last layer
        cell = cell[-1]
        
        # Initialize mask and outputs
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        outputs = []
        selected_indices = []
        
        # First input is the mean of all encoder outputs
        decoder_input = encoder_outputs.mean(dim=1)  # (batch_size, hidden_dim)
        
        for step in range(max_length):
            # LSTM step
            hidden, cell = self.lstm_cell(decoder_input, (hidden, cell))
            
            # Compute attention and get pointer probabilities
            attention_probs, energies = self.attention(hidden, encoder_outputs, mask)
            outputs.append(attention_probs)
            
            # Select next input
            if self.training and teacher_forcing_ratio > 0 and target_sequence is not None:
                # Teacher forcing
                if torch.rand(1).item() < teacher_forcing_ratio:
                    selected_idx = target_sequence[:, step]
                else:
                    selected_idx = torch.multinomial(attention_probs, 1).squeeze(1)
            else:
                # Greedy selection or sampling
                if self.training:
                    selected_idx = torch.multinomial(attention_probs, 1).squeeze(1)
                else:
                    selected_idx = torch.argmax(attention_probs, dim=1)
            
            selected_indices.append(selected_idx)
            
            # Update mask to prevent re-selection - use non-inplace operation
            mask = mask.scatter(1, selected_idx.unsqueeze(1), True)
            
            # Next decoder input is the selected encoder output
            decoder_input = encoder_outputs.gather(1, selected_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, encoder_outputs.size(2))).squeeze(1)
            
        outputs = torch.stack(outputs, dim=1)  # (batch_size, max_length, seq_len)
        selected_indices = torch.stack(selected_indices, dim=1)  # (batch_size, max_length)
        
        return outputs, selected_indices


class PointerNetwork(nn.Module):
    """
    Complete Pointer Network for generating permutations.
    
    Takes cluster embeddings and generates a permutation ordering.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = PointerNetworkEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = PointerNetworkDecoder(hidden_dim, hidden_dim, num_layers)
        
    def forward(self, cluster_embeddings, target_sequence=None, teacher_forcing_ratio=0.0):
        """
        Generate permutation for cluster embeddings.
        
        Args:
            cluster_embeddings: Embeddings of variable clusters (batch_size, num_clusters, input_dim)
            target_sequence: Target permutation for training (batch_size, num_clusters)
            teacher_forcing_ratio: Probability of using teacher forcing during training
            
        Returns:
            pointer_probs: Probability distributions over clusters at each step
            permutation: Generated permutation indices
        """
        batch_size, num_clusters, _ = cluster_embeddings.shape
        
        # Encode cluster embeddings
        encoder_outputs, encoder_state = self.encoder(cluster_embeddings)
        
        # Decode to generate permutation
        pointer_probs, permutation = self.decoder(
            encoder_outputs, 
            encoder_state, 
            num_clusters,
            teacher_forcing_ratio,
            target_sequence
        )
        
        return pointer_probs, permutation
    
    def sample_permutation(self, cluster_embeddings, temperature=1.0, training=False):
        """
        Sample a permutation using temperature-controlled sampling.
        
        Args:
            cluster_embeddings: Cluster embeddings (batch_size, num_clusters, input_dim)
            temperature: Sampling temperature
            training: Whether we're in training mode (enables gradients)
            
        Returns:
            permutation: Sampled permutation
            log_probs: Log probabilities of selected actions
        """
        if not training:
            self.eval()
            with torch.no_grad():
                return self._sample_permutation_impl(cluster_embeddings, temperature)
        else:
            self.train()
            return self._sample_permutation_impl(cluster_embeddings, temperature)

    def _sample_permutation_impl(self, cluster_embeddings, temperature=1.0):
        """
        Implementation of permutation sampling.
        """
        batch_size, num_clusters, _ = cluster_embeddings.shape
        device = cluster_embeddings.device
        
        # Encode
        encoder_outputs, encoder_state = self.encoder(cluster_embeddings)
        
        # Initialize decoder state
        hidden, cell = encoder_state
        hidden = hidden[-1]
        cell = cell[-1]
        
        # Initialize
        mask = torch.zeros(batch_size, num_clusters, dtype=torch.bool, device=device)
        permutation = []
        log_probs = []
        
        decoder_input = encoder_outputs.mean(dim=1)
        
        for step in range(num_clusters):
            # LSTM step
            hidden, cell = self.decoder.lstm_cell(decoder_input, (hidden, cell))
            
            # Get attention scores
            attention_probs, energies = self.decoder.attention(hidden, encoder_outputs, mask)
            
            # Apply temperature
            if temperature != 1.0:
                energies = energies / temperature
                attention_probs = F.softmax(energies, dim=1)
            
            # Sample action
            action = torch.multinomial(attention_probs, 1).squeeze(1)
            permutation.append(action)
            
            # Compute log probability
            log_prob = torch.log(attention_probs.gather(1, action.unsqueeze(1)).squeeze(1) + 1e-8)
            log_probs.append(log_prob)
            
            # Update mask - use non-inplace operation to avoid gradient issues
            mask = mask.scatter(1, action.unsqueeze(1), True)
            
            # Next input
            decoder_input = encoder_outputs.gather(1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, encoder_outputs.size(2))).squeeze(1)
        
        permutation = torch.stack(permutation, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        
        return permutation, log_probs


def compute_permutation_loss(pointer_probs, target_permutation):
    """
    Compute cross-entropy loss for pointer network training.
    
    Args:
        pointer_probs: Predicted probabilities (batch_size, seq_len, num_items)
        target_permutation: Target permutation (batch_size, seq_len)
        
    Returns:
        loss: Cross-entropy loss
    """
    batch_size, seq_len, num_items = pointer_probs.shape
    
    # Reshape for cross-entropy computation
    probs_flat = pointer_probs.view(-1, num_items)
    targets_flat = target_permutation.view(-1)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(probs_flat, targets_flat)
    
    return loss 