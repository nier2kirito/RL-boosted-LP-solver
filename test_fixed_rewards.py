#!/usr/bin/env python3
"""
Simple test to verify the reward function now provides varied rewards.
"""

import torch
import numpy as np
import yaml
from reformulate_lp.models.reformulator import ReformulationSystem
from reformulate_lp.training.reinforcement import REINFORCETrainer
from reformulate_lp.data.dataset import LPDataset

def test_reward_variance():
    """Test that different permutations now give different rewards."""
    print("🧪 TESTING FIXED REWARD FUNCTION")
    print("=" * 50)
    
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset with smaller problems for quick test
    data_config = config['data']
    data_config['synthetic_config']['num_variables_range'] = [20, 50]
    data_config['synthetic_config']['num_constraints_range'] = [10, 30]
    data_config['num_synthetic'] = 10
    
    dataset = LPDataset(
        num_synthetic=data_config['num_synthetic'],
        synthetic_config=data_config['synthetic_config']
    )
    
    # Create model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = config['model']
    model = ReformulationSystem(
        constraint_feature_dim=model_config['constraint_feature_dim'],
        variable_feature_dim=model_config['variable_feature_dim'],
        gnn_hidden_dim=model_config['gnn_hidden_dim'],
        gnn_num_layers=model_config['gnn_num_layers'],
        pointer_hidden_dim=model_config['pointer_hidden_dim'],
        num_clusters=model_config['num_clusters'],
        clustering_method=model_config['clustering_method'],
        dropout=model_config['dropout']
    ).to(device)
    
    trainer = REINFORCETrainer(
        model=model,
        learning_rate=float(config['training']['learning_rate']),
        entropy_weight=float(config['training']['entropy_weight']),
        baseline_buffer_size=int(config['training']['baseline_buffer_size']),
        device=device
    )
    
    # Test on one problem with multiple permutations
    lp_problem = dataset[0]
    rewards = []
    
    print(f"Testing problem with {lp_problem['c'].shape[0]} variables, {lp_problem['A'].shape[0]} constraints")
    
    for i in range(10):
        # Get different permutations by varying temperature
        sample_output = trainer.model.sample_permutation(lp_problem, temperature=1.0 + i * 0.5)
        permutation = sample_output['permutation']
        cluster_assignments = sample_output['cluster_assignments']
        
        # Apply permutation
        reformulated_lp = trainer.model.apply_permutation(lp_problem, permutation, cluster_assignments)
        
        # Get reward
        reward = trainer.reward_function(lp_problem, reformulated_lp)
        rewards.append(reward)
        
        print(f"Permutation {i+1}: reward = {reward:.4f}")
    
    # Analysis
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    reward_range = (np.min(rewards), np.max(rewards))
    
    print(f"\n📊 RESULTS:")
    print(f"Mean reward: {reward_mean:.4f}")
    print(f"Std deviation: {reward_std:.4f}")
    print(f"Range: [{reward_range[0]:.4f}, {reward_range[1]:.4f}]")
    
    if reward_std > 0.1:
        print("✅ SUCCESS: Rewards have good variance - learning signal exists!")
        return True
    else:
        print("❌ FAILED: Rewards still too similar - no learning signal")
        return False

if __name__ == "__main__":
    success = test_reward_variance()
    exit(0 if success else 1) 