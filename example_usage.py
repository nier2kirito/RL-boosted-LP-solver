#!/usr/bin/env python3
"""
Example usage of the Learning to Reformulate system.

This script demonstrates how to use the reformulation system
to improve LP solving performance.
"""

import numpy as np
import torch

from reformulate_lp import ReformulationSystem, LPDataset
from reformulate_lp.data.lp_parser import generate_random_lp
from reformulate_lp.solvers.clp_solver import CLPSolver
from reformulate_lp.training.reinforcement import REINFORCETrainer


def example_basic_usage():
    """Example of basic reformulation system usage."""
    print("="*60)
    print("BASIC USAGE EXAMPLE")
    print("="*60)
    
    # Create a simple LP problem
    lp_problem = generate_random_lp(
        num_variables=20,
        num_constraints=10,
        density=0.3,
        seed=42
    )
    
    print(f"Created LP problem with {lp_problem['num_variables']} variables and {lp_problem['num_constraints']} constraints")
    
    # Initialize the reformulation system
    reformulator = ReformulationSystem(
        gnn_hidden_dim=64,
        pointer_hidden_dim=128,
        num_clusters=10
    )
    
    print(f"Initialized reformulation system with {reformulator.get_num_parameters()} parameters")
    
    # Generate a reformulation (using untrained model)
    reformulated_lp, info = reformulator.reformulate(lp_problem, temperature=1.0)
    
    print(f"Generated reformulation with permutation: {info['permutation']}")
    print(f"Variable reordering: {reformulated_lp['variable_order'][:10]}...")  # Show first 10
    
    # Compare with solver (if available)
    try:
        solver = CLPSolver(verbose=False)
        
        print("\nSolving original problem...")
        original_result = solver.solve_with_timeout(lp_problem, timeout=30.0)
        
        print("\nSolving reformulated problem...")
        reformulated_result = solver.solve_with_timeout(reformulated_lp, timeout=30.0)
        
        if original_result['success'] and reformulated_result['success']:
            print(f"\nResults comparison:")
            print(f"Original - Time: {original_result['solve_time']:.4f}s, Iterations: {original_result.get('iterations', 'N/A')}")
            print(f"Reformulated - Time: {reformulated_result['solve_time']:.4f}s, Iterations: {reformulated_result.get('iterations', 'N/A')}")
            
            if original_result['solve_time'] > 0:
                improvement = (original_result['solve_time'] - reformulated_result['solve_time']) / original_result['solve_time']
                print(f"Time improvement: {improvement:.2%}")
        else:
            print("One or both problems failed to solve")
            
    except Exception as e:
        print(f"Solver not available or failed: {e}")


def example_training():
    """Example of training the reformulation system."""
    print("\n" + "="*60)
    print("TRAINING EXAMPLE")
    print("="*60)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    dataset = LPDataset(
        num_synthetic=50,  # Small dataset for example
        synthetic_config={
            'num_variables_range': (10, 30),
            'num_constraints_range': (5, 15),
            'density_range': (0.2, 0.5),
            'seed': 42
        }
    )
    
    print(f"Created dataset with {len(dataset)} problems")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create model
    model = ReformulationSystem(
        gnn_hidden_dim=32,  # Smaller for faster training
        pointer_hidden_dim=64,
        num_clusters=10
    )
    
    # Create trainer
    try:
        solver = CLPSolver(time_limit=10.0, verbose=False)  # Short time limit for example
        
        trainer = REINFORCETrainer(
            model=model,
            solver=solver,
            learning_rate=1e-3,
            device='cpu'
        )
        
        print("Starting training...")
        
        # Train for a few episodes
        training_history = trainer.train(
            dataset=train_dataset,
            num_episodes=20,  # Very short training for example
            validation_dataset=val_dataset,
            validation_frequency=10
        )
        
        print(f"Training completed. Final rewards: {training_history['episode_rewards'][-5:]}")
        
        # Test the trained model
        print("\nTesting trained model...")
        if len(test_dataset) > 0:
            test_score = trainer.validate(test_dataset, num_samples=5)
            print(f"Test score: {test_score:.4f}")
        
    except Exception as e:
        print(f"Training failed (this is expected without proper solver setup): {e}")


def example_dataset_creation():
    """Example of creating and working with LP datasets."""
    print("\n" + "="*60)
    print("DATASET CREATION EXAMPLE")
    print("="*60)
    
    # Create dataset with different configurations
    configs = [
        {'num_variables_range': (5, 15), 'num_constraints_range': (3, 8), 'density_range': (0.3, 0.6)},
        {'num_variables_range': (20, 50), 'num_constraints_range': (10, 25), 'density_range': (0.1, 0.4)},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        
        dataset = LPDataset(
            num_synthetic=10,
            synthetic_config=config
        )
        
        stats = dataset.get_statistics()
        print(f"Dataset statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: min={value['min']:.1f}, max={value['max']:.1f}, mean={value['mean']:.1f}")
            else:
                print(f"  {key}: {value}")


def example_graph_analysis():
    """Example of analyzing bipartite graph representations."""
    print("\n" + "="*60)
    print("GRAPH ANALYSIS EXAMPLE")
    print("="*60)
    
    from reformulate_lp.data.graph_builder import BipartiteGraphBuilder
    
    # Create LP problem
    lp_problem = generate_random_lp(15, 8, density=0.4, seed=123)
    
    # Build bipartite graph
    graph_builder = BipartiteGraphBuilder()
    graph_data = graph_builder.build_graph(lp_problem)
    
    # Get statistics
    stats = graph_builder.get_graph_statistics(graph_data)
    
    print("Bipartite graph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Convert to NetworkX (if available)
    try:
        import networkx as nx
        G = graph_builder.to_networkx(graph_data)
        print(f"\nNetworkX graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"Is bipartite: {nx.is_bipartite(G)}")
    except ImportError:
        print("NetworkX not available for graph analysis")


def main():
    """Run all examples."""
    print("Learning to Reformulate - Example Usage")
    print("This script demonstrates the key functionality of the system.")
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_basic_usage()
    example_dataset_creation()
    example_graph_analysis()
    example_training()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("For actual training, use: python train.py --config configs/default.yaml")
    print("For evaluation, use: python evaluate.py --model_path <path_to_model>")
    print("="*60)


if __name__ == '__main__':
    main() 