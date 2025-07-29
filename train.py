#!/usr/bin/env python3
"""
Training script for the Learning to Reformulate system.

This script trains the reformulation system using REINFORCE on LP datasets.
"""

import argparse
import yaml
import torch
import numpy as np
import os
from datetime import datetime

from reformulate_lp import ReformulationSystem, LPDataset, Trainer
from reformulate_lp.training.reinforcement import REINFORCETrainer, AdaptiveTemperatureSchedule
from reformulate_lp.data.dataset import NormalizeLP, create_dataloader
from reformulate_lp.data.lp_parser import generate_random_lp
from reformulate_lp.solvers.clp_solver import CLPSolver


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LP reformulation system')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to LP problem dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_arg):
    """Setup training device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return device


def create_datasets(config, data_path=None):
    """Create training and validation datasets."""
    
    # Load real data if available
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        dataset = LPDataset(
            data_path=data_path,
            transform=NormalizeLP() if config['data']['normalize'] else None
        )
    else:
        print("No real data provided, generating synthetic dataset")
        # Generate synthetic dataset
        dataset = LPDataset(
            num_synthetic=config['data']['num_synthetic'],
            synthetic_config=config['data']['synthetic_config'],
            transform=NormalizeLP() if config['data']['normalize'] else None
        )
    
    print("Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['seed']
    )
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_model(config, device):
    """Create the reformulation model."""
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
    )
    
    model = model.to(device)
    print(f"Created model with {model.get_num_parameters()} parameters")
    
    return model


def create_trainer(model, config, device):
    """Create the REINFORCE trainer."""
    training_config = config['training']
    
    # Create solver
    solver = CLPSolver(
        time_limit=training_config['solver_time_limit'],
        verbose=False
    )
    
    # Create trainer
    trainer = REINFORCETrainer(
        model=model,
        solver=solver,
        learning_rate=float(training_config['learning_rate']),
        baseline_learning_rate=float(training_config['baseline_learning_rate']),
        device=device,
        gradient_clip=training_config['gradient_clip']
    )
    
    return trainer


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Output directory: {output_dir}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config, args.data_path)
    
    # Create model
    model = create_model(config, device)
    
    # Create trainer
    trainer = create_trainer(model, config, device)
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_episode, history = trainer.load_checkpoint(args.resume)
        print(f"Resumed from episode {start_episode}")
    
    # Create temperature schedule
    temp_config = config['training']['temperature_schedule']
    temperature_schedule = AdaptiveTemperatureSchedule(
        initial_temp=temp_config['initial_temp'],
        min_temp=temp_config['min_temp'],
        decay_rate=temp_config['decay_rate'],
        decay_frequency=temp_config['decay_frequency'],
        adaptive=temp_config['adaptive']
    )
    
    # Training parameters
    training_config = config['training']
    num_episodes = training_config['num_episodes']
    validation_frequency = training_config['validation_frequency']
    save_frequency = training_config['save_frequency']
    
    # Start training
    print("Starting training...")
    training_history = trainer.train(
        dataset=train_dataset,
        num_episodes=num_episodes,
        temperature_schedule=temperature_schedule,
        validation_dataset=val_dataset,
        validation_frequency=validation_frequency,
        save_frequency=save_frequency,
        checkpoint_path=os.path.join(output_dir, 'checkpoint')
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training history
    import json
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in training_history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                history_json[key] = [v.tolist() for v in value]
            else:
                history_json[key] = value
        json.dump(history_json, f, indent=2)
    
    print(f"Saved training history to {history_path}")
    
    # Final evaluation on test set
    if len(test_dataset) > 0:
        print("Evaluating on test set...")
        test_score = trainer.validate(test_dataset, num_samples=min(50, len(test_dataset)))
        print(f"Test set performance: {test_score:.4f}")
        
        # Save test results
        test_results = {'test_score': test_score}
        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    print("Training completed!")


if __name__ == '__main__':
    main() 