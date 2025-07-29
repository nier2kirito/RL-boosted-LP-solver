#!/usr/bin/env python3
"""
Evaluation script for the Learning to Reformulate system.

This script evaluates a trained model on test datasets and reports performance metrics.
"""

import argparse
import torch
import numpy as np
import json
import os
from typing import Dict, List

from reformulate_lp import ReformulationSystem, LPDataset
from reformulate_lp.data.lp_parser import generate_random_lp
from reformulate_lp.solvers.clp_solver import CLPSolver


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate LP reformulation system')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test dataset directory')
    parser.add_argument('--num_test_problems', type=int, default=100,
                       help='Number of test problems to evaluate')
    parser.add_argument('--output_path', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for evaluation (cpu, cuda, auto)')
    parser.add_argument('--solver_time_limit', type=float, default=60.0,
                       help='Time limit for solver in seconds')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup evaluation device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    return device


def load_model(model_path: str, device: str) -> ReformulationSystem:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Create model with default configuration
    model = ReformulationSystem(
        constraint_feature_dim=5,
        variable_feature_dim=3,
        gnn_hidden_dim=64,
        gnn_num_layers=2,
        pointer_hidden_dim=128,
        num_clusters=20,
        clustering_method='kmeans',
        dropout=0.1
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded with {model.get_num_parameters()} parameters")
    return model


def create_test_dataset(test_data_path: str = None, num_problems: int = 100) -> LPDataset:
    """Create test dataset."""
    if test_data_path and os.path.exists(test_data_path):
        print(f"Loading test data from {test_data_path}")
        dataset = LPDataset(data_path=test_data_path)
    else:
        print(f"Generating {num_problems} synthetic test problems")
        dataset = LPDataset(
            num_synthetic=num_problems,
            synthetic_config={
                'num_variables_range': (10, 100),
                'num_constraints_range': (5, 50),
                'density_range': (0.1, 0.5),
                'seed': 12345  # Different seed for test data
            }
        )
    
    return dataset


def evaluate_single_problem(model: ReformulationSystem, 
                           lp_problem: Dict, 
                           solver: CLPSolver) -> Dict:
    """Evaluate model on a single LP problem."""
    try:
        # Get reformulation
        reformulated_lp, info = model.reformulate(lp_problem, temperature=0.1)
        
        # Solve both problems
        original_result = solver.solve_with_timeout(lp_problem)
        reformulated_result = solver.solve_with_timeout(reformulated_lp)
        
        # Compute metrics
        evaluation = {
            'problem_name': lp_problem.get('name', 'unknown'),
            'num_variables': lp_problem.get('num_variables', len(lp_problem['c'])),
            'num_constraints': lp_problem.get('num_constraints', len(lp_problem['b'])),
            'original_success': original_result['success'],
            'reformulated_success': reformulated_result['success'],
            'original_solve_time': original_result.get('solve_time', 0.0),
            'reformulated_solve_time': reformulated_result.get('solve_time', 0.0),
            'original_iterations': original_result.get('iterations', 0),
            'reformulated_iterations': reformulated_result.get('iterations', 0),
            'original_objective': original_result.get('objective_value', None),
            'reformulated_objective': reformulated_result.get('objective_value', None)
        }
        
        # Compute improvement metrics
        if original_result['success'] and reformulated_result['success']:
            orig_time = original_result['solve_time']
            reform_time = reformulated_result['solve_time']
            orig_iters = original_result.get('iterations', 1)
            reform_iters = reformulated_result.get('iterations', 1)
            
            evaluation.update({
                'time_improvement': (orig_time - reform_time) / orig_time if orig_time > 0 else 0,
                'iteration_improvement': (orig_iters - reform_iters) / orig_iters if orig_iters > 0 else 0,
                'time_ratio': reform_time / orig_time if orig_time > 0 else float('inf'),
                'iteration_ratio': reform_iters / orig_iters if orig_iters > 0 else float('inf'),
                'both_solved': True
            })
        else:
            evaluation.update({
                'time_improvement': -1.0,
                'iteration_improvement': -1.0,
                'time_ratio': float('inf'),
                'iteration_ratio': float('inf'),
                'both_solved': False
            })
        
        evaluation['success'] = True
        
    except Exception as e:
        evaluation = {
            'problem_name': lp_problem.get('name', 'unknown'),
            'success': False,
            'error': str(e),
            'both_solved': False
        }
    
    return evaluation


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics from individual results."""
    successful_results = [r for r in results if r.get('success', False)]
    both_solved_results = [r for r in successful_results if r.get('both_solved', False)]
    
    if len(successful_results) == 0:
        return {'error': 'No successful evaluations'}
    
    metrics = {
        'total_problems': len(results),
        'successful_evaluations': len(successful_results),
        'both_solved_problems': len(both_solved_results),
        'success_rate': len(successful_results) / len(results),
        'both_solved_rate': len(both_solved_results) / len(results) if len(results) > 0 else 0
    }
    
    if len(both_solved_results) > 0:
        time_improvements = [r['time_improvement'] for r in both_solved_results]
        iteration_improvements = [r['iteration_improvement'] for r in both_solved_results]
        time_ratios = [r['time_ratio'] for r in both_solved_results if r['time_ratio'] != float('inf')]
        iteration_ratios = [r['iteration_ratio'] for r in both_solved_results if r['iteration_ratio'] != float('inf')]
        
        metrics.update({
            'avg_time_improvement': np.mean(time_improvements),
            'avg_iteration_improvement': np.mean(iteration_improvements),
            'median_time_improvement': np.median(time_improvements),
            'median_iteration_improvement': np.median(iteration_improvements),
            'time_improvement_std': np.std(time_improvements),
            'iteration_improvement_std': np.std(iteration_improvements),
            'problems_with_time_improvement': sum(1 for x in time_improvements if x > 0),
            'problems_with_iteration_improvement': sum(1 for x in iteration_improvements if x > 0)
        })
        
        if len(time_ratios) > 0:
            metrics.update({
                'avg_time_ratio': np.mean(time_ratios),
                'median_time_ratio': np.median(time_ratios)
            })
        
        if len(iteration_ratios) > 0:
            metrics.update({
                'avg_iteration_ratio': np.mean(iteration_ratios),
                'median_iteration_ratio': np.median(iteration_ratios)
            })
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create test dataset
    test_dataset = create_test_dataset(args.test_data, args.num_test_problems)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create solver
    solver = CLPSolver(time_limit=args.solver_time_limit, verbose=False)
    
    # Evaluate on all test problems
    print("Starting evaluation...")
    results = []
    
    for i in range(len(test_dataset)):
        if i % 10 == 0:
            print(f"Evaluating problem {i+1}/{len(test_dataset)}")
        
        lp_problem = test_dataset[i]
        result = evaluate_single_problem(model, lp_problem, solver)
        results.append(result)
    
    # Compute aggregate metrics
    print("Computing aggregate metrics...")
    aggregate_metrics = compute_aggregate_metrics(results)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total problems evaluated: {aggregate_metrics['total_problems']}")
    print(f"Successful evaluations: {aggregate_metrics['successful_evaluations']}")
    print(f"Both problems solved: {aggregate_metrics['both_solved_problems']}")
    print(f"Success rate: {aggregate_metrics['success_rate']:.3f}")
    print(f"Both solved rate: {aggregate_metrics['both_solved_rate']:.3f}")
    
    if 'avg_time_improvement' in aggregate_metrics:
        print(f"\nPerformance Improvements:")
        print(f"Average time improvement: {aggregate_metrics['avg_time_improvement']:.3f}")
        print(f"Average iteration improvement: {aggregate_metrics['avg_iteration_improvement']:.3f}")
        print(f"Median time improvement: {aggregate_metrics['median_time_improvement']:.3f}")
        print(f"Median iteration improvement: {aggregate_metrics['median_iteration_improvement']:.3f}")
        print(f"Problems with time improvement: {aggregate_metrics['problems_with_time_improvement']}")
        print(f"Problems with iteration improvement: {aggregate_metrics['problems_with_iteration_improvement']}")
        
        if 'avg_time_ratio' in aggregate_metrics:
            print(f"Average time ratio: {aggregate_metrics['avg_time_ratio']:.3f}")
        if 'avg_iteration_ratio' in aggregate_metrics:
            print(f"Average iteration ratio: {aggregate_metrics['avg_iteration_ratio']:.3f}")
    
    # Save results
    output_data = {
        'aggregate_metrics': aggregate_metrics,
        'individual_results': results,
        'evaluation_config': {
            'model_path': args.model_path,
            'test_data': args.test_data,
            'num_test_problems': args.num_test_problems,
            'solver_time_limit': args.solver_time_limit
        }
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output_path}")


if __name__ == '__main__':
    main() 