#!/usr/bin/env python3
"""
Debug script to understand why rewards are constant in LP reformulation training.
"""

import torch
import numpy as np
import yaml
from reformulate_lp.models.reformulator import ReformulationSystem
from reformulate_lp.training.reinforcement import REINFORCETrainer
from reformulate_lp.data.dataset import LPDataset
from reformulate_lp.solvers.clp_solver import CLPSolver
import matplotlib.pyplot as plt


def debug_reward_computation(trainer, lp_problem, num_samples=10):
    """Debug what's happening in reward computation."""
    print("🔍 DEBUGGING REWARD COMPUTATION")
    print("=" * 50)
    
    solver = trainer.solver
    
    # Test multiple permutations on the same problem
    rewards = []
    time_improvements = []
    iter_improvements = []
    solve_times_orig = []
    solve_times_reform = []
    iter_counts_orig = []
    iter_counts_reform = []
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Get a permutation
        sample_output = trainer.model.sample_permutation(lp_problem, temperature=1.0)
        permutation = sample_output['permutation']
        cluster_assignments = sample_output['cluster_assignments']
        
        # Apply permutation
        reformulated_lp = trainer.model.apply_permutation(lp_problem, permutation, cluster_assignments)
        
        # Solve both versions
        original_result = solver.solve(lp_problem)
        reformulated_result = solver.solve(reformulated_lp)
        
        if original_result['success'] and reformulated_result['success']:
            orig_time = original_result.get('solve_time', 0.0)
            reform_time = reformulated_result.get('solve_time', 0.0)
            orig_iters = original_result.get('iterations', 0)
            reform_iters = reformulated_result.get('iterations', 0)
            
            print(f"  Permutation: {permutation.cpu().numpy()}")
            print(f"  Original - Time: {orig_time:.6f}s, Iterations: {orig_iters}")
            print(f"  Reformed - Time: {reform_time:.6f}s, Iterations: {reform_iters}")
            
            # Calculate improvements
            time_improvement = (orig_time - reform_time) / max(orig_time, 1e-8)
            iter_improvement = (orig_iters - reform_iters) / max(orig_iters, 1)
            
            print(f"  Time improvement: {time_improvement:.4f}")
            print(f"  Iter improvement: {iter_improvement:.4f}")
            
            # Get reward
            reward = trainer.reward_function(lp_problem, reformulated_lp)
            print(f"  Final reward: {reward:.4f}")
            
            rewards.append(reward)
            time_improvements.append(time_improvement)
            iter_improvements.append(iter_improvement)
            solve_times_orig.append(orig_time)
            solve_times_reform.append(reform_time)
            iter_counts_orig.append(orig_iters)
            iter_counts_reform.append(reform_iters)
        else:
            print(f"  SOLVER FAILURE - Original: {original_result['success']}, Reformed: {reformulated_result['success']}")
    
    # Analysis
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Average reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"Reward range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    print(f"Average solve time (original): {np.mean(solve_times_orig):.6f}s")
    print(f"Average solve time (reformed): {np.mean(solve_times_reform):.6f}s")
    print(f"Average iterations (original): {np.mean(iter_counts_orig):.1f}")
    print(f"Average iterations (reformed): {np.mean(iter_counts_reform):.1f}")
    print(f"Time improvement range: [{np.min(time_improvements):.4f}, {np.max(time_improvements):.4f}]")
    print(f"Iter improvement range: [{np.min(iter_improvements):.4f}, {np.max(iter_improvements):.4f}]")
    
    return {
        'rewards': rewards,
        'time_improvements': time_improvements,
        'iter_improvements': iter_improvements,
        'solve_times_orig': solve_times_orig,
        'solve_times_reform': solve_times_reform
    }


def debug_reward_function_components(trainer, lp_problem):
    """Debug each component of the reward function."""
    print("\n🔧 DEBUGGING REWARD FUNCTION COMPONENTS")
    print("=" * 50)
    
    # Get one permutation
    sample_output = trainer.model.sample_permutation(lp_problem, temperature=1.0)
    permutation = sample_output['permutation']
    cluster_assignments = sample_output['cluster_assignments']
    reformulated_lp = trainer.model.apply_permutation(lp_problem, permutation, cluster_assignments)
    
    # Solve both
    original_result = trainer.solver.solve(lp_problem)
    reformulated_result = trainer.solver.solve(reformulated_lp)
    
    if not original_result['success']:
        print("❌ Original problem failed to solve")
        return
    
    if not reformulated_result['success']:
        print("❌ Reformulated problem failed to solve")
        return
    
    # Get metrics
    orig_time = original_result.get('solve_time', 0.0)
    reform_time = reformulated_result.get('solve_time', 0.0)
    orig_iters = original_result.get('iterations', 0)
    reform_iters = reformulated_result.get('iterations', 0)
    
    print(f"Original: {orig_time:.6f}s, {orig_iters} iterations")
    print(f"Reformed: {reform_time:.6f}s, {reform_iters} iterations")
    
    # Component-by-component reward calculation
    reward = 0.0
    
    # 1. Time-based reward
    if orig_time > 0.01 and reform_time > 0.01:
        time_improvement = (orig_time - reform_time) / orig_time
        time_reward = np.clip(time_improvement * 15.0, -10.0, 10.0)
        time_contribution = 0.6 * time_reward
        reward += time_contribution
        print(f"✅ Time-based reward: improvement={time_improvement:.4f}, reward={time_reward:.4f}, contribution={time_contribution:.4f}")
    else:
        print(f"❌ Time-based reward SKIPPED (times too small: {orig_time:.6f}s, {reform_time:.6f}s)")
    
    # 2. Iteration-based reward
    if orig_iters > 0 and reform_iters > 0:
        iter_improvement = (orig_iters - reform_iters) / orig_iters
        iter_reward = np.clip(iter_improvement * 10.0, -8.0, 8.0)
        iter_contribution = 0.4 * iter_reward
        reward += iter_contribution
        print(f"✅ Iter-based reward: improvement={iter_improvement:.4f}, reward={iter_reward:.4f}, contribution={iter_contribution:.4f}")
    else:
        print(f"❌ Iter-based reward SKIPPED (iters: {orig_iters}, {reform_iters})")
    
    # 3. Success bonus
    success_bonus = 1.0
    reward += success_bonus
    print(f"✅ Success bonus: {success_bonus}")
    
    # 4. Stability bonus
    if reward > -2.0:
        stability_bonus = 0.5
        reward += stability_bonus
        print(f"✅ Stability bonus: {stability_bonus}")
    else:
        print(f"❌ Stability bonus SKIPPED (reward too low: {reward:.4f})")
    
    # 5. Final clipping
    reward_before_clip = reward
    reward = np.clip(reward, -12.0, 12.0)
    print(f"✅ Final reward: {reward_before_clip:.4f} -> {reward:.4f} (after clipping)")
    
    return reward


def test_problem_difficulty(dataset, num_problems=5):
    """Test if problems are too easy."""
    print("\n🎯 TESTING PROBLEM DIFFICULTY")
    print("=" * 50)
    
    solver = CLPSolver()
    
    for i in range(min(num_problems, len(dataset))):
        lp_problem = dataset[i]
        result = solver.solve(lp_problem)
        
        if result['success']:
            solve_time = result.get('solve_time', 0.0)
            iterations = result.get('iterations', 0)
            num_vars = lp_problem['c'].shape[0]
            num_constraints = lp_problem['A'].shape[0]
            
            print(f"Problem {i}: {num_vars} vars, {num_constraints} constraints")
            print(f"  Solve time: {solve_time:.6f}s, Iterations: {iterations}")
            print(f"  Difficulty: {'TRIVIAL' if solve_time < 0.01 else 'REASONABLE'}")
        else:
            print(f"Problem {i}: FAILED TO SOLVE")


def main():
    print("🚀 DEBUGGING CONSTANT REWARDS IN LP REFORMULATION")
    print("=" * 60)
    
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    print("📊 Creating dataset...")
    data_config = config['data']
    dataset = LPDataset(
        data_path=data_config.get('data_path'),
        num_synthetic=data_config.get('num_synthetic', 100),
        synthetic_config=data_config.get('synthetic_config', {})
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Create model and trainer
    print("🧠 Creating model...")
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
    
    print("🏋️ Creating trainer...")
    trainer = REINFORCETrainer(
        model=model,
        learning_rate=float(config['training']['learning_rate']),
        entropy_weight=float(config['training']['entropy_weight']),
        baseline_buffer_size=int(config['training']['baseline_buffer_size']),
        device=device
    )
    
    # Test problem difficulty
    test_problem_difficulty(dataset)
    
    # Pick a problem for detailed debugging
    lp_problem = dataset[0]
    
    # Debug reward function components
    debug_reward_function_components(trainer, lp_problem)
    
    # Debug multiple permutations
    debug_results = debug_reward_computation(trainer, lp_problem, num_samples=10)
    
    # Final diagnosis
    print("\n🎯 DIAGNOSIS:")
    print("=" * 50)
    
    avg_reward = np.mean(debug_results['rewards'])
    reward_std = np.std(debug_results['rewards'])
    avg_time_orig = np.mean(debug_results['solve_times_orig'])
    
    if avg_time_orig < 0.01:
        print("🔴 ISSUE: Problems are TOO EASY (solve time < 0.01s)")
        print("   → Time-based rewards are disabled")
        print("   → Only iteration-based rewards + fixed bonuses remain")
        print("   → This leads to constant rewards around 1.5")
        
    if reward_std < 0.1:
        print("🔴 ISSUE: Reward variance is TOO LOW")
        print("   → Different permutations give nearly identical rewards")
        print("   → The model has no learning signal")
        
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. Use HARDER problems (more variables/constraints)")
    print(f"2. Remove fixed bonuses (1.0 success + 0.5 stability)")
    print(f"3. Focus purely on relative improvements")
    print(f"4. Consider using iteration ratios instead of absolute differences")


if __name__ == "__main__":
    main() 