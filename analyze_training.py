#!/usr/bin/env python3
"""
Training Analysis Tool for LP Reformulation System
Helps diagnose why permutation learning isn't working
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List


def load_training_history(path: str) -> Dict:
    """Load training history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_rewards(rewards: List[float]) -> Dict:
    """Analyze reward statistics."""
    rewards = np.array(rewards)
    
    # Count failure episodes
    failure_episodes = np.sum(rewards <= -10.0)
    degradation_episodes = np.sum((rewards < 0) & (rewards > -10.0))
    improvement_episodes = np.sum(rewards > 0)
    
    # Compute statistics
    stats = {
        'total_episodes': len(rewards),
        'failure_rate': failure_episodes / len(rewards),
        'degradation_rate': degradation_episodes / len(rewards),
        'improvement_rate': improvement_episodes / len(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'median_reward': np.median(rewards),
        'reward_range': (np.min(rewards), np.max(rewards)),
        'trend': np.polyfit(range(len(rewards)), rewards, 1)[0]  # Linear trend
    }
    
    return stats


def analyze_convergence(history: Dict) -> Dict:
    """Analyze training convergence."""
    rewards = np.array(history['episode_rewards'])
    
    # Moving averages
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        convergence_trend = np.polyfit(range(len(moving_avg)), moving_avg, 1)[0]
        
        # Check for improvement in recent episodes
        recent_rewards = rewards[-window_size:] if len(rewards) >= window_size else rewards
        early_rewards = rewards[:window_size] if len(rewards) >= window_size else rewards[:len(rewards)//2]
        
        recent_mean = np.mean(recent_rewards)
        early_mean = np.mean(early_rewards)
        improvement = recent_mean - early_mean
    else:
        convergence_trend = 0.0
        improvement = 0.0
    
    return {
        'convergence_trend': convergence_trend,
        'improvement_over_time': improvement,
        'is_converging': convergence_trend > 0.001,  # Small positive trend
        'is_improving': improvement > 0.5
    }


def plot_training_analysis(history: Dict, output_dir: str = "analysis_plots"):
    """Create comprehensive training analysis plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    rewards = np.array(history['episode_rewards'])
    episodes = np.arange(len(rewards))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LP Reformulation Training Analysis', fontsize=16)
    
    # 1. Reward over time
    axes[0, 0].plot(episodes, rewards, alpha=0.6, linewidth=0.5, label='Raw Rewards')
    if len(rewards) >= 100:
        window = 100
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reward distribution
    axes[0, 1].hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Break-even')
    axes[0, 1].axvline(x=np.mean(rewards), color='g', linestyle='-', label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 1].set_xlabel('Reward Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Failure analysis
    failure_mask = rewards <= -10.0
    success_mask = rewards > 0
    neutral_mask = (rewards <= 0) & (rewards > -10.0)
    
    categories = ['Failures', 'Neutral/Small Loss', 'Improvements']
    counts = [np.sum(failure_mask), np.sum(neutral_mask), np.sum(success_mask)]
    colors = ['red', 'orange', 'green']
    
    axes[0, 2].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Episode Outcome Breakdown')
    
    # 4. Baseline and advantage analysis
    if 'episode_baselines' in history and len(history['episode_baselines']) > 0:
        baselines = np.array(history['episode_baselines'])
        advantages = rewards - baselines
        
        axes[1, 0].plot(episodes, baselines, 'b-', label='Baseline', alpha=0.8)
        axes[1, 0].plot(episodes, rewards, 'g-', label='Actual Reward', alpha=0.6)
        axes[1, 0].fill_between(episodes, baselines, rewards, alpha=0.3, 
                               where=(rewards >= baselines), color='green', label='Positive Advantage')
        axes[1, 0].fill_between(episodes, baselines, rewards, alpha=0.3, 
                               where=(rewards < baselines), color='red', label='Negative Advantage')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Baseline vs Actual Rewards')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No baseline data available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Baseline Analysis (No Data)')
    
    # 5. Policy loss
    if 'policy_losses' in history and len(history['policy_losses']) > 0:
        losses = np.array(history['policy_losses'])
        axes[1, 1].plot(episodes, losses, 'purple', alpha=0.7)
        if len(losses) >= 50:
            window = 50
            smooth_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(episodes[window-1:], smooth_losses, 'red', linewidth=2, label=f'Smoothed ({window})')
            axes[1, 1].legend()
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Policy Loss')
        axes[1, 1].set_title('Policy Loss Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No policy loss data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Policy Loss (No Data)')
    
    # 6. Learning progress indicators
    window_size = min(200, len(rewards) // 4) if len(rewards) > 100 else max(10, len(rewards) // 2)
    if len(rewards) >= window_size * 2:
        # Compare early vs recent performance
        n_segments = 5
        segment_size = len(rewards) // n_segments
        segment_means = []
        segment_labels = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(rewards)
            segment = rewards[start_idx:end_idx]
            segment_means.append(np.mean(segment))
            segment_labels.append(f'Ep {start_idx}-{end_idx-1}')
        
        axes[1, 2].plot(range(n_segments), segment_means, 'bo-', linewidth=2, markersize=8)
        axes[1, 2].set_xticks(range(n_segments))
        axes[1, 2].set_xticklabels(segment_labels, rotation=45)
        axes[1, 2].set_ylabel('Mean Reward')
        axes[1, 2].set_title('Learning Progress by Segment')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add trend line
        trend_coef = np.polyfit(range(n_segments), segment_means, 1)[0]
        trend_line = np.poly1d(np.polyfit(range(n_segments), segment_means, 1))
        axes[1, 2].plot(range(n_segments), trend_line(range(n_segments)), 'r--', 
                       alpha=0.8, label=f'Trend: {trend_coef:.3f}')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient data for\nlearning progress analysis', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Learning Progress (Insufficient Data)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Analysis plots saved to: {output_dir}/training_analysis.png")


def generate_report(history: Dict) -> str:
    """Generate a comprehensive training analysis report."""
    reward_stats = analyze_rewards(history['episode_rewards'])
    convergence_stats = analyze_convergence(history)
    
    report = f"""
# LP Reformulation Training Analysis Report

## Training Overview
- **Total Episodes**: {reward_stats['total_episodes']:,}
- **Mean Reward**: {reward_stats['mean_reward']:.3f}
- **Reward Std**: {reward_stats['std_reward']:.3f}
- **Reward Range**: [{reward_stats['reward_range'][0]:.2f}, {reward_stats['reward_range'][1]:.2f}]

## Episode Outcomes
- **Failure Rate**: {reward_stats['failure_rate']:.1%} (solver failures)
- **Degradation Rate**: {reward_stats['degradation_rate']:.1%} (negative rewards > -10)
- **Improvement Rate**: {reward_stats['improvement_rate']:.1%} (positive rewards)

## Learning Analysis
- **Overall Trend**: {reward_stats['trend']:.6f} reward/episode
- **Convergence Trend**: {convergence_stats['convergence_trend']:.6f}
- **Recent vs Early Improvement**: {convergence_stats['improvement_over_time']:.3f}
- **Is Converging**: {'✅ YES' if convergence_stats['is_converging'] else '❌ NO'}
- **Is Improving**: {'✅ YES' if convergence_stats['is_improving'] else '❌ NO'}

## Diagnosis

{'### 🔴 CRITICAL ISSUES IDENTIFIED:' if reward_stats['failure_rate'] > 0.3 or not convergence_stats['is_improving'] else '### ✅ Training appears healthy:'}

"""
    
    # Add specific diagnoses
    if reward_stats['failure_rate'] > 0.3:
        report += f"- **High Failure Rate**: {reward_stats['failure_rate']:.1%} of episodes fail completely\n"
        report += "  → Check solver timeouts, problem feasibility, and reward function\n\n"
    
    if reward_stats['std_reward'] > 10:
        report += f"- **Very Noisy Rewards**: Std = {reward_stats['std_reward']:.2f}\n"
        report += "  → Consider reward smoothing, better baseline, or different metrics\n\n"
    
    if not convergence_stats['is_improving']:
        report += "- **No Learning Progress**: Recent performance not better than early episodes\n"
        report += "  → Check learning rate, exploration, or reward signal quality\n\n"
    
    if abs(reward_stats['trend']) < 0.001:
        report += "- **Flat Learning Curve**: No significant trend in rewards\n"
        report += "  → May need curriculum learning, better initialization, or reward shaping\n\n"
    
    # Recommendations
    report += """
## Recommendations

### Immediate Actions:
1. **Reduce Solver Failures**: Increase time limits or use more robust problems
2. **Improve Reward Signal**: Use iteration-based metrics for small problems
3. **Better Baseline**: Implement neural baseline or larger reward buffer
4. **Add Regularization**: Include entropy bonus for better exploration

### Advanced Improvements:
1. **Curriculum Learning**: Start with easier problems, gradually increase difficulty
2. **Multi-Objective Rewards**: Combine time, iterations, and feasibility metrics
3. **Better Clustering**: Try learned clustering instead of k-means
4. **Hierarchical Learning**: Learn cluster ordering and within-cluster ordering separately

### Configuration Changes:
```yaml
training:
  learning_rate: 3e-4        # Increase learning rate
  entropy_weight: 0.02       # Add exploration bonus
  baseline_buffer_size: 200  # Better baseline estimates
  solver_time_limit: 180.0   # Reduce timeouts
  temperature_schedule:
    initial_temp: 2.0        # More initial exploration
    min_temp: 0.05          # Lower final temperature
```
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze LP reformulation training')
    parser.add_argument('training_file', help='Path to training_history.json file')
    parser.add_argument('--output-dir', default='analysis_plots', help='Output directory for plots')
    parser.add_argument('--report-file', default='training_analysis_report.md', help='Output report file')
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing training file: {args.training_file}")
    
    # Load training history
    try:
        history = load_training_history(args.training_file)
        print(f"✅ Loaded {len(history['episode_rewards'])} episodes")
    except Exception as e:
        print(f"❌ Error loading training file: {e}")
        return
    
    # Generate analysis plots
    try:
        plot_training_analysis(history, args.output_dir)
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
    
    # Generate report
    try:
        report = generate_report(history)
        with open(args.report_file, 'w') as f:
            f.write(report)
        print(f"✅ Analysis report saved to: {args.report_file}")
        print("\n" + "="*80)
        print(report)
    except Exception as e:
        print(f"⚠️  Error generating report: {e}")


if __name__ == "__main__":
    main() 