"""
REINFORCE training algorithm for the reformulation system.

Implements the reinforcement learning approach described in the paper
using solver performance as reward signal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time
from tqdm import tqdm

from ..solvers.solver_interface import SolverInterface
from ..solvers.clp_solver import CLPSolver


class REINFORCETrainer:
    """
    REINFORCE trainer for the reformulation system.
    
    Uses solver performance improvement as reward signal to train
    the neural networks via policy gradient methods.
    """
    
    def __init__(self,
                 model,
                 solver: Optional[SolverInterface] = None,
                 learning_rate: float = 1e-4,
                 baseline_learning_rate: float = 1e-3,
                 reward_function: Optional[Callable] = None,
                 device: str = 'cpu',
                 gradient_clip: float = 1.0,
                 entropy_weight: float = 0.01,
                 baseline_buffer_size: int = 100):
        """
        Initialize REINFORCE trainer with improvements.
        
        Args:
            model: ReformulationSystem model
            solver: LP solver interface
            learning_rate: Learning rate for policy networks
            baseline_learning_rate: Learning rate for baseline network
            reward_function: Custom reward function
            device: Training device
            gradient_clip: Gradient clipping value
            entropy_weight: Weight for entropy regularization
            baseline_buffer_size: Size of reward buffer for better baseline
        """
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.entropy_weight = entropy_weight
        
        # Initialize solver
        if solver is None:
            solver = CLPSolver()
        self.solver = solver
        
        # Initialize optimizers - only for policy networks
        policy_params = list(self.model.gnn.parameters()) + list(self.model.pointer_net.parameters())
        self.policy_optimizer = optim.Adam(policy_params, lr=learning_rate)
        
        # Improved baseline with reward buffer for better estimates
        self.reward_buffer = []
        self.baseline_buffer_size = baseline_buffer_size
        self.baseline_value = 0.0
        self.baseline_momentum = 0.9
        
        # Reward function
        if reward_function is None:
            self.reward_function = self._default_reward_function
        else:
            self.reward_function = reward_function
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_baselines': [],
            'policy_losses': [],
            'baseline_losses': [],
            'solving_times_original': [],
            'solving_times_reformulated': [],
            'improvement_ratios': [],
            'entropy_values': []
        }
    
    def train_episode(self, lp_problem: Dict, temperature: float = 1.0) -> Dict:
        """
        Train on a single LP problem episode with improvements.
        
        Args:
            lp_problem: LP problem instance
            temperature: Sampling temperature
            
        Returns:
            episode_info: Training statistics for this episode
        """
        self.model.train()
        
        # Sample permutation and get log probabilities
        sample_output = self.model.sample_permutation(lp_problem, temperature)
        
        permutation = sample_output['permutation']
        log_probs = sample_output['log_probs']
        cluster_assignments = sample_output['cluster_assignments']
        
        # Apply permutation to get reformulated LP
        reformulated_lp = self.model.apply_permutation(lp_problem, permutation, cluster_assignments)
        
        # Compute reward
        reward = self.reward_function(lp_problem, reformulated_lp)
        
        # Update reward buffer for better baseline
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > self.baseline_buffer_size:
            self.reward_buffer.pop(0)
        
        # Compute improved baseline
        if len(self.reward_buffer) >= 5:  # Need some history
            # Use both moving average and buffer statistics
            buffer_mean = np.mean(self.reward_buffer)
            buffer_std = np.std(self.reward_buffer)
            
            # Adaptive baseline combining moving average and recent performance
            self.baseline_value = (
                self.baseline_momentum * self.baseline_value + 
                (1 - self.baseline_momentum) * buffer_mean
            )
            
            # Use standardized advantage for better learning
            advantage = (reward - self.baseline_value) / max(buffer_std, 0.1)
        else:
            # Fallback to simple moving average
            self.baseline_value = self.baseline_momentum * self.baseline_value + (1 - self.baseline_momentum) * reward
            advantage = reward - self.baseline_value
        
        # Compute entropy for regularization
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs)
        
        # Improved policy loss with entropy regularization
        # Weight log_probs by their position importance (later decisions matter more)
        position_weights = torch.linspace(0.5, 1.0, len(log_probs)).to(log_probs.device)
        weighted_log_probs = log_probs * position_weights
        
        policy_loss = -torch.sum(weighted_log_probs) * advantage - self.entropy_weight * entropy
        
        # Update policy networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.gnn.parameters()) + list(self.model.pointer_net.parameters()),
                self.gradient_clip
            )
        self.policy_optimizer.step()
        
        episode_info = {
            'reward': reward,
            'baseline': self.baseline_value,
            'advantage': advantage,
            'policy_loss': policy_loss.detach().item(),
            'baseline_loss': 0.0,  # No baseline loss with moving average
            'entropy': entropy.detach().item(),
            'permutation': permutation.cpu().numpy(),
            'log_probs_sum': torch.sum(log_probs).detach().item(),
            'reward_buffer_size': len(self.reward_buffer)
        }
        
        return episode_info
    
    def train(self, 
              dataset,
              num_episodes: int,
              temperature_schedule: Optional[Callable] = None,
              validation_dataset=None,
              validation_frequency: int = 100,
              save_frequency: int = 1000,
              checkpoint_path: Optional[str] = None) -> Dict:
        """
        Train the model using REINFORCE.
        
        Args:
            dataset: Training dataset
            num_episodes: Number of training episodes
            temperature_schedule: Function that returns temperature for episode
            validation_dataset: Optional validation dataset
            validation_frequency: How often to run validation
            save_frequency: How often to save checkpoints
            checkpoint_path: Path to save checkpoints
            
        Returns:
            training_history: Complete training statistics
        """
        print(f"Starting REINFORCE training for {num_episodes} episodes")
        print(f"Dataset size: {len(dataset)}")
        print(f"Model parameters: {self.model.get_num_parameters()}")
        
        if temperature_schedule is None:
            temperature_schedule = lambda episode: max(0.1, 1.0 * (0.99 ** (episode // 100)))
        
        training_history = {
            'episode_rewards': [],
            'episode_baselines': [],
            'policy_losses': [],
            'baseline_losses': [],
            'validation_scores': [],
            'validation_episodes': []
        }
        
        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Sample problem from dataset
            problem_idx = np.random.randint(len(dataset))
            lp_problem = dataset[problem_idx]
            
            # Get temperature for this episode
            temperature = temperature_schedule(episode)
            
            # Train on this episode
            episode_info = self.train_episode(lp_problem, temperature)
            
            # Record statistics
            training_history['episode_rewards'].append(episode_info['reward'])
            training_history['episode_baselines'].append(episode_info['baseline'])
            training_history['policy_losses'].append(episode_info['policy_loss'])
            training_history['baseline_losses'].append(episode_info['baseline_loss'])
            training_history['entropy_values'] = getattr(training_history, 'entropy_values', [])
            training_history['entropy_values'].append(episode_info['entropy'])
            
            # Validation
            if validation_dataset and (episode + 1) % validation_frequency == 0:
                val_score = self.validate(validation_dataset)
                training_history['validation_scores'].append(val_score)
                training_history['validation_episodes'].append(episode)
                
                print(f"Episode {episode + 1}: Reward={episode_info['reward']:.4f}, "
                      f"Baseline={episode_info['baseline']:.4f}, "
                      f"Validation={val_score:.4f}, "
                      f"Temperature={temperature:.4f}")
            
            # Save checkpoint
            if checkpoint_path and (episode + 1) % save_frequency == 0:
                self.save_checkpoint(checkpoint_path, episode, training_history)
        
        return training_history
    
    def validate(self, validation_dataset, num_samples: Optional[int] = None) -> float:
        """
        Validate the model on validation dataset.
        
        Args:
            validation_dataset: Validation dataset
            num_samples: Number of samples to validate on (None for all)
            
        Returns:
            average_reward: Average reward on validation set
        """
        self.model.eval()
        
        if num_samples is None:
            num_samples = len(validation_dataset)
        else:
            num_samples = min(num_samples, len(validation_dataset))
        
        total_reward = 0.0
        
        with torch.no_grad():
            for i in range(num_samples):
                lp_problem = validation_dataset[i]
                
                # Sample permutation
                sample_output = self.model.sample_permutation(lp_problem, temperature=0.1)
                
                # Apply permutation
                reformulated_lp = self.model.apply_permutation(
                    lp_problem, 
                    sample_output['permutation'], 
                    sample_output['cluster_assignments']
                )
                
                # Compute reward
                reward = self.reward_function(lp_problem, reformulated_lp)
                total_reward += reward
        
        return total_reward / num_samples
    
    def _default_reward_function(self, original_lp: Dict, reformulated_lp: Dict) -> float:
        """
        Fixed reward function that works for easy problems and provides learning signals.
        
        Args:
            original_lp: Original LP problem
            reformulated_lp: Reformulated LP problem
            
        Returns:
            reward: Reward value (positive for improvement, negative for penalties)
        """
        try:
            # Solve both problems
            original_result = self.solver.solve(original_lp)
            reformulated_result = self.solver.solve(reformulated_lp)
            
            # Heavy penalty for solver failures
            if not original_result['success']:
                return -20.0
            
            if not reformulated_result['success']:
                return -15.0
            
            # Get metrics
            orig_time = original_result.get('solve_time', 0.0)
            reform_time = reformulated_result.get('solve_time', 0.0)
            orig_iters = original_result.get('iterations', 0)
            reform_iters = reformulated_result.get('iterations', 0)
            
            reward = 0.0
            
            # 1. Always use time-based reward (remove 0.01s threshold)
            if orig_time > 0 and reform_time > 0:
                # Use relative improvement with higher sensitivity
                time_improvement = (orig_time - reform_time) / orig_time
                time_reward = time_improvement * 50.0  # Increased sensitivity
                reward += time_reward
                
            # 2. Always use iteration-based reward 
            if orig_iters > 0 and reform_iters > 0:
                iter_improvement = (orig_iters - reform_iters) / orig_iters
                iter_reward = iter_improvement * 30.0  # Increased sensitivity
                reward += iter_reward
                
            # 3. NO FIXED BONUSES - they mask the learning signal
            
            # 4. Add small noise to break ties when improvements are identical
            noise = np.random.normal(0, 0.01)  # Small random noise
            reward += noise
            
            # 5. Scale reward to reasonable range
            reward = np.clip(reward, -20.0, 20.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"💥 Exception in reward computation: {e}")
            return -20.0
    
    def save_checkpoint(self, path: str, episode: int, training_history: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'baseline_value': self.baseline_value,  # Save moving average baseline
            'training_history': training_history
        }
        torch.save(checkpoint, f"{path}_episode_{episode}.pt")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        # Load baseline value if available (for backward compatibility)
        if 'baseline_value' in checkpoint:
            self.baseline_value = checkpoint['baseline_value']
        else:
            self.baseline_value = 0.0  # Default value for old checkpoints
        
        return checkpoint['episode'], checkpoint['training_history']


class AdaptiveTemperatureSchedule:
    """
    Adaptive temperature schedule that adjusts based on training progress.
    """
    
    def __init__(self, 
                 initial_temp: float = 1.0,
                 min_temp: float = 0.1,
                 decay_rate: float = 0.99,
                 decay_frequency: int = 100,
                 adaptive: bool = True):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.decay_rate = decay_rate
        self.decay_frequency = decay_frequency
        self.adaptive = adaptive
        
        self.recent_rewards = []
        self.reward_window = 100
        
    def __call__(self, episode: int, recent_reward: Optional[float] = None) -> float:
        """Get temperature for current episode."""
        if recent_reward is not None:
            self.recent_rewards.append(recent_reward)
            if len(self.recent_rewards) > self.reward_window:
                self.recent_rewards.pop(0)
        
        # Base exponential decay
        base_temp = max(
            self.min_temp,
            self.initial_temp * (self.decay_rate ** (episode // self.decay_frequency))
        )
        
        if not self.adaptive or len(self.recent_rewards) < 10:
            return base_temp
        
        # Adaptive adjustment based on recent performance
        recent_mean = np.mean(self.recent_rewards[-10:])
        if len(self.recent_rewards) >= 20:
            older_mean = np.mean(self.recent_rewards[-20:-10])
            
            # If performance is improving, can reduce temperature faster
            if recent_mean > older_mean:
                base_temp *= 0.95
            # If performance is degrading, increase temperature for more exploration
            elif recent_mean < older_mean:
                base_temp *= 1.05
        
        return max(self.min_temp, min(self.initial_temp, base_temp)) 