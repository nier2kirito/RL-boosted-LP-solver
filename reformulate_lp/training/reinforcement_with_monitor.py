"""
Enhanced REINFORCE training with Gradio monitoring integration.

This module extends the base REINFORCE trainer to support real-time
loss monitoring through a Gradio web interface.
"""

import time
from typing import Dict, List, Tuple, Optional, Callable
import threading

from .reinforcement import REINFORCETrainer, AdaptiveTemperatureSchedule
from .gradio_monitor import TrainingMonitor


class REINFORCETrainerWithMonitor(REINFORCETrainer):
    """
    Enhanced REINFORCE trainer with Gradio monitoring support.
    
    Extends the base trainer to provide real-time visualization of
    training metrics through a web interface.
    """
    
    def __init__(self,
                 model,
                 solver=None,
                 learning_rate: float = 1e-4,
                 baseline_learning_rate: float = 1e-3,
                 reward_function: Optional[Callable] = None,
                 device: str = 'cpu',
                 gradient_clip: float = 1.0,
                 monitor_enabled: bool = True,
                 monitor_port: int = 7860,
                 monitor_share: bool = False,
                 monitor_update_frequency: int = 10):
        """
        Initialize enhanced REINFORCE trainer with monitoring.
        
        Args:
            model: ReformulationSystem model
            solver: LP solver interface
            learning_rate: Learning rate for policy networks
            baseline_learning_rate: Learning rate for baseline network
            reward_function: Custom reward function
            device: Training device
            gradient_clip: Gradient clipping value
            monitor_enabled: Whether to enable Gradio monitoring
            monitor_port: Port for Gradio interface
            monitor_share: Whether to create public sharing link
            monitor_update_frequency: How often to update monitor (episodes)
        """
        super().__init__(
            model=model,
            solver=solver,
            learning_rate=learning_rate,
            baseline_learning_rate=baseline_learning_rate,
            reward_function=reward_function,
            device=device,
            gradient_clip=gradient_clip
        )
        
        # Monitor configuration
        self.monitor_enabled = monitor_enabled
        self.monitor_update_frequency = monitor_update_frequency
        
        # Initialize monitor
        if self.monitor_enabled:
            self.monitor = TrainingMonitor(
                title="LP Reformulation Training Monitor",
                port=monitor_port,
                share=monitor_share
            )
            self.monitor_thread = None
        else:
            self.monitor = None
    
    def start_monitor(self, output_dir: Optional[str] = None):
        """Start the Gradio monitoring interface."""
        if not self.monitor_enabled or self.monitor is None:
            print("Monitor is disabled")
            return
        
        if output_dir:
            self.monitor.output_dir = output_dir
        
        print(f"Starting monitoring interface on port {self.monitor.port}")
        
        # Start monitor in separate thread
        self.monitor_thread = threading.Thread(
            target=self.monitor.start_interface,
            daemon=True
        )
        self.monitor_thread.start()
        
        print("Monitor started. Access the interface at:")
        print(f"  Local: http://localhost:{self.monitor.port}")
        if self.monitor.share:
            print("  Public link will be displayed when available")
    
    def train_episode(self, lp_problem: Dict, temperature: float = 1.0) -> Dict:
        """
        Train on a single LP problem episode with monitoring.
        
        Args:
            lp_problem: LP problem instance
            temperature: Sampling temperature
            
        Returns:
            episode_info: Training statistics for this episode
        """
        # Time the episode
        episode_start = time.time()
        
        # Get original problem solving time for comparison
        original_solve_time = self._time_original_solve(lp_problem)
        
        # Run the base training episode
        episode_info = super().train_episode(lp_problem, temperature)
        
        # Time the reformulated problem solving
        reformulated_solve_time = self._time_reformulated_solve(lp_problem, episode_info)
        
        # Calculate improvement metrics
        improvement_ratio = original_solve_time / max(reformulated_solve_time, 1e-6)
        
        # Enhance episode info with additional metrics
        episode_info.update({
            'temperature': temperature,
            'episode_time': time.time() - episode_start,
            'solving_time_original': original_solve_time,
            'solving_time_reformulated': reformulated_solve_time,
            'improvement_ratio': improvement_ratio
        })
        
        return episode_info
    
    def train(self, 
              dataset,
              num_episodes: int,
              temperature_schedule: Optional[Callable] = None,
              validation_dataset=None,
              validation_frequency: int = 100,
              save_frequency: int = 1000,
              checkpoint_path: Optional[str] = None,
              output_dir: Optional[str] = None) -> Dict:
        """
        Train the model using REINFORCE with monitoring.
        
        Args:
            dataset: Training dataset
            num_episodes: Number of training episodes
            temperature_schedule: Function that returns temperature for episode
            validation_dataset: Optional validation dataset
            validation_frequency: How often to run validation
            save_frequency: How often to save checkpoints
            checkpoint_path: Path to save checkpoints
            output_dir: Output directory for monitor data
            
        Returns:
            training_history: Complete training statistics
        """
        # Start monitoring if enabled
        if self.monitor_enabled:
            self.start_monitor(output_dir)
            self.monitor.start_training()
        
        print(f"Starting enhanced REINFORCE training for {num_episodes} episodes")
        print(f"Monitor enabled: {self.monitor_enabled}")
        
        # Run base training with monitoring hooks
        training_history = super().train(
            dataset=dataset,
            num_episodes=num_episodes,
            temperature_schedule=temperature_schedule,
            validation_dataset=validation_dataset,
            validation_frequency=validation_frequency,
            save_frequency=save_frequency,
            checkpoint_path=checkpoint_path
        )
        
        # Stop monitoring
        if self.monitor_enabled:
            self.monitor.stop_training()
            print("Training completed. Monitor interface remains active.")
        
        return training_history
    
    def train_with_live_updates(self,
                               dataset,
                               num_episodes: int,
                               temperature_schedule: Optional[Callable] = None,
                               validation_dataset=None,
                               validation_frequency: int = 100,
                               save_frequency: int = 1000,
                               checkpoint_path: Optional[str] = None,
                               output_dir: Optional[str] = None) -> Dict:
        """
        Enhanced training loop with live monitoring updates.
        """
        import numpy as np
        from tqdm import tqdm
        
        # Start monitoring
        if self.monitor_enabled:
            self.start_monitor(output_dir)
            self.monitor.start_training()
        
        if temperature_schedule is None:
            temperature_schedule = lambda episode: max(0.1, 1.0 * (0.99 ** (episode // 100)))
        
        training_history = {
            'episode_rewards': [],
            'episode_baselines': [],
            'policy_losses': [],
            'baseline_losses': [],
            'validation_scores': [],
            'validation_episodes': [],
            'temperatures': [],
            'solving_times_original': [],
            'solving_times_reformulated': [],
            'improvement_ratios': []
        }
        
        print(f"Starting monitored REINFORCE training for {num_episodes} episodes")
        print(f"Dataset size: {len(dataset)}")
        print(f"Model parameters: {self.model.get_num_parameters()}")
        
        # Training loop with monitoring
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
            training_history['temperatures'].append(temperature)
            
            if 'solving_time_original' in episode_info:
                training_history['solving_times_original'].append(episode_info['solving_time_original'])
            if 'solving_time_reformulated' in episode_info:
                training_history['solving_times_reformulated'].append(episode_info['solving_time_reformulated'])
            if 'improvement_ratio' in episode_info:
                training_history['improvement_ratios'].append(episode_info['improvement_ratio'])
            
            # Update monitor
            if self.monitor_enabled and episode % self.monitor_update_frequency == 0:
                monitor_data = {
                    'episode': episode,
                    'reward': episode_info['reward'],
                    'baseline': episode_info['baseline'],
                    'policy_loss': episode_info['policy_loss'],
                    'baseline_loss': episode_info['baseline_loss'],
                    'temperature': temperature,
                    'solving_time_original': episode_info.get('solving_time_original', 0),
                    'solving_time_reformulated': episode_info.get('solving_time_reformulated', 0),
                    'improvement_ratio': episode_info.get('improvement_ratio', 1.0)
                }
                self.monitor.update(monitor_data)
            
            # Validation
            if validation_dataset and (episode + 1) % validation_frequency == 0:
                val_score = self.validate(validation_dataset)
                training_history['validation_scores'].append(val_score)
                training_history['validation_episodes'].append(episode)
                
                # Update monitor with validation score
                if self.monitor_enabled:
                    validation_data = {
                        'episode': episode,
                        'validation_score': val_score,
                        'reward': episode_info['reward'],
                        'baseline': episode_info['baseline'],
                        'policy_loss': episode_info['policy_loss'],
                        'baseline_loss': episode_info['baseline_loss'],
                        'temperature': temperature
                    }
                    self.monitor.update(validation_data)
                
                print(f"Episode {episode + 1}: Reward={episode_info['reward']:.4f}, "
                      f"Baseline={episode_info['baseline']:.4f}, "
                      f"Validation={val_score:.4f}, "
                      f"Temperature={temperature:.4f}")
            
            # Save checkpoint
            if checkpoint_path and (episode + 1) % save_frequency == 0:
                self.save_checkpoint(checkpoint_path, episode, training_history)
        
        # Stop monitoring
        if self.monitor_enabled:
            self.monitor.stop_training()
            print("\nTraining completed. Monitor interface remains active for inspection.")
        
        return training_history
    
    def _time_original_solve(self, lp_problem: Dict) -> float:
        """Time how long it takes to solve the original problem."""
        try:
            start_time = time.time()
            self.solver.solve_lp(lp_problem)
            return time.time() - start_time
        except Exception:
            return float('inf')  # Return infinity if solve fails
    
    def _time_reformulated_solve(self, lp_problem: Dict, episode_info: Dict) -> float:
        """Time how long it takes to solve the reformulated problem."""
        try:
            # Get the reformulated problem from episode info
            if 'permutation' in episode_info:
                # Apply the permutation to get reformulated LP
                permutation = episode_info['permutation']
                cluster_assignments = episode_info.get('cluster_assignments', None)
                reformulated_lp = self.model.apply_permutation(
                    lp_problem, permutation, cluster_assignments
                )
                
                start_time = time.time()
                self.solver.solve_lp(reformulated_lp)
                return time.time() - start_time
            else:
                return float('inf')
        except Exception:
            return float('inf')  # Return infinity if solve fails
    
    def get_monitor_data(self) -> Dict:
        """Get current monitoring data."""
        if self.monitor_enabled and self.monitor:
            return self.monitor.training_data.copy()
        return {}
    
    def save_monitor_data(self, filepath: str):
        """Save monitoring data to file."""
        if self.monitor_enabled and self.monitor:
            self.monitor._save_data()
            print(f"Monitor data saved")
        else:
            print("Monitor not enabled")
    
    def export_monitor_plots(self, output_dir: str):
        """Export all monitor plots to files."""
        if self.monitor_enabled and self.monitor:
            original_output_dir = self.monitor.output_dir
            self.monitor.output_dir = output_dir
            self.monitor._export_data()
            self.monitor.output_dir = original_output_dir
            print(f"Plots exported to {output_dir}")
        else:
            print("Monitor not enabled") 