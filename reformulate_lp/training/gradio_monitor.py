"""
Gradio interface for monitoring training progress in real-time.

This module provides a web-based interface to visualize training losses,
rewards, and other metrics during the training process.
"""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime


class TrainingMonitor:
    """
    Real-time training monitor using Gradio interface.
    """
    
    def __init__(self, 
                 title: str = "LP Reformulation Training Monitor",
                 port: int = 7860,
                 share: bool = False,
                 output_dir: Optional[str] = None):
        """
        Initialize the training monitor.
        
        Args:
            title: Title for the Gradio interface
            port: Port to run the interface on
            share: Whether to create a public link
            output_dir: Directory to save plots and data
        """
        self.title = title
        self.port = port
        self.share = share
        self.output_dir = output_dir or "monitor_outputs"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training data storage
        self.training_data = {
            'episodes': [],
            'rewards': [],
            'baselines': [],
            'policy_losses': [],
            'baseline_losses': [],
            'validation_scores': [],
            'validation_episodes': [],
            'temperatures': [],
            'solving_times_original': [],
            'solving_times_reformulated': [],
            'improvement_ratios': []
        }
        
        # Thread-safe queue for updates
        self.update_queue = queue.Queue()
        self.is_training = False
        self.training_start_time = None
        
        # Gradio interface components
        self.interface = None
        self.update_interval = 3.0  # seconds
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
    def start_interface(self):
        """Start the Gradio interface."""
        self.interface = self._create_interface()
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Launch interface
        print(f"Launching Gradio interface on port {self.port}")
        self.interface.launch(
            server_port=self.port,
            share=self.share,
            show_error=True,
            quiet=False,
            prevent_thread_lock=True
        )
    
    def _create_interface(self):
        """Create the Gradio interface layout."""
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown("Real-time training monitoring interface")
            
            # Training status
            with gr.Row():
                status_text = gr.Textbox(
                    label="Training Status",
                    value="Not Started",
                    interactive=False
                )
                episode_counter = gr.Number(
                    label="Current Episode",
                    value=0,
                    interactive=False
                )
                elapsed_time = gr.Textbox(
                    label="Elapsed Time",
                    value="00:00:00",
                    interactive=False
                )
            
            # Main metrics plots
            with gr.Row():
                with gr.Column():
                    reward_plot = gr.Plot(label="Reward Progress")
                    loss_plot = gr.Plot(label="Policy Loss")
                with gr.Column():
                    validation_plot = gr.Plot(label="Validation Performance")
                    temperature_plot = gr.Plot(label="Temperature Schedule")
            
            # Performance metrics
            with gr.Row():
                with gr.Column():
                    solving_time_plot = gr.Plot(label="Solving Time Comparison")
                with gr.Column():
                    improvement_plot = gr.Plot(label="Improvement Ratios")
            
            # Statistics table
            with gr.Row():
                stats_table = gr.DataFrame(
                    label="Training Statistics",
                    headers=["Metric", "Current", "Best", "Average", "Std"],
                    datatype=["str", "number", "number", "number", "number"],
                    value=[["No data", "-", "-", "-", "-"]]
                )
            
            # Control buttons
            with gr.Row():
                refresh_btn = gr.Button("Refresh Now", variant="primary")
                save_btn = gr.Button("Save Data", variant="secondary")
                reset_btn = gr.Button("Reset", variant="secondary")
                export_btn = gr.Button("Export All", variant="secondary")
            
            # Instructions
            with gr.Row():
                gr.Markdown("**Instructions:** Click 'Refresh Now' to update all plots and data. The interface shows live training progress from the connected training process.")
            
            # Define refresh function
            def refresh_all():
                """Refresh all interface components."""
                try:
                    plots = self.get_plots()
                    stats = self.get_statistics_table()  
                    status, episode, elapsed = self.get_status_info()
                    
                    return (
                        plots[0],  # reward_plot
                        plots[1],  # loss_plot
                        plots[2],  # validation_plot
                        plots[3],  # temperature_plot
                        plots[4],  # solving_time_plot
                        plots[5],  # improvement_plot
                        stats,     # stats_table
                        status,    # status_text
                        episode,   # episode_counter
                        elapsed    # elapsed_time
                    )
                except Exception as e:
                    print(f"Error in refresh_all: {e}")
                    empty_plots = self._create_empty_plots()
                    return (
                        empty_plots[0], empty_plots[1], empty_plots[2],
                        empty_plots[3], empty_plots[4], empty_plots[5],
                        [["Error", "-", "-", "-", "-"]],
                        "Error",
                        0,
                        "00:00:00"
                    )
            
            def save_data_action():
                """Save current training data."""
                try:
                    result = self._save_data()
                    print(f"Data saved successfully: {result}")
                    return result
                except Exception as e:
                    print(f"Failed to save data: {e}")
                    return "Save failed"
            
            def reset_data_action():
                """Reset all training data."""
                try:
                    result = self._reset_data()
                    print("Data reset successfully")
                    # Return refreshed interface
                    return refresh_all()
                except Exception as e:
                    print(f"Failed to reset data: {e}")
                    return refresh_all()
            
            def export_data_action():
                """Export all data and plots."""
                try:
                    result = self._export_data()
                    print("Data exported successfully")
                    return result
                except Exception as e:
                    print(f"Failed to export data: {e}")
                    return "Export failed"
            
            # Set up button callbacks
            all_outputs = [
                reward_plot, loss_plot, validation_plot,
                temperature_plot, solving_time_plot, improvement_plot,
                stats_table, status_text, episode_counter, elapsed_time
            ]
            
            refresh_btn.click(
                fn=refresh_all,
                outputs=all_outputs
            )
            
            save_btn.click(fn=save_data_action, outputs=[])
            
            reset_btn.click(
                fn=reset_data_action,
                outputs=all_outputs
            )
            
            export_btn.click(fn=export_data_action, outputs=[])
            
            # Initial load of interface
            interface.load(
                fn=refresh_all,
                outputs=all_outputs
            )
            
        return interface
    
    def update(self, episode_data: Dict):
        """
        Update training data with new episode information.
        
        Args:
            episode_data: Dictionary containing episode metrics
        """
        try:
            self.update_queue.put(episode_data)
        except Exception as e:
            print(f"Error updating monitor: {e}")
    
    def start_training(self):
        """Mark the start of training."""
        self.is_training = True
        self.training_start_time = time.time()
    
    def stop_training(self):
        """Mark the end of training."""
        self.is_training = False
    
    def _update_loop(self):
        """Background thread that processes updates."""
        while True:
            try:
                # Process all queued updates
                while not self.update_queue.empty():
                    try:
                        episode_data = self.update_queue.get_nowait()
                        self._process_update(episode_data)
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Error processing update: {e}")
                
                time.sleep(1.0)  # Check queue every second
                
            except Exception as e:
                print(f"Update loop error: {e}")
                time.sleep(1.0)
    
    def _process_update(self, episode_data: Dict):
        """Process a single update."""
        with self.data_lock:
            try:
                episode = len(self.training_data['episodes'])
                
                # Store basic episode data
                self.training_data['episodes'].append(episode)
                self.training_data['rewards'].append(episode_data.get('reward', 0))
                self.training_data['baselines'].append(episode_data.get('baseline', 0))
                self.training_data['policy_losses'].append(episode_data.get('policy_loss', 0))
                self.training_data['baseline_losses'].append(episode_data.get('baseline_loss', 0))
                self.training_data['temperatures'].append(episode_data.get('temperature', 1.0))
                
                # Store performance data if available
                if 'solving_time_original' in episode_data:
                    self.training_data['solving_times_original'].append(episode_data['solving_time_original'])
                if 'solving_time_reformulated' in episode_data:
                    self.training_data['solving_times_reformulated'].append(episode_data['solving_time_reformulated'])
                if 'improvement_ratio' in episode_data:
                    self.training_data['improvement_ratios'].append(episode_data['improvement_ratio'])
                
                # Store validation data
                if 'validation_score' in episode_data:
                    self.training_data['validation_scores'].append(episode_data['validation_score'])
                    self.training_data['validation_episodes'].append(episode)
                    
            except Exception as e:
                print(f"Error in _process_update: {e}")
    
    def get_plots(self) -> Tuple[plt.Figure, ...]:
        """Generate all training plots."""
        with self.data_lock:
            try:
                if not self.training_data['episodes']:
                    return self._create_empty_plots()
                
                # Create plots
                reward_fig = self._create_reward_plot()
                loss_fig = self._create_loss_plot()
                validation_fig = self._create_validation_plot()
                temperature_fig = self._create_temperature_plot()
                solving_time_fig = self._create_solving_time_plot()
                improvement_fig = self._create_improvement_plot()
                
                return (reward_fig, loss_fig, validation_fig, 
                        temperature_fig, solving_time_fig, improvement_fig)
            except Exception as e:
                print(f"Error generating plots: {e}")
                return self._create_empty_plots()
    
    def _create_reward_plot(self) -> plt.Figure:
        """Create reward progress plot."""
        plt.style.use('default')  # Reset style
        fig, ax = plt.subplots(figsize=(8, 5))
        
        episodes = self.training_data['episodes']
        rewards = self.training_data['rewards']
        baselines = self.training_data['baselines']
        
        if episodes:
            ax.plot(episodes, rewards, label='Reward', alpha=0.6, color='blue', linewidth=1)
            ax.plot(episodes, baselines, label='Baseline', color='red', linewidth=2)
            
            # Add moving average
            if len(rewards) > 20:
                window = min(50, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='green', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_loss_plot(self) -> plt.Figure:
        """Create policy loss plot."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        episodes = self.training_data['episodes']
        policy_losses = self.training_data['policy_losses']
        
        if episodes:
            ax.plot(episodes, policy_losses, label='Policy Loss', color='orange', alpha=0.6, linewidth=1)
            
            # Add moving average
            if len(policy_losses) > 20:
                window = min(50, len(policy_losses) // 5)
                moving_avg = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
                ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='red', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_validation_plot(self) -> plt.Figure:
        """Create validation performance plot."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        val_episodes = self.training_data['validation_episodes']
        val_scores = self.training_data['validation_scores']
        
        if val_episodes:
            ax.plot(val_episodes, val_scores, 'o-', label='Validation Score', color='green', markersize=4, linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No validation data yet', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Validation Score')
        ax.set_title('Validation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_temperature_plot(self) -> plt.Figure:
        """Create temperature schedule plot."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        episodes = self.training_data['episodes']
        temperatures = self.training_data['temperatures']
        
        if episodes:
            ax.plot(episodes, temperatures, label='Temperature', color='purple', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Temperature')
        ax.set_title('Temperature Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_solving_time_plot(self) -> plt.Figure:
        """Create solving time comparison plot."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if (self.training_data['solving_times_original'] and 
            self.training_data['solving_times_reformulated']):
            
            episodes = range(len(self.training_data['solving_times_original']))
            original_times = self.training_data['solving_times_original']
            reformulated_times = self.training_data['solving_times_reformulated']
            
            ax.plot(episodes, original_times, label='Original', alpha=0.7, color='red', linewidth=1)
            ax.plot(episodes, reformulated_times, label='Reformulated', alpha=0.7, color='blue', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No solving time data yet', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Solving Time (s)')
        ax.set_title('Solving Time Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_improvement_plot(self) -> plt.Figure:
        """Create improvement ratio plot."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if self.training_data['improvement_ratios']:
            episodes = range(len(self.training_data['improvement_ratios']))
            improvements = self.training_data['improvement_ratios']
            
            ax.plot(episodes, improvements, label='Improvement Ratio', color='green', alpha=0.7, linewidth=1)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
        else:
            ax.text(0.5, 0.5, 'No improvement data yet', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Improvement Ratio')
        ax.set_title('Performance Improvement Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_empty_plots(self):
        """Create empty plots when no data is available."""
        plots = []
        titles = ['Reward Progress', 'Policy Loss', 'Validation Performance', 
                 'Temperature Schedule', 'Solving Time Comparison', 'Improvement Ratios']
        
        for title in titles:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title)
            ax.text(0.5, 0.5, 'Waiting for training data...', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots.append(fig)
        
        return tuple(plots)
    
    def get_statistics_table(self) -> List[List]:
        """Generate training statistics table."""
        with self.data_lock:
            try:
                if not self.training_data['rewards']:
                    return [["No data available", "-", "-", "-", "-"]]
                
                stats = []
                
                # Reward statistics
                rewards = np.array(self.training_data['rewards'])
                stats.append([
                    "Reward",
                    f"{rewards[-1]:.4f}",
                    f"{rewards.max():.4f}",
                    f"{rewards.mean():.4f}",
                    f"{rewards.std():.4f}"
                ])
                
                # Policy loss statistics
                losses = np.array(self.training_data['policy_losses'])
                stats.append([
                    "Policy Loss",
                    f"{losses[-1]:.4f}",
                    f"{losses.min():.4f}",
                    f"{losses.mean():.4f}",
                    f"{losses.std():.4f}"
                ])
                
                # Validation statistics
                if self.training_data['validation_scores']:
                    val_scores = np.array(self.training_data['validation_scores'])
                    stats.append([
                        "Validation",
                        f"{val_scores[-1]:.4f}",
                        f"{val_scores.max():.4f}",
                        f"{val_scores.mean():.4f}",
                        f"{val_scores.std():.4f}"
                    ])
                
                return stats
            except Exception as e:
                print(f"Error generating stats table: {e}")
                return [["Error generating stats", "-", "-", "-", "-"]]
    
    def get_status_info(self) -> Tuple[str, int, str]:
        """Get current training status information."""
        try:
            if not self.is_training:
                status = "Training Stopped" if self.training_data['episodes'] else "Waiting for Training"
            else:
                status = "Training Active"
            
            episode = len(self.training_data['episodes'])
            
            if self.training_start_time:
                elapsed = time.time() - self.training_start_time
                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                elapsed_str = "00:00:00"
            
            return status, episode, elapsed_str
        except Exception as e:
            print(f"Error getting status info: {e}")
            return "Error", 0, "00:00:00"
    
    def _save_data(self):
        """Save current training data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"training_monitor_{timestamp}.json")
            
            with self.data_lock:
                with open(filename, 'w') as f:
                    json.dump(self.training_data, f, indent=2)
            
            return f"Data saved to {filename}"
        except Exception as e:
            print(f"Error saving data: {e}")
            return f"Error saving data: {e}"
    
    def _reset_data(self):
        """Reset all training data."""
        try:
            with self.data_lock:
                for key in self.training_data:
                    self.training_data[key] = []
            return "Data reset successfully"
        except Exception as e:
            print(f"Error resetting data: {e}")
            return f"Error resetting data: {e}"
    
    def _export_data(self):
        """Export data in multiple formats."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON
            json_file = os.path.join(self.output_dir, f"training_export_{timestamp}.json")
            with self.data_lock:
                with open(json_file, 'w') as f:
                    json.dump(self.training_data, f, indent=2)
            
            # Save plots
            plots = self.get_plots()
            plot_names = ['reward', 'loss', 'validation', 'temperature', 'solving_time', 'improvement']
            
            for plot, name in zip(plots, plot_names):
                plot_file = os.path.join(self.output_dir, f"plot_{name}_{timestamp}.png")
                plot.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(plot)  # Close to free memory
            
            return f"Data exported to {self.output_dir}"
        except Exception as e:
            print(f"Error exporting data: {e}")
            return f"Error exporting data: {e}" 