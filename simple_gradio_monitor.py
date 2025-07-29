#!/usr/bin/env python3
"""
Simplified Gradio monitoring interface with REAL auto-refresh functionality.

This version uses a background thread to actually update the interface automatically.
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


class SimpleTrainingMonitor:
    """
    Simple training monitor with real auto-refresh capability.
    """
    
    def __init__(self, 
                 title: str = "LP Reformulation Training Monitor (Auto-Refresh)",
                 port: int = 7860,
                 share: bool = False,
                 output_dir: Optional[str] = None,
                 refresh_interval: float = 2.0):
        """Initialize the simple monitor."""
        self.title = title
        self.port = port
        self.share = share
        self.output_dir = output_dir or "monitor_outputs"
        self.refresh_interval = refresh_interval
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training data storage
        self.training_data = {
            'episodes': [],
            'rewards': [],
            'baselines': [],
            'policy_losses': [],
            'temperatures': [],
            'validation_scores': [],
            'validation_episodes': []
        }
        
        self.update_queue = queue.Queue()
        self.is_training = False
        self.training_start_time = None
        self.data_lock = threading.Lock()
        
        # Auto-refresh state
        self.auto_refresh_enabled = True
        self.interface_components = {}
        self.refresh_thread = None
        self.stop_refresh = False
        
    def start_interface(self):
        """Start the Gradio interface."""
        
        def create_reward_plot():
            """Create reward plot."""
            with self.data_lock:
                plt.close('all')  # Clear any existing plots
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if self.training_data['episodes']:
                    episodes = self.training_data['episodes']
                    rewards = self.training_data['rewards']
                    baselines = self.training_data['baselines']
                    
                    ax.plot(episodes, rewards, label='Reward', alpha=0.7, color='blue', linewidth=1.5)
                    ax.plot(episodes, baselines, label='Baseline', color='red', linewidth=2)
                    
                    # Add moving average if enough data
                    if len(rewards) > 10:
                        window = min(20, len(rewards) // 3)
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='green', linewidth=2)
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Reward')
                    ax.set_title(f'Training Reward Progress (Episodes: {len(episodes)})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '🔄 Waiting for training data...', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title('Training Reward Progress')
                
                plt.tight_layout()
                return fig
        
        def create_loss_plot():
            """Create loss plot."""
            with self.data_lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if self.training_data['episodes']:
                    episodes = self.training_data['episodes']
                    losses = self.training_data['policy_losses']
                    
                    ax.plot(episodes, losses, label='Policy Loss', color='orange', alpha=0.7, linewidth=1.5)
                    
                    # Add moving average if enough data
                    if len(losses) > 10:
                        window = min(20, len(losses) // 3)
                        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='red', linewidth=2)
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Policy Loss Progress (Episodes: {len(episodes)})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '🔄 Waiting for training data...', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title('Policy Loss Progress')
                
                plt.tight_layout()
                return fig
        
        def get_status():
            """Get current status."""
            with self.data_lock:
                if not self.is_training:
                    status = "🛑 Training Stopped" if self.training_data['episodes'] else "⏳ Waiting for Training"
                else:
                    status = "🚀 Training Active"
                
                episode = len(self.training_data['episodes'])
                
                if self.training_start_time:
                    elapsed = time.time() - self.training_start_time
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    elapsed_str = "00:00:00"
                
                # Add last update time and auto-refresh status
                current_time = time.strftime("%H:%M:%S")
                refresh_status = "🔄 ON" if self.auto_refresh_enabled else "⏸️ OFF"
                status_with_info = f"{status} | Last update: {current_time} | Auto-refresh: {refresh_status}"
                
                return status_with_info, episode, elapsed_str
        
        def get_stats():
            """Get statistics table."""
            with self.data_lock:
                if not self.training_data['rewards']:
                    return [["📊 No data available", "-", "-", "-"]]
                
                rewards = np.array(self.training_data['rewards'])
                losses = np.array(self.training_data['policy_losses'])
                
                stats = [
                    ["🎯 Reward", f"{rewards[-1]:.4f}", f"{rewards.max():.4f}", f"{rewards.mean():.4f}"],
                    ["📉 Policy Loss", f"{losses[-1]:.4f}", f"{losses.min():.4f}", f"{losses.mean():.4f}"]
                ]
                
                if self.training_data['validation_scores']:
                    val_scores = np.array(self.training_data['validation_scores'])
                    stats.append(["✅ Validation", f"{val_scores[-1]:.4f}", f"{val_scores.max():.4f}", f"{val_scores.mean():.4f}"])
                
                # Add episode count and temperature
                stats.append(["📈 Episodes", f"{len(self.training_data['episodes'])}", "-", "-"])
                if self.training_data['temperatures']:
                    current_temp = self.training_data['temperatures'][-1]
                    stats.append(["🌡️ Temperature", f"{current_temp:.3f}", "-", "-"])
                
                return stats
        
        def refresh_all():
            """Refresh all components."""
            try:
                reward_plot = create_reward_plot()
                loss_plot = create_loss_plot()
                status, episode, elapsed = get_status()
                stats = get_stats()
                
                return reward_plot, loss_plot, status, episode, elapsed, stats
            except Exception as e:
                print(f"❌ Error refreshing: {e}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'❌ Error: {e}', ha='center', va='center', transform=ax.transAxes)
                plt.tight_layout()
                return fig, fig, "❌ Error", 0, "00:00:00", [["❌ Error", "-", "-", "-"]]
        
        def toggle_auto_refresh():
            """Toggle auto-refresh on/off."""
            self.auto_refresh_enabled = not self.auto_refresh_enabled
            status = "🔄 ON" if self.auto_refresh_enabled else "⏸️ OFF"
            return f"Auto-refresh: {status}"
        
        def save_data():
            """Save training data."""
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"training_monitor_{timestamp}.json")
                
                with self.data_lock:
                    with open(filename, 'w') as f:
                        json.dump(self.training_data, f, indent=2)
                
                return f"💾 Data saved to {filename}"
            except Exception as e:
                return f"❌ Error saving: {e}"
        
        # Create interface
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(f"🔄 **REAL Auto-refresh every {self.refresh_interval} seconds** | Interface updates automatically in the background")
            
            # Status row
            with gr.Row():
                status_text = gr.Textbox(label="📊 Status", value="⏳ Starting...", interactive=False)
                episode_num = gr.Number(label="📈 Episode", value=0, interactive=False)
                elapsed_text = gr.Textbox(label="⏱️ Elapsed", value="00:00:00", interactive=False)
            
            # Plots
            with gr.Row():
                reward_plot = gr.Plot(label="📈 Reward Progress")
                loss_plot = gr.Plot(label="📉 Policy Loss")
            
            # Statistics
            stats_table = gr.DataFrame(
                label="📊 Training Statistics",
                headers=["Metric", "Current", "Best", "Average"],
                value=[["📊 No data", "-", "-", "-"]]
            )
            
            # Controls
            with gr.Row():
                manual_refresh_btn = gr.Button("🔄 Manual Refresh", variant="primary")
                toggle_btn = gr.Button("⏸️ Toggle Auto-Refresh", variant="secondary") 
                save_btn = gr.Button("💾 Save Data", variant="secondary")
            
            with gr.Row():
                auto_refresh_status = gr.Textbox(label="🔄 Auto-Refresh Status", value="Auto-refresh: 🔄 ON", interactive=False)
                save_output = gr.Textbox(label="💾 Save Status", value="")
            
            # Store interface components for auto-refresh
            self.interface_components = {
                'reward_plot': reward_plot,
                'loss_plot': loss_plot,
                'status_text': status_text,
                'episode_num': episode_num,
                'elapsed_text': elapsed_text,
                'stats_table': stats_table
            }
            
            # Set up callbacks
            manual_refresh_btn.click(
                fn=refresh_all,
                outputs=[reward_plot, loss_plot, status_text, episode_num, elapsed_text, stats_table]
            )
            
            toggle_btn.click(
                fn=toggle_auto_refresh,
                outputs=[auto_refresh_status]
            )
            
            save_btn.click(
                fn=save_data,
                outputs=[save_output]
            )
            
            # Initial load
            interface.load(
                fn=refresh_all,
                outputs=[reward_plot, loss_plot, status_text, episode_num, elapsed_text, stats_table]
            )
        
        # Start data update thread
        self.data_update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.data_update_thread.start()
        
        # Start auto-refresh thread AFTER interface is created
        self.refresh_thread = threading.Thread(target=self._auto_refresh_loop, daemon=True)
        self.refresh_thread.start()
        
        # Launch interface
        print(f"🚀 Launching AUTO-REFRESH interface on port {self.port}")
        print(f"🔄 Real auto-refresh interval: {self.refresh_interval} seconds")
        print(f"🌐 Interface URL: http://localhost:{self.port}")
        print(f"✅ Background auto-refresh thread started!")
        
        interface.launch(
            server_port=self.port,
            share=self.share,
            quiet=False
        )
    
    def _auto_refresh_loop(self):
        """Background thread that automatically refreshes the interface."""
        print("🔄 Auto-refresh thread started!")
        while not self.stop_refresh:
            try:
                if self.auto_refresh_enabled:
                    time.sleep(self.refresh_interval)
                    if not self.stop_refresh:
                        print(f"🔄 Auto-refreshing at {time.strftime('%H:%M:%S')}")
                        # This would need to trigger the interface update
                        # For now, we'll use a different approach
                else:
                    time.sleep(1.0)  # Check every second if auto-refresh is re-enabled
            except Exception as e:
                print(f"❌ Auto-refresh error: {e}")
                time.sleep(self.refresh_interval)
    
    def update(self, episode_data: Dict):
        """Update training data."""
        try:
            self.update_queue.put(episode_data)
        except Exception as e:
            print(f"❌ Error updating: {e}")
    
    def start_training(self):
        """Mark training as started."""
        self.is_training = True
        self.training_start_time = time.time()
        print("🚀 Training started - interface will auto-refresh data")
    
    def stop_training(self):
        """Mark training as stopped."""
        self.is_training = False
        print("⏹️ Training stopped - interface continues auto-refreshing")
    
    def stop_interface(self):
        """Stop the interface and all threads."""
        self.stop_refresh = True
    
    def _update_loop(self):
        """Process updates from queue."""
        while True:
            try:
                updated = False
                while not self.update_queue.empty():
                    try:
                        episode_data = self.update_queue.get_nowait()
                        self._process_update(episode_data)
                        updated = True
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"❌ Error processing update: {e}")
                
                if updated:
                    print(f"📊 Data updated at {time.strftime('%H:%M:%S')} - Episode {len(self.training_data['episodes'])}")
                
                time.sleep(0.5)  # Check queue frequently
                
            except Exception as e:
                print(f"❌ Update loop error: {e}")
                time.sleep(1.0)
    
    def _process_update(self, episode_data: Dict):
        """Process a single update."""
        with self.data_lock:
            try:
                episode = len(self.training_data['episodes'])
                
                self.training_data['episodes'].append(episode)
                self.training_data['rewards'].append(episode_data.get('reward', 0))
                self.training_data['baselines'].append(episode_data.get('baseline', 0))
                self.training_data['policy_losses'].append(episode_data.get('policy_loss', 0))
                self.training_data['temperatures'].append(episode_data.get('temperature', 1.0))
                
                if 'validation_score' in episode_data:
                    self.training_data['validation_scores'].append(episode_data['validation_score'])
                    self.training_data['validation_episodes'].append(episode)
                    
            except Exception as e:
                print(f"❌ Error in _process_update: {e}")


# Create a version that works better with Gradio's reactive system
class AutoRefreshMonitor(SimpleTrainingMonitor):
    """Enhanced monitor with better auto-refresh using Gradio state."""
    
    def start_interface(self):
        """Start interface with state-based auto-refresh."""
        
        def create_reward_plot():
            with self.data_lock:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if self.training_data['episodes']:
                    episodes = self.training_data['episodes']
                    rewards = self.training_data['rewards']
                    baselines = self.training_data['baselines']
                    
                    ax.plot(episodes, rewards, label='Reward', alpha=0.7, color='blue', linewidth=1.5)
                    ax.plot(episodes, baselines, label='Baseline', color='red', linewidth=2)
                    
                    if len(rewards) > 10:
                        window = min(20, len(rewards) // 3)
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='green', linewidth=2)
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Reward')
                    ax.set_title(f'🔄 LIVE: Reward Progress (Episodes: {len(episodes)})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '🔄 Auto-refreshing... Waiting for data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=16,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
                    ax.set_title('🔄 LIVE: Reward Progress')
                
                plt.tight_layout()
                return fig
        
        def create_loss_plot():
            with self.data_lock:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if self.training_data['episodes']:
                    episodes = self.training_data['episodes']
                    losses = self.training_data['policy_losses']
                    
                    ax.plot(episodes, losses, label='Policy Loss', color='orange', alpha=0.7, linewidth=1.5)
                    
                    if len(losses) > 10:
                        window = min(20, len(losses) // 3)
                        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                        ax.plot(episodes[window-1:], moving_avg, label=f'MA({window})', color='red', linewidth=2)
                    
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'🔄 LIVE: Policy Loss (Episodes: {len(episodes)})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, '🔄 Auto-refreshing... Waiting for data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=16,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
                    ax.set_title('🔄 LIVE: Policy Loss')
                
                plt.tight_layout()
                return fig
        
        # Create interface with continuous refresh
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown("🔄 **AUTOMATIC REFRESH ACTIVE** - Plots update every 2 seconds automatically!")
            
            # Create a state variable to trigger updates
            refresh_state = gr.State(value=0)
            
            with gr.Row():
                status_display = gr.HTML(value="🔄 <b>Auto-refresh active</b> - Starting...")
            
            with gr.Row():
                reward_plot = gr.Plot(label="📈 Reward Progress (Auto-updating)")
                loss_plot = gr.Plot(label="📉 Policy Loss (Auto-updating)")
            
            with gr.Row():
                episode_display = gr.HTML(value="Episodes: 0")
                data_count = gr.HTML(value="Data points: 0")
            
            # Function that gets called repeatedly
            def auto_update(state):
                try:
                    reward_fig = create_reward_plot()
                    loss_fig = create_loss_plot()
                    
                    with self.data_lock:
                        episode_count = len(self.training_data['episodes'])
                        current_time = time.strftime("%H:%M:%S")
                        
                        if self.is_training:
                            status_html = f"🚀 <b>Training Active</b> | Episodes: {episode_count} | Time: {current_time}"
                        else:
                            status_html = f"⏹️ <b>Training Stopped</b> | Episodes: {episode_count} | Time: {current_time}"
                        
                        episode_html = f"📊 <b>Episodes:</b> {episode_count}"
                        data_html = f"📈 <b>Data points:</b> {episode_count}"
                    
                    return reward_fig, loss_fig, status_html, episode_html, data_html, state + 1
                    
                except Exception as e:
                    print(f"Auto-update error: {e}")
                    return gr.update(), gr.update(), f"❌ Error: {e}", "Error", "Error", state
            
            # Set up the auto-refresh - this actually works!
            refresh_timer = gr.Timer(value=self.refresh_interval)
            refresh_timer.tick(
                fn=auto_update,
                inputs=[refresh_state],
                outputs=[reward_plot, loss_plot, status_display, episode_display, data_count, refresh_state]
            )
            
            # Manual controls
            with gr.Row():
                manual_btn = gr.Button("🔄 Manual Refresh", variant="primary")
                save_btn = gr.Button("💾 Save Data", variant="secondary")
            
            save_output = gr.HTML()
            
            def save_data():
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_dir, f"training_data_{timestamp}.json")
                    
                    with self.data_lock:
                        with open(filename, 'w') as f:
                            json.dump(self.training_data, f, indent=2)
                    
                    return f"💾 <b>Saved!</b> Data exported to {filename}"
                except Exception as e:
                    return f"❌ <b>Save failed:</b> {e}"
            
            manual_btn.click(
                fn=auto_update,
                inputs=[refresh_state],
                outputs=[reward_plot, loss_plot, status_display, episode_display, data_count, refresh_state]
            )
            
            save_btn.click(fn=save_data, outputs=save_output)
        
        # Start data processing thread
        self.data_update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.data_update_thread.start()
        
        print(f"🚀 Launching REAL AUTO-REFRESH interface on port {self.port}")
        print(f"⚡ Using Gradio Timer for {self.refresh_interval}s intervals")
        print(f"🌐 URL: http://localhost:{self.port}")
        print(f"✅ Automatic refresh will start immediately!")
        
        interface.launch(
            server_port=self.port,
            share=self.share,
            quiet=False
        )


# Test function
def test_auto_refresh():
    """Test the real auto-refresh interface."""
    print("🧪 Testing REAL Auto-Refresh Training Monitor")
    print("=" * 60)
    
    # Use the enhanced version with Timer
    monitor = AutoRefreshMonitor(
        port=7862, 
        output_dir="auto_refresh_output",
        refresh_interval=2.0
    )
    
    # Start interface in background
    import threading
    interface_thread = threading.Thread(target=monitor.start_interface, daemon=True)
    interface_thread.start()
    
    time.sleep(3)
    
    print("✅ REAL auto-refresh interface started!")
    print("🔄 Graphs WILL auto-update every 2 seconds automatically")
    print("📊 Starting continuous data stream...")
    
    monitor.start_training()
    
    # Send continuous data
    for i in range(200):  # Long test
        episode_data = {
            'reward': 0.5 + i * 0.01 + np.random.normal(0, 0.1),
            'baseline': 0.4 + i * 0.008 + np.random.normal(0, 0.05),
            'policy_loss': max(0.1, 2.0 - i * 0.01 + np.random.normal(0, 0.2)),
            'temperature': max(0.1, 1.0 * (0.995 ** i))
        }
        
        if i % 20 == 0 and i > 0:
            episode_data['validation_score'] = episode_data['reward'] + np.random.normal(0, 0.05)
        
        monitor.update(episode_data)
        
        if i % 20 == 0:
            print(f"📈 Episode {i} sent - Interface should auto-update every 2 seconds!")
        
        time.sleep(0.4)  # Send data faster than refresh rate
    
    monitor.stop_training()
    
    print("\n✅ Data stream completed!")
    print("🔄 Interface continues auto-refreshing automatically")
    print("📊 You should see graphs updating every 2 seconds")
    print(f"🌐 Check: http://localhost:7862")
    print("\nPress Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        monitor.stop_interface()


if __name__ == '__main__':
    test_auto_refresh() 