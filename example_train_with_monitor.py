#!/usr/bin/env python3
"""
Simple example of using the Gradio monitoring interface during training.

This script demonstrates how to run training with real-time loss monitoring
using a minimal configuration.
"""

import torch
import numpy as np
import time
from reformulate_lp.training.gradio_monitor import TrainingMonitor

def demo_training_with_monitor():
    """Demonstrate the monitoring interface with simulated training data."""
    
    # Create and start the monitor
    monitor = TrainingMonitor(
        title="Demo LP Reformulation Training Monitor",
        port=7860,
        share=False
    )
    
    print("Starting monitoring interface...")
    print("Visit http://localhost:7860 to view the training progress")
    
    # Start the interface in a separate thread
    import threading
    monitor_thread = threading.Thread(target=monitor.start_interface, daemon=True)
    monitor_thread.start()
    
    # Give the interface time to start
    time.sleep(3)
    
    # Start training simulation
    monitor.start_training()
    
    print("Simulating training with fake data...")
    print("The interface will update in real-time as training progresses")
    
    # Simulate training episodes
    num_episodes = 1000
    for episode in range(num_episodes):
        # Simulate training metrics with some realistic patterns
        reward = np.random.normal(0.5 + episode * 0.001, 0.1)
        baseline = reward * 0.8 + np.random.normal(0, 0.05)
        policy_loss = max(0.1, 2.0 - episode * 0.001 + np.random.normal(0, 0.2))
        temperature = max(0.1, 1.0 * (0.995 ** episode))
        
        # Simulate solving times
        original_time = np.random.exponential(2.0) + 1.0
        reformulated_time = original_time * (0.9 - episode * 0.0005) + np.random.normal(0, 0.1)
        improvement_ratio = original_time / max(reformulated_time, 0.1)
        
        episode_data = {
            'episode': episode,
            'reward': reward,
            'baseline': baseline,
            'policy_loss': policy_loss,
            'baseline_loss': 0.0,
            'temperature': temperature,
            'solving_time_original': original_time,
            'solving_time_reformulated': max(0.1, reformulated_time),
            'improvement_ratio': improvement_ratio
        }
        
        # Add validation data occasionally
        if episode % 50 == 0 and episode > 0:
            val_score = reward + np.random.normal(0, 0.05)
            episode_data['validation_score'] = val_score
        
        # Update monitor
        monitor.update(episode_data)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={reward:.3f}, Loss={policy_loss:.3f}, "
                  f"Improvement={improvement_ratio:.2f}x")
        
        # Simulate training time delay
        time.sleep(0.1)  # Adjust this to control simulation speed
    
    # Mark training as complete
    monitor.stop_training()
    
    print("\nTraining simulation completed!")
    print("The monitoring interface remains active.")
    print("You can:")
    print("- View the training plots and statistics")
    print("- Export the data and plots using the buttons")
    print("- Save the current data")
    print("\nPress Ctrl+C to exit")
    
    # Keep the program running to maintain the interface
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    print("=" * 60)
    print("DEMO: LP Reformulation Training Monitor")
    print("=" * 60)
    print("This demo simulates training with fake data to show the interface.")
    print("In real training, use train_with_monitor.py instead.")
    print("=" * 60)
    
    demo_training_with_monitor() 