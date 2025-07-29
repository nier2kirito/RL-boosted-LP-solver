#!/usr/bin/env python3
"""
Test script for the Gradio monitoring interface.

This script tests the interface with a simple data stream to ensure it works correctly.
"""

import time
import numpy as np
from reformulate_lp.training.gradio_monitor import TrainingMonitor


def test_gradio_interface():
    """Test the Gradio interface with a simple data stream."""
    
    print("Testing Gradio Training Monitor Interface")
    print("=" * 50)
    
    # Create monitor
    monitor = TrainingMonitor(
        title="Test Training Monitor",
        port=7861,  # Use different port to avoid conflicts
        share=False,
        output_dir="test_monitor_output"
    )
    
    print("Starting interface...")
    print("Visit http://localhost:7861 to view the interface")
    
    # Start interface in background
    import threading
    interface_thread = threading.Thread(target=monitor.start_interface, daemon=True)
    interface_thread.start()
    
    # Wait for interface to start
    time.sleep(3)
    
    print("Interface started successfully!")
    print("Starting test data stream...")
    
    # Mark training as started
    monitor.start_training()
    
    # Send test data
    for i in range(50):
        # Generate test data
        episode_data = {
            'episode': i,
            'reward': 0.5 + i * 0.01 + np.random.normal(0, 0.1),
            'baseline': 0.4 + i * 0.008 + np.random.normal(0, 0.05),
            'policy_loss': max(0.1, 2.0 - i * 0.02 + np.random.normal(0, 0.2)),
            'baseline_loss': 0.0,
            'temperature': max(0.1, 1.0 * (0.99 ** i)),
            'solving_time_original': np.random.exponential(1.5) + 0.5,
            'solving_time_reformulated': np.random.exponential(1.0) + 0.3,
            'improvement_ratio': 1.0 + np.random.normal(0.2, 0.1)
        }
        
        # Add validation data occasionally
        if i % 10 == 0 and i > 0:
            episode_data['validation_score'] = episode_data['reward'] + np.random.normal(0, 0.05)
        
        # Update monitor
        monitor.update(episode_data)
        
        print(f"Sent episode {i} data")
        time.sleep(0.5)  # Send data every 0.5 seconds
    
    # Mark training as stopped
    monitor.stop_training()
    
    print("\nTest data stream completed!")
    print("The interface should now show:")
    print("- Training status: 'Training Stopped'") 
    print("- Episode count: 50")
    print("- All plots with data")
    print("- Statistics table with values")
    print("\nYou can test the buttons:")
    print("- Refresh Now: Updates all plots")
    print("- Save Data: Saves training data to JSON")
    print("- Reset: Clears all data")
    print("- Export All: Exports data and plots")
    
    print(f"\nInterface running at: http://localhost:7861")
    print("Press Ctrl+C to exit when done testing")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down test...")


if __name__ == '__main__':
    test_gradio_interface() 