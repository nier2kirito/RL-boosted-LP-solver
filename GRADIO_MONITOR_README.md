# Gradio Training Monitor

This document describes how to use the Gradio-based web interface for monitoring training progress in real-time.

## Overview

The Gradio Training Monitor provides a real-time web interface to visualize training metrics during the LP reformulation training process. It displays:

- **Reward Progress**: Training rewards and baselines over episodes
- **Policy Loss**: Policy loss curves with moving averages
- **Validation Performance**: Validation scores at regular intervals
- **Temperature Schedule**: Temperature decay over training
- **Solving Time Comparison**: Original vs. reformulated problem solving times
- **Improvement Ratios**: Performance improvement metrics
- **Training Statistics**: Summary statistics table

## Installation

Make sure you have the required dependencies:

```bash
pip install gradio>=4.0.0
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Using the Enhanced Training Script

The easiest way to use the monitoring interface is with the provided training script:

```bash
python train_with_monitor.py --config configs/default.yaml
```

**Command line options:**
- `--no_monitor`: Disable the monitoring interface
- `--monitor_port 7860`: Set the port for the web interface (default: 7860)
- `--monitor_share`: Create a public sharing link
- `--monitor_update_freq 10`: Update frequency in episodes (default: 10)

**Example with custom settings:**
```bash
python train_with_monitor.py \
    --config configs/default.yaml \
    --monitor_port 8080 \
    --monitor_update_freq 5 \
    --output_dir my_experiment
```

### 2. Using the Monitor in Custom Code

You can integrate the monitor into your own training code:

```python
from reformulate_lp.training.reinforcement_with_monitor import REINFORCETrainerWithMonitor
from reformulate_lp.training.gradio_monitor import TrainingMonitor

# Create trainer with monitoring
trainer = REINFORCETrainerWithMonitor(
    model=model,
    solver=solver,
    monitor_enabled=True,
    monitor_port=7860,
    monitor_update_frequency=10
)

# Train with live updates
training_history = trainer.train_with_live_updates(
    dataset=train_dataset,
    num_episodes=1000,
    validation_dataset=val_dataset,
    output_dir="outputs"
)
```

### 3. Standalone Monitor Usage

You can also use the monitor independently:

```python
from reformulate_lp.training.gradio_monitor import TrainingMonitor

# Create monitor
monitor = TrainingMonitor(
    title="My Training Monitor",
    port=7860,
    output_dir="outputs"
)

# Start interface
monitor.start_interface()
monitor.start_training()

# Update during training
for episode in range(num_episodes):
    # ... training code ...
    
    episode_data = {
        'reward': reward,
        'baseline': baseline,
        'policy_loss': policy_loss,
        'temperature': temperature,
        'validation_score': val_score  # optional
    }
    monitor.update(episode_data)

monitor.stop_training()
```

## Demo

To see the interface in action with simulated data:

```bash
python example_train_with_monitor.py
```

This will start a demo with fake training data to show how the interface works.

## Interface Features

### Real-time Plots

1. **Reward Progress**: Shows training rewards, baselines, and moving averages
2. **Policy Loss**: Policy loss with smoothing
3. **Validation Performance**: Validation scores over time
4. **Temperature Schedule**: Temperature decay visualization
5. **Solving Time Comparison**: Original vs. reformulated solving times
6. **Improvement Ratios**: Performance improvement over episodes

### Status Information

- **Training Status**: Current training state (Active/Stopped)
- **Current Episode**: Episode counter
- **Elapsed Time**: Training duration
- **Statistics Table**: Summary statistics for key metrics

### Control Buttons

- **Save Current Data**: Save training data to JSON
- **Reset Plots**: Clear all current data
- **Export Data**: Export data and plots to files
- **Refresh Now**: Manual refresh of all plots

### Auto-refresh

The interface automatically refreshes every 2 seconds to show the latest training progress.

## Configuration

### Monitor Settings

```python
monitor = TrainingMonitor(
    title="Custom Training Monitor",     # Interface title
    port=7860,                          # Web server port
    share=False,                        # Create public link
    output_dir="outputs"                # Directory for exports
)
```

### Trainer Settings

```python
trainer = REINFORCETrainerWithMonitor(
    # ... standard trainer parameters ...
    monitor_enabled=True,               # Enable monitoring
    monitor_port=7860,                  # Interface port
    monitor_share=False,                # Public sharing
    monitor_update_frequency=10         # Update every N episodes
)
```

## Output Files

The monitor can export various files to the output directory:

- `training_monitor_TIMESTAMP.json`: Raw training data
- `training_export_TIMESTAMP.json`: Complete training export
- `plot_reward_TIMESTAMP.png`: Reward progress plot
- `plot_loss_TIMESTAMP.png`: Loss progress plot
- `plot_validation_TIMESTAMP.png`: Validation plot
- `plot_temperature_TIMESTAMP.png`: Temperature schedule plot
- `plot_solving_time_TIMESTAMP.png`: Solving time comparison
- `plot_improvement_TIMESTAMP.png`: Improvement ratio plot

## Tips

1. **Port Configuration**: If port 7860 is busy, use `--monitor_port` to specify a different port
2. **Update Frequency**: Lower values (e.g., 5) give more frequent updates but may slow training slightly
3. **Public Sharing**: Use `--monitor_share` to get a public URL for remote monitoring
4. **Data Export**: Regularly export data during long training runs using the interface buttons
5. **Multiple Runs**: Each run creates a timestamped output directory to avoid conflicts

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `--monitor_port`
2. **Interface not loading**: Wait a few seconds after starting, then refresh your browser
3. **Plots not updating**: Check that training is active and update frequency is reasonable
4. **Memory issues**: For very long training runs, periodically export and reset data

### Browser Compatibility

The interface works best with modern browsers:
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

### Performance

For best performance:
- Use update frequencies of 5-20 episodes
- Export and reset data periodically for very long runs
- Close other browser tabs when monitoring intensive training

## Integration with Existing Code

To add monitoring to existing training code:

1. Replace `REINFORCETrainer` with `REINFORCETrainerWithMonitor`
2. Add monitor parameters to trainer initialization
3. Use `train_with_live_updates()` instead of `train()`
4. Optionally export monitor data after training

The monitor is designed to be minimally invasive and can be easily disabled by setting `monitor_enabled=False`. 