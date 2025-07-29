"""
Learning to Reformulate for Linear Programming

This package implements the reinforcement learning-based reformulation method
for linear programming problems as described in the paper by Li et al.
"""

from .models.reformulator import ReformulationSystem
from .data.dataset import LPDataset
from .training.trainer import Trainer
from .training.reinforcement import REINFORCETrainer, AdaptiveTemperatureSchedule
from .training.gradio_monitor import TrainingMonitor
from .training.reinforcement_with_monitor import REINFORCETrainerWithMonitor

__version__ = "1.0.0"
__author__ = "Implementation based on Li et al. (2022)"

__all__ = [
    "ReformulationSystem",
    "LPDataset", 
    "Trainer",
    "REINFORCETrainer",
    "AdaptiveTemperatureSchedule", 
    "TrainingMonitor",
    "REINFORCETrainerWithMonitor"
] 