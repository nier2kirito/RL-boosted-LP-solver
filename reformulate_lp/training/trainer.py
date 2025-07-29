"""
Simple trainer wrapper for compatibility.

This provides a unified interface for different training methods.
"""

from .reinforcement import REINFORCETrainer

# For now, just alias the REINFORCE trainer as the main trainer
Trainer = REINFORCETrainer 