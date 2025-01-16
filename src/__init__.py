# src/__init__.py

"""
Package-wide initialization for the custom StableMax + Orthogonal Optimizer project.

Exposes:
    - StableMax, LogStableMax (from stablemax.py)
    - with_orthogonal_gradient (from orthogonal_optimizer.py)
    - MODEL_CONFIG_MAPPING, CustomTrainingArguments, CustomTrainer (from custom_trainer.py)
"""

__version__ = "0.1.0"

from .stablemax import StableMax, LogStableMax
from .orthogonal_optimizer import with_orthogonal_gradient
from .custom_trainer import (
    MODEL_CONFIG_MAPPING,
    CustomTrainingArguments,
    CustomTrainer
)

__all__ = [
    "StableMax",
    "LogStableMax",
    "with_orthogonal_gradient",
    "MODEL_CONFIG_MAPPING",
    "CustomTrainingArguments",
    "CustomTrainer"
]
