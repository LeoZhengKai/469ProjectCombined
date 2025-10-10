"""
DSPy optimizers for the evaluation framework.

This module provides optimized configurations for common DSPy optimizers:
- BootstrapFewShot: Few-shot learning with bootstrap sampling
- MIPROv2: Multi-prompt optimization
- COPRO: Collaborative prompt optimization
- Custom optimizers for specific use cases

Optimizers are configured with sensible defaults and can be customized
via YAML configuration files.
"""

from .bootstrap import BootstrapFewShotOptimizer
from .mipro import MIPROv2Optimizer
from .copro import COPROOptimizer
from .custom import CustomOptimizer

__all__ = [
    "BootstrapFewShotOptimizer",
    "MIPROv2Optimizer", 
    "COPROOptimizer",
    "CustomOptimizer",
]
