"""
DSPy evaluation components for the framework.

This module provides evaluation utilities including:
- Metrics calculation (MAE, accuracy, F1, etc.)
- LLM judges for quality assessment
- Fact-checking helpers
- Schema validation for results

All evaluation components are designed to work with
the artifact system and telemetry collection.
"""

from .metrics import MetricsCalculator
from .judges import LLMJudge, QualityJudge, SafetyJudge
from .factcheck import FactChecker
from .schema import EvaluationSchema, validate_results

__all__ = [
    "MetricsCalculator",
    "LLMJudge",
    "QualityJudge", 
    "SafetyJudge",
    "FactChecker",
    "EvaluationSchema",
    "validate_results",
]
