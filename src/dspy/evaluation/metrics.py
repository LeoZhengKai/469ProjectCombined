"""
Metrics calculation for DSPy evaluation framework.

This module provides comprehensive metrics calculation including:
- Mean Absolute Error (MAE)
- Within ±1 accuracy
- Standard accuracy and F1 scores
- Factuality hit rates
- Custom project-specific metrics

All metrics are designed to work with the artifact system
and provide consistent evaluation across different projects.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    metric_name: str
    value: float
    sample_count: int
    metadata: Optional[Dict[str, Any]] = None


class MetricsCalculator:
    """
    Comprehensive metrics calculator for evaluation results.
    
    Provides methods to calculate various metrics including:
    - Regression metrics (MAE, RMSE, etc.)
    - Classification metrics (accuracy, F1, etc.)
    - Custom project-specific metrics
    - Aggregated statistics
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metric_functions = {
            "mae": self._calculate_mae,
            "rmse": self._calculate_rmse,
            "accuracy": self._calculate_accuracy,
            "f1": self._calculate_f1,
            "within_1": self._calculate_within_1,
            "factuality_hit_rate": self._calculate_factuality_hit_rate,
        }
    
    def calculate_metric(
        self,
        metric_name: str,
        predictions: List[Union[float, int, str]],
        truths: List[Union[float, int, str]],
        **kwargs
    ) -> MetricResult:
        """
        Calculate a specific metric.
        
        Args:
            metric_name: Name of the metric to calculate
            predictions: List of predicted values
            truths: List of ground truth values
            **kwargs: Additional parameters for the metric
            
        Returns:
            MetricResult object
        """
        if metric_name not in self.metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        try:
            value = self.metric_functions[metric_name](predictions, truths, **kwargs)
            return MetricResult(
                metric_name=metric_name,
                value=value,
                sample_count=len(predictions)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to calculate {metric_name}: {e}")
    
    def calculate_all_metrics(
        self,
        predictions: List[Union[float, int, str]],
        truths: List[Union[float, int, str]],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculate multiple metrics at once.
        
        Args:
            predictions: List of predicted values
            truths: List of ground truth values
            metric_names: List of metrics to calculate (all if None)
            
        Returns:
            Dictionary mapping metric names to results
        """
        if metric_names is None:
            metric_names = list(self.metric_functions.keys())
        
        results = {}
        for metric_name in metric_names:
            try:
                results[metric_name] = self.calculate_metric(
                    metric_name, predictions, truths
                )
            except Exception as e:
                # Log error but continue with other metrics
                print(f"Warning: Failed to calculate {metric_name}: {e}")
        
        return results
    
    def _calculate_mae(
        self,
        predictions: List[Union[float, int]],
        truths: List[Union[float, int]],
        **kwargs
    ) -> float:
        """Calculate Mean Absolute Error."""
        if len(predictions) != len(truths):
            raise ValueError("Predictions and truths must have the same length")
        
        predictions = np.array(predictions, dtype=float)
        truths = np.array(truths, dtype=float)
        
        return float(np.mean(np.abs(predictions - truths)))
    
    def _calculate_rmse(
        self,
        predictions: List[Union[float, int]],
        truths: List[Union[float, int]],
        **kwargs
    ) -> float:
        """Calculate Root Mean Square Error."""
        if len(predictions) != len(truths):
            raise ValueError("Predictions and truths must have the same length")
        
        predictions = np.array(predictions, dtype=float)
        truths = np.array(truths, dtype=float)
        
        return float(np.sqrt(np.mean((predictions - truths) ** 2)))
    
    def _calculate_accuracy(
        self,
        predictions: List[Union[str, int]],
        truths: List[Union[str, int]],
        **kwargs
    ) -> float:
        """Calculate accuracy for classification tasks."""
        if len(predictions) != len(truths):
            raise ValueError("Predictions and truths must have the same length")
        
        correct = sum(1 for p, t in zip(predictions, truths) if p == t)
        return correct / len(predictions)
    
    def _calculate_f1(
        self,
        predictions: List[Union[str, int]],
        truths: List[Union[str, int]],
        **kwargs
    ) -> float:
        """Calculate F1 score for binary classification."""
        if len(predictions) != len(truths):
            raise ValueError("Predictions and truths must have the same length")
        
        # Convert to binary (assuming 1 is positive class)
        pred_binary = [1 if p == 1 else 0 for p in predictions]
        truth_binary = [1 if t == 1 else 0 for t in truths]
        
        tp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(pred_binary, truth_binary) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def _calculate_within_1(
        self,
        predictions: List[Union[float, int]],
        truths: List[Union[float, int]],
        **kwargs
    ) -> float:
        """Calculate within ±1 accuracy for regression tasks."""
        if len(predictions) != len(truths):
            raise ValueError("Predictions and truths must have the same length")
        
        predictions = np.array(predictions, dtype=float)
        truths = np.array(truths, dtype=float)
        
        within_1 = np.abs(predictions - truths) <= 1.0
        return float(np.mean(within_1))
    
    def _calculate_factuality_hit_rate(
        self,
        predictions: List[str],
        truths: List[str],
        fact_checker: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        Calculate factuality hit rate using a fact checker.
        
        Args:
            predictions: List of predicted texts
            truths: List of ground truth texts
            fact_checker: Fact checker instance
            **kwargs: Additional parameters
            
        Returns:
            Factuality hit rate (0-1)
        """
        if fact_checker is None:
            # Fallback to simple string matching
            return self._calculate_accuracy(predictions, truths)
        
        try:
            factual_count = 0
            for pred, truth in zip(predictions, truths):
                # Use fact checker to determine if prediction is factual
                is_factual = fact_checker.check_factuality(pred, truth)
                if is_factual:
                    factual_count += 1
            
            return factual_count / len(predictions)
        except Exception:
            # Fallback to simple string matching
            return self._calculate_accuracy(predictions, truths)
    
    def calculate_aggregated_metrics(
        self,
        results: Dict[str, List[Union[float, int, str]]]
    ) -> Dict[str, float]:
        """
        Calculate aggregated metrics from multiple runs.
        
        Args:
            results: Dictionary mapping metric names to lists of values
            
        Returns:
            Dictionary with aggregated statistics
        """
        aggregated = {}
        
        for metric_name, values in results.items():
            if not values:
                continue
            
            values = np.array(values, dtype=float)
            
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))
            aggregated[f"{metric_name}_min"] = float(np.min(values))
            aggregated[f"{metric_name}_max"] = float(np.max(values))
            aggregated[f"{metric_name}_p50"] = float(np.percentile(values, 50))
            aggregated[f"{metric_name}_p95"] = float(np.percentile(values, 95))
        
        return aggregated
    
    def add_custom_metric(
        self,
        name: str,
        metric_function: callable
    ) -> None:
        """
        Add a custom metric function.
        
        Args:
            name: Name of the metric
            metric_function: Function that takes (predictions, truths, **kwargs)
        """
        self.metric_functions[name] = metric_function
    
    def list_available_metrics(self) -> List[str]:
        """
        List all available metrics.
        
        Returns:
            List of metric names
        """
        return list(self.metric_functions.keys())
    
    def validate_metric_inputs(
        self,
        predictions: List[Any],
        truths: List[Any],
        metric_name: str
    ) -> bool:
        """
        Validate inputs for a specific metric.
        
        Args:
            predictions: List of predictions
            truths: List of truths
            metric_name: Name of the metric
            
        Returns:
            True if inputs are valid
        """
        if len(predictions) != len(truths):
            return False
        
        if not predictions:
            return False
        
        # Check if metric exists
        if metric_name not in self.metric_functions:
            return False
        
        return True


# Global metrics calculator instance
metrics_calculator = MetricsCalculator()
