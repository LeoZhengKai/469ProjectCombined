"""
Telemetry and performance monitoring for DSPy evaluation framework.

This module provides comprehensive tracking of:
- Model inference performance (latency, tokens, cost)
- Evaluation metrics and quality scores
- Iteration efficiency and improvement rates
- Resource utilization and optimization progress

All telemetry data is structured and can be exported to various backends
(MLflow, Weights & Biases, local files, etc.).
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque

from .logging import performance_logger


@dataclass
class InferenceMetrics:
    """Metrics for a single model inference."""
    model_name: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: str
    sample_id: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class EvaluationMetrics:
    """Metrics for evaluation results."""
    metric_name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    timestamp: str = None
    sample_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold


@dataclass
class IterationMetrics:
    """Metrics for iteration efficiency."""
    iteration_type: str  # "refine", "optimize", "bootstrap", etc.
    iteration_count: int
    success: bool
    timestamp: str = None
    sample_id: Optional[str] = None
    improvement_delta: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


class TelemetryCollector:
    """
    Collects and aggregates telemetry data for experiments.
    
    Provides methods to:
    - Track model inference performance
    - Record evaluation metrics
    - Monitor iteration efficiency
    - Generate aggregated reports
    - Export data to various backends
    """
    
    def __init__(self, run_id: str):
        """
        Initialize telemetry collector for a specific run.
        
        Args:
            run_id: Unique identifier for the current run
        """
        self.run_id = run_id
        self.inference_metrics: List[InferenceMetrics] = []
        self.evaluation_metrics: List[EvaluationMetrics] = []
        self.iteration_metrics: List[IterationMetrics] = []
        
        # Aggregated statistics
        self._latency_history = deque(maxlen=1000)  # Keep last 1000 latencies
        self._cost_history = deque(maxlen=1000)    # Keep last 1000 costs
        self._iteration_counts = defaultdict(list)  # Track iterations by type
    
    def record_inference(
        self,
        model_name: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        sample_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record model inference metrics.
        
        Args:
            model_name: Name of the model used
            latency_ms: Inference latency in milliseconds
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            cost_usd: Cost in USD
            sample_id: Optional sample identifier
            error: Optional error message if inference failed
        """
        metrics = InferenceMetrics(
            model_name=model_name,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            sample_id=sample_id,
            error=error
        )
        
        self.inference_metrics.append(metrics)
        self._latency_history.append(latency_ms)
        self._cost_history.append(cost_usd)
        
        # Log to performance logger
        performance_logger.log_inference(
            model_name=model_name,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            sample_id=sample_id
        )
    
    def record_evaluation(
        self,
        metric_name: str,
        value: float,
        threshold: Optional[float] = None,
        sample_id: Optional[str] = None
    ) -> None:
        """
        Record evaluation metric results.
        
        Args:
            metric_name: Name of the metric (e.g., "quality", "factuality")
            value: Metric value
            threshold: Optional threshold for this metric
            sample_id: Optional sample identifier
        """
        metrics = EvaluationMetrics(
            metric_name=metric_name,
            value=value,
            threshold=threshold,
            sample_id=sample_id
        )
        
        self.evaluation_metrics.append(metrics)
        
        # Log to performance logger
        performance_logger.log_evaluation(
            metric_name=metric_name,
            value=value,
            sample_id=sample_id,
            threshold=threshold
        )
    
    def record_iteration(
        self,
        iteration_type: str,
        iteration_count: int,
        success: bool = True,
        sample_id: Optional[str] = None,
        improvement_delta: Optional[float] = None
    ) -> None:
        """
        Record iteration efficiency metrics.
        
        Args:
            iteration_type: Type of iteration (refine, optimize, etc.)
            iteration_count: Number of iterations
            success: Whether the iteration was successful
            sample_id: Optional sample identifier
            improvement_delta: Optional improvement achieved
        """
        metrics = IterationMetrics(
            iteration_type=iteration_type,
            iteration_count=iteration_count,
            success=success,
            sample_id=sample_id,
            improvement_delta=improvement_delta
        )
        
        self.iteration_metrics.append(metrics)
        self._iteration_counts[iteration_type].append(iteration_count)
        
        # Log to performance logger
        performance_logger.log_iteration(
            iteration_type=iteration_type,
            iteration_count=iteration_count,
            sample_id=sample_id,
            success=success
        )
    
    @contextmanager
    def time_inference(self, model_name: str, sample_id: Optional[str] = None):
        """
        Context manager to time model inference.
        
        Usage:
            with telemetry.time_inference("gpt-4o-mini", sample_id="123") as timer:
                result = model.generate(input_text)
                timer.record(tokens_in=100, tokens_out=50, cost_usd=0.01)
        
        Args:
            model_name: Name of the model
            sample_id: Optional sample identifier
        """
        start_time = time.time()
        timer = InferenceTimer(model_name, sample_id, start_time, self)
        try:
            yield timer
        except Exception as e:
            # Record error if inference fails
            latency_ms = (time.time() - start_time) * 1000
            self.record_inference(
                model_name=model_name,
                latency_ms=latency_ms,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                sample_id=sample_id,
                error=str(e)
            )
            raise
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for the current run.
        
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.inference_metrics:
            return {}
        
        # Calculate latency statistics
        latencies = [m.latency_ms for m in self.inference_metrics if m.error is None]
        latency_p50 = self._percentile(latencies, 50) if latencies else 0.0
        latency_p95 = self._percentile(latencies, 95) if latencies else 0.0
        
        # Calculate cost statistics
        costs = [m.cost_usd for m in self.inference_metrics if m.error is None]
        total_cost = sum(costs)
        avg_cost_per_sample = total_cost / len(costs) if costs else 0.0
        
        # Calculate token statistics
        total_tokens_in = sum(m.tokens_in for m in self.inference_metrics if m.error is None)
        total_tokens_out = sum(m.tokens_out for m in self.inference_metrics if m.error is None)
        
        # Calculate evaluation metrics
        eval_metrics = {}
        for metric in self.evaluation_metrics:
            metric_name = metric.metric_name
            if metric_name not in eval_metrics:
                eval_metrics[metric_name] = []
            eval_metrics[metric_name].append(metric.value)
        
        # Calculate mean evaluation metrics
        mean_eval_metrics = {}
        for metric_name, values in eval_metrics.items():
            mean_eval_metrics[f"{metric_name}_mean"] = sum(values) / len(values) if values else 0.0
        
        # Calculate iteration efficiency
        iteration_efficiency = {}
        for iteration_type, counts in self._iteration_counts.items():
            if counts:
                iteration_efficiency[f"{iteration_type}_mean"] = sum(counts) / len(counts)
        
        return {
            "run_id": self.run_id,
            "total_samples": len(self.inference_metrics),
            "successful_samples": len([m for m in self.inference_metrics if m.error is None]),
            "failed_samples": len([m for m in self.inference_metrics if m.error is not None]),
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "total_cost_usd": total_cost,
            "cost_per_sample_usd": avg_cost_per_sample,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            **mean_eval_metrics,
            **iteration_efficiency,
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list of numbers."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
    
    def export_to_json(self, file_path: Path) -> None:
        """
        Export all telemetry data to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        data = {
            "run_id": self.run_id,
            "inference_metrics": [asdict(m) for m in self.inference_metrics],
            "evaluation_metrics": [asdict(m) for m in self.evaluation_metrics],
            "iteration_metrics": [asdict(m) for m in self.iteration_metrics],
            "aggregated_metrics": self.get_aggregated_metrics(),
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


class InferenceTimer:
    """Helper class for timing model inference."""
    
    def __init__(self, model_name: str, sample_id: Optional[str], start_time: float, collector: TelemetryCollector):
        self.model_name = model_name
        self.sample_id = sample_id
        self.start_time = start_time
        self.collector = collector
    
    def record(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        """
        Record inference metrics.
        
        Args:
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            cost_usd: Cost in USD
        """
        latency_ms = (time.time() - self.start_time) * 1000
        
        self.collector.record_inference(
            model_name=self.model_name,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            sample_id=self.sample_id
        )


# Global telemetry collector (will be initialized per run)
_telemetry_collectors: Dict[str, TelemetryCollector] = {}


def get_telemetry_collector(run_id: str) -> TelemetryCollector:
    """
    Get or create a telemetry collector for a run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        TelemetryCollector instance
    """
    if run_id not in _telemetry_collectors:
        _telemetry_collectors[run_id] = TelemetryCollector(run_id)
    
    return _telemetry_collectors[run_id]


def clear_telemetry_collector(run_id: str) -> None:
    """
    Clear telemetry data for a run.
    
    Args:
        run_id: Run identifier
    """
    if run_id in _telemetry_collectors:
        del _telemetry_collectors[run_id]
