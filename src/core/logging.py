"""
Structured logging configuration for DSPy evaluation framework.

This module provides JSON-formatted logging that's easy to parse and analyze.
Logs include structured fields for:
- Run IDs and experiment tracking
- Performance metrics (latency, tokens, cost)
- Quality scores and evaluation results
- Error tracking and debugging information
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context variables for tracking run information
run_id_var: ContextVar[Optional[str]] = ContextVar('run_id', default=None)
experiment_name_var: ContextVar[Optional[str]] = ContextVar('experiment_name', default=None)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log records as JSON with consistent structure:
    {
        "timestamp": "2024-01-01T12:00:00Z",
        "level": "INFO",
        "logger": "module.name",
        "message": "Log message",
        "run_id": "abc123",
        "experiment": "sharktank-eval",
        "extra_fields": {...}
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add context variables if available
        run_id = run_id_var.get()
        if run_id:
            log_entry["run_id"] = run_id
            
        experiment_name = experiment_name_var.get()
        if experiment_name:
            log_entry["experiment"] = experiment_name
        
        # Add extra fields from the record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """
    Specialized logger for performance metrics.
    
    Provides convenient methods for logging:
    - Model inference metrics (latency, tokens, cost)
    - Evaluation results (quality scores, accuracy)
    - Iteration counts and efficiency metrics
    """
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
    
    def log_inference(
        self,
        model_name: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        sample_id: Optional[str] = None
    ) -> None:
        """
        Log model inference metrics.
        
        Args:
            model_name: Name of the model used
            latency_ms: Inference latency in milliseconds
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            cost_usd: Cost in USD
            sample_id: Optional sample identifier
        """
        self.logger.info(
            "Model inference completed",
            extra={
                "extra_fields": {
                    "event_type": "inference",
                    "model_name": model_name,
                    "latency_ms": latency_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": cost_usd,
                    "sample_id": sample_id,
                }
            }
        )
    
    def log_evaluation(
        self,
        metric_name: str,
        value: float,
        sample_id: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> None:
        """
        Log evaluation metric results.
        
        Args:
            metric_name: Name of the metric (e.g., "quality", "factuality")
            value: Metric value
            sample_id: Optional sample identifier
            threshold: Optional threshold for this metric
        """
        extra_fields = {
            "event_type": "evaluation",
            "metric_name": metric_name,
            "value": value,
            "sample_id": sample_id,
        }
        
        if threshold is not None:
            extra_fields["threshold"] = threshold
            extra_fields["passed"] = value >= threshold
        
        self.logger.info(
            f"Evaluation metric: {metric_name} = {value}",
            extra={"extra_fields": extra_fields}
        )
    
    def log_iteration(
        self,
        iteration_type: str,
        iteration_count: int,
        sample_id: Optional[str] = None,
        success: bool = True
    ) -> None:
        """
        Log iteration information (refinement, optimization, etc.).
        
        Args:
            iteration_type: Type of iteration (refine, optimize, etc.)
            iteration_count: Number of iterations
            sample_id: Optional sample identifier
            success: Whether the iteration was successful
        """
        self.logger.info(
            f"{iteration_type.title()} iterations: {iteration_count}",
            extra={
                "extra_fields": {
                    "event_type": "iteration",
                    "iteration_type": iteration_type,
                    "iteration_count": iteration_count,
                    "sample_id": sample_id,
                    "success": success,
                }
            }
        )


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[Path] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format ("json" or "text")
        log_file: Optional file to write logs to
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def set_run_context(run_id: str, experiment_name: str) -> None:
    """
    Set context variables for the current run.
    
    Args:
        run_id: Unique identifier for the current run
        experiment_name: Name of the current experiment
    """
    run_id_var.set(run_id)
    experiment_name_var.set(experiment_name)


def clear_run_context() -> None:
    """Clear run context variables."""
    run_id_var.set(None)
    experiment_name_var.set(None)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Performance logger instance
performance_logger = PerformanceLogger()
