"""
Artifact management for DSPy evaluation framework.

This module handles the creation, storage, and retrieval of experiment artifacts:
- Run configurations (config.json)
- Predictions and results (predictions.jsonl)
- Aggregated metrics (metrics.json)
- Comparison results (comparisons/*.json)

Artifacts are stored in a structured format under experiments/runs/<run_id>/
"""

import json
import jsonlines
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from uuid import uuid4

from .ids import generate_run_id, generate_compare_id


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""
    project: str
    model: str
    optimizer: Optional[str] = None
    seed: int = 42
    thresholds: Dict[str, float] = None
    num_samples: int = 100
    split: str = "test"
    created_at: str = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {"quality": 8.0, "fact": 8.5}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class Prediction:
    """Single prediction result."""
    sample_id: Union[str, int]
    input: Dict[str, Any]
    truth: Optional[Dict[str, Any]] = None
    prediction: str = ""
    metrics: Dict[str, float] = None
    perf: Dict[str, float] = None
    iters: Dict[str, int] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.perf is None:
            self.perf = {}
        if self.iters is None:
            self.iters = {}


@dataclass
class RunMetrics:
    """Aggregated metrics for a run."""
    quality_mean: float
    fact_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_sample_usd: float
    improvement_efficiency_mean: float
    total_samples: int
    successful_samples: int
    failed_samples: int
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class Comparison:
    """Comparison between multiple runs."""
    compare_id: str
    run_ids: List[str]
    comparison_type: str  # "quality", "latency", "cost", "overall"
    results: Dict[str, Any]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


class ArtifactManager:
    """
    Manages experiment artifacts (configs, predictions, metrics, comparisons).
    
    Provides methods to:
    - Create and save run artifacts
    - Load existing artifacts
    - Generate comparison reports
    - Validate artifact integrity
    """
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        """
        Initialize artifact manager.
        
        Args:
            experiments_dir: Base directory for experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.runs_dir = self.experiments_dir / "runs"
        self.comparisons_dir = self.experiments_dir / "comparisons"
        
        # Create directories if they don't exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run(
        self,
        config: RunConfig,
        run_id: Optional[str] = None
    ) -> str:
        """
        Create a new run directory and save configuration.
        
        Args:
            config: Run configuration
            run_id: Optional run ID (generated if not provided)
            
        Returns:
            Generated or provided run ID
        """
        if run_id is None:
            run_id = generate_run_id(config.project, config.model)
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = run_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2)
        
        return run_id
    
    def save_predictions(
        self,
        run_id: str,
        predictions: List[Prediction]
    ) -> None:
        """
        Save predictions to predictions.jsonl.
        
        Args:
            run_id: Run identifier
            predictions: List of prediction results
        """
        run_dir = self.runs_dir / run_id
        predictions_path = run_dir / "predictions.jsonl"
        
        with jsonlines.open(predictions_path, 'w') as writer:
            for pred in predictions:
                writer.write(asdict(pred))
    
    def load_predictions(self, run_id: str) -> List[Prediction]:
        """
        Load predictions from predictions.jsonl.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of prediction objects
        """
        run_dir = self.runs_dir / run_id
        predictions_path = run_dir / "predictions.jsonl"
        
        if not predictions_path.exists():
            return []
        
        predictions = []
        with jsonlines.open(predictions_path, 'r') as reader:
            for obj in reader:
                predictions.append(Prediction(**obj))
        
        return predictions
    
    def save_metrics(
        self,
        run_id: str,
        metrics: RunMetrics
    ) -> None:
        """
        Save aggregated metrics to metrics.json.
        
        Args:
            run_id: Run identifier
            metrics: Aggregated metrics
        """
        run_dir = self.runs_dir / run_id
        metrics_path = run_dir / "metrics.json"
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def load_metrics(self, run_id: str) -> Optional[RunMetrics]:
        """
        Load metrics from metrics.json.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Metrics object or None if not found
        """
        run_dir = self.runs_dir / run_id
        metrics_path = run_dir / "metrics.json"
        
        if not metrics_path.exists():
            return None
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return RunMetrics(**data)
    
    def load_config(self, run_id: str) -> Optional[RunConfig]:
        """
        Load run configuration from config.json.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run configuration or None if not found
        """
        run_dir = self.runs_dir / run_id
        config_path = run_dir / "config.json"
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return RunConfig(**data)
    
    def list_runs(self, project: Optional[str] = None) -> List[str]:
        """
        List all run IDs, optionally filtered by project.
        
        Args:
            project: Optional project name to filter by
            
        Returns:
            List of run IDs
        """
        run_ids = []
        
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                config = self.load_config(run_dir.name)
                if config and (project is None or config.project == project):
                    run_ids.append(run_dir.name)
        
        return sorted(run_ids)
    
    def create_comparison(
        self,
        run_ids: List[str],
        comparison_type: str,
        results: Dict[str, Any],
        compare_id: Optional[str] = None
    ) -> str:
        """
        Create a comparison between multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            comparison_type: Type of comparison (quality, latency, cost, overall)
            results: Comparison results
            compare_id: Optional comparison ID (generated if not provided)
            
        Returns:
            Comparison ID
        """
        if compare_id is None:
            compare_id = generate_compare_id(run_ids)
        
        comparison = Comparison(
            compare_id=compare_id,
            run_ids=run_ids,
            comparison_type=comparison_type,
            results=results
        )
        
        comparison_path = self.comparisons_dir / f"{compare_id}.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(comparison), f, indent=2)
        
        return compare_id
    
    def load_comparison(self, compare_id: str) -> Optional[Comparison]:
        """
        Load comparison results.
        
        Args:
            compare_id: Comparison identifier
            
        Returns:
            Comparison object or None if not found
        """
        comparison_path = self.comparisons_dir / f"{compare_id}.json"
        
        if not comparison_path.exists():
            return None
        
        with open(comparison_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Comparison(**data)
    
    def list_comparisons(self) -> List[str]:
        """
        List all comparison IDs.
        
        Returns:
            List of comparison IDs
        """
        compare_ids = []
        
        for comparison_file in self.comparisons_dir.glob("*.json"):
            compare_ids.append(comparison_file.stem)
        
        return sorted(compare_ids)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get a summary of a run including config, metrics, and basic stats.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run summary
        """
        config = self.load_config(run_id)
        metrics = self.load_metrics(run_id)
        predictions = self.load_predictions(run_id)
        
        summary = {
            "run_id": run_id,
            "config": asdict(config) if config else None,
            "metrics": asdict(metrics) if metrics else None,
            "num_predictions": len(predictions),
            "run_dir": str(self.runs_dir / run_id),
        }
        
        return summary


# Global artifact manager instance
artifact_manager = ArtifactManager()
