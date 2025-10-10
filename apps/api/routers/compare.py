"""
Comparison router for DSPy evaluation API.

This module provides endpoints for comparing evaluation results:
- Run comparisons
- Model comparisons
- Optimizer comparisons
- Performance analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.logging import get_logger
from core.artifacts import artifact_manager
from core.ids import generate_compare_id

router = APIRouter()
logger = get_logger(__name__)


class ComparisonRequest(BaseModel):
    """Request model for run comparison."""
    run_ids: List[str]
    comparison_type: str = "overall"
    metrics: Optional[List[str]] = None


class ComparisonResponse(BaseModel):
    """Response model for run comparison."""
    compare_id: str
    run_ids: List[str]
    comparison_type: str
    results: Dict[str, Any]
    created_at: str


@router.post("/runs", response_model=ComparisonResponse)
async def compare_runs(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Compare multiple evaluation runs.
    
    Args:
        request: Comparison request
        background_tasks: FastAPI background tasks
        
    Returns:
        Comparison results
    """
    try:
        logger.info(f"Comparing runs: {request.run_ids}")
        
        # Validate run IDs
        for run_id in request.run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            if not summary["config"]:
                raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        
        # Generate comparison ID
        compare_id = generate_compare_id(request.run_ids)
        
        # Schedule background comparison task
        if background_tasks:
            background_tasks.add_task(
                _run_comparison_background,
                compare_id, request.run_ids, request.comparison_type, request.metrics
            )
        
        return ComparisonResponse(
            compare_id=compare_id,
            run_ids=request.run_ids,
            comparison_type=request.comparison_type,
            results={"status": "processing"},
            created_at="2024-01-01T00:00:00Z"  # This would be actual timestamp in production
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_comparison_background(
    compare_id: str,
    run_ids: List[str],
    comparison_type: str,
    metrics: Optional[List[str]]
):
    """
    Run comparison in background.
    
    Args:
        compare_id: Comparison identifier
        run_ids: List of run IDs to compare
        comparison_type: Type of comparison
        metrics: Optional metrics to focus on
    """
    try:
        logger.info(f"Running background comparison: {compare_id}")
        
        # Load run data
        runs_data = []
        for run_id in run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            runs_data.append(summary)
        
        # Perform comparison
        comparison_results = _perform_comparison(runs_data, comparison_type, metrics)
        
        # Save comparison results
        artifact_manager.create_comparison(
            run_ids=run_ids,
            comparison_type=comparison_type,
            results=comparison_results,
            compare_id=compare_id
        )
        
        logger.info(f"Comparison completed: {compare_id}")
        
    except Exception as e:
        logger.error(f"Error in background comparison {compare_id}: {e}")


def _perform_comparison(
    runs_data: List[Dict[str, Any]],
    comparison_type: str,
    metrics: Optional[List[str]]
) -> Dict[str, Any]:
    """
    Perform the actual comparison between runs.
    
    Args:
        runs_data: List of run data dictionaries
        comparison_type: Type of comparison
        metrics: Optional metrics to focus on
        
    Returns:
        Comparison results dictionary
    """
    if len(runs_data) < 2:
        return {"error": "At least 2 runs required for comparison"}
    
    # Extract metrics from each run
    run_metrics = []
    for run_data in runs_data:
        if run_data["metrics"]:
            run_metrics.append(run_data["metrics"])
        else:
            run_metrics.append({})
    
    # Perform comparison based on type
    if comparison_type == "quality":
        return _compare_quality_metrics(run_metrics, runs_data)
    elif comparison_type == "latency":
        return _compare_latency_metrics(run_metrics, runs_data)
    elif comparison_type == "cost":
        return _compare_cost_metrics(run_metrics, runs_data)
    elif comparison_type == "overall":
        return _compare_overall_metrics(run_metrics, runs_data)
    else:
        return {"error": f"Unknown comparison type: {comparison_type}"}


def _compare_quality_metrics(
    run_metrics: List[Dict[str, Any]],
    runs_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare quality metrics between runs."""
    quality_metrics = ["quality_mean", "fact_mean"]
    
    comparison = {
        "type": "quality",
        "runs": []
    }
    
    for i, (metrics, run_data) in enumerate(zip(run_metrics, runs_data)):
        run_info = {
            "run_id": run_data["run_id"],
            "model": run_data["config"]["model"] if run_data["config"] else "unknown",
            "metrics": {}
        }
        
        for metric in quality_metrics:
            if metric in metrics:
                run_info["metrics"][metric] = metrics[metric]
            else:
                run_info["metrics"][metric] = 0.0
        
        comparison["runs"].append(run_info)
    
    # Find best run for each metric
    for metric in quality_metrics:
        best_run = max(comparison["runs"], key=lambda x: x["metrics"].get(metric, 0))
        comparison[f"best_{metric}"] = {
            "run_id": best_run["run_id"],
            "value": best_run["metrics"][metric]
        }
    
    return comparison


def _compare_latency_metrics(
    run_metrics: List[Dict[str, Any]],
    runs_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare latency metrics between runs."""
    latency_metrics = ["latency_p50_ms", "latency_p95_ms"]
    
    comparison = {
        "type": "latency",
        "runs": []
    }
    
    for i, (metrics, run_data) in enumerate(zip(run_metrics, runs_data)):
        run_info = {
            "run_id": run_data["run_id"],
            "model": run_data["config"]["model"] if run_data["config"] else "unknown",
            "metrics": {}
        }
        
        for metric in latency_metrics:
            if metric in metrics:
                run_info["metrics"][metric] = metrics[metric]
            else:
                run_info["metrics"][metric] = 0.0
        
        comparison["runs"].append(run_info)
    
    # Find fastest run for each metric
    for metric in latency_metrics:
        fastest_run = min(comparison["runs"], key=lambda x: x["metrics"].get(metric, float('inf')))
        comparison[f"fastest_{metric}"] = {
            "run_id": fastest_run["run_id"],
            "value": fastest_run["metrics"][metric]
        }
    
    return comparison


def _compare_cost_metrics(
    run_metrics: List[Dict[str, Any]],
    runs_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare cost metrics between runs."""
    cost_metrics = ["total_cost_usd", "cost_per_sample_usd"]
    
    comparison = {
        "type": "cost",
        "runs": []
    }
    
    for i, (metrics, run_data) in enumerate(zip(run_metrics, runs_data)):
        run_info = {
            "run_id": run_data["run_id"],
            "model": run_data["config"]["model"] if run_data["config"] else "unknown",
            "metrics": {}
        }
        
        for metric in cost_metrics:
            if metric in metrics:
                run_info["metrics"][metric] = metrics[metric]
            else:
                run_info["metrics"][metric] = 0.0
        
        comparison["runs"].append(run_info)
    
    # Find cheapest run for each metric
    for metric in cost_metrics:
        cheapest_run = min(comparison["runs"], key=lambda x: x["metrics"].get(metric, float('inf')))
        comparison[f"cheapest_{metric}"] = {
            "run_id": cheapest_run["run_id"],
            "value": cheapest_run["metrics"][metric]
        }
    
    return comparison


def _compare_overall_metrics(
    run_metrics: List[Dict[str, Any]],
    runs_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare overall metrics between runs."""
    all_metrics = ["quality_mean", "fact_mean", "latency_p50_ms", "latency_p95_ms", "cost_per_sample_usd"]
    
    comparison = {
        "type": "overall",
        "runs": []
    }
    
    for i, (metrics, run_data) in enumerate(zip(run_metrics, runs_data)):
        run_info = {
            "run_id": run_data["run_id"],
            "model": run_data["config"]["model"] if run_data["config"] else "unknown",
            "metrics": {}
        }
        
        for metric in all_metrics:
            if metric in metrics:
                run_info["metrics"][metric] = metrics[metric]
            else:
                run_info["metrics"][metric] = 0.0
        
        comparison["runs"].append(run_info)
    
    # Calculate overall scores (weighted combination)
    for run_info in comparison["runs"]:
        quality_score = run_info["metrics"].get("quality_mean", 0) * 0.3
        fact_score = run_info["metrics"].get("fact_mean", 0) * 0.3
        latency_score = max(0, 10 - run_info["metrics"].get("latency_p50_ms", 0) / 100) * 0.2
        cost_score = max(0, 10 - run_info["metrics"].get("cost_per_sample_usd", 0) * 100) * 0.2
        
        run_info["overall_score"] = quality_score + fact_score + latency_score + cost_score
    
    # Find best overall run
    best_run = max(comparison["runs"], key=lambda x: x["overall_score"])
    comparison["best_overall"] = {
        "run_id": best_run["run_id"],
        "score": best_run["overall_score"]
    }
    
    return comparison


@router.get("/{compare_id}")
async def get_comparison(compare_id: str):
    """
    Get comparison results by ID.
    
    Args:
        compare_id: Comparison identifier
        
    Returns:
        Comparison results
    """
    try:
        comparison = artifact_manager.load_comparison(compare_id)
        
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        return {
            "compare_id": comparison.compare_id,
            "run_ids": comparison.run_ids,
            "comparison_type": comparison.comparison_type,
            "results": comparison.results,
            "created_at": comparison.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison {compare_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_comparisons():
    """
    List all comparison results.
    
    Returns:
        List of comparison information
    """
    try:
        compare_ids = artifact_manager.list_comparisons()
        
        comparisons = []
        for compare_id in compare_ids:
            comparison = artifact_manager.load_comparison(compare_id)
            if comparison:
                comparisons.append({
                    "compare_id": comparison.compare_id,
                    "run_ids": comparison.run_ids,
                    "comparison_type": comparison.comparison_type,
                    "created_at": comparison.created_at
                })
        
        return {
            "comparisons": comparisons,
            "total": len(comparisons)
        }
        
    except Exception as e:
        logger.error(f"Error listing comparisons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{compare_id}")
async def delete_comparison(compare_id: str):
    """
    Delete a comparison result.
    
    Args:
        compare_id: Comparison identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        comparison = artifact_manager.load_comparison(compare_id)
        
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Delete comparison file
        comparison_file = Path("experiments/comparisons") / f"{compare_id}.json"
        if comparison_file.exists():
            comparison_file.unlink()
        
        logger.info(f"Deleted comparison: {compare_id}")
        
        return {
            "compare_id": compare_id,
            "status": "deleted",
            "message": "Comparison deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comparison {compare_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
