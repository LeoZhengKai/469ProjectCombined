#!/usr/bin/env python3
"""
Command-line interface for comparing DSPy evaluation results.

This module provides a CLI for comparing evaluation runs and
generating comparison reports.

Usage:
    python compare.py --runs run1 run2 --type overall
    python compare.py --runs run1 run2 run3 --type quality --output comparison.json
    python compare.py --list
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.logging import setup_logging, get_logger
from core.artifacts import artifact_manager
from core.ids import generate_compare_id


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare DSPy evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --runs run1 run2 --type overall
  %(prog)s --runs run1 run2 run3 --type quality --output comparison.json
  %(prog)s --list
  %(prog)s --compare-id comp123 --format table
        """
    )
    
    # Comparison mode
    comparison_group = parser.add_argument_group("comparison options")
    comparison_group.add_argument(
        "--runs",
        nargs="+",
        help="Run IDs to compare"
    )
    
    comparison_group.add_argument(
        "--type",
        choices=["overall", "quality", "latency", "cost"],
        default="overall",
        help="Type of comparison (default: overall)"
    )
    
    comparison_group.add_argument(
        "--output",
        help="Output file for comparison results"
    )
    
    comparison_group.add_argument(
        "--format",
        choices=["json", "table", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    
    # List mode
    list_group = parser.add_argument_group("list options")
    list_group.add_argument(
        "--list",
        action="store_true",
        help="List available runs and comparisons"
    )
    
    list_group.add_argument(
        "--project",
        help="Filter runs by project"
    )
    
    # Get comparison mode
    get_group = parser.add_argument_group("get comparison options")
    get_group.add_argument(
        "--compare-id",
        help="Get existing comparison by ID"
    )
    
    # General options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def list_runs_and_comparisons(project: Optional[str] = None):
    """
    List available runs and comparisons.
    
    Args:
        project: Optional project filter
    """
    logger = get_logger(__name__)
    
    print("Available Runs:")
    print("=" * 50)
    
    run_ids = artifact_manager.list_runs(project=project)
    
    if not run_ids:
        print("No runs found")
    else:
        for run_id in run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            if summary["config"]:
                config = summary["config"]
                print(f"  {run_id}")
                print(f"    Project: {config.get('project', 'unknown')}")
                print(f"    Model: {config.get('model', 'unknown')}")
                print(f"    Optimizer: {config.get('optimizer', 'none')}")
                print(f"    Samples: {config.get('num_samples', 'unknown')}")
                print(f"    Created: {config.get('created_at', 'unknown')}")
                print()
    
    print("Available Comparisons:")
    print("=" * 50)
    
    compare_ids = artifact_manager.list_comparisons()
    
    if not compare_ids:
        print("No comparisons found")
    else:
        for compare_id in compare_ids:
            comparison = artifact_manager.load_comparison(compare_id)
            if comparison:
                print(f"  {compare_id}")
                print(f"    Type: {comparison.comparison_type}")
                print(f"    Runs: {', '.join(comparison.run_ids)}")
                print(f"    Created: {comparison.created_at}")
                print()


def get_comparison(compare_id: str, format_type: str = "table"):
    """
    Get and display comparison results.
    
    Args:
        compare_id: Comparison identifier
        format_type: Output format
    """
    logger = get_logger(__name__)
    
    comparison = artifact_manager.load_comparison(compare_id)
    
    if not comparison:
        print(f"Comparison not found: {compare_id}")
        return
    
    if format_type == "json":
        print(json.dumps({
            "compare_id": comparison.compare_id,
            "run_ids": comparison.run_ids,
            "comparison_type": comparison.comparison_type,
            "results": comparison.results,
            "created_at": comparison.created_at
        }, indent=2))
    elif format_type == "table":
        _print_comparison_table(comparison)
    elif format_type == "csv":
        _print_comparison_csv(comparison)


def _print_comparison_table(comparison):
    """Print comparison results in table format."""
    print(f"Comparison: {comparison.compare_id}")
    print(f"Type: {comparison.comparison_type}")
    print(f"Runs: {', '.join(comparison.run_ids)}")
    print(f"Created: {comparison.created_at}")
    print()
    
    results = comparison.results
    
    if "runs" in results:
        print("Run Comparison:")
        print("-" * 80)
        
        # Print header
        runs = results["runs"]
        if runs:
            # Get all metric keys
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.get("metrics", {}).keys())
            
            # Print table header
            header = f"{'Run ID':<20} {'Model':<15}"
            for metric in sorted(all_metrics):
                header += f" {metric:<12}"
            print(header)
            print("-" * len(header))
            
            # Print run data
            for run in runs:
                row = f"{run['run_id']:<20} {run.get('model', 'unknown'):<15}"
                for metric in sorted(all_metrics):
                    value = run.get("metrics", {}).get(metric, 0)
                    if isinstance(value, float):
                        row += f" {value:<12.3f}"
                    else:
                        row += f" {str(value):<12}"
                print(row)
    
    # Print best results
    print("\nBest Results:")
    print("-" * 40)
    for key, value in results.items():
        if key.startswith("best_") or key.startswith("fastest_") or key.startswith("cheapest_"):
            print(f"  {key}: {value}")


def _print_comparison_csv(comparison):
    """Print comparison results in CSV format."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["compare_id", "comparison_type", "run_id", "model", "metric", "value"])
    
    # Write data
    results = comparison.results
    if "runs" in results:
        for run in results["runs"]:
            for metric, value in run.get("metrics", {}).items():
                writer.writerow([
                    comparison.compare_id,
                    comparison.comparison_type,
                    run["run_id"],
                    run.get("model", "unknown"),
                    metric,
                    value
                ])
    
    print(output.getvalue())


def compare_runs(
    run_ids: List[str],
    comparison_type: str,
    output_file: Optional[str] = None,
    format_type: str = "table"
) -> str:
    """
    Compare multiple runs.
    
    Args:
        run_ids: List of run IDs to compare
        comparison_type: Type of comparison
        output_file: Optional output file
        format_type: Output format
        
    Returns:
        Comparison ID
    """
    logger = get_logger(__name__)
    
    # Validate run IDs
    for run_id in run_ids:
        summary = artifact_manager.get_run_summary(run_id)
        if not summary["config"]:
            raise ValueError(f"Run not found: {run_id}")
    
    # Generate comparison ID
    compare_id = generate_compare_id(run_ids)
    
    logger.info(f"Comparing runs: {run_ids}")
    logger.info(f"Comparison type: {comparison_type}")
    
    # Load run data
    runs_data = []
    for run_id in run_ids:
        summary = artifact_manager.get_run_summary(run_id)
        runs_data.append(summary)
    
    # Perform comparison
    comparison_results = _perform_comparison(runs_data, comparison_type)
    
    # Save comparison results
    artifact_manager.create_comparison(
        run_ids=run_ids,
        comparison_type=comparison_type,
        results=comparison_results,
        compare_id=compare_id
    )
    
    # Display results
    comparison = artifact_manager.load_comparison(compare_id)
    if comparison:
        if format_type == "table":
            _print_comparison_table(comparison)
        elif format_type == "json":
            print(json.dumps({
                "compare_id": comparison.compare_id,
                "run_ids": comparison.run_ids,
                "comparison_type": comparison.comparison_type,
                "results": comparison.results,
                "created_at": comparison.created_at
            }, indent=2))
        elif format_type == "csv":
            _print_comparison_csv(comparison)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "compare_id": comparison.compare_id,
                "run_ids": comparison.run_ids,
                "comparison_type": comparison.comparison_type,
                "results": comparison.results,
                "created_at": comparison.created_at
            }, f, indent=2)
        
        print(f"\nComparison saved to: {output_file}")
    
    return compare_id


def _perform_comparison(
    runs_data: List[Dict[str, Any]],
    comparison_type: str
) -> Dict[str, Any]:
    """
    Perform the actual comparison between runs.
    
    Args:
        runs_data: List of run data dictionaries
        comparison_type: Type of comparison
        
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


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(level=log_level)
        logger = get_logger(__name__)
        
        # Handle different modes
        if args.list:
            list_runs_and_comparisons(project=args.project)
        elif args.compare_id:
            get_comparison(args.compare_id, args.format)
        elif args.runs:
            compare_id = compare_runs(
                run_ids=args.runs,
                comparison_type=args.type,
                output_file=args.output,
                format_type=args.format
            )
            print(f"Comparison completed: {compare_id}")
        else:
            print("No action specified. Use --help for usage information.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
