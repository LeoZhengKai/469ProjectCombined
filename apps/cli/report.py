#!/usr/bin/env python3
"""
Command-line interface for generating evaluation reports.

This module provides a CLI for generating HTML and CSV reports
from evaluation results and comparisons.

Usage:
    python report.py --run-id run123 --format html
    python report.py --compare-id comp123 --format csv
    python report.py --project sharktank --format html --output report.html
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.logging import setup_logging, get_logger
from core.artifacts import artifact_manager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --run-id run123 --format html
  %(prog)s --compare-id comp123 --format csv
  %(prog)s --project sharktank --format html --output report.html
  %(prog)s --runs run1 run2 run3 --format html --output comparison.html
        """
    )
    
    # Report type
    report_group = parser.add_argument_group("report options")
    report_group.add_argument(
        "--run-id",
        help="Generate report for specific run"
    )
    
    report_group.add_argument(
        "--compare-id",
        help="Generate report for specific comparison"
    )
    
    report_group.add_argument(
        "--project",
        help="Generate report for all runs in project"
    )
    
    report_group.add_argument(
        "--runs",
        nargs="+",
        help="Generate report for specific runs"
    )
    
    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--format",
        choices=["html", "csv", "json"],
        default="html",
        help="Output format (default: html)"
    )
    
    output_group.add_argument(
        "--output",
        help="Output file path"
    )
    
    output_group.add_argument(
        "--template",
        help="Custom HTML template file"
    )
    
    # General options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def generate_run_report(run_id: str, format_type: str, output_file: Optional[str] = None) -> str:
    """
    Generate a report for a specific run.
    
    Args:
        run_id: Run identifier
        format_type: Output format
        output_file: Optional output file
        
    Returns:
        Generated report content
    """
    logger = get_logger(__name__)
    
    # Get run summary
    summary = artifact_manager.get_run_summary(run_id)
    
    if not summary["config"]:
        raise ValueError(f"Run not found: {run_id}")
    
    logger.info(f"Generating {format_type} report for run: {run_id}")
    
    if format_type == "html":
        content = _generate_html_run_report(summary)
    elif format_type == "csv":
        content = _generate_csv_run_report(summary)
    elif format_type == "json":
        content = json.dumps(summary, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to: {output_file}")
    
    return content


def generate_comparison_report(compare_id: str, format_type: str, output_file: Optional[str] = None) -> str:
    """
    Generate a report for a specific comparison.
    
    Args:
        compare_id: Comparison identifier
        format_type: Output format
        output_file: Optional output file
        
    Returns:
        Generated report content
    """
    logger = get_logger(__name__)
    
    # Get comparison
    comparison = artifact_manager.load_comparison(compare_id)
    
    if not comparison:
        raise ValueError(f"Comparison not found: {compare_id}")
    
    logger.info(f"Generating {format_type} report for comparison: {compare_id}")
    
    if format_type == "html":
        content = _generate_html_comparison_report(comparison)
    elif format_type == "csv":
        content = _generate_csv_comparison_report(comparison)
    elif format_type == "json":
        content = json.dumps({
            "compare_id": comparison.compare_id,
            "run_ids": comparison.run_ids,
            "comparison_type": comparison.comparison_type,
            "results": comparison.results,
            "created_at": comparison.created_at
        }, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to: {output_file}")
    
    return content


def generate_project_report(project: str, format_type: str, output_file: Optional[str] = None) -> str:
    """
    Generate a report for all runs in a project.
    
    Args:
        project: Project name
        format_type: Output format
        output_file: Optional output file
        
    Returns:
        Generated report content
    """
    logger = get_logger(__name__)
    
    # Get all runs for project
    run_ids = artifact_manager.list_runs(project=project)
    
    if not run_ids:
        raise ValueError(f"No runs found for project: {project}")
    
    logger.info(f"Generating {format_type} report for project: {project}")
    
    # Get run summaries
    summaries = []
    for run_id in run_ids:
        summary = artifact_manager.get_run_summary(run_id)
        if summary["config"]:
            summaries.append(summary)
    
    if format_type == "html":
        content = _generate_html_project_report(project, summaries)
    elif format_type == "csv":
        content = _generate_csv_project_report(project, summaries)
    elif format_type == "json":
        content = json.dumps({
            "project": project,
            "runs": summaries,
            "total": len(summaries)
        }, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to: {output_file}")
    
    return content


def generate_multi_run_report(run_ids: List[str], format_type: str, output_file: Optional[str] = None) -> str:
    """
    Generate a report for multiple runs.
    
    Args:
        run_ids: List of run identifiers
        format_type: Output format
        output_file: Optional output file
        
    Returns:
        Generated report content
    """
    logger = get_logger(__name__)
    
    # Get run summaries
    summaries = []
    for run_id in run_ids:
        summary = artifact_manager.get_run_summary(run_id)
        if summary["config"]:
            summaries.append(summary)
    
    if not summaries:
        raise ValueError("No valid runs found")
    
    logger.info(f"Generating {format_type} report for {len(summaries)} runs")
    
    if format_type == "html":
        content = _generate_html_multi_run_report(summaries)
    elif format_type == "csv":
        content = _generate_csv_multi_run_report(summaries)
    elif format_type == "json":
        content = json.dumps({
            "runs": summaries,
            "total": len(summaries)
        }, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to: {output_file}")
    
    return content


def _generate_html_run_report(summary: Dict[str, Any]) -> str:
    """Generate HTML report for a single run."""
    config = summary["config"]
    metrics = summary["metrics"]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {summary['run_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
            .config {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Evaluation Report</h1>
            <h2>Run ID: {summary['run_id']}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Configuration</h3>
            <div class="config">
                <p><strong>Project:</strong> {config.get('project', 'unknown')}</p>
                <p><strong>Model:</strong> {config.get('model', 'unknown')}</p>
                <p><strong>Optimizer:</strong> {config.get('optimizer', 'none')}</p>
                <p><strong>Samples:</strong> {config.get('num_samples', 'unknown')}</p>
                <p><strong>Split:</strong> {config.get('split', 'unknown')}</p>
                <p><strong>Seed:</strong> {config.get('seed', 'unknown')}</p>
                <p><strong>Created:</strong> {config.get('created_at', 'unknown')}</p>
            </div>
        </div>
        
        <div class="section">
            <h3>Metrics</h3>
            <div class="metrics">
    """
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><strong>{key}:</strong> {value:.3f}</div>'
            else:
                html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
    else:
        html += '<p>No metrics available</p>'
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h3>Summary</h3>
            <p>Total samples: {num_predictions}</p>
            <p>Run directory: {run_dir}</p>
        </div>
    </body>
    </html>
    """.format(
        num_predictions=summary.get('num_predictions', 0),
        run_dir=summary.get('run_dir', 'unknown')
    )
    
    return html


def _generate_csv_run_report(summary: Dict[str, Any]) -> str:
    """Generate CSV report for a single run."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["run_id", "project", "model", "optimizer", "num_samples", "split", "seed", "created_at"])
    
    # Write config data
    config = summary["config"]
    writer.writerow([
        summary["run_id"],
        config.get("project", ""),
        config.get("model", ""),
        config.get("optimizer", ""),
        config.get("num_samples", ""),
        config.get("split", ""),
        config.get("seed", ""),
        config.get("created_at", "")
    ])
    
    # Write metrics header
    writer.writerow([])
    writer.writerow(["metric", "value"])
    
    # Write metrics data
    metrics = summary["metrics"]
    if metrics:
        for key, value in metrics.items():
            writer.writerow([key, value])
    
    return output.getvalue()


def _generate_html_comparison_report(comparison) -> str:
    """Generate HTML report for a comparison."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparison Report - {comparison.compare_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .best {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Comparison Report</h1>
            <h2>Comparison ID: {comparison.compare_id}</h2>
            <p>Type: {comparison.comparison_type}</p>
            <p>Runs: {', '.join(comparison.run_ids)}</p>
            <p>Created: {comparison.created_at}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Results</h3>
            <pre>{json.dumps(comparison.results, indent=2)}</pre>
        </div>
    </body>
    </html>
    """
    
    return html


def _generate_csv_comparison_report(comparison) -> str:
    """Generate CSV report for a comparison."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["compare_id", "comparison_type", "run_id", "metric", "value"])
    
    # Write comparison data
    results = comparison.results
    if "runs" in results:
        for run in results["runs"]:
            for metric, value in run.get("metrics", {}).items():
                writer.writerow([
                    comparison.compare_id,
                    comparison.comparison_type,
                    run["run_id"],
                    metric,
                    value
                ])
    
    return output.getvalue()


def _generate_html_project_report(project: str, summaries: List[Dict[str, Any]]) -> str:
    """Generate HTML report for a project."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Project Report - {project}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Project Report</h1>
            <h2>Project: {project}</h2>
            <p>Total runs: {len(summaries)}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Runs Summary</h3>
            <table>
                <tr>
                    <th>Run ID</th>
                    <th>Model</th>
                    <th>Optimizer</th>
                    <th>Samples</th>
                    <th>Created</th>
                </tr>
    """
    
    for summary in summaries:
        config = summary["config"]
        html += f"""
                <tr>
                    <td>{summary['run_id']}</td>
                    <td>{config.get('model', 'unknown')}</td>
                    <td>{config.get('optimizer', 'none')}</td>
                    <td>{config.get('num_samples', 'unknown')}</td>
                    <td>{config.get('created_at', 'unknown')}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html


def _generate_csv_project_report(project: str, summaries: List[Dict[str, Any]]) -> str:
    """Generate CSV report for a project."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["project", "run_id", "model", "optimizer", "num_samples", "created_at"])
    
    # Write data
    for summary in summaries:
        config = summary["config"]
        writer.writerow([
            project,
            summary["run_id"],
            config.get("model", ""),
            config.get("optimizer", ""),
            config.get("num_samples", ""),
            config.get("created_at", "")
        ])
    
    return output.getvalue()


def _generate_html_multi_run_report(summaries: List[Dict[str, Any]]) -> str:
    """Generate HTML report for multiple runs."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Run Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multi-Run Report</h1>
            <p>Total runs: {len(summaries)}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>Runs Summary</h3>
            <table>
                <tr>
                    <th>Run ID</th>
                    <th>Project</th>
                    <th>Model</th>
                    <th>Optimizer</th>
                    <th>Samples</th>
                    <th>Created</th>
                </tr>
    """
    
    for summary in summaries:
        config = summary["config"]
        html += f"""
                <tr>
                    <td>{summary['run_id']}</td>
                    <td>{config.get('project', 'unknown')}</td>
                    <td>{config.get('model', 'unknown')}</td>
                    <td>{config.get('optimizer', 'none')}</td>
                    <td>{config.get('num_samples', 'unknown')}</td>
                    <td>{config.get('created_at', 'unknown')}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html


def _generate_csv_multi_run_report(summaries: List[Dict[str, Any]]) -> str:
    """Generate CSV report for multiple runs."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["run_id", "project", "model", "optimizer", "num_samples", "created_at"])
    
    # Write data
    for summary in summaries:
        config = summary["config"]
        writer.writerow([
            summary["run_id"],
            config.get("project", ""),
            config.get("model", ""),
            config.get("optimizer", ""),
            config.get("num_samples", ""),
            config.get("created_at", "")
        ])
    
    return output.getvalue()


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(level=log_level)
        logger = get_logger(__name__)
        
        # Generate report based on arguments
        if args.run_id:
            content = generate_run_report(args.run_id, args.format, args.output)
        elif args.compare_id:
            content = generate_comparison_report(args.compare_id, args.format, args.output)
        elif args.project:
            content = generate_project_report(args.project, args.format, args.output)
        elif args.runs:
            content = generate_multi_run_report(args.runs, args.format, args.output)
        else:
            print("No report type specified. Use --help for usage information.")
            sys.exit(1)
        
        # Print content if no output file specified
        if not args.output:
            print(content)
        
    except KeyboardInterrupt:
        print("\nReport generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
