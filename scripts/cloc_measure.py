#!/usr/bin/env python3
"""
Code complexity measurement script.

This script measures lines of code (LoC) for comparing legacy multi-agent
systems with DSPy programs to evaluate code complexity reduction.

Owner: Zheng Kai
Acceptance Check: Compares LoC between legacy and DSPy paths; writes to experiments/comparisons/<id>.json
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse


class CLOCMeasurer:
    """
    Measures code complexity using cloc (Count Lines of Code).
    
    Provides methods to:
    - Measure LoC for different codebases
    - Compare legacy vs DSPy implementations
    - Generate complexity reduction reports
    """
    
    def __init__(self, cloc_path: Optional[str] = None):
        """
        Initialize the CLOC measurer.
        
        Args:
            cloc_path: Path to cloc executable (if not in PATH)
        """
        self.cloc_path = cloc_path or "cloc"
        self.project_root = Path(__file__).parent.parent
        
    def check_cloc_available(self) -> bool:
        """
        Check if cloc is available.
        
        Returns:
            True if cloc is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.cloc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_cloc(self) -> bool:
        """
        Install cloc if not available.
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Try different installation methods
            install_commands = [
                ["pip", "install", "cloc"],
                ["brew", "install", "cloc"],
                ["apt-get", "install", "-y", "cloc"],
                ["yum", "install", "-y", "cloc"]
            ]
            
            for cmd in install_commands:
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        print(f"✓ cloc installed successfully using {' '.join(cmd)}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("✗ Failed to install cloc automatically")
            print("Please install cloc manually:")
            print("  - macOS: brew install cloc")
            print("  - Ubuntu: apt-get install cloc")
            print("  - CentOS: yum install cloc")
            print("  - Python: pip install cloc")
            return False
            
        except Exception as e:
            print(f"✗ Error installing cloc: {e}")
            return False
    
    def measure_directory(
        self,
        directory: Path,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Measure LoC for a directory.
        
        Args:
            directory: Directory to measure
            exclude_patterns: Patterns to exclude from measurement
            
        Returns:
            Dictionary with LoC measurements
        """
        if not directory.exists():
            return {"error": f"Directory not found: {directory}"}
        
        # Default exclude patterns
        default_excludes = [
            "*.pyc", "*.pyo", "__pycache__", "*.egg-info",
            ".git", ".svn", "node_modules", "venv", "env",
            "*.log", "*.tmp", "*.cache", ".pytest_cache"
        ]
        
        exclude_patterns = exclude_patterns or default_excludes
        
        try:
            # Build cloc command
            cmd = [self.cloc_path, "--json", str(directory)]
            
            # Add exclude patterns
            for pattern in exclude_patterns:
                cmd.extend(["--exclude-dir", pattern])
            
            # Run cloc
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                return {"error": f"cloc failed: {result.stderr}"}
            
            # Parse JSON output
            cloc_data = json.loads(result.stdout)
            
            # Extract relevant metrics
            total_lines = cloc_data.get("SUM", {}).get("code", 0)
            total_files = cloc_data.get("SUM", {}).get("nFiles", 0)
            
            # Get language breakdown
            language_breakdown = {}
            for lang, data in cloc_data.items():
                if lang != "SUM" and isinstance(data, dict):
                    language_breakdown[lang] = {
                        "files": data.get("nFiles", 0),
                        "lines": data.get("code", 0),
                        "blank_lines": data.get("blank", 0),
                        "comment_lines": data.get("comment", 0)
                    }
            
            return {
                "directory": str(directory),
                "total_files": total_files,
                "total_lines": total_lines,
                "language_breakdown": language_breakdown,
                "measurement_time": datetime.now().isoformat(),
                "cloc_version": self._get_cloc_version()
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "cloc measurement timed out"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse cloc JSON output"}
        except Exception as e:
            return {"error": f"Measurement failed: {e}"}
    
    def _get_cloc_version(self) -> str:
        """Get cloc version."""
        try:
            result = subprocess.run(
                [self.cloc_path, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "unknown"
    
    def compare_implementations(
        self,
        legacy_path: Path,
        dspy_path: Path,
        comparison_name: str
    ) -> Dict[str, Any]:
        """
        Compare LoC between legacy and DSPy implementations.
        
        Args:
            legacy_path: Path to legacy implementation
            dspy_path: Path to DSPy implementation
            comparison_name: Name for the comparison
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Measuring legacy implementation: {legacy_path}")
        legacy_metrics = self.measure_directory(legacy_path)
        
        print(f"Measuring DSPy implementation: {dspy_path}")
        dspy_metrics = self.measure_directory(dspy_path)
        
        if "error" in legacy_metrics or "error" in dspy_metrics:
            return {
                "error": "Failed to measure one or both implementations",
                "legacy_error": legacy_metrics.get("error"),
                "dspy_error": dspy_metrics.get("error")
            }
        
        # Calculate reduction metrics
        legacy_lines = legacy_metrics["total_lines"]
        dspy_lines = dspy_metrics["total_lines"]
        
        reduction_absolute = legacy_lines - dspy_lines
        reduction_percentage = (reduction_absolute / legacy_lines * 100) if legacy_lines > 0 else 0
        
        # Compare language breakdowns
        language_comparison = {}
        all_languages = set(legacy_metrics["language_breakdown"].keys()) | set(dspy_metrics["language_breakdown"].keys())
        
        for lang in all_languages:
            legacy_lang = legacy_metrics["language_breakdown"].get(lang, {"lines": 0, "files": 0})
            dspy_lang = dspy_metrics["language_breakdown"].get(lang, {"lines": 0, "files": 0})
            
            language_comparison[lang] = {
                "legacy_lines": legacy_lang["lines"],
                "dspy_lines": dspy_lang["lines"],
                "reduction_lines": legacy_lang["lines"] - dspy_lang["lines"],
                "reduction_percentage": (
                    (legacy_lang["lines"] - dspy_lang["lines"]) / legacy_lang["lines"] * 100
                    if legacy_lang["lines"] > 0 else 0
                )
            }
        
        comparison_result = {
            "comparison_name": comparison_name,
            "legacy_implementation": {
                "path": str(legacy_path),
                "total_files": legacy_metrics["total_files"],
                "total_lines": legacy_lines,
                "language_breakdown": legacy_metrics["language_breakdown"]
            },
            "dspy_implementation": {
                "path": str(dspy_path),
                "total_files": dspy_metrics["total_files"],
                "total_lines": dspy_lines,
                "language_breakdown": dspy_metrics["language_breakdown"]
            },
            "reduction_metrics": {
                "absolute_reduction": reduction_absolute,
                "percentage_reduction": reduction_percentage,
                "complexity_ratio": dspy_lines / legacy_lines if legacy_lines > 0 else 0
            },
            "language_comparison": language_comparison,
            "measurement_time": datetime.now().isoformat(),
            "cloc_version": self._get_cloc_version()
        }
        
        return comparison_result
    
    def save_comparison(
        self,
        comparison_result: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save comparison results to file.
        
        Args:
            comparison_result: Comparison results to save
            output_dir: Output directory (default: experiments/comparisons)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = self.project_root / "experiments" / "comparisons"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = comparison_result["comparison_name"].replace(" ", "_").lower()
        filename = f"{comparison_name}_cloc_comparison_{timestamp}.json"
        
        output_path = output_dir / filename
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2)
        
        print(f"✓ Comparison saved to: {output_path}")
        return output_path
    
    def generate_report(self, comparison_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from comparison results.
        
        Args:
            comparison_result: Comparison results
            
        Returns:
            Human-readable report string
        """
        if "error" in comparison_result:
            return f"Error: {comparison_result['error']}"
        
        report = []
        report.append(f"Code Complexity Comparison: {comparison_result['comparison_name']}")
        report.append("=" * 60)
        
        # Overall metrics
        legacy_lines = comparison_result["legacy_implementation"]["total_lines"]
        dspy_lines = comparison_result["dspy_implementation"]["total_lines"]
        reduction = comparison_result["reduction_metrics"]
        
        report.append(f"Legacy Implementation: {legacy_lines:,} lines")
        report.append(f"DSPy Implementation: {dspy_lines:,} lines")
        report.append(f"Absolute Reduction: {reduction['absolute_reduction']:,} lines")
        report.append(f"Percentage Reduction: {reduction['percentage_reduction']:.1f}%")
        report.append(f"Complexity Ratio: {reduction['complexity_ratio']:.2f}")
        report.append("")
        
        # Language breakdown
        report.append("Language Breakdown:")
        report.append("-" * 30)
        
        for lang, metrics in comparison_result["language_comparison"].items():
            if metrics["legacy_lines"] > 0 or metrics["dspy_lines"] > 0:
                report.append(f"{lang}:")
                report.append(f"  Legacy: {metrics['legacy_lines']:,} lines")
                report.append(f"  DSPy: {metrics['dspy_lines']:,} lines")
                report.append(f"  Reduction: {metrics['reduction_lines']:,} lines ({metrics['reduction_percentage']:.1f}%)")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Measure code complexity using cloc")
    parser.add_argument("--legacy-path", type=str, required=True, help="Path to legacy implementation")
    parser.add_argument("--dspy-path", type=str, required=True, help="Path to DSPy implementation")
    parser.add_argument("--comparison-name", type=str, required=True, help="Name for the comparison")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--cloc-path", type=str, help="Path to cloc executable")
    parser.add_argument("--install-cloc", action="store_true", help="Install cloc if not available")
    parser.add_argument("--report-only", action="store_true", help="Only generate report, don't save")
    
    args = parser.parse_args()
    
    # Initialize measurer
    measurer = CLOCMeasurer(args.cloc_path)
    
    # Check cloc availability
    if not measurer.check_cloc_available():
        print("✗ cloc not found")
        if args.install_cloc:
            if not measurer.install_cloc():
                sys.exit(1)
        else:
            print("Use --install-cloc to install cloc automatically")
            sys.exit(1)
    
    # Convert paths
    legacy_path = Path(args.legacy_path)
    dspy_path = Path(args.dspy_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Run comparison
    print(f"Comparing implementations:")
    print(f"  Legacy: {legacy_path}")
    print(f"  DSPy: {dspy_path}")
    print(f"  Name: {args.comparison_name}")
    
    comparison_result = measurer.compare_implementations(
        legacy_path, dspy_path, args.comparison_name
    )
    
    # Generate and display report
    report = measurer.generate_report(comparison_result)
    print("\n" + report)
    
    # Save results if not report-only
    if not args.report_only:
        output_path = measurer.save_comparison(comparison_result, output_dir)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
