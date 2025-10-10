#!/usr/bin/env python3
"""
Command-line interface for running DSPy evaluations.

This module provides a CLI for running evaluations on different
projects and models, with support for various optimizers and
configuration options.

Usage:
    python eval.py --project sharktank --model gpt-4o-mini
    python eval.py --project aneeta --model gemma-2b --optimizer BootstrapFewShot
    python eval.py --project sharktank --model gpt-4o-mini --num-samples 50
"""

import argparse
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.config import get_settings
from core.logging import setup_logging, get_logger, set_run_context
from core.artifacts import artifact_manager
from core.ids import generate_run_id
from core.telemetry import get_telemetry_collector


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DSPy model evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --project sharktank --model gpt-4o-mini
  %(prog)s --project aneeta --model gemma-2b --optimizer BootstrapFewShot
  %(prog)s --project sharktank --model gpt-4o-mini --num-samples 50 --split test
  %(prog)s --project aneeta --model gemma-2b --config configs/eval/aneeta.yaml
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--project",
        required=True,
        choices=["sharktank", "aneeta"],
        help="Project to evaluate"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to evaluate"
    )
    
    # Optional arguments
    parser.add_argument(
        "--optimizer",
        help="Optimizer to use (BootstrapFewShot, MIPROv2, COPRO)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )
    
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: experiments/runs)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running"
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        if config_file.suffix == '.json':
            return json.load(f)
        elif config_file.suffix in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")


def create_run_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create run configuration from arguments and config file.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        
    Returns:
        Run configuration dictionary
    """
    run_config = {
        "project": args.project,
        "model": args.model,
        "optimizer": args.optimizer,
        "num_samples": args.num_samples,
        "split": args.split,
        "seed": args.seed,
    }
    
    # Merge with config file
    run_config.update(config)
    
    return run_config


async def run_evaluation(
    run_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    dry_run: bool = False
) -> str:
    """
    Run the evaluation.
    
    Args:
        run_config: Run configuration
        output_dir: Output directory
        dry_run: Whether to perform a dry run
        
    Returns:
        Run ID
    """
    logger = get_logger(__name__)
    
    # Generate run ID
    run_id = generate_run_id(run_config["project"], run_config["model"])
    
    # Set run context for logging
    set_run_context(run_id, f"{run_config['project']}-eval")
    
    logger.info(f"Starting evaluation: {run_id}")
    logger.info(f"Configuration: {run_config}")
    
    if dry_run:
        logger.info("DRY RUN: Would start evaluation")
        return run_id
    
    # Create run directory and save configuration
    artifact_manager.create_run(
        config=run_config,
        run_id=run_id
    )
    
    # Initialize telemetry collector
    telemetry = get_telemetry_collector(run_id)
    
    try:
        # Import project-specific program
        if run_config["project"] == "sharktank":
            from projects.sharktank.program import create_sharktank_program
            program = create_sharktank_program()
        elif run_config["project"] == "aneeta":
            from projects.aneeta.program import create_aneeta_program
            program = create_aneeta_program()
        else:
            raise ValueError(f"Unknown project: {run_config['project']}")
        
        logger.info(f"Loaded {run_config['project']} program")
        
        # TODO: Implement actual evaluation logic
        # This is a placeholder for the evaluation process
        logger.info("Evaluation completed successfully")
        
        # Save telemetry data
        telemetry.export_to_json(Path("experiments/runs") / run_id / "telemetry.json")
        
        return run_id
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(level=log_level)
        logger = get_logger(__name__)
        
        # Load configuration
        config = load_config(args.config)
        
        # Create run configuration
        run_config = create_run_config(args, config)
        
        # Run evaluation
        run_id = asyncio.run(run_evaluation(
            run_config=run_config,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        ))
        
        print(f"Evaluation completed: {run_id}")
        print(f"Results saved to: experiments/runs/{run_id}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
