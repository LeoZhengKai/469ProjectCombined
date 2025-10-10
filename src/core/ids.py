"""
ID generation utilities for DSPy evaluation framework.

This module provides functions to generate unique identifiers for:
- Run IDs (experiment runs)
- Comparison IDs (comparisons between runs)
- Sample IDs (individual evaluation samples)

IDs are designed to be:
- Unique and collision-resistant
- Human-readable with meaningful prefixes
- Sortable by creation time
- Compatible with filesystem naming conventions
"""

import hashlib
import time
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


def generate_run_id(project: str, model: str, timestamp: Optional[float] = None) -> str:
    """
    Generate a unique run ID for an experiment.
    
    Format: <project>-<model>-<timestamp>-<short_hash>
    Example: sharktank-gpt4o-mini-20240101-120000-a1b2c3d4
    
    Args:
        project: Project name (e.g., "sharktank", "aneeta")
        model: Model name (e.g., "gpt-4o-mini", "claude-3-haiku")
        timestamp: Optional timestamp (current time if not provided)
        
    Returns:
        Unique run ID string
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Format timestamp as YYYYMMDD-HHMMSS
    dt = datetime.fromtimestamp(timestamp)
    time_str = dt.strftime("%Y%m%d-%H%M%S")
    
    # Generate short hash for uniqueness
    hash_input = f"{project}-{model}-{timestamp}-{uuid4()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Clean model name for filesystem compatibility
    clean_model = model.replace("/", "-").replace(":", "-").replace("_", "-")
    
    return f"{project}-{clean_model}-{time_str}-{short_hash}"


def generate_compare_id(run_ids: List[str], timestamp: Optional[float] = None) -> str:
    """
    Generate a unique comparison ID for comparing multiple runs.
    
    Format: compare-<timestamp>-<hash_of_run_ids>
    Example: compare-20240101-120000-a1b2c3d4
    
    Args:
        run_ids: List of run IDs being compared
        timestamp: Optional timestamp (current time if not provided)
        
    Returns:
        Unique comparison ID string
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Format timestamp as YYYYMMDD-HHMMSS
    dt = datetime.fromtimestamp(timestamp)
    time_str = dt.strftime("%Y%m%d-%H%M%S")
    
    # Generate hash from run IDs for uniqueness
    run_ids_str = "-".join(sorted(run_ids))
    hash_input = f"compare-{run_ids_str}-{timestamp}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    return f"compare-{time_str}-{short_hash}"


def generate_sample_id(prefix: str = "sample", timestamp: Optional[float] = None) -> str:
    """
    Generate a unique sample ID for individual evaluation samples.
    
    Format: <prefix>-<timestamp>-<short_hash>
    Example: sample-20240101-120000-a1b2c3d4
    
    Args:
        prefix: Prefix for the sample ID (default: "sample")
        timestamp: Optional timestamp (current time if not provided)
        
    Returns:
        Unique sample ID string
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Format timestamp as YYYYMMDD-HHMMSS
    dt = datetime.fromtimestamp(timestamp)
    time_str = dt.strftime("%Y%m%d-%H%M%S")
    
    # Generate short hash for uniqueness
    hash_input = f"{prefix}-{timestamp}-{uuid4()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    return f"{prefix}-{time_str}-{short_hash}"


def parse_run_id(run_id: str) -> dict:
    """
    Parse a run ID to extract its components.
    
    Args:
        run_id: Run ID string to parse
        
    Returns:
        Dictionary with parsed components:
        - project: Project name
        - model: Model name
        - timestamp: Timestamp string
        - hash: Short hash
        - is_valid: Whether the ID format is valid
    """
    try:
        parts = run_id.split("-")
        if len(parts) < 4:
            return {"is_valid": False}
        
        # Last part is the hash
        hash_part = parts[-1]
        
        # Second to last part is the timestamp
        timestamp_part = parts[-2]
        
        # Everything before timestamp is project-model
        project_model = "-".join(parts[:-2])
        
        # Split project and model (assume first part is project)
        project_model_parts = project_model.split("-", 1)
        if len(project_model_parts) < 2:
            return {"is_valid": False}
        
        project = project_model_parts[0]
        model = project_model_parts[1]
        
        return {
            "project": project,
            "model": model,
            "timestamp": timestamp_part,
            "hash": hash_part,
            "is_valid": True
        }
    except Exception:
        return {"is_valid": False}


def validate_run_id(run_id: str) -> bool:
    """
    Validate that a run ID has the correct format.
    
    Args:
        run_id: Run ID string to validate
        
    Returns:
        True if valid, False otherwise
    """
    parsed = parse_run_id(run_id)
    return parsed.get("is_valid", False)


def get_run_id_timestamp(run_id: str) -> Optional[float]:
    """
    Extract timestamp from a run ID.
    
    Args:
        run_id: Run ID string
        
    Returns:
        Timestamp as float, or None if invalid
    """
    parsed = parse_run_id(run_id)
    if not parsed.get("is_valid", False):
        return None
    
    try:
        # Convert YYYYMMDD-HHMMSS back to timestamp
        timestamp_str = parsed["timestamp"]
        dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
        return dt.timestamp()
    except ValueError:
        return None


def sort_run_ids_by_time(run_ids: List[str]) -> List[str]:
    """
    Sort run IDs by their creation time (oldest first).
    
    Args:
        run_ids: List of run ID strings
        
    Returns:
        Sorted list of run IDs
    """
    def get_timestamp(run_id: str) -> float:
        timestamp = get_run_id_timestamp(run_id)
        return timestamp if timestamp is not None else 0.0
    
    return sorted(run_ids, key=get_timestamp)


def filter_run_ids_by_project(run_ids: List[str], project: str) -> List[str]:
    """
    Filter run IDs by project name.
    
    Args:
        run_ids: List of run ID strings
        project: Project name to filter by
        
    Returns:
        List of run IDs for the specified project
    """
    filtered = []
    for run_id in run_ids:
        parsed = parse_run_id(run_id)
        if parsed.get("is_valid", False) and parsed.get("project") == project:
            filtered.append(run_id)
    
    return filtered


def filter_run_ids_by_model(run_ids: List[str], model: str) -> List[str]:
    """
    Filter run IDs by model name.
    
    Args:
        run_ids: List of run ID strings
        model: Model name to filter by
        
    Returns:
        List of run IDs for the specified model
    """
    filtered = []
    for run_id in run_ids:
        parsed = parse_run_id(run_id)
        if parsed.get("is_valid", False) and parsed.get("model") == model:
            filtered.append(run_id)
    
    return filtered
