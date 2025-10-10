"""
Health check router for DSPy evaluation API.

This module provides health check endpoints for monitoring
the API service status and dependencies.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.config import get_settings
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "dspy-eval-api",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with system information.
    
    Returns:
        Detailed health status
    """
    try:
        settings = get_settings()
        
        # Check if experiments directory exists and is writable
        experiments_dir = Path("experiments")
        experiments_status = "healthy" if experiments_dir.exists() and experiments_dir.is_dir() else "unhealthy"
        
        # Check if datasets directory exists
        datasets_dir = Path("datasets")
        datasets_status = "healthy" if datasets_dir.exists() and datasets_dir.is_dir() else "unhealthy"
        
        return {
            "status": "healthy",
            "service": "dspy-eval-api",
            "version": "1.0.0",
            "configuration": {
                "model_provider": settings.model.provider,
                "model_name": settings.model.model_name,
                "rag_enabled": settings.rag.enabled,
                "mlflow_enabled": settings.mlflow_tracking_uri is not None
            },
            "dependencies": {
                "experiments_dir": experiments_status,
                "datasets_dir": datasets_status,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for service startup.
    
    Returns:
        Readiness status
    """
    try:
        settings = get_settings()
        
        # Check critical dependencies
        experiments_dir = Path("experiments")
        if not experiments_dir.exists():
            raise HTTPException(status_code=503, detail="Experiments directory not available")
        
        # Check if we can create a test file
        test_file = experiments_dir / "health_check_test.tmp"
        try:
            test_file.write_text("health check")
            test_file.unlink()
        except Exception:
            raise HTTPException(status_code=503, detail="Experiments directory not writable")
        
        return {
            "status": "ready",
            "service": "dspy-eval-api",
            "message": "Service is ready to accept requests"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {e}")


@router.get("/live")
async def liveness_check():
    """
    Liveness check for service monitoring.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "service": "dspy-eval-api",
        "timestamp": "2024-01-01T00:00:00Z"  # This would be actual timestamp in production
    }
