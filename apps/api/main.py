"""
FastAPI main application for DSPy evaluation framework.

This module provides HTTP endpoints for:
- Running model evaluations
- Comparing evaluation results
- Managing experiments and artifacts
- Health checks and system status

The API serves as a RESTful interface to the DSPy evaluation
framework, allowing remote execution of evaluations and comparisons.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, Optional, List
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.config import get_settings
from core.logging import setup_logging, get_logger
from core.artifacts import artifact_manager
from core.ids import generate_run_id

# Import routers
from routers import health, sharktank, aneeta, compare

# Initialize FastAPI app
app = FastAPI(
    title="DSPy Evaluation API",
    description="API for running DSPy model evaluations and comparisons",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redocs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(sharktank.router, prefix="/sharktank", tags=["sharktank"])
app.include_router(aneeta.router, prefix="/aneeta", tags=["aneeta"])
app.include_router(compare.router, prefix="/compare", tags=["compare"])


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting DSPy Evaluation API")
    
    # Initialize settings
    settings = get_settings()
    logger.info(f"Configuration loaded: {settings.model.model_name}")
    
    # Create necessary directories
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/runs").mkdir(exist_ok=True)
    Path("experiments/comparisons").mkdir(exist_ok=True)
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down DSPy Evaluation API")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DSPy Evaluation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "sharktank": "/sharktank",
            "aneeta": "/aneeta",
            "compare": "/compare"
        }
    }


@app.get("/status")
async def get_status():
    """Get system status and configuration."""
    try:
        settings = get_settings()
        
        # Check if experiments directory exists and is writable
        experiments_dir = Path("experiments")
        experiments_writable = experiments_dir.exists() and experiments_dir.is_dir()
        
        # Get recent runs count
        recent_runs = len(artifact_manager.list_runs())
        
        return {
            "status": "healthy",
            "configuration": {
                "model": settings.model.model_name,
                "provider": settings.model.provider,
                "experiments_dir": str(settings.experiments_dir),
                "rag_enabled": settings.rag.enabled
            },
            "system": {
                "experiments_writable": experiments_writable,
                "recent_runs": recent_runs,
                "total_comparisons": len(artifact_manager.list_comparisons())
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval")
async def run_evaluation(
    project: str,
    model: str,
    optimizer: Optional[str] = None,
    num_samples: int = 100,
    background_tasks: BackgroundTasks = None
):
    """
    Run a model evaluation for a specific project.
    
    Args:
        project: Project name (sharktank, aneeta)
        model: Model name to evaluate
        optimizer: Optional optimizer to use
        num_samples: Number of samples to evaluate
        background_tasks: FastAPI background tasks
        
    Returns:
        Evaluation run information
    """
    try:
        # Generate run ID
        run_id = generate_run_id(project, model)
        
        logger.info(f"Starting evaluation: {run_id}")
        
        # Schedule background evaluation task
        if background_tasks:
            background_tasks.add_task(
                _run_evaluation_background,
                run_id, project, model, optimizer, num_samples
            )
        
        return {
            "run_id": run_id,
            "project": project,
            "model": model,
            "optimizer": optimizer,
            "num_samples": num_samples,
            "status": "started",
            "message": "Evaluation started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_evaluation_background(
    run_id: str,
    project: str,
    model: str,
    optimizer: Optional[str],
    num_samples: int
):
    """
    Run evaluation in background.
    
    Args:
        run_id: Run identifier
        project: Project name
        model: Model name
        optimizer: Optimizer name
        num_samples: Number of samples
    """
    try:
        logger.info(f"Running background evaluation: {run_id}")
        
        # Import project-specific evaluation logic
        if project == "sharktank":
            from projects.sharktank.program import create_sharktank_program
            program = create_sharktank_program()
        elif project == "aneeta":
            from projects.aneeta.program import create_aneeta_program
            program = create_aneeta_program()
        else:
            raise ValueError(f"Unknown project: {project}")
        
        # TODO: Implement actual evaluation logic
        # This is a placeholder for the evaluation process
        logger.info(f"Evaluation completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Error in background evaluation {run_id}: {e}")


@app.get("/runs")
async def list_runs(project: Optional[str] = None):
    """
    List all evaluation runs.
    
    Args:
        project: Optional project filter
        
    Returns:
        List of run information
    """
    try:
        run_ids = artifact_manager.list_runs(project=project)
        
        runs = []
        for run_id in run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            runs.append(summary)
        
        return {
            "runs": runs,
            "total": len(runs),
            "project_filter": project
        }
        
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """
    Get detailed information about a specific run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Detailed run information
    """
    try:
        summary = artifact_manager.get_run_summary(run_id)
        
        if not summary["config"]:
            raise HTTPException(status_code=404, detail="Run not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """
    Delete a specific run and its artifacts.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Check if run exists
        summary = artifact_manager.get_run_summary(run_id)
        if not summary["config"]:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Delete run directory
        run_dir = Path(summary["run_dir"])
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        
        logger.info(f"Deleted run: {run_id}")
        
        return {
            "run_id": run_id,
            "status": "deleted",
            "message": "Run and artifacts deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/comparisons")
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


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
