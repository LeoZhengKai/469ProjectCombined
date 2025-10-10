"""
SharkTank project router for DSPy evaluation API.

This module provides endpoints specific to the SharkTank project:
- Pitch generation
- Fact-checking
- Quality assessment
- Evaluation runs
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
from core.ids import generate_run_id

router = APIRouter()
logger = get_logger(__name__)


class PitchRequest(BaseModel):
    """Request model for pitch generation."""
    product_facts: str
    guidelines: str
    target_quality: Optional[float] = 8.0
    target_factuality: Optional[float] = 8.5


class PitchResponse(BaseModel):
    """Response model for pitch generation."""
    pitch: str
    quality_score: float
    fact_score: float
    refinement_iterations: int
    meets_quality_threshold: bool
    meets_fact_threshold: bool
    quality_feedback: str
    fact_issues: str


class FactCheckRequest(BaseModel):
    """Request model for fact-checking."""
    pitch: str
    product_facts: str


class FactCheckResponse(BaseModel):
    """Response model for fact-checking."""
    score: float
    issues: str
    meets_threshold: bool


class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    pitch: str
    criteria: Optional[str] = None


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    score: float
    feedback: str
    meets_threshold: bool


@router.post("/generate-pitch", response_model=PitchResponse)
async def generate_pitch(request: PitchRequest):
    """
    Generate an investor-ready pitch.
    
    Args:
        request: Pitch generation request
        
    Returns:
        Generated pitch with quality and factuality scores
    """
    try:
        logger.info("Generating SharkTank pitch")
        
        # Import SharkTank program
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        # Generate pitch
        result = program.forward(
            product_facts=request.product_facts,
            guidelines=request.guidelines,
            target_quality=request.target_quality,
            target_factuality=request.target_factuality
        )
        
        return PitchResponse(
            pitch=result.pitch,
            quality_score=result.quality_score,
            fact_score=result.fact_score,
            refinement_iterations=result.refinement_iterations,
            meets_quality_threshold=result.meets_quality_threshold,
            meets_fact_threshold=result.meets_fact_threshold,
            quality_feedback=result.final_quality_feedback,
            fact_issues=result.final_fact_issues
        )
        
    except Exception as e:
        logger.error(f"Error generating pitch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_pitch(request: FactCheckRequest):
    """
    Fact-check a pitch against product facts.
    
    Args:
        request: Fact-checking request
        
    Returns:
        Fact-checking results
    """
    try:
        logger.info("Fact-checking SharkTank pitch")
        
        # Import SharkTank program
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        # Perform fact-check
        result = program.fact_check_pitch(
            pitch=request.pitch,
            product_facts=request.product_facts
        )
        
        return FactCheckResponse(
            score=result["score"],
            issues=result["issues"],
            meets_threshold=result["meets_threshold"]
        )
        
    except Exception as e:
        logger.error(f"Error fact-checking pitch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-quality", response_model=QualityAssessmentResponse)
async def assess_pitch_quality(request: QualityAssessmentRequest):
    """
    Assess the quality of a pitch.
    
    Args:
        request: Quality assessment request
        
    Returns:
        Quality assessment results
    """
    try:
        logger.info("Assessing SharkTank pitch quality")
        
        # Import SharkTank program
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        # Assess quality
        result = program.assess_pitch_quality(
            pitch=request.pitch,
            criteria=request.criteria
        )
        
        return QualityAssessmentResponse(
            score=result["score"],
            feedback=result["feedback"],
            meets_threshold=result["meets_threshold"]
        )
        
    except Exception as e:
        logger.error(f"Error assessing pitch quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refine-pitch")
async def refine_pitch(
    pitch: str,
    feedback: str
):
    """
    Refine a pitch based on feedback.
    
    Args:
        pitch: Original pitch to refine
        feedback: Feedback for improvement
        
    Returns:
        Refined pitch
    """
    try:
        logger.info("Refining SharkTank pitch")
        
        # Import SharkTank program
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        # Refine pitch
        refined_pitch = program.refine_pitch(
            pitch=pitch,
            feedback=feedback
        )
        
        return {
            "original_pitch": pitch,
            "refined_pitch": refined_pitch,
            "feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"Error refining pitch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_sharktank(
    model: str,
    optimizer: Optional[str] = None,
    num_samples: int = 100,
    background_tasks: BackgroundTasks = None
):
    """
    Run evaluation for SharkTank project.
    
    Args:
        model: Model name to evaluate
        optimizer: Optional optimizer to use
        num_samples: Number of samples to evaluate
        background_tasks: FastAPI background tasks
        
    Returns:
        Evaluation run information
    """
    try:
        # Generate run ID
        run_id = generate_run_id("sharktank", model)
        
        logger.info(f"Starting SharkTank evaluation: {run_id}")
        
        # Schedule background evaluation task
        if background_tasks:
            background_tasks.add_task(
                _run_sharktank_evaluation_background,
                run_id, model, optimizer, num_samples
            )
        
        return {
            "run_id": run_id,
            "project": "sharktank",
            "model": model,
            "optimizer": optimizer,
            "num_samples": num_samples,
            "status": "started",
            "message": "SharkTank evaluation started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting SharkTank evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_sharktank_evaluation_background(
    run_id: str,
    model: str,
    optimizer: Optional[str],
    num_samples: int
):
    """
    Run SharkTank evaluation in background.
    
    Args:
        run_id: Run identifier
        model: Model name
        optimizer: Optimizer name
        num_samples: Number of samples
    """
    try:
        logger.info(f"Running background SharkTank evaluation: {run_id}")
        
        # Import SharkTank program
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        # TODO: Implement actual evaluation logic
        # This is a placeholder for the evaluation process
        logger.info(f"SharkTank evaluation completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Error in background SharkTank evaluation {run_id}: {e}")


@router.get("/runs")
async def list_sharktank_runs():
    """
    List all SharkTank evaluation runs.
    
    Returns:
        List of SharkTank run information
    """
    try:
        run_ids = artifact_manager.list_runs(project="sharktank")
        
        runs = []
        for run_id in run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            runs.append(summary)
        
        return {
            "runs": runs,
            "total": len(runs),
            "project": "sharktank"
        }
        
    except Exception as e:
        logger.error(f"Error listing SharkTank runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_sharktank_run(run_id: str):
    """
    Get detailed information about a specific SharkTank run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Detailed SharkTank run information
    """
    try:
        summary = artifact_manager.get_run_summary(run_id)
        
        if not summary["config"] or summary["config"]["project"] != "sharktank":
            raise HTTPException(status_code=404, detail="SharkTank run not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SharkTank run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/program-info")
async def get_sharktank_program_info():
    """
    Get information about the SharkTank program configuration.
    
    Returns:
        Program information
    """
    try:
        from projects.sharktank.program import create_sharktank_program
        program = create_sharktank_program()
        
        return program.get_program_info()
        
    except Exception as e:
        logger.error(f"Error getting SharkTank program info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
