"""
ANEETA project router for DSPy evaluation API.

This module provides endpoints specific to the ANEETA project:
- Question answering
- Safety checking
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


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str
    context: Optional[str] = None
    safety_check: bool = True
    quality_check: bool = True


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    safety_score: float
    quality_score: float
    bias_score: float
    is_valid: bool
    safety_concerns: List[str]
    quality_feedback: str
    validation_issues: List[str]
    privacy_concerns: List[str]
    detected_bias: List[str]
    safety_recommendation: str
    retrieved_context: str


class SafetyCheckRequest(BaseModel):
    """Request model for safety checking."""
    response: str


class SafetyCheckResponse(BaseModel):
    """Response model for safety checking."""
    score: float
    concerns: List[str]
    recommendation: str
    meets_threshold: bool


class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    response: str
    criteria: Optional[str] = None


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    score: float
    feedback: str
    meets_threshold: bool


@router.post("/answer-question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question with safety and quality checks.
    
    Args:
        request: Question answering request
        
    Returns:
        Answer with safety and quality scores
    """
    try:
        logger.info("Answering ANEETA question")
        
        # Import ANEETA program
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        # Answer question
        result = program.forward(
            question=request.question,
            context=request.context,
            safety_check=request.safety_check,
            quality_check=request.quality_check
        )
        
        return QuestionResponse(
            answer=result.answer,
            safety_score=result.safety_score,
            quality_score=result.quality_score,
            bias_score=result.bias_score,
            is_valid=result.is_valid,
            safety_concerns=result.safety_concerns,
            quality_feedback=result.quality_feedback,
            validation_issues=result.validation_issues,
            privacy_concerns=result.privacy_concerns,
            detected_bias=result.detected_bias,
            safety_recommendation=result.safety_recommendation,
            retrieved_context=result.retrieved_context
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safety-check", response_model=SafetyCheckResponse)
async def safety_check_response(request: SafetyCheckRequest):
    """
    Check a response for safety concerns.
    
    Args:
        request: Safety checking request
        
    Returns:
        Safety checking results
    """
    try:
        logger.info("Safety checking ANEETA response")
        
        # Import ANEETA program
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        # Perform safety check
        result = program.safety_check_response(
            response=request.response
        )
        
        return SafetyCheckResponse(
            score=result["score"],
            concerns=result["concerns"],
            recommendation=result["recommendation"],
            meets_threshold=result["meets_threshold"]
        )
        
    except Exception as e:
        logger.error(f"Error safety checking response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-quality", response_model=QualityAssessmentResponse)
async def assess_response_quality(request: QualityAssessmentRequest):
    """
    Assess the quality of a response.
    
    Args:
        request: Quality assessment request
        
    Returns:
        Quality assessment results
    """
    try:
        logger.info("Assessing ANEETA response quality")
        
        # Import ANEETA program
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        # Assess quality
        result = program.assess_response_quality(
            response=request.response,
            criteria=request.criteria
        )
        
        return QualityAssessmentResponse(
            score=result["score"],
            feedback=result["feedback"],
            meets_threshold=result["meets_threshold"]
        )
        
    except Exception as e:
        logger.error(f"Error assessing response quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-response")
async def validate_response(
    response: str,
    requirements: Optional[str] = None
):
    """
    Validate a response against requirements.
    
    Args:
        response: Response to validate
        requirements: Requirements to check against
        
    Returns:
        Validation results
    """
    try:
        logger.info("Validating ANEETA response")
        
        # Import ANEETA program
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        # Validate response
        result = program.validate_response(
            response=response,
            requirements=requirements
        )
        
        return {
            "response": response,
            "is_valid": result["is_valid"],
            "violations": result["violations"],
            "requirements": requirements or "accurate, complete, safe, helpful"
        }
        
    except Exception as e:
        logger.error(f"Error validating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_aneeta(
    model: str,
    optimizer: Optional[str] = None,
    num_samples: int = 100,
    background_tasks: BackgroundTasks = None
):
    """
    Run evaluation for ANEETA project.
    
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
        run_id = generate_run_id("aneeta", model)
        
        logger.info(f"Starting ANEETA evaluation: {run_id}")
        
        # Schedule background evaluation task
        if background_tasks:
            background_tasks.add_task(
                _run_aneeta_evaluation_background,
                run_id, model, optimizer, num_samples
            )
        
        return {
            "run_id": run_id,
            "project": "aneeta",
            "model": model,
            "optimizer": optimizer,
            "num_samples": num_samples,
            "status": "started",
            "message": "ANEETA evaluation started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting ANEETA evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_aneeta_evaluation_background(
    run_id: str,
    model: str,
    optimizer: Optional[str],
    num_samples: int
):
    """
    Run ANEETA evaluation in background.
    
    Args:
        run_id: Run identifier
        model: Model name
        optimizer: Optimizer name
        num_samples: Number of samples
    """
    try:
        logger.info(f"Running background ANEETA evaluation: {run_id}")
        
        # Import ANEETA program
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        # TODO: Implement actual evaluation logic
        # This is a placeholder for the evaluation process
        logger.info(f"ANEETA evaluation completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Error in background ANEETA evaluation {run_id}: {e}")


@router.get("/runs")
async def list_aneeta_runs():
    """
    List all ANEETA evaluation runs.
    
    Returns:
        List of ANEETA run information
    """
    try:
        run_ids = artifact_manager.list_runs(project="aneeta")
        
        runs = []
        for run_id in run_ids:
            summary = artifact_manager.get_run_summary(run_id)
            runs.append(summary)
        
        return {
            "runs": runs,
            "total": len(runs),
            "project": "aneeta"
        }
        
    except Exception as e:
        logger.error(f"Error listing ANEETA runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_aneeta_run(run_id: str):
    """
    Get detailed information about a specific ANEETA run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Detailed ANEETA run information
    """
    try:
        summary = artifact_manager.get_run_summary(run_id)
        
        if not summary["config"] or summary["config"]["project"] != "aneeta":
            raise HTTPException(status_code=404, detail="ANEETA run not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ANEETA run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/program-info")
async def get_aneeta_program_info():
    """
    Get information about the ANEETA program configuration.
    
    Returns:
        Program information
    """
    try:
        from projects.aneeta.program import create_aneeta_program
        program = create_aneeta_program()
        
        return program.get_program_info()
        
    except Exception as e:
        logger.error(f"Error getting ANEETA program info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
