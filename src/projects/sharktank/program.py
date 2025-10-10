"""
SharkTank DSPy program implementation.

This module implements the main DSPy program for the SharkTank project,
composing Draft→(tools)→FactCheck→Refine modules to create investor-ready pitches.

The program is designed to:
- Generate initial pitches from product facts
- Fact-check pitches against source material
- Assess pitch quality using judges
- Refine pitches based on feedback
- Optimize the entire pipeline using DSPy optimizers
"""

import dspy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .signatures import (
    PitchGenerationSignature,
    PitchFactCheckSignature,
    PitchQualitySignature,
    PitchRefinementSignature
)
from ...dspy.modules import (
    DraftModule, FactCheckModule, QualityAssessmentModule, RefinementModule
)


class SharkTankProgram(dspy.Module):
    """
    Main DSPy program for SharkTank pitch generation.
    
    This program composes multiple modules to create a complete
    pipeline for generating, fact-checking, and refining pitches.
    """
    
    def __init__(
        self,
        draft_module: Optional[DraftModule] = None,
        fact_check_module: Optional[FactCheckModule] = None,
        quality_module: Optional[QualityAssessmentModule] = None,
        refinement_module: Optional[RefinementModule] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the SharkTank program.
        
        Args:
            draft_module: Module for generating initial pitches
            fact_check_module: Module for fact-checking pitches
            quality_module: Module for quality assessment
            refinement_module: Module for pitch refinement
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        
        # Initialize modules with project-specific signatures
        self.draft_module = draft_module or DraftModule(
            signature=PitchGenerationSignature()
        )
        self.fact_check_module = fact_check_module or FactCheckModule(
            signature=PitchFactCheckSignature()
        )
        self.quality_module = quality_module or QualityAssessmentModule(
            signature=PitchQualitySignature()
        )
        self.refinement_module = refinement_module or RefinementModule(
            signature=PitchRefinementSignature()
        )
        
        # Quality thresholds
        self.quality_threshold = self.config.get("quality_threshold", 8.0)
        self.fact_threshold = self.config.get("fact_threshold", 8.5)
        self.max_refinement_iterations = self.config.get("max_refinement_iterations", 3)
    
    def forward(
        self,
        product_facts: str,
        guidelines: str,
        target_quality: Optional[float] = None,
        target_factuality: Optional[float] = None
    ) -> dspy.Prediction:
        """
        Generate a high-quality pitch through the complete pipeline.
        
        Args:
            product_facts: Product facts to base the pitch on
            guidelines: Guidelines for pitch structure and content
            target_quality: Target quality score (uses threshold if not provided)
            target_factuality: Target factuality score (uses threshold if not provided)
            
        Returns:
            DSPy prediction with the final pitch and metrics
        """
        # Set target scores
        if target_quality is None:
            target_quality = self.quality_threshold
        if target_factuality is None:
            target_factuality = self.fact_threshold
        
        # Step 1: Generate initial pitch
        draft_result = self.draft_module.forward(
            product_facts=product_facts,
            guidelines=guidelines
        )
        current_pitch = draft_result.pitch
        
        # Step 2: Fact-check the pitch
        fact_check_result = self.fact_check_module.forward(
            pitch=current_pitch,
            product_facts=product_facts
        )
        fact_score = float(fact_check_result.score)
        
        # Step 3: Assess quality
        quality_result = self.quality_module.forward(
            content=current_pitch,
            criteria="clarity, persuasiveness, market opportunity, business model, financial projections"
        )
        quality_score = float(quality_result.score)
        
        # Step 4: Refine if needed
        refinement_iterations = 0
        refinement_history = []
        
        while (quality_score < target_quality or fact_score < target_factuality) and \
              refinement_iterations < self.max_refinement_iterations:
            
            # Generate feedback for refinement
            feedback = self._generate_refinement_feedback(
                quality_score, fact_score, target_quality, target_factuality,
                quality_result.feedback, fact_check_result.issues
            )
            
            # Refine the pitch
            refinement_result = self.refinement_module.forward(
                original_content=current_pitch,
                feedback=feedback
            )
            current_pitch = refinement_result.refined_content
            
            # Re-evaluate the refined pitch
            fact_check_result = self.fact_check_module.forward(
                pitch=current_pitch,
                product_facts=product_facts
            )
            fact_score = float(fact_check_result.score)
            
            quality_result = self.quality_module.forward(
                content=current_pitch,
                criteria="clarity, persuasiveness, market opportunity, business model, financial projections"
            )
            quality_score = float(quality_result.score)
            
            refinement_iterations += 1
            refinement_history.append({
                "iteration": refinement_iterations,
                "quality_score": quality_score,
                "fact_score": fact_score,
                "pitch": current_pitch
            })
        
        # Create final prediction
        return dspy.Prediction(
            pitch=current_pitch,
            quality_score=quality_score,
            fact_score=fact_score,
            refinement_iterations=refinement_iterations,
            refinement_history=refinement_history,
            meets_quality_threshold=quality_score >= target_quality,
            meets_fact_threshold=fact_score >= target_factuality,
            final_quality_feedback=quality_result.feedback,
            final_fact_issues=fact_check_result.issues
        )
    
    def _generate_refinement_feedback(
        self,
        quality_score: float,
        fact_score: float,
        target_quality: float,
        target_factuality: float,
        quality_feedback: str,
        fact_issues: str
    ) -> str:
        """
        Generate feedback for pitch refinement.
        
        Args:
            quality_score: Current quality score
            fact_score: Current factuality score
            target_quality: Target quality score
            target_factuality: Target factuality score
            quality_feedback: Quality assessment feedback
            fact_issues: Factual issues identified
            
        Returns:
            Refinement feedback string
        """
        feedback_parts = []
        
        if quality_score < target_quality:
            feedback_parts.append(f"Quality needs improvement (current: {quality_score:.1f}, target: {target_quality:.1f})")
            feedback_parts.append(f"Quality feedback: {quality_feedback}")
        
        if fact_score < target_factuality:
            feedback_parts.append(f"Factuality needs improvement (current: {fact_score:.1f}, target: {target_factuality:.1f})")
            feedback_parts.append(f"Factual issues: {fact_issues}")
        
        return ". ".join(feedback_parts)
    
    def generate_pitch_only(
        self,
        product_facts: str,
        guidelines: str
    ) -> str:
        """
        Generate a pitch without fact-checking or refinement.
        
        Args:
            product_facts: Product facts to base the pitch on
            guidelines: Guidelines for pitch structure and content
            
        Returns:
            Generated pitch string
        """
        draft_result = self.draft_module.forward(
            product_facts=product_facts,
            guidelines=guidelines
        )
        return draft_result.pitch
    
    def fact_check_pitch(
        self,
        pitch: str,
        product_facts: str
    ) -> Dict[str, Any]:
        """
        Fact-check a pitch against product facts.
        
        Args:
            pitch: Pitch to fact-check
            product_facts: Product facts to validate against
            
        Returns:
            Fact-check results dictionary
        """
        result = self.fact_check_module.forward(
            pitch=pitch,
            product_facts=product_facts
        )
        
        return {
            "score": float(result.score),
            "issues": result.issues,
            "meets_threshold": float(result.score) >= self.fact_threshold
        }
    
    def assess_pitch_quality(
        self,
        pitch: str,
        criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess the quality of a pitch.
        
        Args:
            pitch: Pitch to assess
            criteria: Quality criteria (uses default if not provided)
            
        Returns:
            Quality assessment results dictionary
        """
        if criteria is None:
            criteria = "clarity, persuasiveness, market opportunity, business model, financial projections"
        
        result = self.quality_module.forward(
            content=pitch,
            criteria=criteria
        )
        
        return {
            "score": float(result.score),
            "feedback": result.feedback,
            "meets_threshold": float(result.score) >= self.quality_threshold
        }
    
    def refine_pitch(
        self,
        pitch: str,
        feedback: str
    ) -> str:
        """
        Refine a pitch based on feedback.
        
        Args:
            pitch: Original pitch to refine
            feedback: Feedback for improvement
            
        Returns:
            Refined pitch string
        """
        result = self.refinement_module.forward(
            original_content=pitch,
            feedback=feedback
        )
        return result.refined_content
    
    def get_program_info(self) -> Dict[str, Any]:
        """
        Get information about the program configuration.
        
        Returns:
            Program information dictionary
        """
        return {
            "program_type": "SharkTankProgram",
            "quality_threshold": self.quality_threshold,
            "fact_threshold": self.fact_threshold,
            "max_refinement_iterations": self.max_refinement_iterations,
            "modules": {
                "draft": type(self.draft_module).__name__,
                "fact_check": type(self.fact_check_module).__name__,
                "quality": type(self.quality_module).__name__,
                "refinement": type(self.refinement_module).__name__
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update program configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update thresholds if provided
        if "quality_threshold" in new_config:
            self.quality_threshold = new_config["quality_threshold"]
        if "fact_threshold" in new_config:
            self.fact_threshold = new_config["fact_threshold"]
        if "max_refinement_iterations" in new_config:
            self.max_refinement_iterations = new_config["max_refinement_iterations"]


class SharkTankProgramBuilder:
    """
    Builder class for creating SharkTank programs with custom configurations.
    
    Provides a fluent interface for configuring and building
    SharkTank programs with different modules and settings.
    """
    
    def __init__(self):
        """Initialize the program builder."""
        self.config = {}
        self.draft_module = None
        self.fact_check_module = None
        self.quality_module = None
        self.refinement_module = None
    
    def with_quality_threshold(self, threshold: float) -> "SharkTankProgramBuilder":
        """
        Set the quality threshold.
        
        Args:
            threshold: Quality threshold value
            
        Returns:
            Self for method chaining
        """
        self.config["quality_threshold"] = threshold
        return self
    
    def with_fact_threshold(self, threshold: float) -> "SharkTankProgramBuilder":
        """
        Set the factuality threshold.
        
        Args:
            threshold: Factuality threshold value
            
        Returns:
            Self for method chaining
        """
        self.config["fact_threshold"] = threshold
        return self
    
    def with_max_refinement_iterations(self, iterations: int) -> "SharkTankProgramBuilder":
        """
        Set the maximum number of refinement iterations.
        
        Args:
            iterations: Maximum refinement iterations
            
        Returns:
            Self for method chaining
        """
        self.config["max_refinement_iterations"] = iterations
        return self
    
    def with_draft_module(self, module: DraftModule) -> "SharkTankProgramBuilder":
        """
        Set the draft module.
        
        Args:
            module: Draft module instance
            
        Returns:
            Self for method chaining
        """
        self.draft_module = module
        return self
    
    def with_fact_check_module(self, module: FactCheckModule) -> "SharkTankProgramBuilder":
        """
        Set the fact-check module.
        
        Args:
            module: Fact-check module instance
            
        Returns:
            Self for method chaining
        """
        self.fact_check_module = module
        return self
    
    def with_quality_module(self, module: QualityAssessmentModule) -> "SharkTankProgramBuilder":
        """
        Set the quality assessment module.
        
        Args:
            module: Quality assessment module instance
            
        Returns:
            Self for method chaining
        """
        self.quality_module = module
        return self
    
    def with_refinement_module(self, module: RefinementModule) -> "SharkTankProgramBuilder":
        """
        Set the refinement module.
        
        Args:
            module: Refinement module instance
            
        Returns:
            Self for method chaining
        """
        self.refinement_module = module
        return self
    
    def build(self) -> SharkTankProgram:
        """
        Build the SharkTank program.
        
        Returns:
            Configured SharkTankProgram instance
        """
        return SharkTankProgram(
            draft_module=self.draft_module,
            fact_check_module=self.fact_check_module,
            quality_module=self.quality_module,
            refinement_module=self.refinement_module,
            config=self.config
        )


# Utility functions

def create_sharktank_program(config: Optional[Dict[str, Any]] = None) -> SharkTankProgram:
    """
    Create a SharkTank program with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SharkTankProgram instance
    """
    return SharkTankProgram(config=config)


def create_sharktank_program_from_config(config: Dict[str, Any]) -> SharkTankProgram:
    """
    Create a SharkTank program from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SharkTankProgram instance
    """
    builder = SharkTankProgramBuilder()
    
    # Apply configuration
    if "quality_threshold" in config:
        builder.with_quality_threshold(config["quality_threshold"])
    if "fact_threshold" in config:
        builder.with_fact_threshold(config["fact_threshold"])
    if "max_refinement_iterations" in config:
        builder.with_max_refinement_iterations(config["max_refinement_iterations"])
    
    return builder.build()
