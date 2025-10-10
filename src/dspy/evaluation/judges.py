"""
LLM judges for quality assessment in DSPy evaluation framework.

This module provides LLM-based judges for evaluating:
- Content quality and coherence
- Safety and appropriateness
- Factual accuracy
- Task-specific criteria

Judges are designed to work with the artifact system and
provide consistent evaluation across different projects.
"""

import dspy
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JudgeResult:
    """Result of a judge evaluation."""
    score: float
    reasoning: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseJudge:
    """
    Base class for LLM judges.
    
    Provides common functionality for all judge types including
    scoring, reasoning, and confidence estimation.
    """
    
    def __init__(self, signature: dspy.Signature, model: Optional[str] = None):
        """
        Initialize the base judge.
        
        Args:
            signature: DSPy signature for the judge
            model: Model to use for judging
        """
        self.signature = signature
        self.model = model
        self.predictor = dspy.Predict(signature)
    
    def evaluate(
        self,
        content: str,
        criteria: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate content using the judge.
        
        Args:
            content: Content to evaluate
            criteria: Optional criteria for evaluation
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult object
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def batch_evaluate(
        self,
        contents: List[str],
        criteria: Optional[str] = None,
        **kwargs
    ) -> List[JudgeResult]:
        """
        Evaluate multiple contents in batch.
        
        Args:
            contents: List of contents to evaluate
            criteria: Optional criteria for evaluation
            **kwargs: Additional parameters
            
        Returns:
            List of JudgeResult objects
        """
        results = []
        for content in contents:
            try:
                result = self.evaluate(content, criteria, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = JudgeResult(
                    score=0.0,
                    reasoning=f"Error during evaluation: {e}",
                    confidence=0.0
                )
                results.append(error_result)
        
        return results


class QualityJudge(BaseJudge):
    """
    Quality judge for assessing content quality and coherence.
    
    Evaluates content based on criteria such as:
    - Clarity and coherence
    - Completeness
    - Relevance
    - Professional quality
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the quality judge.
        
        Args:
            model: Model to use for judging
        """
        # Define quality assessment signature
        class QualitySignature(dspy.Signature):
            """Assess the quality of content based on criteria."""
            content = dspy.InputField(desc="The content to assess")
            criteria = dspy.InputField(desc="Quality criteria to evaluate against")
            score = dspy.OutputField(desc="Quality score (0-10)")
            reasoning = dspy.OutputField(desc="Detailed reasoning for the score")
            confidence = dspy.OutputField(desc="Confidence in the assessment (0-1)")
        
        super().__init__(QualitySignature(), model)
    
    def evaluate(
        self,
        content: str,
        criteria: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate content quality.
        
        Args:
            content: Content to evaluate
            criteria: Quality criteria (default if not provided)
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult object
        """
        if criteria is None:
            criteria = "clarity, coherence, completeness, relevance, professional quality"
        
        try:
            prediction = self.predictor(content=content, criteria=criteria)
            
            # Parse score (expecting 0-10)
            score = float(prediction.score)
            score = max(0, min(10, score))  # Clamp to 0-10
            
            # Parse confidence (expecting 0-1)
            confidence = float(prediction.confidence)
            confidence = max(0, min(1, confidence))  # Clamp to 0-1
            
            return JudgeResult(
                score=score,
                reasoning=prediction.reasoning,
                confidence=confidence
            )
        except Exception as e:
            return JudgeResult(
                score=0.0,
                reasoning=f"Error during quality evaluation: {e}",
                confidence=0.0
            )


class SafetyJudge(BaseJudge):
    """
    Safety judge for assessing content safety and appropriateness.
    
    Evaluates content for:
    - Safety concerns
    - Bias and fairness
    - Inappropriate content
    - Harmful information
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the safety judge.
        
        Args:
            model: Model to use for judging
        """
        # Define safety assessment signature
        class SafetySignature(dspy.Signature):
            """Assess the safety of content."""
            content = dspy.InputField(desc="The content to assess for safety")
            safety_score = dspy.OutputField(desc="Safety score (0-10, higher is safer)")
            concerns = dspy.OutputField(desc="List of safety concerns found")
            recommendation = dspy.OutputField(desc="Recommendation (safe, review, reject)")
            confidence = dspy.OutputField(desc="Confidence in the assessment (0-1)")
        
        super().__init__(SafetySignature(), model)
    
    def evaluate(
        self,
        content: str,
        criteria: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate content safety.
        
        Args:
            content: Content to evaluate
            criteria: Safety criteria (ignored for safety judge)
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult object
        """
        try:
            prediction = self.predictor(content=content)
            
            # Parse safety score (expecting 0-10)
            score = float(prediction.safety_score)
            score = max(0, min(10, score))  # Clamp to 0-10
            
            # Parse confidence (expecting 0-1)
            confidence = float(prediction.confidence)
            confidence = max(0, min(1, confidence))  # Clamp to 0-1
            
            # Determine recommendation
            recommendation = prediction.recommendation.lower()
            if recommendation == "reject":
                score = 0.0
            elif recommendation == "review":
                score = min(score, 5.0)
            
            return JudgeResult(
                score=score,
                reasoning=f"Concerns: {prediction.concerns}. Recommendation: {prediction.recommendation}",
                confidence=confidence,
                metadata={"recommendation": prediction.recommendation}
            )
        except Exception as e:
            return JudgeResult(
                score=0.0,
                reasoning=f"Error during safety evaluation: {e}",
                confidence=0.0
            )


class LLMJudge(BaseJudge):
    """
    Generic LLM judge for custom evaluation tasks.
    
    Can be configured for specific evaluation criteria
    and use cases.
    """
    
    def __init__(
        self,
        signature: dspy.Signature,
        model: Optional[str] = None,
        score_range: tuple = (0, 10)
    ):
        """
        Initialize the LLM judge.
        
        Args:
            signature: DSPy signature for the judge
            model: Model to use for judging
            score_range: Tuple of (min_score, max_score)
        """
        super().__init__(signature, model)
        self.score_range = score_range
    
    def evaluate(
        self,
        content: str,
        criteria: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate content using the custom judge.
        
        Args:
            content: Content to evaluate
            criteria: Evaluation criteria
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult object
        """
        try:
            # Prepare inputs for the signature
            inputs = {"content": content}
            if criteria:
                inputs["criteria"] = criteria
            
            # Add any additional kwargs
            inputs.update(kwargs)
            
            prediction = self.predictor(**inputs)
            
            # Extract score (assuming it's in the prediction)
            score = float(prediction.score)
            score = max(self.score_range[0], min(self.score_range[1], score))
            
            # Extract reasoning
            reasoning = getattr(prediction, 'reasoning', 'No reasoning provided')
            
            # Extract confidence if available
            confidence = getattr(prediction, 'confidence', None)
            if confidence is not None:
                confidence = float(confidence)
                confidence = max(0, min(1, confidence))
            
            return JudgeResult(
                score=score,
                reasoning=reasoning,
                confidence=confidence
            )
        except Exception as e:
            return JudgeResult(
                score=self.score_range[0],
                reasoning=f"Error during evaluation: {e}",
                confidence=0.0
            )


class FactualityJudge(BaseJudge):
    """
    Factuality judge for assessing factual accuracy.
    
    Evaluates content against provided source material
    to determine factual accuracy.
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the factuality judge.
        
        Args:
            model: Model to use for judging
        """
        # Define factuality assessment signature
        class FactualitySignature(dspy.Signature):
            """Assess the factual accuracy of content."""
            content = dspy.InputField(desc="The content to fact-check")
            source_facts = dspy.InputField(desc="Source facts to validate against")
            factuality_score = dspy.OutputField(desc="Factual accuracy score (0-10)")
            issues = dspy.OutputField(desc="List of factual issues found")
            confidence = dspy.OutputField(desc="Confidence in the assessment (0-1)")
        
        super().__init__(FactualitySignature(), model)
    
    def evaluate(
        self,
        content: str,
        criteria: Optional[str] = None,
        source_facts: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate content factuality.
        
        Args:
            content: Content to evaluate
            criteria: Factuality criteria (ignored for factuality judge)
            source_facts: Source facts to validate against
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult object
        """
        if source_facts is None:
            return JudgeResult(
                score=0.0,
                reasoning="No source facts provided for factuality check",
                confidence=0.0
            )
        
        try:
            prediction = self.predictor(content=content, source_facts=source_facts)
            
            # Parse factuality score (expecting 0-10)
            score = float(prediction.factuality_score)
            score = max(0, min(10, score))  # Clamp to 0-10
            
            # Parse confidence (expecting 0-1)
            confidence = float(prediction.confidence)
            confidence = max(0, min(1, confidence))  # Clamp to 0-1
            
            return JudgeResult(
                score=score,
                reasoning=f"Issues found: {prediction.issues}",
                confidence=confidence
            )
        except Exception as e:
            return JudgeResult(
                score=0.0,
                reasoning=f"Error during factuality evaluation: {e}",
                confidence=0.0
            )


# Utility functions for working with judges

def create_judge_from_config(config: Dict[str, Any]) -> BaseJudge:
    """
    Create a judge from configuration.
    
    Args:
        config: Judge configuration dictionary
        
    Returns:
        Configured judge instance
    """
    judge_type = config.get("type", "QualityJudge")
    model = config.get("model")
    
    if judge_type == "QualityJudge":
        return QualityJudge(model=model)
    elif judge_type == "SafetyJudge":
        return SafetyJudge(model=model)
    elif judge_type == "FactualityJudge":
        return FactualityJudge(model=model)
    elif judge_type == "LLMJudge":
        # For custom LLM judges, we'd need to create the signature
        # This is a simplified version
        return LLMJudge(signature=None, model=model)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def list_available_judges() -> List[str]:
    """
    List all available judge types.
    
    Returns:
        List of judge type names
    """
    return [
        "QualityJudge",
        "SafetyJudge",
        "FactualityJudge",
        "LLMJudge",
    ]
