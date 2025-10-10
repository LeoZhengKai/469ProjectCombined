"""
ANEETA DSPy program implementation.

This module implements the main DSPy program for the ANEETA project,
recreating the Multi-Agent System (MAS) with DSPy components for
question answering, safety checking, and local model optimization.

The program is designed to:
- Answer questions with appropriate context
- Perform safety checks on responses
- Coordinate multiple agents for complex tasks
- Optimize local model performance
- Ensure privacy and bias protection
"""

import dspy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .signatures import (
    QuestionAnsweringSignature,
    SafetyCheckSignature,
    MultiAgentSystemSignature,
    LocalModelOptimizationSignature,
    ContextRetrievalSignature,
    AnswerValidationSignature,
    PrivacyProtectionSignature,
    BiasDetectionSignature,
    QualityAssuranceSignature
)
from ...dspy.modules import (
    DraftModule, SafetyCheckModule, RAGModule, ValidationModule,
    ClassificationModule, QualityAssessmentModule
)


class ANEETAProgram(dspy.Module):
    """
    Main DSPy program for ANEETA question answering system.
    
    This program recreates the Multi-Agent System (MAS) using DSPy
    components for safe, accurate, and efficient question answering.
    """
    
    def __init__(
        self,
        qa_module: Optional[DraftModule] = None,
        safety_module: Optional[SafetyCheckModule] = None,
        rag_module: Optional[RAGModule] = None,
        validation_module: Optional[ValidationModule] = None,
        quality_module: Optional[QualityAssessmentModule] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ANEETA program.
        
        Args:
            qa_module: Module for question answering
            safety_module: Module for safety checking
            rag_module: Module for retrieval-augmented generation
            validation_module: Module for answer validation
            quality_module: Module for quality assessment
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        
        # Initialize modules with project-specific signatures
        self.qa_module = qa_module or DraftModule(
            signature=QuestionAnsweringSignature()
        )
        self.safety_module = safety_module or SafetyCheckModule(
            signature=SafetyCheckSignature()
        )
        self.rag_module = rag_module or RAGModule(
            signature=ContextRetrievalSignature(),
            top_k=self.config.get("rag_top_k", 5)
        )
        self.validation_module = validation_module or ValidationModule(
            signature=AnswerValidationSignature()
        )
        self.quality_module = quality_module or QualityAssessmentModule(
            signature=QualityAssuranceSignature()
        )
        
        # Safety and quality thresholds
        self.safety_threshold = self.config.get("safety_threshold", 7.0)
        self.quality_threshold = self.config.get("quality_threshold", 8.0)
        self.max_iterations = self.config.get("max_iterations", 3)
        
        # Privacy and bias settings
        self.privacy_level = self.config.get("privacy_level", "high")
        self.bias_detection_enabled = self.config.get("bias_detection_enabled", True)
    
    def forward(
        self,
        question: str,
        context: Optional[str] = None,
        safety_check: bool = True,
        quality_check: bool = True
    ) -> dspy.Prediction:
        """
        Answer a question with safety and quality checks.
        
        Args:
            question: Question to answer
            context: Optional context information
            safety_check: Whether to perform safety checking
            quality_check: Whether to perform quality assessment
            
        Returns:
            DSPy prediction with the answer and metrics
        """
        # Step 1: Retrieve context if RAG is enabled
        if self.rag_module and self.rag_module.retriever:
            rag_result = self.rag_module.forward(question=question, context=context or "")
            retrieved_context = rag_result.context
        else:
            retrieved_context = context or ""
        
        # Step 2: Generate initial answer
        qa_result = self.qa_module.forward(
            question=question,
            context=retrieved_context
        )
        current_answer = qa_result.answer
        
        # Step 3: Safety check
        safety_score = 0.0
        safety_concerns = []
        safety_recommendation = "safe"
        
        if safety_check:
            safety_result = self.safety_module.forward(content=current_answer)
            safety_score = float(safety_result.safety_score)
            safety_concerns = safety_result.concerns.split(", ") if safety_result.concerns else []
            safety_recommendation = safety_result.recommendation
            
            # If safety check fails, generate a safer response
            if safety_score < self.safety_threshold:
                current_answer = self._generate_safer_response(question, current_answer, safety_concerns)
        
        # Step 4: Quality assessment
        quality_score = 0.0
        quality_feedback = ""
        
        if quality_check:
            quality_result = self.quality_module.forward(
                content=current_answer,
                criteria="accuracy, completeness, clarity, helpfulness"
            )
            quality_score = float(quality_result.score)
            quality_feedback = quality_result.feedback
        
        # Step 5: Answer validation
        validation_result = self.validation_module.forward(
            content=current_answer,
            requirements="accurate, complete, safe, helpful"
        )
        is_valid = validation_result.is_valid
        validation_issues = validation_result.violations.split(", ") if validation_result.violations else []
        
        # Step 6: Privacy protection
        privacy_result = self._apply_privacy_protection(current_answer)
        protected_answer = privacy_result["protected_response"]
        privacy_concerns = privacy_result["concerns"]
        
        # Step 7: Bias detection
        bias_score = 0.0
        detected_bias = []
        
        if self.bias_detection_enabled:
            bias_result = self._detect_bias(protected_answer)
            bias_score = bias_result["bias_score"]
            detected_bias = bias_result["detected_bias"]
        
        # Create final prediction
        return dspy.Prediction(
            answer=protected_answer,
            safety_score=safety_score,
            quality_score=quality_score,
            bias_score=bias_score,
            is_valid=is_valid,
            safety_concerns=safety_concerns,
            quality_feedback=quality_feedback,
            validation_issues=validation_issues,
            privacy_concerns=privacy_concerns,
            detected_bias=detected_bias,
            safety_recommendation=safety_recommendation,
            retrieved_context=retrieved_context
        )
    
    def _generate_safer_response(
        self,
        question: str,
        original_answer: str,
        safety_concerns: List[str]
    ) -> str:
        """
        Generate a safer response based on safety concerns.
        
        Args:
            question: Original question
            original_answer: Original answer with safety issues
            safety_concerns: List of safety concerns
            
        Returns:
            Safer response
        """
        # Create a safety-focused prompt
        safety_prompt = f"""
        Original question: {question}
        Original answer: {original_answer}
        Safety concerns: {', '.join(safety_concerns)}
        
        Please provide a safer, more appropriate response that addresses the safety concerns
        while still being helpful and accurate.
        """
        
        # Use the QA module to generate a safer response
        safer_result = self.qa_module.forward(
            question=safety_prompt,
            context="Generate a safe and appropriate response"
        )
        
        return safer_result.answer
    
    def _apply_privacy_protection(self, response: str) -> Dict[str, Any]:
        """
        Apply privacy protection to a response.
        
        Args:
            response: Response to protect
            
        Returns:
            Dictionary with protected response and concerns
        """
        # Simple privacy protection (in a real system, this would be more sophisticated)
        privacy_concerns = []
        protected_response = response
        
        # Check for common privacy-sensitive patterns
        privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        import re
        for pattern in privacy_patterns:
            if re.search(pattern, response):
                privacy_concerns.append("Potential sensitive information detected")
                # Replace with placeholder
                protected_response = re.sub(pattern, "[REDACTED]", protected_response)
        
        return {
            "protected_response": protected_response,
            "concerns": privacy_concerns
        }
    
    def _detect_bias(self, response: str) -> Dict[str, Any]:
        """
        Detect bias in a response.
        
        Args:
            response: Response to check for bias
            
        Returns:
            Dictionary with bias score and detected issues
        """
        # Simple bias detection (in a real system, this would be more sophisticated)
        bias_score = 10.0  # Start with perfect score
        detected_bias = []
        
        # Check for common bias indicators
        bias_indicators = [
            "stereotypical",
            "discriminatory",
            "prejudiced",
            "unfair",
            "biased"
        ]
        
        response_lower = response.lower()
        for indicator in bias_indicators:
            if indicator in response_lower:
                bias_score -= 2.0
                detected_bias.append(f"Potential bias indicator: {indicator}")
        
        return {
            "bias_score": max(0.0, bias_score),
            "detected_bias": detected_bias
        }
    
    def answer_question_only(
        self,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Answer a question without safety or quality checks.
        
        Args:
            question: Question to answer
            context: Optional context information
            
        Returns:
            Answer string
        """
        qa_result = self.qa_module.forward(
            question=question,
            context=context or ""
        )
        return qa_result.answer
    
    def safety_check_response(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Perform safety check on a response.
        
        Args:
            response: Response to check
            
        Returns:
            Safety check results dictionary
        """
        result = self.safety_module.forward(content=response)
        
        return {
            "score": float(result.safety_score),
            "concerns": result.concerns.split(", ") if result.concerns else [],
            "recommendation": result.recommendation,
            "meets_threshold": float(result.safety_score) >= self.safety_threshold
        }
    
    def assess_response_quality(
        self,
        response: str,
        criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess the quality of a response.
        
        Args:
            response: Response to assess
            criteria: Quality criteria (uses default if not provided)
            
        Returns:
            Quality assessment results dictionary
        """
        if criteria is None:
            criteria = "accuracy, completeness, clarity, helpfulness"
        
        result = self.quality_module.forward(
            content=response,
            criteria=criteria
        )
        
        return {
            "score": float(result.score),
            "feedback": result.feedback,
            "meets_threshold": float(result.score) >= self.quality_threshold
        }
    
    def validate_response(
        self,
        response: str,
        requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a response against requirements.
        
        Args:
            response: Response to validate
            requirements: Requirements to check against
            
        Returns:
            Validation results dictionary
        """
        if requirements is None:
            requirements = "accurate, complete, safe, helpful"
        
        result = self.validation_module.forward(
            content=response,
            requirements=requirements
        )
        
        return {
            "is_valid": result.is_valid,
            "violations": result.violations.split(", ") if result.violations else []
        }
    
    def get_program_info(self) -> Dict[str, Any]:
        """
        Get information about the program configuration.
        
        Returns:
            Program information dictionary
        """
        return {
            "program_type": "ANEETAProgram",
            "safety_threshold": self.safety_threshold,
            "quality_threshold": self.quality_threshold,
            "max_iterations": self.max_iterations,
            "privacy_level": self.privacy_level,
            "bias_detection_enabled": self.bias_detection_enabled,
            "modules": {
                "qa": type(self.qa_module).__name__,
                "safety": type(self.safety_module).__name__,
                "rag": type(self.rag_module).__name__,
                "validation": type(self.validation_module).__name__,
                "quality": type(self.quality_module).__name__
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
        if "safety_threshold" in new_config:
            self.safety_threshold = new_config["safety_threshold"]
        if "quality_threshold" in new_config:
            self.quality_threshold = new_config["quality_threshold"]
        if "max_iterations" in new_config:
            self.max_iterations = new_config["max_iterations"]
        if "privacy_level" in new_config:
            self.privacy_level = new_config["privacy_level"]
        if "bias_detection_enabled" in new_config:
            self.bias_detection_enabled = new_config["bias_detection_enabled"]


class ANEETAProgramBuilder:
    """
    Builder class for creating ANEETA programs with custom configurations.
    
    Provides a fluent interface for configuring and building
    ANEETA programs with different modules and settings.
    """
    
    def __init__(self):
        """Initialize the program builder."""
        self.config = {}
        self.qa_module = None
        self.safety_module = None
        self.rag_module = None
        self.validation_module = None
        self.quality_module = None
    
    def with_safety_threshold(self, threshold: float) -> "ANEETAProgramBuilder":
        """
        Set the safety threshold.
        
        Args:
            threshold: Safety threshold value
            
        Returns:
            Self for method chaining
        """
        self.config["safety_threshold"] = threshold
        return self
    
    def with_quality_threshold(self, threshold: float) -> "ANEETAProgramBuilder":
        """
        Set the quality threshold.
        
        Args:
            threshold: Quality threshold value
            
        Returns:
            Self for method chaining
        """
        self.config["quality_threshold"] = threshold
        return self
    
    def with_max_iterations(self, iterations: int) -> "ANEETAProgramBuilder":
        """
        Set the maximum number of iterations.
        
        Args:
            iterations: Maximum iterations
            
        Returns:
            Self for method chaining
        """
        self.config["max_iterations"] = iterations
        return self
    
    def with_privacy_level(self, level: str) -> "ANEETAProgramBuilder":
        """
        Set the privacy protection level.
        
        Args:
            level: Privacy level (low, medium, high)
            
        Returns:
            Self for method chaining
        """
        self.config["privacy_level"] = level
        return self
    
    def with_bias_detection(self, enabled: bool) -> "ANEETAProgramBuilder":
        """
        Enable or disable bias detection.
        
        Args:
            enabled: Whether to enable bias detection
            
        Returns:
            Self for method chaining
        """
        self.config["bias_detection_enabled"] = enabled
        return self
    
    def with_qa_module(self, module: DraftModule) -> "ANEETAProgramBuilder":
        """
        Set the question answering module.
        
        Args:
            module: QA module instance
            
        Returns:
            Self for method chaining
        """
        self.qa_module = module
        return self
    
    def with_safety_module(self, module: SafetyCheckModule) -> "ANEETAProgramBuilder":
        """
        Set the safety check module.
        
        Args:
            module: Safety check module instance
            
        Returns:
            Self for method chaining
        """
        self.safety_module = module
        return self
    
    def with_rag_module(self, module: RAGModule) -> "ANEETAProgramBuilder":
        """
        Set the RAG module.
        
        Args:
            module: RAG module instance
            
        Returns:
            Self for method chaining
        """
        self.rag_module = module
        return self
    
    def with_validation_module(self, module: ValidationModule) -> "ANEETAProgramBuilder":
        """
        Set the validation module.
        
        Args:
            module: Validation module instance
            
        Returns:
            Self for method chaining
        """
        self.validation_module = module
        return self
    
    def with_quality_module(self, module: QualityAssessmentModule) -> "ANEETAProgramBuilder":
        """
        Set the quality assessment module.
        
        Args:
            module: Quality assessment module instance
            
        Returns:
            Self for method chaining
        """
        self.quality_module = module
        return self
    
    def build(self) -> ANEETAProgram:
        """
        Build the ANEETA program.
        
        Returns:
            Configured ANEETAProgram instance
        """
        return ANEETAProgram(
            qa_module=self.qa_module,
            safety_module=self.safety_module,
            rag_module=self.rag_module,
            validation_module=self.validation_module,
            quality_module=self.quality_module,
            config=self.config
        )


# Utility functions

def create_aneeta_program(config: Optional[Dict[str, Any]] = None) -> ANEETAProgram:
    """
    Create an ANEETA program with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ANEETAProgram instance
    """
    return ANEETAProgram(config=config)


def create_aneeta_program_from_config(config: Dict[str, Any]) -> ANEETAProgram:
    """
    Create an ANEETA program from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ANEETAProgram instance
    """
    builder = ANEETAProgramBuilder()
    
    # Apply configuration
    if "safety_threshold" in config:
        builder.with_safety_threshold(config["safety_threshold"])
    if "quality_threshold" in config:
        builder.with_quality_threshold(config["quality_threshold"])
    if "max_iterations" in config:
        builder.with_max_iterations(config["max_iterations"])
    if "privacy_level" in config:
        builder.with_privacy_level(config["privacy_level"])
    if "bias_detection_enabled" in config:
        builder.with_bias_detection(config["bias_detection_enabled"])
    
    return builder.build()
