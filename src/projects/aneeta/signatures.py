"""
ANEETA project-specific signatures for DSPy evaluation framework.

This module contains DSPy signatures specific to the ANEETA project:
- Question answering signatures
- Safety check signatures
- Multi-agent system signatures
- Local model optimization signatures

Signatures are designed to work with the ANEETA program and
provide clear input/output contracts for optimization.
"""

import dspy
from typing import List, Optional


class QuestionAnsweringSignature(dspy.Signature):
    """
    Answer questions based on provided context and knowledge.
    
    This signature is used to answer questions in the ANEETA system
    with appropriate context and safety considerations.
    """
    question = dspy.InputField(desc="The question to be answered")
    context = dspy.InputField(desc="Relevant context information")
    answer = dspy.OutputField(desc="The answer to the question")


class SafetyCheckSignature(dspy.Signature):
    """
    Check content for safety concerns and appropriateness.
    
    This signature evaluates content for safety issues including:
    - Harmful content
    - Bias and fairness
    - Inappropriate material
    - Privacy concerns
    """
    content = dspy.InputField(desc="The content to check for safety")
    safety_score = dspy.OutputField(desc="Safety score (0-10, higher is safer)")
    concerns = dspy.OutputField(desc="List of safety concerns found")
    recommendation = dspy.OutputField(desc="Recommendation (safe, review, reject)")


class MultiAgentSystemSignature(dspy.Signature):
    """
    Coordinate multiple agents for complex task completion.
    
    This signature manages the coordination of multiple agents
    in the ANEETA system for handling complex queries.
    """
    task = dspy.InputField(desc="The task to be completed")
    available_agents = dspy.InputField(desc="List of available agents")
    coordination_plan = dspy.OutputField(desc="Plan for agent coordination")
    execution_order = dspy.OutputField(desc="Order of agent execution")


class LocalModelOptimizationSignature(dspy.Signature):
    """
    Optimize local model performance and efficiency.
    
    This signature provides guidance for optimizing local models
    including quantization and performance tuning.
    """
    model_config = dspy.InputField(desc="Current model configuration")
    optimization_goals = dspy.InputField(desc="Optimization goals (speed, quality, memory)")
    optimized_config = dspy.OutputField(desc="Optimized model configuration")
    expected_improvements = dspy.OutputField(desc="Expected performance improvements")


class ContextRetrievalSignature(dspy.Signature):
    """
    Retrieve relevant context for question answering.
    
    This signature finds and retrieves relevant context information
    from knowledge bases and documents.
    """
    question = dspy.InputField(desc="The question to find context for")
    knowledge_base = dspy.InputField(desc="Available knowledge base")
    retrieved_context = dspy.OutputField(desc="Retrieved relevant context")
    relevance_score = dspy.OutputField(desc="Relevance score for retrieved context")


class AnswerValidationSignature(dspy.Signature):
    """
    Validate answers for accuracy and completeness.
    
    This signature checks answers for correctness and completeness
    based on the provided context and question.
    """
    question = dspy.InputField(desc="The original question")
    answer = dspy.InputField(desc="The answer to validate")
    context = dspy.InputField(desc="Context used for answering")
    validation_score = dspy.OutputField(desc="Validation score (0-10)")
    issues = dspy.OutputField(desc="List of validation issues")


class PrivacyProtectionSignature(dspy.Signature):
    """
    Ensure privacy protection in responses.
    
    This signature checks responses for privacy concerns and
    ensures appropriate protection of sensitive information.
    """
    response = dspy.InputField(desc="The response to check for privacy")
    privacy_level = dspy.InputField(desc="Required privacy level")
    protected_response = dspy.OutputField(desc="Privacy-protected response")
    privacy_concerns = dspy.OutputField(desc="List of privacy concerns addressed")


class BiasDetectionSignature(dspy.Signature):
    """
    Detect and mitigate bias in responses.
    
    This signature identifies potential bias in responses and
    provides suggestions for mitigation.
    """
    response = dspy.InputField(desc="The response to check for bias")
    bias_categories = dspy.InputField(desc="Categories of bias to check for")
    bias_score = dspy.OutputField(desc="Bias score (0-10, lower is better)")
    detected_bias = dspy.OutputField(desc="List of detected bias issues")
    mitigation_suggestions = dspy.OutputField(desc="Suggestions for bias mitigation")


class PerformanceOptimizationSignature(dspy.Signature):
    """
    Optimize system performance and efficiency.
    
    This signature provides recommendations for improving
    system performance and resource utilization.
    """
    current_performance = dspy.InputField(desc="Current performance metrics")
    optimization_targets = dspy.InputField(desc="Performance targets to achieve")
    optimization_plan = dspy.OutputField(desc="Plan for performance optimization")
    expected_gains = dspy.OutputField(desc="Expected performance gains")


class ErrorHandlingSignature(dspy.Signature):
    """
    Handle errors and provide graceful degradation.
    
    This signature manages error handling and provides
    fallback responses when primary systems fail.
    """
    error_context = dspy.InputField(desc="Context of the error")
    fallback_options = dspy.InputField(desc="Available fallback options")
    error_response = dspy.OutputField(desc="Appropriate error response")
    recovery_actions = dspy.OutputField(desc="Actions to recover from the error")


class QualityAssuranceSignature(dspy.Signature):
    """
    Ensure quality standards in system responses.
    
    This signature validates responses against quality criteria
    and provides feedback for improvement.
    """
    response = dspy.InputField(desc="The response to assess")
    quality_criteria = dspy.InputField(desc="Quality criteria to evaluate against")
    quality_score = dspy.OutputField(desc="Quality score (0-10)")
    quality_feedback = dspy.OutputField(desc="Feedback on response quality")


# Utility functions for working with ANEETA signatures

def get_aneeta_signature_by_name(name: str) -> Optional[dspy.Signature]:
    """
    Get an ANEETA signature by its name.
    
    Args:
        name: Name of the signature class
        
    Returns:
        Signature class or None if not found
    """
    signature_map = {
        "QuestionAnsweringSignature": QuestionAnsweringSignature,
        "SafetyCheckSignature": SafetyCheckSignature,
        "MultiAgentSystemSignature": MultiAgentSystemSignature,
        "LocalModelOptimizationSignature": LocalModelOptimizationSignature,
        "ContextRetrievalSignature": ContextRetrievalSignature,
        "AnswerValidationSignature": AnswerValidationSignature,
        "PrivacyProtectionSignature": PrivacyProtectionSignature,
        "BiasDetectionSignature": BiasDetectionSignature,
        "PerformanceOptimizationSignature": PerformanceOptimizationSignature,
        "ErrorHandlingSignature": ErrorHandlingSignature,
        "QualityAssuranceSignature": QualityAssuranceSignature,
    }
    
    return signature_map.get(name)


def list_aneeta_signatures() -> List[str]:
    """
    List all available ANEETA signature names.
    
    Returns:
        List of signature names
    """
    return [
        "QuestionAnsweringSignature",
        "SafetyCheckSignature",
        "MultiAgentSystemSignature",
        "LocalModelOptimizationSignature",
        "ContextRetrievalSignature",
        "AnswerValidationSignature",
        "PrivacyProtectionSignature",
        "BiasDetectionSignature",
        "PerformanceOptimizationSignature",
        "ErrorHandlingSignature",
        "QualityAssuranceSignature",
    ]


def create_aneeta_signature(
    name: str,
    input_fields: List[str],
    output_fields: List[str],
    descriptions: Optional[dict] = None
) -> dspy.Signature:
    """
    Create a custom ANEETA signature dynamically.
    
    Args:
        name: Name for the signature
        input_fields: List of input field names
        output_fields: List of output field names
        descriptions: Optional descriptions for fields
        
    Returns:
        Custom signature class
    """
    if descriptions is None:
        descriptions = {}
    
    # Create field definitions
    field_definitions = {}
    
    for field in input_fields:
        desc = descriptions.get(field, f"The {field} input")
        field_definitions[field] = dspy.InputField(desc=desc)
    
    for field in output_fields:
        desc = descriptions.get(field, f"The {field} output")
        field_definitions[field] = dspy.OutputField(desc=desc)
    
    # Create the signature class
    signature_class = type(name, (dspy.Signature,), field_definitions)
    
    return signature_class
