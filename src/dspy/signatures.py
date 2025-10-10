"""
Shared base signatures for DSPy evaluation framework.

This module contains reusable DSPy signatures that can be used across
multiple projects. Signatures define typed input/output contracts that
DSPy optimizers can learn against.

Base signatures provide:
- Generic question-answering patterns
- Fact-checking and validation patterns
- Refinement and improvement patterns
- Safety and quality assessment patterns
"""

import dspy
from typing import List, Optional


class QAInput(dspy.Signature):
    """
    Generic question-answering input signature.
    
    This signature can be used as a base for various QA tasks
    across different projects (SharkTank, ANEETA, etc.).
    """
    question = dspy.InputField(desc="The question to be answered")
    context = dspy.InputField(desc="Relevant context information", format=lambda x: str(x) if x else "")
    answer = dspy.OutputField(desc="The answer to the question")


class FactCheckInput(dspy.Signature):
    """
    Generic fact-checking input signature.
    
    Used to validate factual accuracy of generated content
    against provided source material.
    """
    content = dspy.InputField(desc="The content to be fact-checked")
    source_facts = dspy.InputField(desc="Source facts to validate against")
    score = dspy.OutputField(desc="Factual accuracy score (0-10)")
    issues = dspy.OutputField(desc="List of factual issues found")


class QualityAssessment(dspy.Signature):
    """
    Generic quality assessment signature.
    
    Used to evaluate the quality of generated content
    based on various criteria (clarity, coherence, etc.).
    """
    content = dspy.InputField(desc="The content to be assessed")
    criteria = dspy.InputField(desc="Quality criteria to evaluate against")
    score = dspy.OutputField(desc="Quality score (0-10)")
    feedback = dspy.OutputField(desc="Detailed feedback on quality")


class RefinementInput(dspy.Signature):
    """
    Generic content refinement signature.
    
    Used to improve existing content based on feedback
    or specific requirements.
    """
    original_content = dspy.InputField(desc="The original content to refine")
    feedback = dspy.InputField(desc="Feedback or requirements for improvement")
    refined_content = dspy.OutputField(desc="The improved content")


class SafetyCheck(dspy.Signature):
    """
    Generic safety check signature.
    
    Used to assess content for safety concerns,
    bias, or inappropriate material.
    """
    content = dspy.InputField(desc="The content to be checked for safety")
    safety_score = dspy.OutputField(desc="Safety score (0-10, higher is safer)")
    concerns = dspy.OutputField(desc="List of safety concerns found")
    recommendation = dspy.OutputField(desc="Recommendation (safe, review, reject)")


class ComparisonInput(dspy.Signature):
    """
    Generic comparison signature.
    
    Used to compare two pieces of content or solutions
    based on specified criteria.
    """
    item_a = dspy.InputField(desc="First item to compare")
    item_b = dspy.InputField(desc="Second item to compare")
    criteria = dspy.InputField(desc="Criteria for comparison")
    winner = dspy.OutputField(desc="Which item is better (A or B)")
    reasoning = dspy.OutputField(desc="Reasoning for the comparison")


class ToolUseInput(dspy.Signature):
    """
    Generic tool use signature.
    
    Used when the model needs to decide whether and how
    to use external tools or APIs.
    """
    task = dspy.InputField(desc="The task to be performed")
    available_tools = dspy.InputField(desc="List of available tools")
    tool_choice = dspy.OutputField(desc="Which tool to use (or 'none')")
    tool_input = dspy.OutputField(desc="Input parameters for the chosen tool")


class ReasoningInput(dspy.Signature):
    """
    Generic reasoning signature.
    
    Used for step-by-step reasoning tasks where
    the model needs to show its thinking process.
    """
    problem = dspy.InputField(desc="The problem to solve")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    conclusion = dspy.OutputField(desc="Final conclusion or answer")


class ValidationInput(dspy.Signature):
    """
    Generic validation signature.
    
    Used to validate whether content meets specific
    requirements or constraints.
    """
    content = dspy.InputField(desc="The content to validate")
    requirements = dspy.InputField(desc="Requirements to check against")
    is_valid = dspy.OutputField(desc="Whether the content is valid (true/false)")
    violations = dspy.OutputField(desc="List of requirement violations")


class SummarizationInput(dspy.Signature):
    """
    Generic summarization signature.
    
    Used to create summaries of content based on
    specific length or focus requirements.
    """
    content = dspy.InputField(desc="The content to summarize")
    length_requirement = dspy.InputField(desc="Length requirement (e.g., 'short', 'medium', 'long')")
    focus = dspy.InputField(desc="What to focus on in the summary")
    summary = dspy.OutputField(desc="The generated summary")


class ClassificationInput(dspy.Signature):
    """
    Generic classification signature.
    
    Used to classify content into predefined categories
    based on specific criteria.
    """
    content = dspy.InputField(desc="The content to classify")
    categories = dspy.InputField(desc="Available categories")
    classification = dspy.OutputField(desc="The chosen category")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")


# Utility functions for working with signatures

def get_signature_by_name(name: str) -> Optional[dspy.Signature]:
    """
    Get a signature by its name.
    
    Args:
        name: Name of the signature class
        
    Returns:
        Signature class or None if not found
    """
    signature_map = {
        "QAInput": QAInput,
        "FactCheckInput": FactCheckInput,
        "QualityAssessment": QualityAssessment,
        "RefinementInput": RefinementInput,
        "SafetyCheck": SafetyCheck,
        "ComparisonInput": ComparisonInput,
        "ToolUseInput": ToolUseInput,
        "ReasoningInput": ReasoningInput,
        "ValidationInput": ValidationInput,
        "SummarizationInput": SummarizationInput,
        "ClassificationInput": ClassificationInput,
    }
    
    return signature_map.get(name)


def list_available_signatures() -> List[str]:
    """
    List all available signature names.
    
    Returns:
        List of signature names
    """
    return [
        "QAInput",
        "FactCheckInput", 
        "QualityAssessment",
        "RefinementInput",
        "SafetyCheck",
        "ComparisonInput",
        "ToolUseInput",
        "ReasoningInput",
        "ValidationInput",
        "SummarizationInput",
        "ClassificationInput",
    ]


def create_custom_signature(
    name: str,
    input_fields: List[str],
    output_fields: List[str],
    descriptions: Optional[Dict[str, str]] = None
) -> dspy.Signature:
    """
    Create a custom signature dynamically.
    
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
