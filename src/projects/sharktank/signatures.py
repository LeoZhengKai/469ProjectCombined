"""
SharkTank project-specific signatures for DSPy evaluation framework.

This module contains DSPy signatures specific to the SharkTank project:
- Pitch generation signatures
- Fact-checking signatures for pitches
- Quality assessment signatures
- Refinement signatures for pitch improvement

Signatures are designed to work with the SharkTank program and
provide clear input/output contracts for optimization.
"""

import dspy
from typing import List, Optional


class PitchGenerationSignature(dspy.Signature):
    """
    Generate an investor-ready pitch grounded in provided product facts.
    
    This signature is used to generate pitches for SharkTank-style
    presentations based on product information and guidelines.
    """
    product_facts = dspy.InputField(desc="Key facts about the product or service")
    guidelines = dspy.InputField(desc="Guidelines for pitch structure and content")
    pitch = dspy.OutputField(desc="The generated investor pitch")


class PitchFactCheckSignature(dspy.Signature):
    """
    Score factual alignment between pitch and product facts.
    
    This signature evaluates how well a pitch aligns with the
    provided product facts and identifies any discrepancies.
    """
    pitch = dspy.InputField(desc="The pitch to fact-check")
    product_facts = dspy.InputField(desc="Product facts to validate against")
    score = dspy.OutputField(desc="Factual alignment score (0-10)")
    issues = dspy.OutputField(desc="List of factual issues or discrepancies")


class PitchQualitySignature(dspy.Signature):
    """
    Assess the quality of a pitch based on investor criteria.
    
    This signature evaluates pitch quality based on criteria such as:
    - Clarity and persuasiveness
    - Market opportunity presentation
    - Business model clarity
    - Financial projections
    """
    pitch = dspy.InputField(desc="The pitch to assess")
    criteria = dspy.InputField(desc="Quality criteria for evaluation")
    score = dspy.OutputField(desc="Quality score (0-10)")
    feedback = dspy.OutputField(desc="Detailed feedback on pitch quality")


class PitchRefinementSignature(dspy.Signature):
    """
    Refine a pitch based on feedback and requirements.
    
    This signature takes an existing pitch and feedback to produce
    an improved version that better meets investor expectations.
    """
    original_pitch = dspy.InputField(desc="The original pitch to refine")
    feedback = dspy.InputField(desc="Feedback or requirements for improvement")
    refined_pitch = dspy.OutputField(desc="The improved pitch")


class MarketAnalysisSignature(dspy.Signature):
    """
    Analyze market opportunity and competitive landscape.
    
    This signature evaluates market potential and competitive positioning
    based on product information and market data.
    """
    product_info = dspy.InputField(desc="Information about the product or service")
    market_data = dspy.InputField(desc="Available market data and trends")
    analysis = dspy.OutputField(desc="Market analysis and opportunity assessment")
    recommendations = dspy.OutputField(desc="Strategic recommendations")


class BusinessModelSignature(dspy.Signature):
    """
    Evaluate and improve business model clarity.
    
    This signature assesses business model presentation and provides
    suggestions for improvement.
    """
    business_model = dspy.InputField(desc="Description of the business model")
    revenue_streams = dspy.InputField(desc="Revenue streams and monetization")
    evaluation = dspy.OutputField(desc="Business model evaluation")
    improvements = dspy.OutputField(desc="Suggested improvements")


class FinancialProjectionSignature(dspy.Signature):
    """
    Assess financial projections and assumptions.
    
    This signature evaluates the realism and clarity of financial
    projections presented in pitches.
    """
    projections = dspy.InputField(desc="Financial projections and assumptions")
    market_size = dspy.InputField(desc="Market size and growth data")
    assessment = dspy.OutputField(desc="Assessment of projection realism")
    concerns = dspy.OutputField(desc="Concerns or questions about projections")


class InvestorQuestionsSignature(dspy.Signature):
    """
    Generate potential investor questions for a pitch.
    
    This signature anticipates questions that investors might ask
    based on the pitch content and business model.
    """
    pitch = dspy.InputField(desc="The pitch content")
    business_model = dspy.InputField(desc="Business model information")
    questions = dspy.OutputField(desc="List of potential investor questions")
    priority = dspy.OutputField(desc="Priority level for each question")


class PitchComparisonSignature(dspy.Signature):
    """
    Compare two pitches and determine which is better.
    
    This signature compares pitches based on specific criteria
    and provides reasoning for the comparison.
    """
    pitch_a = dspy.InputField(desc="First pitch to compare")
    pitch_b = dspy.InputField(desc="Second pitch to compare")
    criteria = dspy.InputField(desc="Criteria for comparison")
    winner = dspy.OutputField(desc="Which pitch is better (A or B)")
    reasoning = dspy.OutputField(desc="Reasoning for the comparison")


class PitchValidationSignature(dspy.Signature):
    """
    Validate a pitch against SharkTank requirements.
    
    This signature checks if a pitch meets the specific requirements
    and format expected for SharkTank presentations.
    """
    pitch = dspy.InputField(desc="The pitch to validate")
    requirements = dspy.InputField(desc="SharkTank requirements and format")
    is_valid = dspy.OutputField(desc="Whether the pitch meets requirements")
    violations = dspy.OutputField(desc="List of requirement violations")


class PitchSummarizationSignature(dspy.Signature):
    """
    Create a summary of a pitch for quick review.
    
    This signature generates concise summaries of pitches
    highlighting key points and value propositions.
    """
    pitch = dspy.InputField(desc="The pitch to summarize")
    length_requirement = dspy.InputField(desc="Length requirement for summary")
    focus = dspy.InputField(desc="What to focus on in the summary")
    summary = dspy.OutputField(desc="The generated summary")


# Utility functions for working with SharkTank signatures

def get_sharktank_signature_by_name(name: str) -> Optional[dspy.Signature]:
    """
    Get a SharkTank signature by its name.
    
    Args:
        name: Name of the signature class
        
    Returns:
        Signature class or None if not found
    """
    signature_map = {
        "PitchGenerationSignature": PitchGenerationSignature,
        "PitchFactCheckSignature": PitchFactCheckSignature,
        "PitchQualitySignature": PitchQualitySignature,
        "PitchRefinementSignature": PitchRefinementSignature,
        "MarketAnalysisSignature": MarketAnalysisSignature,
        "BusinessModelSignature": BusinessModelSignature,
        "FinancialProjectionSignature": FinancialProjectionSignature,
        "InvestorQuestionsSignature": InvestorQuestionsSignature,
        "PitchComparisonSignature": PitchComparisonSignature,
        "PitchValidationSignature": PitchValidationSignature,
        "PitchSummarizationSignature": PitchSummarizationSignature,
    }
    
    return signature_map.get(name)


def list_sharktank_signatures() -> List[str]:
    """
    List all available SharkTank signature names.
    
    Returns:
        List of signature names
    """
    return [
        "PitchGenerationSignature",
        "PitchFactCheckSignature",
        "PitchQualitySignature",
        "PitchRefinementSignature",
        "MarketAnalysisSignature",
        "BusinessModelSignature",
        "FinancialProjectionSignature",
        "InvestorQuestionsSignature",
        "PitchComparisonSignature",
        "PitchValidationSignature",
        "PitchSummarizationSignature",
    ]


def create_sharktank_signature(
    name: str,
    input_fields: List[str],
    output_fields: List[str],
    descriptions: Optional[dict] = None
) -> dspy.Signature:
    """
    Create a custom SharkTank signature dynamically.
    
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
