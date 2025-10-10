"""
DSPy modules and wrappers for the evaluation framework.

This module contains DSPy module implementations that wrap common patterns
used across projects. Modules provide reusable building blocks for:
- Draft generation and refinement
- Fact-checking and validation
- Quality assessment and improvement
- Tool integration and RAG support

Modules are designed to be:
- Composable and reusable
- Optimizable by DSPy optimizers
- Configurable via YAML settings
- Testable and debuggable
"""

import dspy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .signatures import (
    QAInput, FactCheckInput, QualityAssessment, RefinementInput,
    SafetyCheck, ComparisonInput, ToolUseInput, ReasoningInput,
    ValidationInput, SummarizationInput, ClassificationInput
)


class DraftModule(dspy.Module):
    """
    Generic draft generation module.
    
    This module generates initial drafts of content based on
    input requirements and context. It can be optimized for
    different types of content generation tasks.
    """
    
    def __init__(self, signature: dspy.Signature = QAInput):
        """
        Initialize the draft module.
        
        Args:
            signature: DSPy signature to use for generation
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, **kwargs) -> dspy.Prediction:
        """
        Generate a draft based on input parameters.
        
        Args:
            **kwargs: Input parameters matching the signature
            
        Returns:
            DSPy prediction with generated content
        """
        return self.predictor(**kwargs)


class FactCheckModule(dspy.Module):
    """
    Fact-checking module for validating generated content.
    
    This module checks the factual accuracy of content against
    provided source material and returns scores and issues.
    """
    
    def __init__(self, signature: dspy.Signature = FactCheckInput):
        """
        Initialize the fact-check module.
        
        Args:
            signature: DSPy signature to use for fact-checking
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str, source_facts: str) -> dspy.Prediction:
        """
        Check facts in the provided content.
        
        Args:
            content: Content to fact-check
            source_facts: Source facts to validate against
            
        Returns:
            DSPy prediction with score and issues
        """
        return self.predictor(content=content, source_facts=source_facts)


class QualityAssessmentModule(dspy.Module):
    """
    Quality assessment module for evaluating content quality.
    
    This module assesses the quality of generated content
    based on specified criteria and provides scores and feedback.
    """
    
    def __init__(self, signature: dspy.Signature = QualityAssessment):
        """
        Initialize the quality assessment module.
        
        Args:
            signature: DSPy signature to use for assessment
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str, criteria: str) -> dspy.Prediction:
        """
        Assess the quality of the provided content.
        
        Args:
            content: Content to assess
            criteria: Quality criteria to evaluate against
            
        Returns:
            DSPy prediction with score and feedback
        """
        return self.predictor(content=content, criteria=criteria)


class RefinementModule(dspy.Module):
    """
    Content refinement module for improving existing content.
    
    This module takes existing content and feedback to produce
    improved versions that better meet requirements.
    """
    
    def __init__(self, signature: dspy.Signature = RefinementInput):
        """
        Initialize the refinement module.
        
        Args:
            signature: DSPy signature to use for refinement
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, original_content: str, feedback: str) -> dspy.Prediction:
        """
        Refine the provided content based on feedback.
        
        Args:
            original_content: Original content to refine
            feedback: Feedback or requirements for improvement
            
        Returns:
            DSPy prediction with refined content
        """
        return self.predictor(original_content=original_content, feedback=feedback)


class SafetyCheckModule(dspy.Module):
    """
    Safety check module for assessing content safety.
    
    This module evaluates content for safety concerns,
    bias, or inappropriate material.
    """
    
    def __init__(self, signature: dspy.Signature = SafetyCheck):
        """
        Initialize the safety check module.
        
        Args:
            signature: DSPy signature to use for safety checking
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str) -> dspy.Prediction:
        """
        Check the safety of the provided content.
        
        Args:
            content: Content to check for safety
            
        Returns:
            DSPy prediction with safety score and concerns
        """
        return self.predictor(content=content)


class RAGModule(dspy.Module):
    """
    Retrieval-Augmented Generation module.
    
    This module integrates with vector databases (like Milvus)
    to retrieve relevant context for generation tasks.
    """
    
    def __init__(self, signature: dspy.Signature = QAInput, top_k: int = 5):
        """
        Initialize the RAG module.
        
        Args:
            signature: DSPy signature to use for generation
            top_k: Number of retrieved documents to use
        """
        super().__init__()
        self.signature = signature
        self.top_k = top_k
        self.predictor = dspy.Predict(signature)
        self.retriever = None  # Will be set up by the system
    
    def forward(self, question: str, context: str = "") -> dspy.Prediction:
        """
        Generate an answer using retrieved context.
        
        Args:
            question: Question to answer
            context: Optional additional context
            
        Returns:
            DSPy prediction with generated answer
        """
        # If retriever is available, retrieve relevant documents
        if self.retriever:
            retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
            context = f"{context}\n\nRetrieved context:\n{retrieved_docs}"
        
        return self.predictor(question=question, context=context)
    
    def set_retriever(self, retriever: Any) -> None:
        """
        Set the retriever for this module.
        
        Args:
            retriever: Retriever instance (e.g., MilvusRetriever)
        """
        self.retriever = retriever


class ReActModule(dspy.Module):
    """
    ReAct (Reasoning and Acting) module for tool use.
    
    This module implements the ReAct pattern for reasoning
    about problems and using tools to solve them.
    """
    
    def __init__(self, signature: dspy.Signature = ReasoningInput):
        """
        Initialize the ReAct module.
        
        Args:
            signature: DSPy signature to use for reasoning
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
        self.tools = {}  # Available tools
    
    def forward(self, problem: str) -> dspy.Prediction:
        """
        Solve a problem using reasoning and tool use.
        
        Args:
            problem: Problem to solve
            
        Returns:
            DSPy prediction with reasoning and conclusion
        """
        return self.predictor(problem=problem)
    
    def add_tool(self, name: str, tool: Any) -> None:
        """
        Add a tool to the module.
        
        Args:
            name: Name of the tool
            tool: Tool instance
        """
        self.tools[name] = tool


class ComparisonModule(dspy.Module):
    """
    Comparison module for evaluating multiple options.
    
    This module compares two or more items based on
    specified criteria and provides reasoning.
    """
    
    def __init__(self, signature: dspy.Signature = ComparisonInput):
        """
        Initialize the comparison module.
        
        Args:
            signature: DSPy signature to use for comparison
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, item_a: str, item_b: str, criteria: str) -> dspy.Prediction:
        """
        Compare two items based on criteria.
        
        Args:
            item_a: First item to compare
            item_b: Second item to compare
            criteria: Criteria for comparison
            
        Returns:
            DSPy prediction with winner and reasoning
        """
        return self.predictor(item_a=item_a, item_b=item_b, criteria=criteria)


class ValidationModule(dspy.Module):
    """
    Validation module for checking content against requirements.
    
    This module validates whether content meets specific
    requirements or constraints.
    """
    
    def __init__(self, signature: dspy.Signature = ValidationInput):
        """
        Initialize the validation module.
        
        Args:
            signature: DSPy signature to use for validation
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str, requirements: str) -> dspy.Prediction:
        """
        Validate content against requirements.
        
        Args:
            content: Content to validate
            requirements: Requirements to check against
            
        Returns:
            DSPy prediction with validation result
        """
        return self.predictor(content=content, requirements=requirements)


class SummarizationModule(dspy.Module):
    """
    Summarization module for creating content summaries.
    
    This module creates summaries of content based on
    length and focus requirements.
    """
    
    def __init__(self, signature: dspy.Signature = SummarizationInput):
        """
        Initialize the summarization module.
        
        Args:
            signature: DSPy signature to use for summarization
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str, length_requirement: str = "medium", focus: str = "") -> dspy.Prediction:
        """
        Create a summary of the provided content.
        
        Args:
            content: Content to summarize
            length_requirement: Length requirement (short, medium, long)
            focus: What to focus on in the summary
            
        Returns:
            DSPy prediction with generated summary
        """
        return self.predictor(
            content=content,
            length_requirement=length_requirement,
            focus=focus
        )


class ClassificationModule(dspy.Module):
    """
    Classification module for categorizing content.
    
    This module classifies content into predefined categories
    based on specific criteria.
    """
    
    def __init__(self, signature: dspy.Signature = ClassificationInput):
        """
        Initialize the classification module.
        
        Args:
            signature: DSPy signature to use for classification
        """
        super().__init__()
        self.signature = signature
        self.predictor = dspy.Predict(signature)
    
    def forward(self, content: str, categories: List[str]) -> dspy.Prediction:
        """
        Classify the provided content.
        
        Args:
            content: Content to classify
            categories: Available categories
            
        Returns:
            DSPy prediction with classification and confidence
        """
        categories_str = ", ".join(categories)
        return self.predictor(content=content, categories=categories_str)


# Utility functions for working with modules

def create_module_from_config(config: Dict[str, Any]) -> dspy.Module:
    """
    Create a module from configuration.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Configured DSPy module
    """
    module_type = config.get("type", "DraftModule")
    signature_name = config.get("signature", "QAInput")
    
    # Get signature
    from .signatures import get_signature_by_name
    signature = get_signature_by_name(signature_name)
    if signature is None:
        raise ValueError(f"Unknown signature: {signature_name}")
    
    # Create module
    module_map = {
        "DraftModule": DraftModule,
        "FactCheckModule": FactCheckModule,
        "QualityAssessmentModule": QualityAssessmentModule,
        "RefinementModule": RefinementModule,
        "SafetyCheckModule": SafetyCheckModule,
        "RAGModule": RAGModule,
        "ReActModule": ReActModule,
        "ComparisonModule": ComparisonModule,
        "ValidationModule": ValidationModule,
        "SummarizationModule": SummarizationModule,
        "ClassificationModule": ClassificationModule,
    }
    
    if module_type not in module_map:
        raise ValueError(f"Unknown module type: {module_type}")
    
    module_class = module_map[module_type]
    
    # Create module with signature
    if module_type == "RAGModule":
        top_k = config.get("top_k", 5)
        return module_class(signature=signature, top_k=top_k)
    else:
        return module_class(signature=signature)


def list_available_modules() -> List[str]:
    """
    List all available module types.
    
    Returns:
        List of module type names
    """
    return [
        "DraftModule",
        "FactCheckModule",
        "QualityAssessmentModule",
        "RefinementModule",
        "SafetyCheckModule",
        "RAGModule",
        "ReActModule",
        "ComparisonModule",
        "ValidationModule",
        "SummarizationModule",
        "ClassificationModule",
    ]
