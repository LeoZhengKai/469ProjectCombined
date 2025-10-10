"""
Fact-checking utilities for DSPy evaluation framework.

This module provides fact-checking capabilities including:
- Text-based fact verification
- Source material validation
- Factual accuracy scoring
- Issue identification and reporting

Fact-checking is designed to work with the artifact system
and provide consistent evaluation across different projects.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FactCheckResult:
    """Result of a fact-checking operation."""
    is_factual: bool
    score: float
    issues: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FactualClaim:
    """A factual claim to be checked."""
    claim: str
    source: Optional[str] = None
    context: Optional[str] = None


class FactChecker:
    """
    Fact-checking utility for validating content accuracy.
    
    Provides methods to:
    - Check factual claims against source material
    - Score factual accuracy
    - Identify specific issues
    - Generate detailed reports
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the fact checker.
        
        Args:
            strict_mode: Whether to use strict fact-checking rules
        """
        self.strict_mode = strict_mode
        self.issue_patterns = [
            r"(\d+)\s*(?:percent|%)",  # Percentage claims
            r"(?:in|on)\s+(\d{4})",   # Date claims
            r"(?:worth|valued at)\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",  # Value claims
            r"(?:founded|established|created)\s+(?:in\s+)?(\d{4})",  # Founding claims
        ]
    
    def check_factuality(
        self,
        content: str,
        source_facts: str,
        threshold: float = 0.7
    ) -> FactCheckResult:
        """
        Check the factuality of content against source facts.
        
        Args:
            content: Content to fact-check
            source_facts: Source facts to validate against
            threshold: Minimum score for considering content factual
            
        Returns:
            FactCheckResult object
        """
        try:
            # Extract factual claims from content
            claims = self._extract_claims(content)
            
            # Check each claim against source facts
            issues = []
            total_claims = len(claims)
            factual_claims = 0
            
            for claim in claims:
                is_factual, claim_issues = self._check_claim(claim, source_facts)
                if is_factual:
                    factual_claims += 1
                else:
                    issues.extend(claim_issues)
            
            # Calculate overall score
            if total_claims == 0:
                score = 1.0  # No claims to check
            else:
                score = factual_claims / total_claims
            
            # Determine if content is factual
            is_factual = score >= threshold
            
            # Calculate confidence based on claim coverage
            confidence = min(1.0, total_claims / 10.0)  # More claims = higher confidence
            
            return FactCheckResult(
                is_factual=is_factual,
                score=score,
                issues=issues,
                confidence=confidence,
                metadata={
                    "total_claims": total_claims,
                    "factual_claims": factual_claims,
                    "threshold": threshold
                }
            )
        except Exception as e:
            return FactCheckResult(
                is_factual=False,
                score=0.0,
                issues=[f"Error during fact-checking: {e}"],
                confidence=0.0
            )
    
    def _extract_claims(self, content: str) -> List[FactualClaim]:
        """
        Extract factual claims from content.
        
        Args:
            content: Content to extract claims from
            
        Returns:
            List of FactualClaim objects
        """
        claims = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains factual patterns
            for pattern in self.issue_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    claim = FactualClaim(
                        claim=sentence,
                        context=content
                    )
                    claims.append(claim)
                    break  # Only add sentence once
        
        # If no specific patterns found, treat each sentence as a potential claim
        if not claims and self.strict_mode:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Avoid very short sentences
                    claim = FactualClaim(
                        claim=sentence,
                        context=content
                    )
                    claims.append(claim)
        
        return claims
    
    def _check_claim(
        self,
        claim: FactualClaim,
        source_facts: str
    ) -> Tuple[bool, List[str]]:
        """
        Check a single claim against source facts.
        
        Args:
            claim: Factual claim to check
            source_facts: Source facts to validate against
            
        Returns:
            Tuple of (is_factual, issues)
        """
        issues = []
        
        # Simple keyword-based checking
        claim_lower = claim.claim.lower()
        source_lower = source_facts.lower()
        
        # Check for contradictory keywords
        contradictions = [
            ("increased", "decreased"),
            ("higher", "lower"),
            ("more", "less"),
            ("better", "worse"),
            ("successful", "failed"),
        ]
        
        for pos, neg in contradictions:
            if pos in claim_lower and neg in source_lower:
                issues.append(f"Contradiction detected: '{pos}' vs '{neg}'")
                return False, issues
        
        # Check for specific factual patterns
        for pattern in self.issue_patterns:
            claim_matches = re.findall(pattern, claim.claim, re.IGNORECASE)
            source_matches = re.findall(pattern, source_facts, re.IGNORECASE)
            
            for claim_match in claim_matches:
                if claim_match not in source_matches:
                    issues.append(f"Unverified claim: '{claim_match}'")
                    return False, issues
        
        # If no issues found, consider it factual
        return True, issues
    
    def batch_check_factuality(
        self,
        contents: List[str],
        source_facts: str,
        threshold: float = 0.7
    ) -> List[FactCheckResult]:
        """
        Check factuality of multiple contents in batch.
        
        Args:
            contents: List of contents to fact-check
            source_facts: Source facts to validate against
            threshold: Minimum score for considering content factual
            
        Returns:
            List of FactCheckResult objects
        """
        results = []
        for content in contents:
            try:
                result = self.check_factuality(content, source_facts, threshold)
                results.append(result)
            except Exception as e:
                error_result = FactCheckResult(
                    is_factual=False,
                    score=0.0,
                    issues=[f"Error during fact-checking: {e}"],
                    confidence=0.0
                )
                results.append(error_result)
        
        return results
    
    def generate_fact_check_report(
        self,
        results: List[FactCheckResult],
        contents: List[str]
    ) -> str:
        """
        Generate a detailed fact-check report.
        
        Args:
            results: List of fact-check results
            contents: List of corresponding contents
            
        Returns:
            Formatted report string
        """
        if len(results) != len(contents):
            raise ValueError("Results and contents must have the same length")
        
        report = []
        report.append("=" * 50)
        report.append("FACT-CHECK REPORT")
        report.append("=" * 50)
        
        total_contents = len(contents)
        factual_contents = sum(1 for r in results if r.is_factual)
        avg_score = sum(r.score for r in results) / total_contents if total_contents > 0 else 0
        
        report.append(f"Total contents checked: {total_contents}")
        report.append(f"Factual contents: {factual_contents}")
        report.append(f"Factual accuracy: {avg_score:.2%}")
        report.append("")
        
        # Detailed results
        for i, (result, content) in enumerate(zip(results, contents)):
            report.append(f"Content {i+1}:")
            report.append(f"  Score: {result.score:.2f}")
            report.append(f"  Factual: {'Yes' if result.is_factual else 'No'}")
            report.append(f"  Confidence: {result.confidence:.2f}")
            
            if result.issues:
                report.append("  Issues:")
                for issue in result.issues:
                    report.append(f"    - {issue}")
            
            # Truncate content for display
            content_preview = content[:100] + "..." if len(content) > 100 else content
            report.append(f"  Content: {content_preview}")
            report.append("")
        
        return "\n".join(report)
    
    def save_fact_check_results(
        self,
        results: List[FactCheckResult],
        file_path: Path
    ) -> None:
        """
        Save fact-check results to a file.
        
        Args:
            results: List of fact-check results
            file_path: Path to save results
        """
        import json
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "is_factual": result.is_factual,
                "score": result.score,
                "issues": result.issues,
                "confidence": result.confidence,
                "metadata": result.metadata
            })
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_fact_check_results(self, file_path: Path) -> List[FactCheckResult]:
        """
        Load fact-check results from a file.
        
        Args:
            file_path: Path to load results from
            
        Returns:
            List of FactCheckResult objects
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            result = FactCheckResult(
                is_factual=item["is_factual"],
                score=item["score"],
                issues=item["issues"],
                confidence=item["confidence"],
                metadata=item.get("metadata")
            )
            results.append(result)
        
        return results
    
    def get_fact_check_summary(self, results: List[FactCheckResult]) -> Dict[str, Any]:
        """
        Get a summary of fact-check results.
        
        Args:
            results: List of fact-check results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        total = len(results)
        factual = sum(1 for r in results if r.is_factual)
        avg_score = sum(r.score for r in results) / total
        avg_confidence = sum(r.confidence for r in results) / total
        
        # Count issues
        total_issues = sum(len(r.issues) for r in results)
        
        return {
            "total_contents": total,
            "factual_contents": factual,
            "factual_accuracy": avg_score,
            "average_confidence": avg_confidence,
            "total_issues": total_issues,
            "issues_per_content": total_issues / total if total > 0 else 0
        }


# Global fact checker instance
fact_checker = FactChecker()
