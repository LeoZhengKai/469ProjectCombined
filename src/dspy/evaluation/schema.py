"""
Schema validation for DSPy evaluation results.

This module provides schema validation for:
- Evaluation results
- Metrics data
- Artifact structures
- Configuration files

Schemas ensure consistency and data integrity across
the evaluation framework.
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import jsonschema


@dataclass
class EvaluationSchema:
    """Schema definition for evaluation results."""
    name: str
    version: str
    description: str
    schema: Dict[str, Any]


class SchemaValidator:
    """
    Schema validator for evaluation results and artifacts.
    
    Provides methods to validate:
    - Evaluation results
    - Metrics data
    - Configuration files
    - Artifact structures
    """
    
    def __init__(self):
        """Initialize the schema validator."""
        self.schemas = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self) -> None:
        """Load default schemas for common data structures."""
        
        # Evaluation result schema
        evaluation_result_schema = {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "project": {"type": "string"},
                "model": {"type": "string"},
                "optimizer": {"type": ["string", "null"]},
                "timestamp": {"type": "string"},
                "results": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sample_id": {"type": ["string", "number"]},
                                    "input": {"type": "object"},
                                    "truth": {"type": ["object", "null"]},
                                    "prediction": {"type": "string"},
                                    "metrics": {"type": "object"},
                                    "perf": {"type": "object"},
                                    "iters": {"type": "object"}
                                },
                                "required": ["sample_id", "input", "prediction"]
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "properties": {
                                "quality_mean": {"type": "number"},
                                "fact_mean": {"type": "number"},
                                "latency_p50_ms": {"type": "number"},
                                "latency_p95_ms": {"type": "number"},
                                "cost_per_sample_usd": {"type": "number"},
                                "improvement_efficiency_mean": {"type": "number"}
                            }
                        }
                    },
                    "required": ["predictions", "metrics"]
                }
            },
            "required": ["run_id", "project", "model", "results"]
        }
        
        self.schemas["evaluation_result"] = EvaluationSchema(
            name="evaluation_result",
            version="1.0",
            description="Schema for evaluation results",
            schema=evaluation_result_schema
        )
        
        # Metrics schema
        metrics_schema = {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "total_samples": {"type": "integer"},
                "successful_samples": {"type": "integer"},
                "failed_samples": {"type": "integer"},
                "latency_p50_ms": {"type": "number"},
                "latency_p95_ms": {"type": "number"},
                "total_cost_usd": {"type": "number"},
                "cost_per_sample_usd": {"type": "number"},
                "total_tokens_in": {"type": "integer"},
                "total_tokens_out": {"type": "integer"},
                "quality_mean": {"type": "number"},
                "fact_mean": {"type": "number"},
                "improvement_efficiency_mean": {"type": "number"}
            },
            "required": ["run_id", "total_samples"]
        }
        
        self.schemas["metrics"] = EvaluationSchema(
            name="metrics",
            version="1.0",
            description="Schema for metrics data",
            schema=metrics_schema
        )
        
        # Configuration schema
        config_schema = {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "model": {"type": "string"},
                "optimizer": {"type": ["string", "null"]},
                "seed": {"type": "integer"},
                "thresholds": {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "number"},
                        "fact": {"type": "number"}
                    }
                },
                "num_samples": {"type": "integer"},
                "split": {"type": "string"},
                "created_at": {"type": "string"}
            },
            "required": ["project", "model", "seed"]
        }
        
        self.schemas["config"] = EvaluationSchema(
            name="config",
            version="1.0",
            description="Schema for configuration files",
            schema=config_schema
        )
        
        # Comparison schema
        comparison_schema = {
            "type": "object",
            "properties": {
                "compare_id": {"type": "string"},
                "run_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "comparison_type": {"type": "string"},
                "results": {"type": "object"},
                "created_at": {"type": "string"}
            },
            "required": ["compare_id", "run_ids", "comparison_type", "results"]
        }
        
        self.schemas["comparison"] = EvaluationSchema(
            name="comparison",
            version="1.0",
            description="Schema for comparison results",
            schema=comparison_schema
        )
    
    def validate(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        schema_name: str
    ) -> bool:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema to use
            
        Returns:
            True if valid, False otherwise
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        schema = self.schemas[schema_name].schema
        
        try:
            jsonschema.validate(data, schema)
            return True
        except jsonschema.ValidationError:
            return False
        except Exception:
            return False
    
    def validate_with_errors(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        schema_name: str
    ) -> List[str]:
        """
        Validate data against a schema and return errors.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema to use
            
        Returns:
            List of validation errors (empty if valid)
        """
        if schema_name not in self.schemas:
            return [f"Unknown schema: {schema_name}"]
        
        schema = self.schemas[schema_name].schema
        errors = []
        
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def validate_file(
        self,
        file_path: Path,
        schema_name: str
    ) -> bool:
        """
        Validate a JSON file against a schema.
        
        Args:
            file_path: Path to the JSON file
            schema_name: Name of the schema to use
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.validate(data, schema_name)
        except Exception:
            return False
    
    def validate_file_with_errors(
        self,
        file_path: Path,
        schema_name: str
    ) -> List[str]:
        """
        Validate a JSON file against a schema and return errors.
        
        Args:
            file_path: Path to the JSON file
            schema_name: Name of the schema to use
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.validate_with_errors(data, schema_name)
        except Exception as e:
            return [f"File error: {e}"]
    
    def add_schema(
        self,
        name: str,
        version: str,
        description: str,
        schema: Dict[str, Any]
    ) -> None:
        """
        Add a custom schema.
        
        Args:
            name: Schema name
            version: Schema version
            description: Schema description
            schema: JSON schema definition
        """
        self.schemas[name] = EvaluationSchema(
            name=name,
            version=version,
            description=description,
            schema=schema
        )
    
    def get_schema(self, name: str) -> Optional[EvaluationSchema]:
        """
        Get a schema by name.
        
        Args:
            name: Schema name
            
        Returns:
            EvaluationSchema object or None if not found
        """
        return self.schemas.get(name)
    
    def list_schemas(self) -> List[str]:
        """
        List all available schema names.
        
        Returns:
            List of schema names
        """
        return list(self.schemas.keys())
    
    def save_schema(self, name: str, file_path: Path) -> None:
        """
        Save a schema to a file.
        
        Args:
            name: Schema name
            file_path: Path to save schema
        """
        if name not in self.schemas:
            raise ValueError(f"Schema not found: {name}")
        
        schema = self.schemas[name]
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(schema), f, indent=2)
    
    def load_schema(self, file_path: Path) -> None:
        """
        Load a schema from a file.
        
        Args:
            file_path: Path to schema file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        schema = EvaluationSchema(**data)
        self.schemas[schema.name] = schema


# Utility functions

def validate_results(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    schema_name: str = "evaluation_result"
) -> bool:
    """
    Validate evaluation results against a schema.
    
    Args:
        data: Data to validate
        schema_name: Name of the schema to use
        
    Returns:
        True if valid, False otherwise
    """
    validator = SchemaValidator()
    return validator.validate(data, schema_name)


def validate_results_with_errors(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    schema_name: str = "evaluation_result"
) -> List[str]:
    """
    Validate evaluation results and return errors.
    
    Args:
        data: Data to validate
        schema_name: Name of the schema to use
        
    Returns:
        List of validation errors (empty if valid)
    """
    validator = SchemaValidator()
    return validator.validate_with_errors(data, schema_name)


def validate_file(
    file_path: Path,
    schema_name: str = "evaluation_result"
) -> bool:
    """
    Validate a JSON file against a schema.
    
    Args:
        file_path: Path to the JSON file
        schema_name: Name of the schema to use
        
    Returns:
        True if valid, False otherwise
    """
    validator = SchemaValidator()
    return validator.validate_file(file_path, schema_name)


def validate_file_with_errors(
    file_path: Path,
    schema_name: str = "evaluation_result"
) -> List[str]:
    """
    Validate a JSON file and return errors.
    
    Args:
        file_path: Path to the JSON file
        schema_name: Name of the schema to use
        
    Returns:
        List of validation errors (empty if valid)
    """
    validator = SchemaValidator()
    return validator.validate_file_with_errors(file_path, schema_name)


# Global schema validator instance
schema_validator = SchemaValidator()
