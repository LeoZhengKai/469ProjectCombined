"""
Core configuration management for DSPy evaluation framework.

This module handles loading configuration from:
- Environment variables (.env files)
- YAML configuration files
- Command-line arguments

Configuration hierarchy:
1. Default values (hardcoded)
2. YAML config files (configs/base.yaml, configs/models/*.yaml, etc.)
3. Environment variables
4. Command-line arguments (highest priority)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class ModelConfig(BaseSettings):
    """Configuration for language models."""
    provider: str = Field(default="openai", description="Model provider (openai, together, local)")
    model_name: str = Field(default="gpt-4o-mini", description="Specific model to use")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    
    class Config:
        env_prefix = "MODEL_"


class EvaluationConfig(BaseSettings):
    """Configuration for evaluation runs."""
    project: str = Field(default="sharktank", description="Project name (sharktank, aneeta)")
    split: str = Field(default="test", description="Dataset split to evaluate on")
    num_samples: int = Field(default=100, description="Number of samples to evaluate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    thresholds: Dict[str, float] = Field(
        default={"quality": 8.0, "fact": 8.5}, 
        description="Quality thresholds for evaluation"
    )
    
    class Config:
        env_prefix = "EVAL_"


class RAGConfig(BaseSettings):
    """Configuration for Retrieval-Augmented Generation."""
    enabled: bool = Field(default=False, description="Whether to use RAG")
    vector_db_url: str = Field(default="http://localhost:19530", description="Milvus connection URL")
    collection_name: str = Field(default="embeddings", description="Vector collection name")
    top_k: int = Field(default=5, description="Number of retrieved documents")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    
    class Config:
        env_prefix = "RAG_"


class Settings(BaseSettings):
    """Main application settings."""
    # Core paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)
    experiments_dir: Path = Field(default=Path("experiments"))
    datasets_dir: Path = Field(default=Path("datasets"))
    configs_dir: Path = Field(default=Path("configs"))
    
    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Evaluation configuration
    eval: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # RAG configuration
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json, text)")
    
    # MLflow tracking
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="dspy-eval", description="MLflow experiment name")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones.
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def get_settings(config_overrides: Optional[Dict[str, Any]] = None) -> Settings:
    """
    Get application settings with optional overrides.
    
    Args:
        config_overrides: Optional dictionary to override settings
        
    Returns:
        Settings instance
    """
    settings = Settings()
    
    if config_overrides:
        # Apply overrides (this is a simplified approach)
        for key, value in config_overrides.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    
    return settings


# Global settings instance
settings = get_settings()
