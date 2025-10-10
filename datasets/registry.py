"""
Dataset registry for DSPy evaluation framework.

This module provides a centralized registry for datasets used across
different projects. It handles dataset loading, validation, and
splitting for consistent evaluation.

Owner: Zheng Kai (for SharkTank), (fill in) (for ANEETA)
Acceptance Check: get_dataset("sharktank" | "aneeta") returns properly formatted data
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import jsonlines
from dataclasses import dataclass
from enum import Enum


class DatasetSplit(Enum):
    """Dataset split enumeration."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    version: str
    owner: str
    total_samples: int
    splits: Dict[str, int]
    features: List[str]
    created_at: str
    updated_at: str


class DatasetRegistry:
    """
    Registry for managing datasets across projects.
    
    Provides methods to:
    - Register and load datasets
    - Validate dataset formats
    - Split datasets for evaluation
    - Track dataset metadata
    """
    
    def __init__(self, datasets_dir: Path = Path("datasets")):
        """
        Initialize the dataset registry.
        
        Args:
            datasets_dir: Base directory for datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.registered_datasets = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the dataset registry from configuration."""
        registry_file = self.datasets_dir / "registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                self.registered_datasets = json.load(f)
        else:
            # Initialize with default datasets
            self.registered_datasets = {
                "sharktank": {
                    "name": "sharktank",
                    "description": "SharkTank pitch generation dataset",
                    "version": "1.0.0",
                    "owner": "Zheng Kai",
                    "loader_module": "datasets.sharktank.loader",
                    "processed_path": "datasets/sharktank/processed",
                    "features": ["product_facts", "guidelines", "pitch", "quality_score", "fact_score"],
                    "splits": {"train": 0.7, "validation": 0.15, "test": 0.15}
                },
                "aneeta": {
                    "name": "aneeta",
                    "description": "ANEETA question answering dataset",
                    "version": "1.0.0",
                    "owner": "(fill in)",
                    "loader_module": "datasets.aneeta.loader",
                    "processed_path": "datasets/aneeta/processed",
                    "features": ["question", "context", "answer", "safety_score", "quality_score"],
                    "splits": {"train": 0.7, "validation": 0.15, "test": 0.15}
                }
            }
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save the dataset registry to configuration."""
        registry_file = self.datasets_dir / "registry.json"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registered_datasets, f, indent=2)
    
    def register_dataset(
        self,
        name: str,
        description: str,
        version: str,
        owner: str,
        loader_module: str,
        processed_path: str,
        features: List[str],
        splits: Dict[str, float]
    ) -> None:
        """
        Register a new dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            version: Dataset version
            owner: Dataset owner
            loader_module: Python module path for the loader
            processed_path: Path to processed data
            features: List of feature names
            splits: Dataset splits with proportions
        """
        self.registered_datasets[name] = {
            "name": name,
            "description": description,
            "version": version,
            "owner": owner,
            "loader_module": loader_module,
            "processed_path": processed_path,
            "features": features,
            "splits": splits
        }
        self._save_registry()
    
    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """
        Get information about a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            DatasetInfo object or None if not found
        """
        if name not in self.registered_datasets:
            return None
        
        info = self.registered_datasets[name]
        return DatasetInfo(
            name=info["name"],
            description=info["description"],
            version=info["version"],
            owner=info["owner"],
            total_samples=0,  # Will be calculated when loading
            splits=info["splits"],
            features=info["features"],
            created_at="2024-01-01T00:00:00Z",  # Placeholder
            updated_at="2024-01-01T00:00:00Z"  # Placeholder
        )
    
    def list_datasets(self) -> List[str]:
        """
        List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.registered_datasets.keys())
    
    def get_dataset(
        self,
        name: str,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Load a dataset.
        
        Args:
            name: Dataset name
            split: Dataset split to load
            num_samples: Number of samples to load (None for all)
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            
        Returns:
            List of dataset samples
        """
        if name not in self.registered_datasets:
            raise ValueError(f"Dataset not found: {name}")
        
        dataset_info = self.registered_datasets[name]
        
        # Import the loader module
        try:
            module_path = dataset_info["loader_module"]
            module = __import__(module_path, fromlist=["load_dataset"])
            loader_func = getattr(module, "load_dataset")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import loader for dataset {name}: {e}")
        
        # Load the dataset
        data = loader_func(
            split=split.value,
            num_samples=num_samples,
            shuffle=shuffle,
            seed=seed
        )
        
        return data
    
    def validate_dataset(self, name: str) -> Dict[str, Any]:
        """
        Validate a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Validation results dictionary
        """
        if name not in self.registered_datasets:
            return {"valid": False, "error": f"Dataset not found: {name}"}
        
        try:
            # Load a small sample to validate
            sample_data = self.get_dataset(name, split=DatasetSplit.TEST, num_samples=5)
            
            if not sample_data:
                return {"valid": False, "error": "No data found"}
            
            # Check required features
            dataset_info = self.registered_datasets[name]
            required_features = dataset_info["features"]
            
            missing_features = []
            for feature in required_features:
                if feature not in sample_data[0]:
                    missing_features.append(feature)
            
            if missing_features:
                return {
                    "valid": False,
                    "error": f"Missing features: {missing_features}"
                }
            
            return {
                "valid": True,
                "sample_count": len(sample_data),
                "features": list(sample_data[0].keys()),
                "required_features": required_features
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_dataset_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics about a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset statistics dictionary
        """
        if name not in self.registered_datasets:
            return {"error": f"Dataset not found: {name}"}
        
        try:
            # Load all data to calculate stats
            all_data = self.get_dataset(name, split=DatasetSplit.ALL)
            
            if not all_data:
                return {"error": "No data found"}
            
            dataset_info = self.registered_datasets[name]
            features = dataset_info["features"]
            
            stats = {
                "total_samples": len(all_data),
                "features": features,
                "feature_stats": {}
            }
            
            # Calculate basic stats for each feature
            for feature in features:
                if feature in all_data[0]:
                    values = [sample[feature] for sample in all_data if feature in sample]
                    
                    if isinstance(values[0], str):
                        stats["feature_stats"][feature] = {
                            "type": "string",
                            "count": len(values),
                            "unique_count": len(set(values)),
                            "avg_length": sum(len(str(v)) for v in values) / len(values)
                        }
                    elif isinstance(values[0], (int, float)):
                        stats["feature_stats"][feature] = {
                            "type": "numeric",
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "mean": sum(values) / len(values)
                        }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}


# Global dataset registry instance
dataset_registry = DatasetRegistry()


# Convenience functions
def get_dataset(
    name: str,
    split: DatasetSplit = DatasetSplit.TEST,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Get a dataset from the registry.
    
    Args:
        name: Dataset name
        split: Dataset split
        num_samples: Number of samples
        shuffle: Whether to shuffle
        seed: Random seed
        
    Returns:
        List of dataset samples
    """
    return dataset_registry.get_dataset(name, split, num_samples, shuffle, seed)


def list_datasets() -> List[str]:
    """List all registered datasets."""
    return dataset_registry.list_datasets()


def validate_dataset(name: str) -> Dict[str, Any]:
    """Validate a dataset."""
    return dataset_registry.validate_dataset(name)


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get information about a dataset."""
    return dataset_registry.get_dataset_info(name)
