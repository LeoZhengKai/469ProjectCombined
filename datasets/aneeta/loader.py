"""
ANEETA dataset loader.

This module loads and processes ANEETA datasets for evaluation.
It handles data loading, splitting, and formatting for the DSPy
evaluation framework.

Owner: (fill in)
Acceptance Check: get_dataset("aneeta") returns properly formatted data
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import jsonlines
import random
from enum import Enum


class ANEETASplit(Enum):
    """ANEETA dataset split enumeration."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


class ANEETALoader:
    """
    Loader for ANEETA datasets.
    
    Handles loading, splitting, and formatting of ANEETA data
    for evaluation purposes.
    """
    
    def __init__(self, data_dir: Path = Path("datasets/aneeta")):
        """
        Initialize the ANEETA loader.
        
        Args:
            data_dir: Directory containing ANEETA data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Load ANEETA dataset.
        
        Args:
            split: Dataset split to load
            num_samples: Number of samples to load (None for all)
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            
        Returns:
            List of dataset samples
        """
        # Set random seed for reproducibility
        if shuffle:
            random.seed(seed)
        
        # Load data based on split
        if split == "all":
            data = self._load_all_data()
        else:
            data = self._load_split_data(split)
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(data)
        
        # Limit samples if requested
        if num_samples is not None:
            data = data[:num_samples]
        
        return data
    
    def _load_all_data(self) -> List[Dict[str, Any]]:
        """Load all data from processed files."""
        data = []
        
        # Load from processed files
        processed_files = [
            "questions.jsonl",
            "answers.json",
            "evaluations.jsonl"
        ]
        
        for filename in processed_files:
            file_path = self.processed_dir / filename
            if file_path.exists():
                if filename.endswith(".jsonl"):
                    with jsonlines.open(file_path, 'r') as reader:
                        data.extend(list(reader))
                elif filename.endswith(".json"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
        
        return data
    
    def _load_split_data(self, split: str) -> List[Dict[str, Any]]:
        """Load data for a specific split."""
        split_file = self.processed_dir / f"{split}.jsonl"
        
        if split_file.exists():
            with jsonlines.open(split_file, 'r') as reader:
                return list(reader)
        else:
            # Fallback: load all data and split manually
            all_data = self._load_all_data()
            return self._split_data(all_data, split)
    
    def _split_data(self, data: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        """
        Split data into train/validation/test.
        
        Args:
            data: All data
            split: Split to return
            
        Returns:
            Data for the specified split
        """
        if not data:
            return []
        
        # Define split proportions
        split_proportions = {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15
        }
        
        # Calculate split indices
        total_samples = len(data)
        train_end = int(total_samples * split_proportions["train"])
        val_end = train_end + int(total_samples * split_proportions["validation"])
        
        # Return appropriate split
        if split == "train":
            return data[:train_end]
        elif split == "validation":
            return data[train_end:val_end]
        elif split == "test":
            return data[val_end:]
        else:
            return data
    
    def create_sample_data(self) -> None:
        """
        Create sample data for testing.
        
        TODO: (fill in owner) - Replace with actual data loading from raw sources
        """
        sample_data = [
            {
                "sample_id": "an_001",
                "question": "What are the benefits of renewable energy?",
                "context": "Renewable energy sources like solar and wind power are becoming increasingly popular as alternatives to fossil fuels.",
                "answer": "Renewable energy offers several key benefits: 1) Environmental sustainability by reducing greenhouse gas emissions, 2) Energy independence from fossil fuel imports, 3) Long-term cost savings as technology improves, 4) Job creation in green industries, and 5) Reduced air pollution improving public health.",
                "safety_score": 9.0,
                "quality_score": 8.5,
                "bias_score": 9.0,
                "topic": "environment",
                "difficulty": "medium"
            },
            {
                "sample_id": "an_002",
                "question": "How does machine learning work?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
                "answer": "Machine learning works by training algorithms on large datasets to identify patterns and make predictions. The process involves: 1) Data collection and preprocessing, 2) Algorithm selection based on the problem type, 3) Model training using training data, 4) Validation and testing to assess performance, and 5) Deployment for real-world applications.",
                "safety_score": 8.5,
                "quality_score": 8.0,
                "bias_score": 8.5,
                "topic": "technology",
                "difficulty": "medium"
            },
            {
                "sample_id": "an_003",
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe known for its culture, cuisine, and history.",
                "answer": "The capital of France is Paris. Paris is located in the north-central part of the country and is the largest city in France, serving as the country's political, economic, and cultural center.",
                "safety_score": 9.5,
                "quality_score": 9.0,
                "bias_score": 9.5,
                "topic": "geography",
                "difficulty": "easy"
            }
        ]
        
        # Save sample data to processed files
        self._save_sample_data(sample_data)
    
    def _save_sample_data(self, data: List[Dict[str, Any]]) -> None:
        """Save sample data to processed files."""
        # Save as JSONL for easy loading
        with jsonlines.open(self.processed_dir / "test.jsonl", 'w') as writer:
            writer.write_all(data)
        
        # Also save as JSON for reference
        with open(self.processed_dir / "sample_data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate ANEETA data format.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results
        """
        if not data:
            return {"valid": False, "error": "No data provided"}
        
        required_fields = ["question", "answer"]
        optional_fields = ["context", "safety_score", "quality_score", "bias_score", "topic", "difficulty"]
        
        validation_results = {
            "valid": True,
            "total_samples": len(data),
            "missing_fields": [],
            "invalid_samples": []
        }
        
        for i, sample in enumerate(data):
            sample_issues = []
            
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    sample_issues.append(f"Missing required field: {field}")
            
            # Check data types and values
            if "safety_score" in sample:
                if not isinstance(sample["safety_score"], (int, float)) or not (0 <= sample["safety_score"] <= 10):
                    sample_issues.append("Invalid safety_score (must be 0-10)")
            
            if "quality_score" in sample:
                if not isinstance(sample["quality_score"], (int, float)) or not (0 <= sample["quality_score"] <= 10):
                    sample_issues.append("Invalid quality_score (must be 0-10)")
            
            if "bias_score" in sample:
                if not isinstance(sample["bias_score"], (int, float)) or not (0 <= sample["bias_score"] <= 10):
                    sample_issues.append("Invalid bias_score (must be 0-10)")
            
            if sample_issues:
                validation_results["invalid_samples"].append({
                    "sample_index": i,
                    "sample_id": sample.get("sample_id", f"sample_{i}"),
                    "issues": sample_issues
                })
                validation_results["valid"] = False
        
        return validation_results
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Data information dictionary
        """
        try:
            data = self._load_all_data()
            
            if not data:
                return {"error": "No data found"}
            
            # Calculate basic statistics
            info = {
                "total_samples": len(data),
                "fields": list(data[0].keys()) if data else [],
                "safety_scores": [],
                "quality_scores": [],
                "bias_scores": [],
                "topics": [],
                "difficulties": []
            }
            
            # Extract fields for statistics
            for sample in data:
                if "safety_score" in sample:
                    info["safety_scores"].append(sample["safety_score"])
                if "quality_score" in sample:
                    info["quality_scores"].append(sample["quality_score"])
                if "bias_score" in sample:
                    info["bias_scores"].append(sample["bias_score"])
                if "topic" in sample:
                    info["topics"].append(sample["topic"])
                if "difficulty" in sample:
                    info["difficulties"].append(sample["difficulty"])
            
            # Calculate statistics
            for field in ["safety_scores", "quality_scores", "bias_scores"]:
                if info[field]:
                    info[f"{field}_stats"] = {
                        "min": min(info[field]),
                        "max": max(info[field]),
                        "mean": sum(info[field]) / len(info[field]),
                        "count": len(info[field])
                    }
            
            # Calculate topic and difficulty distributions
            if info["topics"]:
                topic_counts = {}
                for topic in info["topics"]:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                info["topic_distribution"] = topic_counts
            
            if info["difficulties"]:
                difficulty_counts = {}
                for difficulty in info["difficulties"]:
                    difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                info["difficulty_distribution"] = difficulty_counts
            
            return info
            
        except Exception as e:
            return {"error": str(e)}


# Global loader instance
aneeta_loader = ANEETALoader()


# Convenience function for dataset registry
def load_dataset(
    split: str = "test",
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load ANEETA dataset.
    
    Args:
        split: Dataset split to load
        num_samples: Number of samples to load
        shuffle: Whether to shuffle
        seed: Random seed
        
    Returns:
        List of dataset samples
    """
    return aneeta_loader.load_dataset(split, num_samples, shuffle, seed)
