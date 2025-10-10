"""
SharkTank dataset loader.

This module loads and processes SharkTank datasets for evaluation.
It handles data loading, splitting, and formatting for the DSPy
evaluation framework.

Owner: Zheng Kai
Acceptance Check: get_dataset("sharktank") returns properly formatted data
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import jsonlines
import random
from enum import Enum


class SharkTankSplit(Enum):
    """SharkTank dataset split enumeration."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"


class SharkTankLoader:
    """
    Loader for SharkTank datasets.
    
    Handles loading, splitting, and formatting of SharkTank data
    for evaluation purposes.
    """
    
    def __init__(self, data_dir: Path = Path("datasets/sharktank")):
        """
        Initialize the SharkTank loader.
        
        Args:
            data_dir: Directory containing SharkTank data
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
        Load SharkTank dataset.
        
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
            "pitches.jsonl",
            "products.json",
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
        
        TODO: Zheng Kai - Replace with actual data loading from raw sources
        """
        sample_data = [
            {
                "sample_id": "st_001",
                "product_facts": "AI-powered fitness app that tracks workouts and provides personalized recommendations",
                "guidelines": "Focus on market opportunity, business model, and competitive advantage",
                "pitch": "FitAI is revolutionizing personal fitness with AI-powered workout tracking and personalized recommendations. Our app addresses the $96B global fitness market by providing users with tailored workout plans that adapt to their progress and preferences.",
                "quality_score": 8.5,
                "fact_score": 9.0,
                "market_size": 96,
                "team_size": 5,
                "revenue_model": "freemium"
            },
            {
                "sample_id": "st_002",
                "product_facts": "Sustainable packaging solution made from agricultural waste",
                "guidelines": "Emphasize environmental impact, scalability, and cost-effectiveness",
                "pitch": "EcoPack transforms agricultural waste into sustainable packaging solutions, addressing the $300B global packaging market while reducing environmental impact. Our cost-effective solution offers 40% cost savings compared to traditional packaging.",
                "quality_score": 8.0,
                "fact_score": 8.5,
                "market_size": 300,
                "team_size": 8,
                "revenue_model": "b2b"
            },
            {
                "sample_id": "st_003",
                "product_facts": "Blockchain-based supply chain transparency platform",
                "guidelines": "Highlight transparency, security, and industry applications",
                "pitch": "ChainTrace provides blockchain-based supply chain transparency, enabling companies to track products from source to consumer. Our platform addresses the $45B supply chain management market with immutable, secure tracking solutions.",
                "quality_score": 7.5,
                "fact_score": 8.0,
                "market_size": 45,
                "team_size": 12,
                "revenue_model": "saas"
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
        Validate SharkTank data format.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results
        """
        if not data:
            return {"valid": False, "error": "No data provided"}
        
        required_fields = ["product_facts", "guidelines"]
        optional_fields = ["pitch", "quality_score", "fact_score", "market_size", "team_size", "revenue_model"]
        
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
            if "quality_score" in sample:
                if not isinstance(sample["quality_score"], (int, float)) or not (0 <= sample["quality_score"] <= 10):
                    sample_issues.append("Invalid quality_score (must be 0-10)")
            
            if "fact_score" in sample:
                if not isinstance(sample["fact_score"], (int, float)) or not (0 <= sample["fact_score"] <= 10):
                    sample_issues.append("Invalid fact_score (must be 0-10)")
            
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
                "quality_scores": [],
                "fact_scores": [],
                "market_sizes": [],
                "team_sizes": []
            }
            
            # Extract numeric fields for statistics
            for sample in data:
                if "quality_score" in sample:
                    info["quality_scores"].append(sample["quality_score"])
                if "fact_score" in sample:
                    info["fact_scores"].append(sample["fact_score"])
                if "market_size" in sample:
                    info["market_sizes"].append(sample["market_size"])
                if "team_size" in sample:
                    info["team_sizes"].append(sample["team_size"])
            
            # Calculate statistics
            for field in ["quality_scores", "fact_scores", "market_sizes", "team_sizes"]:
                if info[field]:
                    info[f"{field}_stats"] = {
                        "min": min(info[field]),
                        "max": max(info[field]),
                        "mean": sum(info[field]) / len(info[field]),
                        "count": len(info[field])
                    }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}


# Global loader instance
sharktank_loader = SharkTankLoader()


# Convenience function for dataset registry
def load_dataset(
    split: str = "test",
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load SharkTank dataset.
    
    Args:
        split: Dataset split to load
        num_samples: Number of samples to load
        shuffle: Whether to shuffle
        seed: Random seed
        
    Returns:
        List of dataset samples
    """
    return sharktank_loader.load_dataset(split, num_samples, shuffle, seed)
