#!/usr/bin/env python3
"""
Sample data setup script.

This script creates sample data for testing the DSPy evaluation framework.
It generates sample datasets for both SharkTank and ANEETA projects.

Owner: Zheng Kai
Acceptance Check: Sample data is created and can be loaded by the dataset loaders
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any
import argparse
import random
from datetime import datetime


def create_sharktank_sample_data() -> List[Dict[str, Any]]:
    """
    Create sample SharkTank data.
    
    Returns:
        List of sample SharkTank data
    """
    sample_data = [
        {
            "sample_id": "st_001",
            "product_facts": "AI-powered fitness app that tracks workouts and provides personalized recommendations based on user progress and preferences",
            "guidelines": "Focus on market opportunity, business model, and competitive advantage. Highlight the AI technology and personalization features.",
            "pitch": "FitAI is revolutionizing personal fitness with AI-powered workout tracking and personalized recommendations. Our app addresses the $96B global fitness market by providing users with tailored workout plans that adapt to their progress and preferences. With 40% higher user retention than traditional fitness apps, FitAI is positioned to capture significant market share.",
            "quality_score": 8.5,
            "fact_score": 9.0,
            "market_size": 96,
            "team_size": 5,
            "revenue_model": "freemium",
            "created_at": "2024-01-15T10:30:00Z"
        },
        {
            "sample_id": "st_002",
            "product_facts": "Sustainable packaging solution made from agricultural waste that is biodegradable and cost-effective",
            "guidelines": "Emphasize environmental impact, scalability, and cost-effectiveness. Highlight the sustainable nature and market demand.",
            "pitch": "EcoPack transforms agricultural waste into sustainable packaging solutions, addressing the $300B global packaging market while reducing environmental impact. Our cost-effective solution offers 40% cost savings compared to traditional packaging while being 100% biodegradable. With major retailers committing to sustainable packaging, EcoPack is positioned for rapid growth.",
            "quality_score": 8.0,
            "fact_score": 8.5,
            "market_size": 300,
            "team_size": 8,
            "revenue_model": "b2b",
            "created_at": "2024-01-16T14:20:00Z"
        },
        {
            "sample_id": "st_003",
            "product_facts": "Blockchain-based supply chain transparency platform that enables real-time tracking of products from source to consumer",
            "guidelines": "Highlight transparency, security, and industry applications. Focus on the blockchain technology and supply chain benefits.",
            "pitch": "ChainTrace provides blockchain-based supply chain transparency, enabling companies to track products from source to consumer with immutable, secure records. Our platform addresses the $45B supply chain management market with real-time tracking, fraud prevention, and compliance automation. With increasing demand for supply chain transparency, ChainTrace is well-positioned for growth.",
            "quality_score": 7.5,
            "fact_score": 8.0,
            "market_size": 45,
            "team_size": 12,
            "revenue_model": "saas",
            "created_at": "2024-01-17T09:15:00Z"
        },
        {
            "sample_id": "st_004",
            "product_facts": "VR-based training platform for healthcare professionals that provides immersive surgical simulations",
            "guidelines": "Focus on healthcare applications, training effectiveness, and market opportunity. Highlight the VR technology and training benefits.",
            "pitch": "MedVR transforms healthcare training with immersive VR surgical simulations, addressing the $2.8B medical training market. Our platform provides realistic surgical scenarios, reducing training costs by 60% while improving skill retention by 40%. With healthcare institutions investing heavily in training technology, MedVR is positioned for significant growth.",
            "quality_score": 8.8,
            "fact_score": 9.2,
            "market_size": 2.8,
            "team_size": 15,
            "revenue_model": "b2b",
            "created_at": "2024-01-18T16:45:00Z"
        },
        {
            "sample_id": "st_005",
            "product_facts": "Smart home automation system that uses AI to learn user preferences and optimize energy consumption",
            "guidelines": "Emphasize AI technology, energy savings, and smart home market. Highlight the learning capabilities and optimization features.",
            "pitch": "SmartHomeAI revolutionizes home automation with AI-powered energy optimization, addressing the $84B smart home market. Our system learns user preferences and automatically optimizes energy consumption, reducing costs by 25% while improving comfort. With the smart home market growing rapidly, SmartHomeAI is positioned to capture significant market share.",
            "quality_score": 8.2,
            "fact_score": 8.7,
            "market_size": 84,
            "team_size": 10,
            "revenue_model": "hardware_saas",
            "created_at": "2024-01-19T11:30:00Z"
        }
    ]
    
    return sample_data


def create_aneeta_sample_data() -> List[Dict[str, Any]]:
    """
    Create sample ANEETA data.
    
    Returns:
        List of sample ANEETA data
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
            "difficulty": "medium",
            "created_at": "2024-01-15T10:30:00Z"
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
            "difficulty": "medium",
            "created_at": "2024-01-16T14:20:00Z"
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
            "difficulty": "easy",
            "created_at": "2024-01-17T09:15:00Z"
        },
        {
            "sample_id": "an_004",
            "question": "What are the ethical implications of artificial intelligence?",
            "context": "Artificial intelligence is rapidly advancing and being integrated into various aspects of society.",
            "answer": "AI raises several ethical concerns: 1) Privacy and data protection, 2) Bias and fairness in decision-making, 3) Job displacement and economic impact, 4) Accountability and transparency, 5) Safety and control of autonomous systems, and 6) The need for responsible AI development and governance.",
            "safety_score": 8.0,
            "quality_score": 8.8,
            "bias_score": 8.2,
            "topic": "ethics",
            "difficulty": "hard",
            "created_at": "2024-01-18T16:45:00Z"
        },
        {
            "sample_id": "an_005",
            "question": "How does photosynthesis work?",
            "context": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "answer": "Photosynthesis occurs in two stages: 1) Light-dependent reactions where chlorophyll absorbs light energy and converts it to chemical energy (ATP and NADPH), and 2) Light-independent reactions (Calvin cycle) where CO2 is fixed and converted into glucose using the energy from the first stage. This process occurs primarily in the chloroplasts of plant cells.",
            "safety_score": 9.5,
            "quality_score": 9.2,
            "bias_score": 9.5,
            "topic": "biology",
            "difficulty": "medium",
            "created_at": "2024-01-19T11:30:00Z"
        }
    ]
    
    return sample_data


def save_sample_data(
    data: List[Dict[str, Any]],
    output_dir: Path,
    filename: str
) -> None:
    """
    Save sample data to files.
    
    Args:
        data: Data to save
        output_dir: Output directory
        filename: Base filename (without extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = output_dir / f"{filename}.jsonl"
    with jsonlines.open(jsonl_path, 'w') as writer:
        writer.write_all(data)
    
    # Save as JSON
    json_path = output_dir / f"{filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {len(data)} samples to {jsonl_path} and {json_path}")


def create_dataset_splits(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create train/validation/test splits from data.
    
    Args:
        data: Data to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Dictionary with splits
    """
    # Shuffle data
    random.shuffle(data)
    
    # Calculate split indices
    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Create splits
    splits = {
        "train": data[:train_end],
        "validation": data[train_end:val_end],
        "test": data[val_end:]
    }
    
    return splits


def setup_sharktank_data(output_dir: Path) -> None:
    """
    Setup SharkTank sample data.
    
    Args:
        output_dir: Output directory
    """
    print("Setting up SharkTank sample data...")
    
    # Create sample data
    sample_data = create_sharktank_sample_data()
    
    # Create splits
    splits = create_dataset_splits(sample_data)
    
    # Save splits
    for split_name, split_data in splits.items():
        save_sample_data(split_data, output_dir, split_name)
    
    # Save all data
    save_sample_data(sample_data, output_dir, "all")
    
    print(f"✓ SharkTank data setup complete: {len(sample_data)} total samples")


def setup_aneeta_data(output_dir: Path) -> None:
    """
    Setup ANEETA sample data.
    
    Args:
        output_dir: Output directory
    """
    print("Setting up ANEETA sample data...")
    
    # Create sample data
    sample_data = create_aneeta_sample_data()
    
    # Create splits
    splits = create_dataset_splits(sample_data)
    
    # Save splits
    for split_name, split_data in splits.items():
        save_sample_data(split_data, output_dir, split_name)
    
    # Save all data
    save_sample_data(sample_data, output_dir, "all")
    
    print(f"✓ ANEETA data setup complete: {len(sample_data)} total samples")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Setup sample data for DSPy evaluation")
    parser.add_argument("--project", choices=["sharktank", "aneeta", "all"], default="all", help="Project to setup")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: datasets/<project>/processed)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.project == "all":
            output_dir = Path("datasets")
        else:
            output_dir = Path(f"datasets/{args.project}/processed")
    
    # Setup data based on project
    if args.project in ["sharktank", "all"]:
        sharktank_dir = output_dir / "sharktank" / "processed"
        setup_sharktank_data(sharktank_dir)
    
    if args.project in ["aneeta", "all"]:
        aneeta_dir = output_dir / "aneeta" / "processed"
        setup_aneeta_data(aneeta_dir)
    
    print("\n✓ Sample data setup complete!")
    print("You can now test the dataset loaders:")
    print("  python -c \"from datasets.registry import get_dataset; print(get_dataset('sharktank', num_samples=2))\"")
    print("  python -c \"from datasets.registry import get_dataset; print(get_dataset('aneeta', num_samples=2))\"")


if __name__ == "__main__":
    main()
