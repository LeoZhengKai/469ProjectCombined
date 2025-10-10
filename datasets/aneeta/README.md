# ANEETA Dataset

This directory contains the ANEETA dataset for question answering and evaluation.

## Overview

The ANEETA dataset is designed for evaluating question answering systems, including:
- Question understanding and analysis
- Answer generation and quality assessment
- Safety and bias detection
- Multi-agent system evaluation

## Dataset Structure

```
datasets/aneeta/
├── README.md              # This file
├── loader.py              # Dataset loader
├── raw/                   # Raw data files
│   ├── questions.json     # Question database
│   ├── answers.jsonl      # Sample answers
│   └── evaluations.jsonl  # Evaluation results
└── processed/             # Processed data files
    ├── train.jsonl        # Training split
    ├── validation.jsonl   # Validation split
    ├── test.jsonl         # Test split
    └── sample_data.json   # Sample data for testing
```

## Data Format

Each sample in the dataset contains:

```json
{
  "sample_id": "an_001",
  "question": "What are the benefits of renewable energy?",
  "context": "Renewable energy sources like solar and wind power...",
  "answer": "Renewable energy offers several key benefits...",
  "safety_score": 9.0,
  "quality_score": 8.5,
  "bias_score": 9.0,
  "topic": "environment",
  "difficulty": "medium"
}
```

### Fields Description

- **sample_id**: Unique identifier for the sample
- **question**: The question to be answered
- **context**: Optional context information
- **answer**: Reference or generated answer
- **safety_score**: Safety score (0-10)
- **quality_score**: Quality score (0-10)
- **bias_score**: Bias score (0-10)
- **topic**: Topic category
- **difficulty**: Difficulty level (easy/medium/hard)

## Usage

### Loading the Dataset

```python
from datasets.aneeta.loader import load_dataset

# Load test split
data = load_dataset(split="test", num_samples=100)

# Load all data
data = load_dataset(split="all")
```

### Using with Dataset Registry

```python
from datasets.registry import get_dataset

# Load via registry
data = get_dataset("aneeta", split="test", num_samples=50)
```

## Data Sources

- **Question Database**: Curated questions from various domains
- **Answer Examples**: Sample answers from human experts
- **Evaluation Data**: Human evaluations of answer quality, safety, and bias

## Owner

**(fill in)** - Responsible for dataset preparation, validation, and maintenance.

## Acceptance Criteria

- [ ] `get_dataset("aneeta")` returns properly formatted data
- [ ] All required fields are present in samples
- [ ] Safety, quality, and bias scores are within valid ranges (0-10)
- [ ] Dataset splits are properly balanced
- [ ] Data validation passes all checks

## TODO Items

- [ ] Load actual question data from external sources
- [ ] Generate more diverse answer examples
- [ ] Add human evaluation data
- [ ] Implement data augmentation techniques
- [ ] Add metadata for each sample
- [ ] Create data visualization tools

## Data Quality

The dataset is designed to be:
- **Representative**: Covers diverse topics and difficulty levels
- **High Quality**: All answers are evaluated for quality, safety, and bias
- **Balanced**: Equal representation across different topics and difficulties
- **Up-to-date**: Regularly updated with new questions and answers

## Contributing

When adding new data:
1. Ensure all required fields are present
2. Validate safety, quality, and bias scores
3. Follow the established data format
4. Update the dataset statistics
5. Test the loader with new data

## Version History

- **v1.0.0**: Initial dataset with sample data
- **v1.1.0**: Added safety and bias scoring
- **v1.2.0**: Added topic and difficulty classification
