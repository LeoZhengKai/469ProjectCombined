# SharkTank Dataset

This directory contains the SharkTank dataset for pitch generation and evaluation.

## Overview

The SharkTank dataset is designed for evaluating pitch generation systems, including:
- Product fact extraction and analysis
- Pitch generation and refinement
- Fact-checking and quality assessment
- Market analysis and business model evaluation

## Dataset Structure

```
datasets/sharktank/
├── README.md              # This file
├── loader.py              # Dataset loader
├── raw/                   # Raw data files
│   ├── products.json      # Product information
│   ├── pitches.jsonl      # Sample pitches
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
  "sample_id": "st_001",
  "product_facts": "AI-powered fitness app that tracks workouts...",
  "guidelines": "Focus on market opportunity, business model...",
  "pitch": "FitAI is revolutionizing personal fitness...",
  "quality_score": 8.5,
  "fact_score": 9.0,
  "market_size": 96,
  "team_size": 5,
  "revenue_model": "freemium"
}
```

### Fields Description

- **sample_id**: Unique identifier for the sample
- **product_facts**: Key facts about the product or service
- **guidelines**: Guidelines for pitch structure and content
- **pitch**: Generated or reference pitch
- **quality_score**: Quality score (0-10)
- **fact_score**: Factuality score (0-10)
- **market_size**: Market size in billions USD
- **team_size**: Number of team members
- **revenue_model**: Revenue model type

## Usage

### Loading the Dataset

```python
from datasets.sharktank.loader import load_dataset

# Load test split
data = load_dataset(split="test", num_samples=100)

# Load all data
data = load_dataset(split="all")
```

### Using with Dataset Registry

```python
from datasets.registry import get_dataset

# Load via registry
data = get_dataset("sharktank", split="test", num_samples=50)
```

## Data Sources

- **Product Database**: Curated product information from various sources
- **Pitch Examples**: Sample pitches from successful startups
- **Evaluation Data**: Human evaluations of pitch quality and factuality

## Owner

**Zheng Kai** - Responsible for dataset preparation, validation, and maintenance.

## Acceptance Criteria

- [ ] `get_dataset("sharktank")` returns properly formatted data
- [ ] All required fields are present in samples
- [ ] Quality and factuality scores are within valid ranges (0-10)
- [ ] Dataset splits are properly balanced
- [ ] Data validation passes all checks

## TODO Items

- [ ] Load actual product data from external sources
- [ ] Generate more diverse pitch examples
- [ ] Add human evaluation data
- [ ] Implement data augmentation techniques
- [ ] Add metadata for each sample
- [ ] Create data visualization tools

## Data Quality

The dataset is designed to be:
- **Representative**: Covers diverse product categories and business models
- **High Quality**: All pitches are evaluated for quality and factuality
- **Balanced**: Equal representation across different market segments
- **Up-to-date**: Regularly updated with new products and pitches

## Contributing

When adding new data:
1. Ensure all required fields are present
2. Validate quality and factuality scores
3. Follow the established data format
4. Update the dataset statistics
5. Test the loader with new data

## Version History

- **v1.0.0**: Initial dataset with sample data
- **v1.1.0**: Added market size and team size fields
- **v1.2.0**: Added revenue model classification
