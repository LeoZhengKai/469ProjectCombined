# DSPy Evaluation Framework

A comprehensive framework for evaluating DSPy models with support for multiple projects, optimizers, and evaluation metrics.

## Overview

This framework provides a structured approach to evaluating DSPy models across different projects:

- **SharkTank**: Pitch generation and evaluation
- **ANEETA**: Question answering and safety assessment

## Project Structure

```
dspy-eval/
├── apps/                    # Applications
│   ├── api/                # FastAPI service (run models & evals via HTTP)
│   │   ├── main.py
│   │   └── routers/ {health.py, sharktank.py, aneeta.py, compare.py}
│   └── cli/                # Command-line entrypoints (mirrors API)
│       ├── eval.py         # run one eval (project, model) → artifacts/
│       ├── compare.py      # compare N run_ids → comparisons/
│       └── report.py       # optional HTML/CSV reports
├── src/                    # Source code
│   ├── core/               # framework-agnostic utilities
│   │   ├── config.py       # .env + YAML → Settings
│   │   ├── logging.py      # JSON logging
│   │   ├── artifacts.py    # run folders (predictions.jsonl, metrics.json, config.json)
│   │   ├── ids.py          # run_id / compare_id helpers
│   │   └── telemetry.py    # measure p95 latency, token/cost, iterations
│   ├── dspy/               # DSPy glue shared by all projects
│   │   ├── signatures.py   # shared base signatures
│   │   ├── modules.py      # Draft / FactCheck / Refine / ReAct wrappers
│   │   ├── optimizers/     # BootstrapFewShot, MIPROv2 setup
│   │   └── evaluation/     # metrics & judges used across projects
│   │       ├── metrics.py  # MAE, within±1, accuracy, F1, factuality hit rate
│   │       ├── judges.py    # rubric LLM judge (or stub)
│   │       ├── factcheck.py # fact-check helpers
│   │       └── schema.py    # canonical metrics.json schema
│   ├── adapters/           # wrap existing legacy code so we don't refactor it
│   │   ├── sharktank.py    # generate_pitch(), fact_check(), …
│   │   └── aneeta.py       # answer_question(), safety_check(), …
│   └── projects/           # Project-specific code
│       ├── sharktank/
│       │   ├── program.py  # DSPy program (compose Draft→(tools)→FactCheck→Refine)
│       │   ├── signatures.py # project-specific signatures
│       │   └── configs/ default.yaml # thresholds, tool ceilings, judge rubric
│       └── aneeta/
│           ├── program.py
│           ├── signatures.py
│           └── configs/ default.yaml
├── datasets/               # data registry + loaders
│   ├── registry.py         # get_dataset("sharktank" | "aneeta")
│   ├── sharktank/ {loader.py, README.md, raw/, processed/}
│   └── aneeta/   {loader.py, README.md, raw/, processed/}
├── configs/                # Configuration files
│   ├── base.yaml           # shared defaults (quality/fact thresholds, paths)
│   ├── models/ {openai.yaml, local.yaml, together.yaml,…}
│   ├── rag.yaml            # retrieval settings (if used)
│   └── eval/
│       ├── sharktank.yaml  # which split, how many samples, which optimizer(s)
│       └── aneeta.yaml
├── experiments/            # Experiment artifacts
│   ├── runs/               # artifacts per run_id (auto-created)
│   │   └── <run_id>/
│   │       ├── config.json # {project, model, optimizer, seed, thresholds, …}
│   │       ├── predictions.jsonl # one line per sample (input, truth, prediction, latency, cost)
│   │       └── metrics.json # aggregate metrics for the run
│   └── comparisons/
│       └── <compare_id>.json # side-by-side A vs B tables
├── infra/                  # Infrastructure
│   ├── mlflow/ docker-compose.yaml # optional tracking server
│   ├── milvus/ docker-compose.yaml # optional vector DB for RAG
│   ├── docker/ Dockerfile  # container for API/CLI
│   └── gpu_cluster/        # GPU cluster configurations
├── notebooks/              # Jupyter notebooks
│   └── colab_setup.ipynb   # Google Colab setup and examples
├── scripts/                # one-off utilities (ingest, cloc-measure, etc.)
├── tests/                  # pytest: unit + smoke tests
├── .github/workflows/ci.yml # lint + type-check + tests
├── Makefile                # make api | eval | compare | test | lint | typecheck
├── pyproject.toml           # deps + tools (ruff, black, mypy, pytest)
├── env.example             # safe env template
└── README.md
```

**Why this structure is clearer:**
- `apps/` = ways to run (API + CLI)
- `src/core/` = plumbing (config/logging/artifacts/telemetry)
- `src/dspy/` = DSPy-specific building blocks + evaluation helpers
- `src/adapters/` = bridge your existing SharkTank/ANEETA code into the new world
- `src/projects/` = the DSPy programs per project (clean, minimal)
- `datasets/` & `experiments/` cleanly separate inputs from outputs

## Features

- 🚀 **Multi-Project Support**: Evaluate different DSPy projects with consistent interfaces
- 🔧 **Multiple Optimizers**: BootstrapFewShot, MIPROv2, COPRO, and custom optimizers
- 📊 **Comprehensive Metrics**: Quality, factuality, safety, bias, and performance metrics
- 📈 **MLflow Integration**: Experiment tracking and model management (optional)
- 🔍 **Vector Search**: Milvus integration for RAG capabilities (optional)
- 🎯 **CLI & API**: Both command-line and REST API interfaces
- 📋 **Structured Artifacts**: Consistent experiment artifacts and comparisons
- ☁️ **Google Colab Support**: Cloud-based experiments and development
- 🖥️ **GPU Cluster Support**: SLURM integration for large-scale training

## Quick Start

### Prerequisites

- Python 3.9+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 469Project
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Run a sample evaluation**
   ```bash
   make eval-sharktank
   ```

### Environment Setup

1. **Copy environment template**
   ```bash
   cp .env.example .env
   ```

2. **Configure your API keys**
   ```bash
   # Edit .env file
   OPENAI_API_KEY=your_openai_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```

## Usage

### Command Line Interface

#### Running Evaluations

```bash
# SharkTank evaluation
python apps/cli/eval.py --project sharktank --model gpt-4o-mini --num-samples 100

# ANEETA evaluation
python apps/cli/eval.py --project aneeta --model gemma-2b --num-samples 100

# With optimizer
python apps/cli/eval.py --project sharktank --model gpt-4o-mini --optimizer BootstrapFewShot
```

#### Comparing Results

```bash
# List available runs
python apps/cli/compare.py --list

# Compare runs
python apps/cli/compare.py --runs run1 run2 --type overall

# Get existing comparison
python apps/cli/compare.py --compare-id comp123 --format table
```

#### Generating Reports

```bash
# Generate HTML report
python apps/cli/report.py --run-id run123 --format html --output report.html

# Generate project report
python apps/cli/report.py --project sharktank --format html --output project_report.html
```

### API Interface

#### Start the API Server

```bash
make run-api
# or
python apps/api/main.py
```

#### API Endpoints

- **Health Check**: `GET /health`
- **Status**: `GET /status`
- **Run Evaluation**: `POST /eval`
- **List Runs**: `GET /runs`
- **Get Run**: `GET /runs/{run_id}`
- **Compare Runs**: `POST /compare/runs`
- **SharkTank**: `POST /sharktank/generate-pitch`
- **ANEETA**: `POST /aneeta/answer-question`

#### Example API Usage

```bash
# Start evaluation
curl -X POST "http://localhost:8000/eval" \
  -H "Content-Type: application/json" \
  -d '{"project": "sharktank", "model": "gpt-4o-mini", "num_samples": 10}'

# Generate pitch
curl -X POST "http://localhost:8000/sharktank/generate-pitch" \
  -H "Content-Type: application/json" \
  -d '{
    "product_facts": "AI-powered fitness app",
    "guidelines": "Focus on market opportunity and business model"
  }'
```

## Project Structure

```
dspy-eval/
├── apps/                    # Applications
│   ├── api/                # FastAPI service
│   └── cli/                # Command-line tools
├── src/                    # Source code
│   ├── core/               # Core utilities
│   ├── dspy/               # DSPy components
│   ├── adapters/           # Legacy code adapters
│   └── projects/           # Project-specific code
├── datasets/               # Data registry
├── configs/                # Configuration files
├── experiments/            # Experiment artifacts
├── infra/                  # Infrastructure
└── tests/                  # Test suite
```

## Configuration

### Model Configuration

Configure models in `configs/models/`:

- `openai.yaml`: OpenAI models (GPT-4, GPT-3.5)
- `local.yaml`: Local models (Gemma, Llama)
- `together.yaml`: Together AI models

### Project Configuration

Configure projects in `configs/eval/`:

- `sharktank.yaml`: SharkTank evaluation settings
- `aneeta.yaml`: ANEETA evaluation settings

### Base Configuration

Global settings in `configs/base.yaml`:

- Quality thresholds
- Evaluation settings
- Performance settings
- Security settings

## Infrastructure

### Deployment Options

#### Option 1: Local Development (Recommended)

For development and testing, run everything locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python apps/api/main.py

# Run CLI evaluation
python apps/cli/eval.py --project sharktank --model gpt-4o-mini --num-samples 10

# Use local MLflow (optional)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### Option 2: Google Colab (For Experiments)

Use the provided Jupyter notebook for cloud-based experiments:

```bash
# Upload notebooks/colab_setup.ipynb to Google Colab
# Follow the notebook instructions for setup
```

#### Option 3: GPU Clusters (For Large-Scale Training)

For GPU-accelerated training and evaluation:

```bash
# SLURM cluster
sbatch infra/gpu_cluster/slurm_config.sh
```

### Services

- **API Server**: http://localhost:8000
- **MLflow** (optional): http://localhost:5000

## Development

### Setup Development Environment

```bash
make dev-setup
```

### Running Tests

```bash
# All tests
make test

# Unit tests
make test-unit

# Integration tests
make test-integration
```

### Code Quality

```bash
# Linting
make lint

# Type checking
make typecheck

# Formatting
make format
```

### Setting Up Sample Data

```bash
# Create sample datasets for testing
python scripts/setup_sample_data.py --project all

# Validate datasets
python scripts/validate_datasets.py
```

## DSPy Signatures

Think of a DSPy signature as a typed function contract the optimizer can learn against. It declares inputs/outputs only (no logic), so everything else—modules, optimizers, refinement—plugs in cleanly.

- **Shared base signatures** (`src/dspy/signatures.py`): reusable shapes you might want across projects (e.g., a generic QAInput)
- **Project-specific signatures** (`src/projects/<name>/signatures.py`): the exact I/O for that task (e.g., pitch generation)

### Example Signature

```python
# src/projects/sharktank/signatures.py
import dspy

class PitchSignature(dspy.Signature):
    """Generate an investor-ready pitch grounded in provided product facts."""
    product_facts = dspy.InputField()
    guidelines    = dspy.InputField()
    pitch         = dspy.OutputField()

class FactCheckSig(dspy.Signature):
    """Score factual alignment between pitch and product facts."""
    pitch         = dspy.InputField()
    product_facts = dspy.InputField()
    score         = dspy.OutputField()   # 0–10
    issues        = dspy.OutputField()   # list/summary
```

**Why signatures matter:**
- **Stable boundaries**: everything calls the program via clear inputs/outputs
- **Optimizer-ready**: DSPy compiles prompt/weights to maximize your metrics for this signature
- **Low coupling**: swap models or modules without breaking the API surface
- **Testable**: signatures make great unit-test targets (validate required fields, shapes, etc.)

## Projects

### SharkTank

SharkTank pitch generation and evaluation:

- **Goal**: Generate investor-ready pitches
- **Metrics**: Quality, factuality, safety
- **Optimizers**: BootstrapFewShot, MIPROv2, COPRO
- **Features**: Fact-checking, quality assessment, refinement

### ANEETA

ANEETA question answering and safety system:

- **Goal**: Safe, accurate question answering
- **Metrics**: Safety, quality, bias detection
- **Optimizers**: MIPROv2, BootstrapFewShot, COPRO
- **Features**: Safety checking, bias detection, privacy protection

## Task Delegation

### SharkTank Tasks

| Task | Where to work | Owner | Acceptance Check |
|------|---------------|-------|------------------|
| Set up vanilla SharkTank for eval | `src/adapters/sharktank.py`, `datasets/sharktank/loader.py` | **Zheng Kai** | `python apps/cli/eval.py --project sharktank` runs & writes `experiments/runs/<id>/` |
| Set up & load Milvus DB | `infra/milvus/docker-compose.yaml`, `configs/rag.yaml`, `src/dspy/modules.py` | **Zheng Kai** | `rag.enabled=true` retrieves; context shows in `predictions.jsonl` |
| Set up MLflow for DSPy | `infra/mlflow/docker-compose.yaml`, `src/core/telemetry.py` (MLflow hook) | **Zheng Kai** | runs logged to MLflow; metrics visible |
| Recreate Pitch Tank in DSPy | `src/projects/sharktank/program.py`, `.../signatures.py` | **Isaiah** | Meets thresholds; passes tests |
| Prompt-optimise SharkTank | `src/dspy/optimizers/` configs; `configs/eval/sharktank.yaml` | **Zheng Kai** | "Optimizer-only" beats vanilla on quality/fact OR latency/cost |
| Fine-tune SharkTank with DSPy | (optional) `src/dspy/optimizers/finetune.py` | **Isaiah** | Clear gains vs optimizer-only; report cost/ops delta |
| Prepare dataset for prompt/fine-tune | `datasets/sharktank/processed/` | **Zheng Kai** | Documented splits; leakage check |
| Evaluate performance | `apps/cli/eval.py`, `apps/cli/compare.py` | **Zheng Kai + Isaiah** | `metrics.json` & comparison table produced |

### ANEETA Tasks

| Task | Where to work | Owner | Acceptance Check |
|------|---------------|-------|------------------|
| Set up vanilla ANEETA | `src/adapters/aneeta.py`, `datasets/aneeta/loader.py` | **(fill in)** | Eval runs write artifacts |
| Initialize MLflow tracking | `infra/mlflow/`, `src/core/telemetry.py` | **Yanjie** | Runs visible in MLflow |
| Local vector store + embedder | `configs/rag.yaml`, `src/dspy/modules.py` | **Yanjie** | Retrieval shows in context; latency split recorded |
| Compare Gemma quantizations | `configs/models/local.yaml` variants | **Yanjie** | Table with quality/latency/cost trade-offs |
| DSPy re-creation of MAS | `src/projects/aneeta/program.py`, `.../signatures.py` | **Benjamin** | Meets safety + quality thresholds |
| Run MIPROv2/Bootstrap/COPRO | `src/dspy/optimizers/` | **Benjamin** | Best config recorded & reproducible |
| A/B tests vs current ANEETA | `apps/cli/compare.py`, `experiments/comparisons/` | **De Wang & Benjamin** | Comparison JSON/CSV + short report committed |

**Pro tip**: Put owner initials in each project's `configs/default.yaml` (e.g., `owner: zhengkai`) so it's obvious who to ping.

## Metrics and Artifacts

### Where Metrics Live

Your core + DSPy metrics go here:

- **Task quality & factuality** → `src/dspy/evaluation/metrics.py` + `judges.py` + `factcheck.py`
  - Pitch Quality Score (1–10): rubric judge in `judges.py`
  - Fact-Check Score (0–10): `factcheck.py` (exact method is pluggable)
- **Latency & cost** → `src/core/telemetry.py`
  - capture per-request `latency_ms`, aggregate p50 / p95
  - capture token usage / API cost (or local inference time → $ estimate)
  - write to `predictions.jsonl` (per-sample) and `metrics.json` (aggregates)
- **Improvement Efficiency** → also `src/core/telemetry.py`
  - count refine/critique iterations to hit threshold (e.g., quality ≥ 8)
- **Code Complexity Reduction** → `scripts/cloc_measure.py` (simple cloc wrapper)
  - compare LoC between legacy multi-agent path and DSPy program path
  - write to `experiments/comparisons/<id>.json`

### Quality Metrics

- **Quality Score**: Overall content quality (0-10)
- **Factuality Score**: Factual accuracy (0-10)
- **Safety Score**: Safety assessment (0-10)
- **Bias Score**: Bias detection (0-10)

### Performance Metrics

- **Latency**: P50, P95 response times
- **Cost**: Per-sample and total costs
- **Tokens**: Input/output token usage
- **Memory**: Resource utilization

### Efficiency Metrics

- **Improvement Efficiency**: Iterations to reach threshold
- **Convergence Rate**: Optimization convergence
- **Success Rate**: Evaluation success rate

## Artifacts

### Schema (steady & human-readable)

#### `experiments/runs/<run_id>/config.json`
```json
{
  "project": "sharktank",
  "model": "gpt-4o-mini",
  "optimizer": "MIPROv2",
  "seed": 42,
  "thresholds": {"quality": 8.0, "fact": 8.5}
}
```

#### `predictions.jsonl` (one line per sample)
```json
{"sample_id": 123, "input": {...}, "truth": {... or null},
 "prediction": "...", "metrics": {"quality": 8.6, "fact": 8.9},
 "perf": {"latency_ms": 1420, "tokens_in": 2900, "tokens_out": 750, "cost_usd": 0.028},
 "iters": {"refine": 2}}
```

#### `metrics.json` (aggregates)
```json
{"quality_mean": 8.4, "fact_mean": 8.7,
 "latency_p50_ms": 980, "latency_p95_ms": 1720,
 "cost_per_sample_usd": 0.027,
 "improvement_efficiency_mean": 1.6}
```

### Run Artifacts

Each evaluation run produces:

- `config.json`: Run configuration
- `predictions.jsonl`: Individual predictions
- `metrics.json`: Aggregated metrics
- `telemetry.json`: Performance data

### Comparison Artifacts

Comparisons produce:

- `comparison.json`: Comparison results
- `report.html`: HTML report
- `report.csv`: CSV data

## Troubleshooting

### Common Issues

1. **API not starting**
   ```bash
   make health-check
   # Check if port 8000 is available
   lsof -i :8000
   ```

2. **Permission errors**
   ```bash
   chmod +x apps/cli/*.py
   chmod +x scripts/*.py
   ```

3. **Import errors**
   ```bash
   # Make sure you're in the project root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python apps/api/main.py
```

### Reset Environment

```bash
make dev-reset
make clean
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## TL;DR

- **Use the clean tree above**. It's small but complete: API/CLI, DSPy glue, adapters for legacy code, datasets registry, and a clean artifacts pattern for comparisons.
- **Signatures are typed I/O contracts** that let DSPy optimize your program while keeping your boundary stable.
- **Metrics from your proposal live** in `dspy/evaluation` and `core/telemetry`, written per-run so comparison is trivial.
- **Your delegation maps neatly** to directories and acceptance checks—just follow the "Where to work" column.

## Changelog

### Version 1.0.0

- Initial release
- SharkTank and ANEETA projects
- Multiple optimizers support
- Comprehensive metrics
- MLflow integration (optional)
- CLI and API interfaces
- Google Colab support
- GPU cluster configurations (SLURM)
- Task delegation framework