Love it—let’s lock this down so your team can work fast and stay tidy.

⸻

1) Clear, simple, and comprehensive repo structure

dspy-eval/
├─ apps/
│  ├─ api/                           # FastAPI service (run models & evals via HTTP)
│  │  ├─ main.py
│  │  └─ routers/ {health.py, sharktank.py, aneeta.py, compare.py}
│  └─ cli/                           # Command-line entrypoints (mirrors API)
│     ├─ eval.py                     # run one eval (project, model) → artifacts/
│     ├─ compare.py                  # compare N run_ids → comparisons/
│     └─ report.py                   # optional HTML/CSV reports
│
├─ src/
│  ├─ core/                          # framework-agnostic utilities
│  │  ├─ config.py                   # .env + YAML → Settings
│  │  ├─ logging.py                  # JSON logging
│  │  ├─ artifacts.py                # run folders (predictions.jsonl, metrics.json, config.json)
│  │  ├─ ids.py                      # run_id / compare_id helpers
│  │  └─ telemetry.py                # measure p95 latency, token/cost, iterations
│  │
│  ├─ dspy/                          # DSPy glue shared by all projects
│  │  ├─ signatures.py               # shared base signatures (see §2)
│  │  ├─ modules.py                  # Draft / FactCheck / Refine / ReAct wrappers
│  │  ├─ optimizers/                 # BootstrapFewShot, MIPROv2 setup
│  │  └─ evaluation/                 # metrics & judges used across projects
│  │     ├─ metrics.py               # MAE, within±1, accuracy, F1, factuality hit rate
│  │     ├─ judges.py                # rubric LLM judge (or stub)
│  │     ├─ factcheck.py             # fact-check helpers
│  │     └─ schema.py                # canonical metrics.json schema
│  │
│  ├─ adapters/                      # wrap existing legacy code so we don’t refactor it
│  │  ├─ sharktank.py                # generate_pitch(), fact_check(), …
│  │  └─ aneeta.py                   # answer_question(), safety_check(), …
│  │
│  └─ projects/
│     ├─ sharktank/
│     │  ├─ program.py               # DSPy program (compose Draft→(tools)→FactCheck→Refine)
│     │  ├─ signatures.py            # project-specific signatures (see §2)
│     │  └─ configs/ default.yaml    # thresholds, tool ceilings, judge rubric
│     └─ aneeta/
│        ├─ program.py
│        ├─ signatures.py
│        └─ configs/ default.yaml
│
├─ datasets/                         # data registry + loaders
│  ├─ registry.py                    # get_dataset("sharktank" | "aneeta")
│  ├─ sharktank/ {loader.py, README.md, raw/, processed/}
│  └─ aneeta/   {loader.py, README.md, raw/, processed/}
│
├─ configs/
│  ├─ base.yaml                      # shared defaults (quality/fact thresholds, paths)
│  ├─ models/ {openai.yaml, local.yaml, together.yaml,…}
│  ├─ rag.yaml                       # retrieval settings (if used)
│  └─ eval/
│     ├─ sharktank.yaml              # which split, how many samples, which optimizer(s)
│     └─ aneeta.yaml
│
├─ experiments/
│  ├─ runs/                          # artifacts per run_id (auto-created)
│  │  └─ <run_id>/
│  │     ├─ config.json              # {project, model, optimizer, seed, thresholds, …}
│  │     ├─ predictions.jsonl        # one line per sample (input, truth, prediction, latency, cost)
│  │     └─ metrics.json             # aggregate metrics for the run
│  └─ comparisons/
│     └─ <compare_id>.json           # side-by-side A vs B tables
│
├─ infra/
│  ├─ mlflow/ docker-compose.yaml    # optional tracking server
│  ├─ milvus/ docker-compose.yaml    # optional vector DB for RAG
│  └─ docker/ Dockerfile             # container for API/CLI
│
├─ tests/                            # pytest: unit + smoke tests
├─ scripts/                          # one-off utilities (ingest, cloc-measure, etc.)
├─ .github/workflows/ci.yml          # lint + type-check + tests
├─ Makefile                          # make api | eval | compare | test | lint | typecheck
├─ pyproject.toml                    # deps + tools (ruff, black, mypy, pytest)
├─ .env.example                      # safe env template
└─ README.md

Why this is clearer:
	•	apps/ = ways to run (API + CLI).
	•	src/core/ = plumbing (config/logging/artifacts/telemetry).
	•	src/dspy/ = DSPy-specific building blocks + evaluation helpers.
	•	src/adapters/ = bridge your existing SharkTank/ANEETA code into the new world.
	•	src/projects/ = the DSPy programs per project (clean, minimal).
	•	datasets/ & experiments/ cleanly separate inputs from outputs.

⸻

2) What do the signatures.py files do?

Think of a DSPy signature as a typed function contract the optimizer can learn against. It declares inputs/outputs only (no logic), so everything else—modules, optimizers, refinement—plugs in cleanly.
	•	Shared base signatures (src/dspy/signatures.py): reusable shapes you might want across projects (e.g., a generic QAInput).
	•	Project-specific signatures (src/projects/<name>/signatures.py): the exact I/O for that task (e.g., pitch generation).

Example (tiny, readable):

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

Why this matters:
	•	Stable boundaries: everything calls the program via clear inputs/outputs.
	•	Optimizer-ready: DSPy compiles prompt/weights to maximize your metrics for this signature.
	•	Low coupling: swap models or modules without breaking the API surface.
	•	Testable: signatures make great unit-test targets (validate required fields, shapes, etc.).

⸻

3) Where your proposal metrics live (and how they’re logged)

Your core + DSPy metrics go here:
	•	Task quality & factuality → src/dspy/evaluation/metrics.py + judges.py + factcheck.py
	•	Pitch Quality Score (1–10): rubric judge in judges.py
	•	Fact-Check Score (0–10): factcheck.py (exact method is pluggable)
	•	Latency & cost → src/core/telemetry.py
	•	capture per-request latency_ms, aggregate p50 / p95;
	•	capture token usage / API cost (or local inference time → $ estimate);
	•	write to predictions.jsonl (per-sample) and metrics.json (aggregates).
	•	Improvement Efficiency → also src/core/telemetry.py
	•	count refine/critique iterations to hit threshold (e.g., quality ≥ 8).
	•	Code Complexity Reduction → scripts/cloc_measure.py (simple cloc wrapper)
	•	compare LoC between legacy multi-agent path and DSPy program path; write to experiments/comparisons/<id>.json.

Schema (steady & human-readable):
	•	experiments/runs/<run_id>/config.json

{
  "project": "sharktank",
  "model": "gpt-4o-mini",
  "optimizer": "MIPROv2",
  "seed": 42,
  "thresholds": {"quality": 8.0, "fact": 8.5}
}


	•	predictions.jsonl (one line per sample)

{"sample_id": 123, "input": {...}, "truth": {... or null},
 "prediction": "...", "metrics": {"quality": 8.6, "fact": 8.9},
 "perf": {"latency_ms": 1420, "tokens_in": 2900, "tokens_out": 750, "cost_usd": 0.028},
 "iters": {"refine": 2}}


	•	metrics.json (aggregates)

{"quality_mean": 8.4, "fact_mean": 8.7,
 "latency_p50_ms": 980, "latency_p95_ms": 1720,
 "cost_per_sample_usd": 0.027,
 "improvement_efficiency_mean": 1.6}



⸻

4) Use your task delegation to drive folders & ownership

Here’s how your assignments map to the structure with clear “you own this” pointers + acceptance checks.

SharkTank

Task	Where to work	Owner	Acceptance check
Set up vanilla SharkTank for eval	src/adapters/sharktank.py, datasets/sharktank/loader.py	Zheng Kai	python apps/cli/eval.py --project sharktank runs & writes experiments/runs/<id>/
Set up & load Milvus DB	infra/milvus/docker-compose.yaml, configs/rag.yaml, src/dspy/modules.py	Zheng Kai	rag.enabled=true retrieves; context shows in predictions.jsonl
Set up MLflow for DSPy	infra/mlflow/docker-compose.yaml, src/core/telemetry.py (MLflow hook)	Zheng Kai	runs logged to MLflow; metrics visible
Recreate Pitch Tank in DSPy	src/projects/sharktank/program.py, .../signatures.py	Isaiah	Meets thresholds; passes tests
Prompt-optimise SharkTank	src/dspy/optimizers/ configs; configs/eval/sharktank.yaml	Zheng Kai	“Optimizer-only” beats vanilla on quality/fact OR latency/cost
Fine-tune SharkTank with DSPy	(optional) src/dspy/optimizers/finetune.py	Isaiah	Clear gains vs optimizer-only; report cost/ops delta
Prepare dataset for prompt/fine-tune	datasets/sharktank/processed/	Zheng Kai	Documented splits; leakage check
Evaluate performance	apps/cli/eval.py, apps/cli/compare.py	Zheng Kai + Isaiah	metrics.json & comparison table produced

ANEETA

Task	Where to work	Owner	Acceptance check
Set up vanilla ANEETA	src/adapters/aneeta.py, datasets/aneeta/loader.py	(fill in)	Eval runs write artifacts
Initialize MLflow tracking	infra/mlflow/, src/core/telemetry.py	Yanjie	Runs visible in MLflow
Local vector store + embedder	configs/rag.yaml, src/dspy/modules.py	Yanjie	Retrieval shows in context; latency split recorded
Compare Gemma quantizations	configs/models/local.yaml variants	Yanjie	Table with quality/latency/cost trade-offs
DSPy re-creation of MAS	src/projects/aneeta/program.py, .../signatures.py	Benjamin	Meets safety + quality thresholds
Run MIPROv2/Bootstrap/COPRO	src/dspy/optimizers/	Benjamin	Best config recorded & reproducible
A/B tests vs current ANEETA	apps/cli/compare.py, experiments/comparisons/	De Wang & Benjamin	Comparison JSON/CSV + short report committed

Pro tip: put owner initials in each project’s configs/default.yaml (e.g., owner: zhengkai) so it’s obvious who to ping.

⸻

TL;DR
	•	Use the clean tree above. It’s small but complete: API/CLI, DSPy glue, adapters for legacy code, datasets registry, and a clean artifacts pattern for comparisons.
	•	Signatures are typed I/O contracts that let DSPy optimize your program while keeping your boundary stable.
	•	Metrics from your proposal live in dspy/evaluation and core/telemetry, written per-run so comparison is trivial.
	•	Your delegation maps neatly to directories and acceptance checks—just follow the “Where to work” column.

If you want, I can generate this exact structure as a ZIP (with stubs and TODOs) so you can git init and start committing immediately.