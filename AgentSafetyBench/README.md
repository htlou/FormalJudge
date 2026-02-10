# AgentSafetyBench

Behavioral Safety Evaluation Benchmark for FormalJudge

This directory contains the AgentSafetyBench component of the [FormalJudge](../README.md) framework, adapted from [Agent-SafetyBench](https://arxiv.org/abs/2412.14470) with extensions for formal verification and iterative refinement.

## Overview

AgentSafetyBench provides a comprehensive agent safety evaluation benchmark with:
- **350+ diverse environments** covering real-world agent scenarios
- **1000+ benchmark samples** with risk categories and failure modes
- **Formal verification pipeline** using Dafny specifications
- **Iterative refinement** with verification feedback

## Directory Structure

```
AgentSafetyBench/
├── data/
│   └── released_data.json      # Benchmark dataset (~65K lines)
├── environments/               # 350+ environment simulators
│   ├── *.py                    # Python implementations
│   ├── *.json                  # Tool specifications
│   └── BaseEnv.py              # Abstract base class
├── evaluation/                 # Agent evaluation framework
│   ├── eval.py                 # Single-worker evaluation
│   ├── eval_parallel.py        # Ray-based parallel evaluation
│   └── model_api/              # Model API implementations
├── formal_verification/        # Formal verification pipeline
│   ├── run_verification.py     # Entry point
│   ├── prompts/                # Agent prompts
│   ├── executors/              # Verification executors
│   └── pipeline/               # Verification orchestration
├── iterative_refinement/       # Iterative refinement pipeline
│   ├── run_refinement.py       # Entry point
│   ├── pipeline.py             # API model pipeline
│   ├── vllm_pipeline.py        # vLLM model pipeline
│   └── config.py               # Configuration
└── score/                      # Safety scoring
    └── eval_with_shield.py     # ShieldAgent evaluation
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Agent Evaluation

Evaluate an agent model on the benchmark:

```bash
cd evaluation
bash eval.sh
```

Configure the model in `eval.sh`:
```bash
python -u eval.py --model_name gpt-5
```

Supported models: Claude, GPT, Gemini, Qwen, Llama, DeepSeek, GLM-4

### 2. Safety Scoring

Compute safety scores using [ShieldAgent](https://huggingface.co/thu-coai/ShieldAgent):

```bash
cd score
bash eval_with_shield.sh
```

### 3. Formal Verification

Run formal verification on agent outputs:

```bash
cd formal_verification

# Dafny formal verification
bash scripts/run_dafny.sh

# Python verification
bash scripts/run_python.sh

# Natural language judgment
bash scripts/run_nl.sh
```

Command-line options:
```bash
python run_verification.py \
    --data_path ../data/released_data.json \
    --agent_output ../evaluation/evaluation_results/gpt-5/gen_res.json \
    --output_dir ./results \
    --language dafny \
    --model_name gpt-4o \
    --num_workers 8
```

### 4. Iterative Refinement

Improve agent safety through iterative feedback:

```bash
cd iterative_refinement

# API models (Claude, GPT, Gemini)
bash run_api_refinement.sh claude-opus-4-5 3 8 dafny

# vLLM models (Qwen)
bash run_vllm_refinement.sh qwen-2.5-7b http://localhost:8001/v1 3 dafny
```

## Formal Verification Pipeline

FormalJudge uses a hierarchical 3-agent pipeline:

```
Agent #1 (NL Decomposition)
    ↓ Breaks down safety requirements
Agent #2 (Formal Translation)
    ↓ Generates Dafny/Python/NL specs
Agent #3 (Trace Abstraction)
    ↓ Extracts values from traces
Executor (Verify)
    ↓ Produces YES/NO verdict
```

### Verification Languages

| Language | Description |
|----------|-------------|
| `dafny` | Full formal verification with Dafny |
| `python` | Python-based verification |
| `natural_language` | Natural language judgment |
| `llm_cot` | LLM with chain-of-thought |
| `llm_fewshot` | LLM with few-shot examples |

## Environments

The benchmark includes 350+ diverse environments:

- **Communication**: Email, SMS, Slack, PublicForum
- **Financial**: AssetManagement, RetailFraudDetection, AntiMoneyLaundering
- **Healthcare**: PredictiveHealthAnalytics, RemoteSurgerySafety
- **Smart Systems**: SmartCityTrafficFlow, SmartHomeSecuritySystem
- **Security**: SecurityManager, SafeChildContentFilter
- **Advanced**: QuantumCommunicationNetwork, AsteroidMiningOperation

Each environment provides:
- Python implementation (`*.py`) with tool methods
- JSON specification (`*.json`) with tool schemas

## Output Formats

### Evaluation Output
```json
[{
  "id": 0,
  "instruction": "...",
  "environments": [...],
  "risks": [...],
  "output": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
}]
```

### Verification Output
```json
{
  "stage1_decomposition.json": "Agent #1 outputs",
  "stage2_spec.json": "Agent #2 outputs",
  "stage3_verification_input.json": "Agent #3 outputs",
  "execution_results.json": "Final verdicts",
  "summary.json": "Statistics"
}
```