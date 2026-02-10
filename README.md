# FormalJudge

A Neuro-Symbolic Paradigm for Agentic Oversight

This repository contains the official implementation for the paper **"FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight"** by Jiayi Zhou, Hantao Lou, Yang Sheng, Jie Fu, and Yaodong Yang.

## Overview

FormalJudge is a neuro-symbolic framework that combines formal verification with large language models for robust agent safety evaluation. Unlike traditional LLM-as-a-Judge approaches, FormalJudge leverages formal specifications (Dafny) to provide verifiable safety guarantees.

### Key Results

- **16.6% average improvement** over LLM-as-a-Judge baselines across 3 benchmarks and 7 agent models
- **Weak-to-strong detection**: A 7B judge model achieves >90% accuracy detecting deceptive behaviors from 72B agents
- **Stable performance** across varying attack sophistication levels
- **Near-linear safety improvement** through iterative refinement: agents improve from 70.7% to 99.8% accuracy over three rounds with Dafny specification feedback

## Repository Structure

```
FormalJudge/
├── AgentSafetyBench/           # Behavioral safety benchmark (available now)
│   ├── data/                   # Benchmark dataset
│   ├── environments/           # 350+ agent environment simulators
│   ├── evaluation/             # Agent evaluation framework
│   ├── formal_verification/    # Formal verification pipeline
│   ├── iterative_refinement/   # Iterative refinement with feedback
│   └── score/                  # Safety scoring module
├── VitaBench/                  # Multi-domain constraint adherence (coming soon)
└── UpwardDeceivers/            # Deception detection benchmark (coming soon)
```

## Benchmarks

| Benchmark | Focus | Status |
|-----------|-------|--------|
| AgentSafetyBench | Behavioral safety evaluation | Available |
| VitaBench | Multi-domain constraint adherence | Coming soon |
| UpwardDeceivers | Deception detection | Coming soon |

## Quick Start

### Installation

```bash
pip install -r AgentSafetyBench/requirements.txt
```

### Agent Evaluation

```bash
cd AgentSafetyBench/evaluation
bash eval.sh
```

### Formal Verification

```bash
cd AgentSafetyBench/formal_verification
bash scripts/run_dafny.sh
```

### Iterative Refinement

```bash
cd AgentSafetyBench/iterative_refinement
bash run_api_refinement.sh claude-opus-4-5 3 8 dafny
```

## Method

FormalJudge employs a hierarchical 3-agent pipeline for formal verification:

1. **Agent #1 (NL Decomposition)**: Breaks down high-level safety requirements into verifiable sub-properties
2. **Agent #2 (Formal Translation)**: Translates requirements into Dafny/Python/NL specifications
3. **Agent #3 (Trace Abstraction)**: Extracts concrete values from agent execution traces
4. **Executor**: Compiles and runs formal verification to produce YES/NO verdicts

### Verification Languages

- `dafny`: Full formal verification with Dafny specifications
- `python`: Python-based verification (without formal proofs)
- `natural_language`: Natural language judgment
- `llm_cot`: LLM with chain-of-thought reasoning
- `llm_fewshot`: LLM with few-shot examples

## Supported Models

FormalJudge supports evaluation of multiple agent models:

**API Models**: Claude (Opus, Sonnet), GPT-4/5, Gemini, DeepSeek

**Local Models**: Qwen (7B-72B), Llama 3, GLM-4

## Citation

Citation information will be updated after the paper is posted to arXiv.

## License

This project is released under the MIT License. See individual benchmark directories for specific licensing information.
