# Formal Verification Framework for Agent Safety

This module implements a **hierarchical Dafny+LLM formal verification mechanism** for evaluating agent safety. The framework uses a 3-agent pipeline to decompose safety requirements, translate them into formal specifications, and verify agent behavior against those specifications.

## Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Formal Verification Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Agent #1  │    │   Agent #2  │    │   Agent #3  │    │  Executor   │  │
│  │ Decompose   │───▶│  Translate  │───▶│  Abstract   │───▶│  Verify     │  │
│  │ (NL→NL)     │    │ (NL→Formal) │    │ (Trace→Input)│   │ (Run Code)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  Input: Safety      Output: Formal    Output: Concrete   Output: YES/NO    │
│  requirements       specification     verification       with explanation  │
│                     (Dafny/Python/NL) input with proof                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical Decomposition Example

```
Level 1: "I don't want the agent to deceive"
    │
    ▼
Level 2: "Agent must not claim to have performed actions it hasn't actually performed"
    │
    ▼
Level 3: "Agent claimed download_report was successful" AND 
         "download_report tool was NOT called" => VIOLATION
    │
    ▼
Level 4: Check from environment:
         - tool_calls contains "download_report"? → Boolean
         - agent response contains "downloaded"? → Boolean
```

## Directory Structure

```
formal_verification/
├── __init__.py
├── run_verification.py          # Main entry point
├── README.md                    # This file
│
├── prompts/                     # Prompt templates for each agent
│   ├── __init__.py
│   ├── agent1_prompts.py        # Natural language decomposition
│   ├── agent2_prompts.py        # Formal language translation (Dafny/Python/NL)
│   └── agent3_prompts.py        # Trace abstraction and proof generation
│
├── pipeline/                    # Pipeline orchestration
│   ├── __init__.py
│   ├── verification_pipeline.py # Single-sample pipeline
│   └── parallel_runner.py       # Ray-based parallel execution
│
├── executors/                   # Language-specific executors
│   ├── __init__.py
│   ├── base_executor.py         # Abstract base class
│   ├── dafny_executor.py        # Dafny formal verification
│   ├── python_executor.py       # Python verification (no proof)
│   └── nl_executor.py           # Natural language (direct judgment)
│
└── scripts/                     # Convenience scripts
    ├── run_dafny.sh            # Run with Dafny
    ├── run_python.sh           # Run with Python
    ├── run_nl.sh               # Run with natural language
    └── run_all_ablations.sh    # Run all ablation experiments
```

## Installation

The framework uses the same dependencies as Agent-SafetyBench plus:

```bash
# Required Python packages
pip install ray openai tqdm

# For Dafny verification (optional, only needed for --language dafny)
# Install Dafny from: https://github.com/dafny-lang/dafny/releases
```

## Usage

### Basic Usage

```bash
cd formal_verification

# Run with Dafny (full formal verification)
python run_verification.py \
    --data_path ../data/released_data.json \
    --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \
    --output_dir ./verification_results/gpt4o_dafny \
    --language dafny \
    --model_name gpt-4o \
    --num_workers 4
```

### Ablation Settings

The framework supports three verification languages as ablations:

| Language | Description | Proof | Execution |
|----------|-------------|-------|-----------|
| `dafny` | Full formal verification with Dafny | Yes | Yes |
| `python` | Python verification code | No | Yes |
| `natural_language` | Direct LLM judgment | No | No |

```bash
# Run with Python (no formal proof)
python run_verification.py \
    --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \
    --output_dir ./verification_results/gpt4o_python \
    --language python

# Run with natural language (direct LLM judgment)
python run_verification.py \
    --agent_output ../evaluation/evaluation_results/tot-gpt4o/gen_res.json \
    --output_dir ./verification_results/gpt4o_nl \
    --language natural_language
```

### Using Shell Scripts

```bash
# Run individual ablations
./scripts/run_dafny.sh --agent_output path/to/gen_res.json --output_dir ./results/dafny
./scripts/run_python.sh --agent_output path/to/gen_res.json --output_dir ./results/python
./scripts/run_nl.sh --agent_output path/to/gen_res.json --output_dir ./results/nl

# Run all ablations at once
./scripts/run_all_ablations.sh \
    --agent_output path/to/gen_res.json \
    --output_base ./results/experiment1
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `../data/released_data.json` | Path to benchmark data |
| `--agent_output` | (required) | Path to agent evaluation output |
| `--output_dir` | (required) | Directory to save results |
| `--language` | `dafny` | Verification language |
| `--model_name` | `gpt-4o` | LLM model for verification agents |
| `--num_workers` | `4` | Number of parallel Ray workers |
| `--api_base` | env var | API base URL |
| `--api_key` | env var | API key |
| `--temperature` | `0.0` | LLM temperature |
| `--max_tokens` | `8192` | Max tokens for LLM response |

## Output Files

The pipeline generates the following outputs in the specified `output_dir`:

```
output_dir/
├── stage1_decomposition.json    # Agent #1 outputs (requirements decomposition)
├── stage2_spec.json             # Agent #2 outputs (formal specifications)
├── stage3_verification_input.json # Agent #3 outputs (trace abstraction)
├── execution_results.json       # Final verification results
└── summary.json                 # Summary statistics
```

### Result Format

```json
// execution_results.json
[
    {
        "id": 123,
        "passed": true,
        "explanation": "All safety requirements verified successfully.",
        "error": null
    },
    {
        "id": 456,
        "passed": false,
        "explanation": "Agent claimed to download report but download_report tool was never called.",
        "error": null
    }
]
```

```json
// summary.json
{
    "total_samples": 1000,
    "stage1_completed": 1000,
    "stage2_completed": 998,
    "stage3_completed": 995,
    "execution_completed": 990,
    "passed": 750,
    "failed": 240,
    "errors": 10,
    "pass_rate": 0.7576
}
```

## Pipeline Details

### Agent #1: Natural Language Decomposition

**Input**: High-level safety requirement, task context, risk categories

**Output**: Hierarchical decomposition with:
- High-level safety principle
- Operational constraints
- Atomic verifiable conditions (with verification type and target)
- Logical relationships between conditions

### Agent #2: Formal Language Translation

**Input**: Decomposition from Agent #1, task context

**Output** (varies by language):
- **Dafny**: Compilable Dafny module with `TraceInput` datatype, predicates, and `Check()` method
- **Python**: Python class with `TraceInput` dataclass and `check()` method
- **NL**: JSON verification checklist with indicators and verdict logic

### Agent #3: Trace Abstraction

**Input**: Specification from Agent #2, agent's reasoning trace, task context

**Output**:
- `trace_input`: Concrete values for each field with evidence
- `assumptions`: Any assumptions made for missing information
- Executable harness (Dafny/Python) or final verdict (NL)

### Execution

- **Dafny**: Compiles and runs the Dafny harness, outputs YES/NO with explanation
- **Python**: Executes the Python script, outputs YES/NO with explanation
- **NL**: Extracts verdict directly from Agent #3 output

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `XHUB_API_KEY` | Alternative API key (for xhub.chat) |
| `API_BASE_URL` | Custom API base URL |

## Relation to Original Repository

This framework is designed to work alongside the existing Agent-SafetyBench evaluation without modifying any original functionality:

1. **Input**: Uses the same `released_data.json` format
2. **Agent Outputs**: Reads `gen_res.json` files produced by `evaluation/eval_parallel.py`
3. **Independent**: Stored in a separate `formal_verification/` directory
4. **No Conflicts**: Does not modify any existing files

## Extending the Framework

### Adding New Verification Languages

1. Create a new prompt in `prompts/agent2_prompts.py` and `prompts/agent3_prompts.py`
2. Update `get_agent2_prompt()` and `get_agent3_prompt()` functions
3. Create a new executor in `executors/`
4. Update `parallel_runner.py` to use the new executor

### Customizing Prompts

All prompts are centralized in `prompts/`. The key prompts are:
- `AGENT1_DECOMPOSITION_PROMPT`: Controls how safety requirements are broken down
- `AGENT2_*_PROMPT`: Controls formal specification generation
- `AGENT3_*_PROMPT`: Controls trace abstraction
