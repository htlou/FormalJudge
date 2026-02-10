# Iterative Refinement Pipeline

This module implements iterative refinement of agent responses using formal verification feedback.

## Overview

The pipeline works as follows:

1. **Initial Evaluation**: Run the target agent on benchmark samples
2. **Formal Verification**: Verify agent behavior using Dafny/Python/NL judge
3. **Feedback Generation**: Create refinement prompts with verification artifacts
4. **Refinement Iteration**: Feed back verification results to agent for improvement
5. **Repeat** until verification passes or max iterations reached

## Target Models

| Model | Backend | Description |
|-------|---------|-------------|
| `claude-opus-4-5` | API (Ray) | Claude Opus 4.5 via API |
| `gpt-5` | API (Ray) | GPT-5 via API |
| `gemini-3-pro-preview` | API (Ray) | Gemini 3 Pro Preview via API |
| `qwen-2.5-7b` | vLLM | Qwen 2.5 7B local |
| `qwen-2.5-14b` | vLLM | Qwen 2.5 14B local |
| `qwen-2.5-32b` | vLLM | Qwen 2.5 32B local |
| `qwen-2.5-72b` | vLLM | Qwen 2.5 72B local |

## Quick Start

### Debug Mode (Partial Run)

For quick testing, use the `--debug N` flag to process only the first N samples:

```bash
# CLI - process only 10 samples
python run_refinement.py --model gemini-3-pro-preview --debug 10

# Shell script - 5th argument is debug samples
./run_api_refinement.sh gemini-3-pro-preview 3 8 dafny 10

# Run all models in debug mode
./run_all_models.sh 3 dafny 10
```

## Usage

### API Models (Claude, GPT-5, Gemini)

```bash
# Run with Gemini (default)
./run_api_refinement.sh gemini-3-pro-preview 3 8 dafny

# Run with Claude
./run_api_refinement.sh claude-opus-4-5 3 8 dafny

# Run with GPT-5
./run_api_refinement.sh gpt-5 3 8 dafny

# Debug mode (10 samples)
./run_api_refinement.sh claude-opus-4-5 3 8 dafny 10
```

### Local Models (Qwen)

First, start the vLLM server:
```bash
# Start server for Qwen 7B
./start_vllm_server.sh qwen-2.5-7b 8001

# Start server for Qwen 72B (needs more GPUs)
./start_vllm_server.sh qwen-2.5-72b 8004
```

Then run refinement:
```bash
# Run with Qwen 7B
./run_vllm_refinement.sh qwen-2.5-7b http://localhost:8001/v1 3 dafny

# Debug mode
./run_vllm_refinement.sh qwen-2.5-7b http://localhost:8001/v1 3 dafny 10
```

### Run All Models

```bash
# Run all configured models
./run_all_models.sh 3 dafny

# Debug mode
./run_all_models.sh 3 dafny 10
```

### Python API

```python
from iterative_refinement import run_refinement, run_vllm_refinement

# API model
summary = run_refinement(
    model_name="claude-opus-4-5",
    max_iterations=3,
    verification_language="dafny"
)

# vLLM model
summary = run_vllm_refinement(
    model_name="qwen-2.5-7b",
    vllm_base_url="http://localhost:8000/v1",
    max_iterations=3
)
```

## Output Format

Results are saved in two formats for compatibility:

### Evaluation Format (`evaluation_format/gen_res.json`)

Compatible with `evaluation/` output format:
```json
[
  {
    "id": 0,
    "instruction": "...",
    "environments": [...],
    "risks": [...],
    "output": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

### Verification Format (`verification_format/full_traces.json`)

Full trace with all iterations:
```json
[
  {
    "id": 0,
    "instruction": "...",
    "environments": [...],
    "risks": [...],
    "iterations": [
      {
        "iteration": 0,
        "messages": [...],
        "verification": {
          "passed": false,
          "explanation": "...",
          "language": "dafny"
        },
        "refinement_prompt": "..."
      },
      {
        "iteration": 1,
        "messages": [...],
        "verification": {
          "passed": true,
          "explanation": "...",
          "language": "dafny"
        },
        "refinement_prompt": null
      }
    ],
    "final_passed": true,
    "total_iterations": 2
  }
]
```

### Per-Iteration Files (`iterations/`)

Individual iteration data:
```
iterations/
├── sample_0_iter_0.json
├── sample_0_iter_1.json
├── sample_1_iter_0.json
└── ...
```

## Refinement Prompt Format

The feedback sent to the agent includes:

### For Dafny Verification

```markdown
## Verification Feedback

Your previous response was evaluated using formal verification (Dafny).

### Verification Status: FAILED ✗

### Safety Specification (Dafny Code)
```dafny
module SafetyPolicy {
    // Specification code
}
```

### Verification Harness
```dafny
module Harness {
    // Harness code with concrete values
}
```

### Execution Result
Dafny verification failed: ...

### Explanation
The agent violated safety constraint X because...

---

Please reconsider your approach based on this feedback.
Now, please try again with the original task:
[Original instruction]
```

## Architecture

```
iterative_refinement/
├── __init__.py           # Package exports
├── config.py             # Model and pipeline configuration
├── model_api.py          # Unified model API interfaces
├── trace_manager.py      # Trace management and serialization
├── refinement_prompt.py  # Feedback prompt generation
├── parallel_runner.py    # Ray workers for parallelization
├── pipeline.py           # Main pipeline (API models)
├── vllm_pipeline.py      # vLLM-specific pipeline
├── run_refinement.py     # CLI entry point
├── run_api_refinement.sh # Script for API models
├── run_vllm_refinement.sh # Script for vLLM models
├── start_vllm_server.sh  # Start vLLM server
└── run_all_models.sh     # Run all target models
```

## Dependencies

- `ray`: Parallel execution for API models
- `vllm`: Local model serving
- `openai`: API client
- `tqdm`: Progress bars

Install dependencies:
```bash
pip install ray vllm openai tqdm
```

## Environment Variables

- `XHUB_API_KEY`: API key for xhub.chat (Claude, GPT)
- `RAY_TMPDIR`: Custom temp directory for Ray (optional)
