# Experiments

All three experiments use the same action schema, prompt builders, and methods.

| Experiment | Held fixed | Varied | Default configuration |
|---|---|---|---|
| [Multi-agent](01_multi_agent/README.md) | Verifier profile | Source-agent actions | 2 agents × 2 records × 2 methods |
| [Adversarial](02_adversarial/README.md) | Seed task and verifier | L1–L4 perturbation | 4 variants × 2 methods |
| [Cross-verifier](03_cross_verifier/README.md) | Action records | Verifier profile | 4 records × 2 profiles × 2 methods |

## Methods

- `hybrid`: semantic assessment, LaCoT generation, Dafny verification, and LLM
  aggregation. Up to three model requests per action.
- `semantic_only`: the semantic branch with the 60-point decision threshold.
  One model request per action.

## Commands

```bash
python experiments/01_multi_agent/run.py plan --output outputs/exp1_plan
python experiments/02_adversarial/run.py plan --output outputs/exp2_plan
python experiments/03_cross_verifier/run.py plan --output outputs/exp3_plan
```

`plan` writes the prompt inputs and job matrix. `run` executes the matrix using
configured model endpoints and Dafny, and requires `--execute` plus
`--max-api-calls`. Use a fresh output directory for each run.

Each directory's README gives the execution command and environment variables.
Use `--config path/to/config.json` to select other action files or model profiles.

## Data and outputs

Data records are stored in `samples.jsonl`; annotations are in `labels.json`.
Their fields and sources are described in [DATA.md](../DATA.md). Labels and
grouping metadata are used for summary calculation, not prompt construction.

A run writes `manifest.json`, per-job stage files, `results.jsonl`, and
`run_status.json`. To summarize an existing run:

```bash
python experiments/01_multi_agent/run.py summarize --output outputs/exp1_run
```

Summaries report completed, failed and missing jobs, coverage, and action-level
Accuracy, Precision, Recall and F1. BLOCK is the positive class. Metrics use
completed predictions with supplied labels; compare coverage alongside scores.
The manifest fingerprints configs and inputs to keep summaries tied to the run.
