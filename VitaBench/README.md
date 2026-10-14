# FormalJudge — VitaBench

This directory provides three action-verification workflows: multi-agent
comparison, adversarial evaluation, and cross-verifier comparison. They share
LaCoT/Dafny generation, semantic assessment, and aggregation prompts.

## Layout

- `experiments/`: experiment entry points, configurations, and data examples.
- `promptkit/` and `prompts/`: prompt construction and model/solver adapters.
- `experimentkit/`: experiment execution and metric calculation.
- `examples/`: action data and rendered prompts.
- `tests/`: automated tests.

See [PACKAGE_GUIDE.md](PACKAGE_GUIDE.md) for the file map,
[METHOD.md](METHOD.md) for the pipeline, and [DATA.md](DATA.md) for data fields.

## Setup

Use Python 3.10 or newer. The Python code uses the standard library.
For model calls, configure a chat-completions-compatible endpoint:

- `FORMALJUDGE_BASE_URL`: API base URL, including the version path if needed.
- `FORMALJUDGE_MODEL`: served model name.
- `FORMALJUDGE_API_KEY`: optional authentication token.

The formal branch uses an installed Dafny 4 executable.

## Inspect the prompts

```bash
python run_sample.py render --stage semantic \
  --input examples/action.json --output outputs/semantic_prompt.txt
python run_sample.py render --stage formal \
  --input examples/action.json --output outputs/formal_prompt.txt
```

Rendered data examples are in [examples/rendered_prompts/](examples/rendered_prompts/).
The aggregation prompt is constructed in `promptkit/aggregation.py` from the
semantic and formal stage records.

## Run an action

```bash
python run_sample.py infer --stage semantic \
  --input examples/action.json --output outputs/semantic.json
python run_sample.py infer --stage formal \
  --input examples/action.json --output outputs/formal_generation.json \
  --dafny-output outputs/action.dfy
python run_sample.py verify --code outputs/action.dfy \
  --dafny /path/to/dafny --output outputs/formal.json
python run_sample.py infer --stage aggregation \
  --input examples/action.json \
  --semantic outputs/semantic.json --formal outputs/formal.json \
  --output outputs/decision.json
```

Each `infer` command makes one model request. Use a new output path for each run.

## Experiments

| Experiment | Variable | Entry point |
|---|---|---|
| Multi-agent | Source-agent action records | `experiments/01_multi_agent/run.py` |
| Adversarial | L1–L4 perturbation target | `experiments/02_adversarial/run.py` |
| Cross-verifier | Verifier model | `experiments/03_cross_verifier/run.py` |

Each entry supports `plan`, `run`, and `summarize`. See
[experiments/README.md](experiments/README.md) for configurations and commands.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover prompt rendering, parsing, experiment matrices, execution controls,
and summary calculations. Model and solver interfaces use test fixtures.

[Citation](CITATION.md) · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
