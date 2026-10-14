# File map

Start with `README.md`, then `experiments/README.md` and `examples/action.json`.
The rendered semantic and formal prompts are in `examples/rendered_prompts/`;
the aggregation prompt builder is in `promptkit/aggregation.py`.

| Location | Contents |
|---|---|
| `README.md` | Setup and commands |
| `METHOD.md` | Verification stages and method definitions |
| `DATA.md` | Data fields, example index, and label sources |
| `run_sample.py` | Single-action render, infer, and verify commands |
| `experiments/01_multi_agent/` | Source-agent comparison |
| `experiments/02_adversarial/` | Seed action and L1–L4 variants |
| `experiments/03_cross_verifier/` | Shared action set across verifier profiles |
| `experimentkit/design.py` | Configuration validation and job matrix |
| `experimentkit/runner.py` | Prompt planning and sequential execution |
| `experimentkit/attacks.py` | Product-search perturbations |
| `experimentkit/summary.py` | Grouped action metrics and coverage |
| `experimentkit/cli.py` | Shared experiment CLI |
| `promptkit/semantic.py` | Semantic rubric and prompt construction |
| `promptkit/aggregation.py` | Evidence-to-aggregation prompt construction |
| `promptkit/render.py` | Input and evidence adapters for the three stages |
| `promptkit/records.py` | Response parsing, score validation, and code extraction |
| `promptkit/client.py` | Model endpoint client |
| `promptkit/formal.py` | Dafny invocation and status handling |
| `prompts/` | LaCoT template and formal layer definitions |
| `reference/schemas.py` | ActionContext and ToolCall input structures |
| `examples/` | Data example, index, rendered prompts, and Dafny examples |
| `tests/` | Unit tests and their fixtures |
| `CITATION.md`, `LICENSE`, `THIRD_PARTY_*` | Citation and attribution |

Each experiment directory includes `run.py`, `config.json`, `samples.jsonl`,
`labels.json`, and a README. The adversarial directory also includes `seed.json`
and `build_examples.py`.

Execution writes a manifest, per-stage artifacts, result records and a run
status file under the selected output directory. `.gitignore` excludes output
folders, local credentials and Python caches. `.gitattributes` preserves prompt
asset whitespace. `__init__.py` files mark Python packages.
