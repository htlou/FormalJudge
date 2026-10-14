# Data examples

The multi-agent and cross-verifier configurations use four action records from
VitaBench task IDs `10806004` and `10807004`, with GPT-4o and
Claude Sonnet as the source agents. `examples/index.json` lists the task, agent,
and action index for each record. `examples/action.json` contains the product
search action used in the walkthrough.

## Record format

Each `samples.jsonl` row contains:

- `sample_id`: the record identifier.
- `source_agent` or `attack_level`: grouping metadata where applicable.
- `action`: task context, conversation history, assistant content, tool call,
  and tool definition.

The exported history contains the user and assistant messages preceding the
selected first tool call. `user_intent` combines the prior user messages;
`task_scenario` records the environment time. Tool parameter schemas follow the
corresponding VitaBench function signatures.

## Labels

`labels.json` is keyed by `sample_id`. For the selected trace records, labels
come from saved Claude Opus annotations, with FAIL mapped to BLOCK. For
adversarial variants, labels follow the perturbation rule in `build_examples.py`.
The `origin` field identifies the label source. Labels are used by the summary
step and are kept outside model inputs.

## Adversarial variants

Experiment 2 starts from the recorded product-search action in `seed.json`.
`build_examples.py` replaces the requested product keywords and adds the
corresponding explanation or history changes for L1–L4. The generated rows
retain their seed identifier.
