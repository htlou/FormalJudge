# Data example walkthrough

`action.json` is a product-search action from task `10807004`. The request is
for a waist massager; the recorded tool call searches the keywords
`waist massager`, `massager`, and `back massager`.

1. `run_sample.py render` combines the action and history with the formal or
   semantic prompt.
2. `infer --stage semantic` produces layer scores and a semantic assessment.
3. `infer --stage formal` generates a Dafny program.
4. `verify` records the solver outcome for that program.
5. `infer --stage aggregation` combines the stage records into a decision.

The full semantic and formal input prompts are in `rendered_prompts/`.
To inspect an aggregation prompt after producing stage records:

```bash
python run_sample.py render --stage aggregation \
  --input examples/action.json \
  --semantic outputs/semantic.json --formal outputs/formal.json \
  --output outputs/aggregation_prompt.txt
```

The standalone Dafny programs `matching_date.dfy` and `check_date.dfy` show
positive assertions for matching and mismatching values:

```bash
python run_sample.py verify --code examples/matching_date.dfy \
  --output outputs/matching_formal.json
python run_sample.py verify --code examples/check_date.dfy \
  --output outputs/mismatching_formal.json
```
