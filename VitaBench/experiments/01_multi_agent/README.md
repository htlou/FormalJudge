# Experiment 1: Multi-agent verification

Use one verifier profile to evaluate actions from different source agents.
The data examples contain GPT-4o and Claude Sonnet records for two task IDs.
Both `hybrid` and `semantic_only` process each record; summaries group by source
agent and method.

```bash
python experiments/01_multi_agent/run.py plan --output outputs/exp1_plan
```

Configure `FORMALJUDGE_BASE_URL`, `FORMALJUDGE_MODEL`, and optionally
`FORMALJUDGE_API_KEY`, then run:

```bash
python experiments/01_multi_agent/run.py run --execute --max-api-calls 16 \
  --dafny /path/to/dafny --output outputs/exp1_run
```

The default configuration contains eight jobs and up to sixteen model requests.
Change `config.json` or supply `--config` to evaluate other exported actions.
