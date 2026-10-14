# Experiment 3: Cross-verifier comparison

Evaluate the same action records and methods across different verifier
profiles. Every profile receives the same four records. A profile selects the
model used for all LLM stages of a hybrid job.

```bash
python experiments/03_cross_verifier/run.py plan --output outputs/exp3_plan
```

| Profile | Endpoint variable | Model variable | Optional key variable |
|---|---|---|---|
| judge_a | `JUDGE_A_BASE_URL` | `JUDGE_A_MODEL` | `JUDGE_A_API_KEY` |
| judge_b | `JUDGE_B_BASE_URL` | `JUDGE_B_MODEL` | `JUDGE_B_API_KEY` |

Configure distinct endpoint/model pairs, then run:

```bash
python experiments/03_cross_verifier/run.py run --execute --max-api-calls 32 \
  --dafny /path/to/dafny --output outputs/exp3_run
```

The default configuration contains sixteen jobs and up to thirty-two model
requests. Summaries group by verifier and method.
