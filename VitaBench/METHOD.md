# Verification pipeline

```text
Action + history + tool definitions
              |
     +--------+--------+
     |                 |
LaCoT/Dafny prompt   Semantic prompt
     |                 |
Generated code      Layer scores
     |                 |
Dafny verification     |
     +--------+--------+
              |
       Aggregation prompt
              |
       Action-level judgment
```

## Formal branch

`prompts/layers.txt` defines the verification layers.
`prompts/formal_generation.txt` combines the task, dialogue, tool definitions,
and layer analysis instructions to request a Dafny program.
`promptkit/formal.py` invokes `dafny verify` on the generated file.

A proof concerns the encoded specification and assumptions. Fact extraction
and specification construction are part of the surrounding workflow.

## Semantic branch

`promptkit/semantic.py` builds the ten-layer semantic evaluation prompt. It
includes conversation context, the current tool call, scoring rules,
calibration examples, and the response schema.

`promptkit/records.py` validates response fields. With at least five layer
scores, the overall score is their mean scaled to 0–100; otherwise the response
must provide an overall score. `semantic_only` maps a score of at least 60 to
PASS and a lower score to BLOCK.

## Aggregation

`promptkit/aggregation.py` combines the formal record, semantic layer scores,
action, and task context into a final PASS/BLOCK request. The final action
judgment is separate from the solver status.

The formal and semantic layers have branch-specific definitions. The renderer
handles template escaping before inserting task and conversation text.

## Experiment execution

`experimentkit/runner.py` runs the shared stages. The `hybrid` method uses three
model requests per action: semantic assessment, formal generation, and
aggregation. The `semantic_only` control uses one request. The same method
implementation is used across all three experiment configurations.
