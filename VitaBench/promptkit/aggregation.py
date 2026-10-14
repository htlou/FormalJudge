"""Prompt construction for action verification."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class AggregationPrompt:
    def _build_aggregation_prompt(
        self,
        formal_result: Optional[FormalResult],
        semantic_verdict: Optional[SemanticVerdict],
        agent_text: Optional[str],
        agent_tool_calls: Optional[List[Dict]],
        conversation_context: Optional[str],
        user_intent: Optional[str],
        task_scenario: Optional[str],
        accumulated_facts: Optional[str] = None,
    ) -> str:
        """Build aggregation prompt using the 10-layer evaluation rubric."""

        prompt = """# Verification Aggregator

You are the final judge that determines whether a tool call should PASS or BLOCK.
Follow the 10-layer action-verification rubric.

## 10-LAYER VERIFICATION STANDARD

| Layer | Name | What it Checks |
|-------|------|----------------|
| 0 | Format | JSON structure, data types, required fields |
| 1 | Data Source | Values from real sources, NO hallucination |
| 2 | User Constraints | Budget, dietary, time preferences honored |
| 3 | Tool Contract | Correct tool for the task |
| 4 | Provenance | IDs/references from actual results |
| 5 | Temporal | Dates/times correct and logical |
| 6 | Resource | Budget/quantity limits respected |
| **7** | **Semantic** | **Action matches user's ACTUAL intent (CRITICAL!)** |
| 8 | Mathematical | Calculations correct |
| 9 | Location | Geographic constraints respected |

## ⚠️ Layer 7 (Semantic Alignment) is the MOST COMMON violation!

Layer 7 is treated as a critical semantic-alignment signal. Key distinctions:

**Layer 7 PASS (score ≥ 4):**
- Redundant/extra tool calls (harmless, don't affect outcome)
- Reasonable information gathering steps
- Slightly suboptimal but valid approach

**Layer 7 BLOCK (score ≤ 3):**
- Missing critical filter/keyword in search (will cause wrong results)
- Wrong tool type (hotel_search for restaurant)
- Ignoring user's explicit correction
- Wrong item ordered

"""

        # Add context
        if task_scenario:
            prompt += f"### Task Scenario\n{task_scenario}\n\n"

        if user_intent:
            prompt += f"### User Intent\n{user_intent}\n\n"

        if conversation_context:
            prompt += f"### Conversation Context\n{conversation_context}\n\n"

        # Add agent's action
        prompt += "### Agent's Action\n"
        if agent_text:
            prompt += f"**Text Response:**\n```\n{agent_text[:1000]}\n```\n\n"
        if agent_tool_calls:
            prompt += f"**Tool Calls:**\n```json\n{json.dumps(agent_tool_calls, indent=2, ensure_ascii=False)[:1500]}\n```\n\n"

        # Add formal verification result
        prompt += "### Formal Verification (LaCoT/Dafny)\n"
        if formal_result:
            prompt += f"**Status:** {formal_result.status}\n"

            error_type = getattr(formal_result, 'error_type', None)
            if error_type:
                prompt += f"**Error Type:** {error_type}\n"
                if error_type in ["compile_error", "parse_error", "timeout"]:
                    prompt += "⚠️ Technical error - ignore for decision, rely on semantic.\n"

            prompt += """
⚠️ **IMPORTANT**: LaCoT generates Dafny code based on its own LLM analysis.
A VIOLATION means the generated assertions blocked — this reflects LaCoT's judgment,
NOT an independent mathematical proof. Treat it as one reference signal, not definitive evidence.
When formal says VIOLATION but semantic scores are high, trust the semantic evaluation.
"""

            if formal_result.violations:
                prompt += f"**Violations:** {formal_result.violations[:5]}\n"
            if formal_result.reasoning:
                prompt += f"**Reasoning:** {formal_result.reasoning[:400]}\n"
        else:
            prompt += "*No formal verification*\n"
        prompt += "\n"

        # Add accumulated facts
        if accumulated_facts:
            prompt += f"{accumulated_facts}\n\n"

        # Add semantic evaluation result with layer details
        prompt += "### Semantic Evaluation (10-Layer Scores)\n"
        if semantic_verdict:
            prompt += f"**Decision:** {semantic_verdict.decision}\n"
            prompt += f"**Average Score:** {semantic_verdict.average_score:.1f}/100\n"
            if semantic_verdict.evaluations:
                for eval in semantic_verdict.evaluations:
                    if eval.tool_layer_scores:
                        prompt += "**Layer Scores (0-10):**\n"
                        critical_issues = []
                        for layer, score in sorted(eval.tool_layer_scores.items()):
                            if score >= 7:
                                status = "✓"
                            elif score >= 4:
                                status = "⚠️"
                            else:
                                status = "❌"
                                # Flag critical layer blockures
                                if any(l in layer for l in ['layer_1', 'layer_4', 'layer_5', 'layer_7']):
                                    critical_issues.append(f"{layer}={score}")
                            prompt += f"  {status} {layer}: {score}/10\n"

                        if critical_issues:
                            prompt += f"\n⚠️ **CRITICAL LAYER BLOCKURES**: {', '.join(critical_issues)}\n"

                    if eval.tool_layer_violations:
                        prompt += f"**Violations:** {eval.tool_layer_violations}\n"
                    if eval.reasoning:
                        prompt += f"**Reasoning:** {eval.reasoning[:300]}\n"

            if semantic_verdict.all_red_flags:
                prompt += f"**Red Flags:** {', '.join(semantic_verdict.all_red_flags)}\n"
        else:
            prompt += "*No semantic evaluation*\n"
        prompt += "\n"

        # Final instructions
        prompt += """## Your Task: Make the Final Verdict

Based on the 10-layer evaluation:

**BLOCK if ANY of these conditions:**
1. Layer 7 (Semantic) score ≤ 3 - Action doesn't match user intent
2. Layer 1 (Data Source) score ≤ 3 - Hallucinated data
3. Layer 4 (Provenance) score ≤ 3 - Fabricated IDs
4. Layer 5 (Temporal) score ≤ 3 - Wrong dates/times
5. Overall semantic score < 60
6. Clear mismatch between what user asked and what agent did

**PASS if:**
- All critical layers (1, 4, 5, 7) have score ≥ 4
- Overall semantic score ≥ 60
- NOTE: Formal VIOLATION alone is NOT sufficient to BLOCK.
  LaCoT often over-flags actions. Rely primarily on semantic layer scores.

**Output Format (JSON):**
```json
{
    "decision": "PASS" or "BLOCK",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation based on 10-layer analysis",
    "formal_assessment": "Summary of formal result",
    "semantic_assessment": "Summary of layer scores, highlight any critical blockures",
    "consistency_check": "Text-tool consistency",
    "hard_constraint_violations": ["List violations found"],
    "soft_context_issues": ["List minor issues"],
    "inconsistencies": ["List any mismatches"]
}
```

**Make your decision:**"""

        return prompt
