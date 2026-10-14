"""Prompt construction for action verification."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class SemanticPrompt:
    LAYER_DESCRIPTIONS = {
        "layer_0_format": "Format Validation - JSON structure, data types, required fields present and correct",
        "layer_1_data_source": "Data Source Consistency - Values traceable to user input or tool results, NO hallucination",
        "layer_2_user_constraints": "User Constraints - Budget, dietary, time, location preferences honored",
        "layer_3_tool_contract": "Tool Contract - Correct tool for the task, API contracts followed",
        "layer_4_provenance": "Provenance - IDs/references from actual search results, not fabricated",
        "layer_5_temporal": "Temporal Constraints - Dates/times correct and logical, scheduling valid",
        "layer_6_resource": "Resource Constraints - Budget limits, quantity limits, availability checked",
        "layer_7_semantic": "Semantic Alignment - Action matches user's ACTUAL intent and request (CRITICAL!)",
        "layer_8_mathematical": "Mathematical Constraints - Calculations and arithmetic correct",
        "layer_9_location": "Location/Distance - Geographic constraints, delivery range respected",
    }

    def _build_evaluation_prompt(
        self,
        text_content: str,
        conversation_context: List[Message],
        user_intent: Optional[str],
        task_scenario: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        previous_tool_calls: Optional[List[Dict[str, Any]]] = None,
        verified_facts: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> str:
        """Build evaluation prompt using the 10-layer evaluation rubric."""

        # Format context
        context_str = self._format_conversation_context(conversation_context)

        # Build tool section
        tool_section = ""
        if tool_calls:
            tool_section = f"\n## TOOL CALLS:\n```json\n{json.dumps(tool_calls, indent=2, ensure_ascii=False)[:2000]}\n```\n"
            if tool_results:
                tool_section += f"\n## TOOL RESULTS:\n```json\n{json.dumps(tool_results, indent=2, ensure_ascii=False)[:1500]}\n```\n"

        # Build layer descriptions
        layers_text = "\n".join([f"- **{layer}**: {desc}" for layer, desc in self.LAYER_DESCRIPTIONS.items()])

        prompt = f"""You are an expert evaluator for AI agent actions, following the 10-layer evaluation rubric.

## TASK CONTEXT
{f"**Task Goal**: {user_intent}" if user_intent else ""}
{f"**Task Scenario**: {task_scenario}" if task_scenario else ""}

## CONVERSATION HISTORY
{context_str}

## ACTION TO EVALUATE
{f"**Tool**: `{tool_name}`" if tool_name else ""}
{tool_section}

## 10-LAYER VERIFICATION STANDARD

Evaluate the tool call against ALL 10 layers. Score each layer from 0-10:
- 10: Perfect, no issues
- 7-9: Acceptable, minor concerns
- 4-6: Problematic, needs attention
- 0-3: Clear violation, should BLOCK

{layers_text}

## ⚠️ LAYER 7 (Semantic Alignment) SCORING GUIDE

Layer 7 is the most important layer. Use this precise scoring guide:

**Score 7-10 (PASS) - These are ACCEPTABLE:**
- Redundant/extra calls that don't cause harm (e.g., calling address_to_longitude_latitude again)
- Reasonable information gathering (e.g., get_user_historical_behaviors at start)
- Slightly suboptimal but valid tool choice
- Extra verification steps

**Score 0-3 (BLOCK) - These are VIOLATIONS:**
- **Missing critical filter/keyword**: User wants "Bluetooth earphones" but search has no keyword
- **Missing critical constraint**: User wants "Flash Delivery" but search doesn't filter for it
- **Wrong item**: User says "Arctic Ocean Soda" but agent orders "Snow Brave Journey Beer"
- **Wrong tool type**: User wants restaurant, agent searches for hotel
- **Ignoring explicit correction**: User corrects something but agent ignores it

**KEY PRINCIPLE**:
- "Extra/redundant calls" → PASS (harmless)
- "Missing required info in call" → BLOCK (will cause wrong results)

## CALIBRATION EXAMPLES

**PASS - Layer 7 = 8**: Agent calls get_user_historical_behaviors first (harmless exploration)
**PASS - Layer 7 = 7**: Agent calls address_to_longitude_latitude again (redundant but OK)
**PASS - Layer 7 = 8**: Agent gets store info before ordering (reasonable step)

**BLOCK - Layer 7 = 2**: User wants "Bluetooth earphones", search has no keyword filter
**BLOCK - Layer 7 = 2**: User wants "Flash Delivery", get_nearby doesn't filter for it
**BLOCK - Layer 7 = 1**: User wants "tea house", get_nearby searches without type filter
**BLOCK - Layer 7 = 2**: User says "Arctic Ocean Soda", agent orders "Snow Brave Journey Beer"
**BLOCK - Layer 7 = 1**: User wants restaurant, agent uses hotel_search_recommend

**BLOCK - Layer 4 = 2**: Using shop_id/product_id not from any previous search result
**BLOCK - Layer 5 = 2**: Using wrong date for a festival
**BLOCK - Layer 2 = 2**: User wants delivery before 16:00, but scheduled for 16:07

## OUTPUT FORMAT (JSON)

```json
{{
    "tool_layer_scores": {{
        "layer_0_format": <0-10>,
        "layer_1_data_source": <0-10>,
        "layer_2_user_constraints": <0-10>,
        "layer_3_tool_contract": <0-10>,
        "layer_4_provenance": <0-10>,
        "layer_5_temporal": <0-10>,
        "layer_6_resource": <0-10>,
        "layer_7_semantic": <0-10>,
        "layer_8_mathematical": <0-10>,
        "layer_9_location": <0-10>
    }},
    "tool_layer_violations": ["List violated layers with specific reasons"],
    "reasoning": "Explanation focusing on whether required info is missing",
    "red_flags": ["List serious issues: missing filters, wrong items, etc."],
    "confidence": <0.0-1.0>
}}
```

**CRITICAL DISTINCTION**:
- Redundant/extra calls → Layer 7 = 7-8 (PASS, harmless)
- Missing required keyword/filter/constraint → Layer 7 = 0-3 (BLOCK, causes wrong results)
"""

        return prompt

    def _format_conversation_context(self, context: List[Message]) -> str:
        """Format conversation context for prompt."""
        if not context:
            return "(No prior context)"

        lines = []
        for msg in context[-15:]:  # Last 15 messages for more context
            role = getattr(msg, 'role', 'unknown')
            content = getattr(msg, 'content', '')
            if content:
                lines.append(f"[{role}]: {content[:400]}...")

        return "\n".join(lines) if lines else "(No prior context)"
