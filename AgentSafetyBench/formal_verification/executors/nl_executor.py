"""
Natural language executor for verification (no execution, result from Agent 3).
"""
import json
import re
from typing import Tuple
from .base_executor import BaseExecutor


class NaturalLanguageExecutor(BaseExecutor):
    """Executor for natural language verification (passthrough from Agent 3)."""

    def __init__(self, work_dir: str = "/tmp/formal_verification/nl"):
        super().__init__(work_dir)

    def execute(self, spec: str, harness: str) -> Tuple[bool, str]:
        """
        Execute natural language verification.
        For NL mode, the 'harness' is the Agent 3 output with verdict.

        Args:
            spec: The safety guidelines (not used in execution)
            harness: The Agent 3 output with verdict (text or JSON)

        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        try:
            # Try to parse as the new simple format first
            # Format: VERDICT: YES/NO\nEXPLANATION: ...
            verdict_match = re.search(r'VERDICT:\s*(YES|NO)', harness, re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).upper()
                passed = verdict == 'YES'

                # Extract explanation
                explanation_match = re.search(r'EXPLANATION:\s*(.+)', harness, re.IGNORECASE | re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    explanation = harness

                return passed, explanation

            # Fallback: Try to parse as JSON (legacy format)
            try:
                result = json.loads(harness)
                verdict = result.get('final_verdict', 'NO').upper()
                passed = verdict == 'YES'
                explanation = result.get('explanation', 'No explanation provided')
                return passed, explanation
            except json.JSONDecodeError:
                pass

            # Last resort: Look for YES/NO anywhere in the output
            if 'YES' in harness.upper() and 'NO' not in harness.upper():
                return True, harness
            else:
                return False, harness

        except Exception as e:
            return False, f"Error processing NL verification: {str(e)}"

    def validate_spec(self, spec: str) -> Tuple[bool, str]:
        """
        Validate NL specification (safety guidelines).

        Args:
            spec: The safety guidelines

        Returns:
            Tuple of (valid: bool, error_message: str)
        """
        # For the simplified NL mode, any non-empty spec is valid
        if spec and spec.strip():
            return True, "Specification is valid"
        return False, "Empty specification"

