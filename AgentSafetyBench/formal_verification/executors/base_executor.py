"""
Base class for verification executors.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import json


class BaseExecutor(ABC):
    """Base class for executing verification code."""
    
    def __init__(self, work_dir: str = "/tmp/formal_verification"):
        self.work_dir = work_dir
    
    @abstractmethod
    def execute(self, spec: str, harness: str) -> Tuple[bool, str]:
        """
        Execute the verification code.
        
        Args:
            spec: The specification/policy code
            harness: The harness code with concrete values
            
        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        pass
    
    @abstractmethod
    def validate_spec(self, spec: str) -> Tuple[bool, str]:
        """
        Validate that the specification code is syntactically correct.
        
        Args:
            spec: The specification code
            
        Returns:
            Tuple of (valid: bool, error_message: str)
        """
        pass
    
    def parse_output(self, output: str) -> Tuple[bool, str]:
        """
        Parse the output from verification execution.
        Expected format: YES/NO on a line, rest is explanation.
        Handles Dafny's extra output by searching for YES/NO pattern.

        Args:
            output: Raw output from execution

        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        if not output or not output.strip():
            return False, "No output from verification"

        # Split into lines and search for standalone YES/NO
        lines = output.strip().split('\n')
        passed = None
        result_line_idx = -1

        for idx, line in enumerate(lines):
            stripped = line.strip().upper()
            # Check for exact match (standalone YES or NO)
            if stripped == 'YES':
                passed = True
                result_line_idx = idx
                break
            elif stripped == 'NO':
                passed = False
                result_line_idx = idx
                break
            # Also check if line starts with YES or NO followed by space/punctuation
            elif stripped.startswith('YES ') or stripped.startswith('YES:') or stripped.startswith('YES.'):
                passed = True
                result_line_idx = idx
                break
            elif stripped.startswith('NO ') or stripped.startswith('NO:') or stripped.startswith('NO.'):
                passed = False
                result_line_idx = idx
                break

        if passed is None:
            return False, f"Could not parse YES/NO from output: {output[:500]}"

        # Extract explanation (everything after the YES/NO line)
        if result_line_idx != -1:
            explanation_lines = lines[result_line_idx + 1:]
            explanation = '\n'.join(explanation_lines).strip()
        else:
            explanation = output.strip()

        return passed, explanation
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """Save verification result to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

