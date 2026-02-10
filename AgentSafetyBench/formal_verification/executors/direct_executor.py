"""
Direct executor - no verification, always passes.

Used as a baseline to measure agent behavior without refinement feedback.
"""
from typing import Tuple
from .base_executor import BaseExecutor


class DirectExecutor(BaseExecutor):
    """
    Executor that always passes - no verification is performed.

    This is used for the 'direct' evaluation mode where we want to measure
    agent behavior without any verification or refinement feedback.
    """

    def __init__(self, work_dir: str = "/tmp/formal_verification/direct"):
        super().__init__(work_dir)

    def execute(self, spec: str, harness: str) -> Tuple[bool, str]:
        """
        Execute direct evaluation (always passes).

        Args:
            spec: Not used
            harness: Not used

        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        return True, "Direct evaluation mode - no verification performed"

    def validate_spec(self, spec: str) -> Tuple[bool, str]:
        """
        Validate specification (always valid in direct mode).

        Args:
            spec: The specification (not used)

        Returns:
            Tuple of (valid: bool, error_message: str)
        """
        return True, "Direct mode - no specification validation"
