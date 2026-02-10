"""
Python executor for verification (without formal proof).
"""
import os
import subprocess
import tempfile
import sys
from typing import Tuple
from .base_executor import BaseExecutor


class PythonExecutor(BaseExecutor):
    """Executor for Python verification code."""
    
    def __init__(self, work_dir: str = "/tmp/formal_verification/python"):
        super().__init__(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        
    def execute(self, spec: str, harness: str) -> Tuple[bool, str]:
        """
        Execute Python verification.
        
        Args:
            spec: The Python specification module
            harness: The Python script with concrete values
            
        Returns:
            Tuple of (passed: bool, explanation: str)
        """
        # Create temporary directory for this execution
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            spec_path = os.path.join(tmpdir, "safety_policy.py")
            harness_path = os.path.join(tmpdir, "harness.py")
            
            # Write spec file
            with open(spec_path, 'w', encoding='utf-8') as f:
                f.write(spec)
            
            # Write harness file
            with open(harness_path, 'w', encoding='utf-8') as f:
                f.write(harness)
            
            try:
                # Run the verification script
                result = subprocess.run(
                    [sys.executable, harness_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )
                
                if result.returncode != 0:
                    return False, f"Execution failed:\n{result.stderr}\n{result.stdout}"
                
                return self.parse_output(result.stdout)
                
            except subprocess.TimeoutExpired:
                return False, "Python execution timed out"
            except Exception as e:
                return False, f"Execution error: {str(e)}"
    
    def validate_spec(self, spec: str) -> Tuple[bool, str]:
        """
        Validate Python specification syntax.
        
        Args:
            spec: The Python specification code
            
        Returns:
            Tuple of (valid: bool, error_message: str)
        """
        try:
            compile(spec, '<string>', 'exec')
            return True, "Specification is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

