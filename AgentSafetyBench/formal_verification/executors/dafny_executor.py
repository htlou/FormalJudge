"""
Dafny executor for formal verification.
"""
import os
import subprocess
import tempfile
import re
from typing import Tuple, Optional, Dict
from .base_executor import BaseExecutor


class DafnyExecutor(BaseExecutor):
    """Executor for Dafny formal verification."""

    def __init__(self, work_dir: str = "/tmp/formal_verification/dafny", cache_dir: Optional[str] = None, timeout: int = 120):
        super().__init__(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.timeout = timeout  # Configurable timeout for parallel execution
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def normalize_harness(self, harness: str) -> str:
        """
        Normalize the harness to fix common issues:
        - Ensure proper import opened SafetyPolicy
        - Fix method call syntax
        - Ensure module wrapper
        - Fix print statement syntax
        - Fix named argument syntax (: -> :=) for datatype constructors

        Args:
            harness: Raw harness code from Agent 3

        Returns:
            Normalized harness code
        """
        normalized = harness.strip()

        # Fix named argument syntax in datatype constructors
        # GPT-4o often generates `fieldName: value` but Dafny requires `fieldName := value`
        # This pattern matches: whitespace + identifier + colon + space + value (true/false/number)
        # but NOT already correct := syntax
        normalized = self._fix_named_arguments(normalized)
        
        # Remove any existing include statements (we'll add it back at the end)
        normalized = re.sub(r'include\s+"[^"]+"\s*\n*', '', normalized, flags=re.IGNORECASE)
        
        # Check if harness has a module wrapper
        has_module = bool(re.search(r'\bmodule\s+\w+\s*\{', normalized, re.IGNORECASE))
        
        # Check if it has import opened SafetyPolicy
        has_import_opened = bool(re.search(r'import\s+opened\s+SafetyPolicy', normalized, re.IGNORECASE))
        
        # If no module wrapper, wrap Main() in a module
        if not has_module:
            # Find Main() method declaration
            main_match = re.search(r'\bmethod\s+Main\s*\([^)]*\)\s*\{', normalized, re.IGNORECASE | re.DOTALL)
            if main_match:
                # Find the matching closing brace
                brace_count = 0
                pos = main_match.end() - 1
                main_end = len(normalized)
                
                while pos < len(normalized):
                    if normalized[pos] == '{':
                        brace_count += 1
                    elif normalized[pos] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            main_end = pos + 1
                            break
                    pos += 1
                
                main_method = normalized[main_match.start():main_end]
                before_main = normalized[:main_match.start()].rstrip()
                after_main = normalized[main_end:].strip()
                
                # Wrap in module with import opened
                normalized = f"""{before_main}

module Harness {{
    import opened SafetyPolicy

{main_method}
}}"""
                if after_main:
                    normalized += "\n" + after_main
                has_module = True
        
        # Add import opened if missing (and we have a module)
        if has_module and not has_import_opened:
            # Find module declaration and insert import after opening brace
            module_match = re.search(r'\bmodule\s+\w+\s*\{', normalized, re.IGNORECASE)
            if module_match:
                module_end = module_match.end()
                # Check if there's already an import (without opened)
                if not re.search(r'import\s+SafetyPolicy', normalized[module_end:module_end+200], re.IGNORECASE):
                    normalized = normalized[:module_end] + "\n    import opened SafetyPolicy\n" + normalized[module_end:]
        
        # Fix method call syntax: var (result, explanation) := Check(...) -> var result, explanation := Check(...)
        normalized = re.sub(
            r'var\s+\((\w+),\s*(\w+)\)\s*:=\s*Check\s*\(',
            r'var \1, \2 := Check(',
            normalized
        )
        
        # Fix print statement syntax: print("text") -> print "text"
        normalized = re.sub(
            r'print\s*\(\s*"([^"]+)"\s*\)',
            r'print "\1"',
            normalized
        )
        # Fix print(variable) -> print variable
        normalized = re.sub(
            r'print\s*\(\s*(\w+)\s*\)',
            r'print \1',
            normalized
        )
        
        # Fix if statement syntax: if (condition) -> if condition (but preserve parentheses in complex expressions)
        # Only fix simple cases to avoid breaking complex conditions
        normalized = re.sub(
            r'if\s+\((\w+\s*==\s*\w+)\)\s*\{',
            r'if \1 {',
            normalized
        )
        
        # Ensure include statement at the top
        if not normalized.strip().startswith('include'):
            normalized = 'include "safety_policy.dfy"\n\n' + normalized
        
        return normalized
        
    def execute(self, spec: str, harness: str, sample_id: Optional[int] = None) -> Tuple[bool, str, Dict[str, str]]:
        """
        Execute Dafny verification.
        
        Args:
            spec: The Dafny specification module
            harness: The Dafny harness with Main() method
            sample_id: Optional sample ID for caching files
            
        Returns:
            Tuple of (passed: bool, explanation: str, file_paths: dict)
            file_paths contains: 'spec_path', 'harness_path', 'normalized_harness_path'
        """
        file_paths = {}
        tmpdir = None
        
        # Save to cache if cache_dir and sample_id are provided
        if self.cache_dir and sample_id is not None:
            sample_cache_dir = os.path.join(self.cache_dir, f"sample_{sample_id}")
            os.makedirs(sample_cache_dir, exist_ok=True)
            
            spec_path = os.path.join(sample_cache_dir, "safety_policy.dfy")
            harness_path = os.path.join(sample_cache_dir, "harness_original.dfy")
            normalized_harness_path = os.path.join(sample_cache_dir, "harness.dfy")
            
            # Write spec file
            with open(spec_path, 'w', encoding='utf-8') as f:
                f.write(spec)
            file_paths['spec_path'] = spec_path
            
            # Write original harness
            with open(harness_path, 'w', encoding='utf-8') as f:
                f.write(harness)
            file_paths['harness_original_path'] = harness_path
            
            # Normalize and write harness file
            normalized_harness = self.normalize_harness(harness)
            with open(normalized_harness_path, 'w', encoding='utf-8') as f:
                f.write(normalized_harness)
            file_paths['harness_path'] = normalized_harness_path
            
            # Use cached paths for execution
            execution_spec_path = spec_path
            execution_harness_path = normalized_harness_path
        else:
            # Create temporary directory for this execution
            tmpdir = tempfile.mkdtemp(dir=self.work_dir)
            execution_spec_path = os.path.join(tmpdir, "safety_policy.dfy")
            execution_harness_path = os.path.join(tmpdir, "harness.dfy")
            
            # Write spec file
            with open(execution_spec_path, 'w', encoding='utf-8') as f:
                f.write(spec)
            
            # Normalize and write harness file
            normalized_harness = self.normalize_harness(harness)
            with open(execution_harness_path, 'w', encoding='utf-8') as f:
                f.write(normalized_harness)
        
        try:
            # Run dafny with Python target for faster execution
            # --target:py compiles to Python instead of C# (faster)
            # --no-verify skips verification (already done separately if needed)
            try:
                run_result = subprocess.run(
                    ["dafny", "run", "--target:py", "--no-verify", execution_harness_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if run_result.returncode != 0:
                    error_output = run_result.stderr or run_result.stdout
                    if len(error_output) > 2000:
                        error_output = error_output[:2000] + "\n... (error message truncated)"
                    return False, f"Dafny failed:\n{error_output}", file_paths

                passed, explanation = self.parse_output(run_result.stdout)
                return passed, explanation, file_paths

            except subprocess.TimeoutExpired:
                return False, f"Dafny execution timed out (exceeded {self.timeout}s)", file_paths
            except FileNotFoundError:
                return False, "Dafny not found. Please install Dafny and add it to PATH.", file_paths
        finally:
            # Clean up temporary directory if not using cache
            if not (self.cache_dir and sample_id is not None):
                import shutil
                if os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir)

    def _fix_named_arguments(self, code: str) -> str:
        """
        Fix named argument syntax in Dafny datatype constructors.

        GPT-4o often generates Python/TypeScript style `fieldName: value` but Dafny
        requires `fieldName := value` for named datatype constructor arguments.

        This method carefully fixes this issue while avoiding false positives like:
        - Type annotations (e.g., `var x: int`)
        - Module imports (e.g., `import opened SafetyPolicy`)
        - Method signatures (e.g., `returns (result: Answer)`)

        Args:
            code: Dafny code that may contain incorrect named argument syntax

        Returns:
            Code with corrected named argument syntax
        """
        lines = code.split('\n')
        fixed_lines = []
        in_datatype_constructor = False
        paren_depth = 0

        for line in lines:
            # Track if we're inside a datatype constructor call (e.g., TraceInput(...))
            # Look for pattern: identifier followed by opening paren
            if re.search(r'\b[A-Z]\w*\s*\($', line.rstrip()) or re.search(r'\b[A-Z]\w*\s*\(', line):
                # Check if this looks like a datatype constructor (starts with uppercase)
                constructor_match = re.search(r'\b([A-Z]\w*)\s*\(', line)
                if constructor_match:
                    # Count parens to track nesting
                    for char in line:
                        if char == '(':
                            paren_depth += 1
                            in_datatype_constructor = True
                        elif char == ')':
                            paren_depth -= 1
                            if paren_depth == 0:
                                in_datatype_constructor = False

            # If we're inside a datatype constructor, fix the named argument syntax
            if in_datatype_constructor or paren_depth > 0:
                # Pattern: identifier followed by colon and space, then a value
                # Match: `fieldName: true` or `fieldName: false` or `fieldName: 123`
                # But NOT: `fieldName :=` (already correct)
                # Use word boundary to avoid matching inside strings
                line = re.sub(
                    r'(\s+)(\w+):\s+(true|false|\d+)\s*(,?)(\s*)$',
                    r'\1\2 := \3\4\5',
                    line
                )
                # Also handle mid-line cases (not at end of line)
                line = re.sub(
                    r'(\s+)(\w+):\s+(true|false|\d+)(,)',
                    r'\1\2 := \3\4',
                    line
                )

            # Update paren tracking for lines we've already processed
            for char in line:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                    if paren_depth <= 0:
                        paren_depth = 0
                        in_datatype_constructor = False

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def validate_spec(self, spec: str) -> Tuple[bool, str]:
        """
        Validate Dafny specification syntax.
        
        Args:
            spec: The Dafny specification code
            
        Returns:
            Tuple of (valid: bool, error_message: str)
        """
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            spec_path = os.path.join(tmpdir, "safety_policy.dfy")
            
            with open(spec_path, 'w', encoding='utf-8') as f:
                f.write(spec)
            
            try:
                result = subprocess.run(
                    ["dafny", "verify", spec_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    return True, "Specification is valid"
                else:
                    return False, f"Specification invalid:\n{result.stderr}\n{result.stdout}"
                    
            except subprocess.TimeoutExpired:
                return False, "Dafny verification timed out"
            except FileNotFoundError:
                return False, "Dafny not found. Please install Dafny and add it to PATH."

