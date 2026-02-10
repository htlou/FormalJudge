"""
Verification Pipeline: Orchestrates the 3-agent verification process.
"""
import json
import re
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import os
import time

from ..prompts import (
    AGENT1_SYSTEM_PROMPT,
    AGENT1_DECOMPOSITION_PROMPT,
    get_agent2_system_prompt,
    get_agent2_user_prompt_template,
    get_agent3_system_prompt,
    get_agent3_user_prompt_template
)


class VerificationPipeline:
    """
    Orchestrates the 3-agent formal verification pipeline.
    
    Agent #1: Natural language decomposition
    Agent #2: Formal language translation (Dafny/Python/NL)
    Agent #3: Trace abstraction and proof generation
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        language: str = "dafny",
        api_base: str = None,
        api_key: str = None,
        generation_config: Dict[str, Any] = None
    ):
        """
        Initialize the verification pipeline.
        
        Args:
            model_name: Name of the LLM to use
            language: Verification language ('dafny', 'python', 'natural_language')
            api_base: API base URL (defaults to environment variable)
            api_key: API key (defaults to environment variable)
            generation_config: Generation configuration for the LLM
        """
        self.model_name = model_name
        self.language = language
        
        # Set up API client
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api3.xhub.chat/v1")
        self.api_key = api_key or os.getenv("XHUB_API_KEY")
        
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=120
        )
        
        self.generation_config = generation_config or {
            "temperature": 1.0,
            "max_tokens": 8192
        }
        
    def _call_llm(self, messages: list, max_retries: int = 3) -> Optional[str]:
        """Call the LLM API with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **self.generation_config
                )
                if response and response.choices:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_code_block(self, text: str, language: str = None) -> Optional[str]:
        """Extract code block from LLM response."""
        if language:
            pattern = rf'```{language}\s*\n?([\s\S]*?)\n?```'
        else:
            pattern = r'```(?:\w+)?\s*\n?([\s\S]*?)\n?```'
        
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def _remove_main_method(self, code: str) -> str:
        """
        Remove Main() method declarations from Dafny code.
        
        This method handles:
        - Various whitespace patterns
        - Comments before/after the method declaration
        - Nested braces in method body
        - Multiple Main() methods (removes all)
        
        Args:
            code: Dafny code string
            
        Returns:
            Code with all Main() methods removed
        """
        if not code or self.language != "dafny":
            return code
        
        lines = code.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line contains "method Main()"
            # Match patterns like:
            #   method Main() {
            #   method Main() {
            #   method Main()  // comment
            #   // comment
            #   method Main() {
            main_match = re.search(r'method\s+Main\s*\(\s*\)', line, re.IGNORECASE)
            
            if main_match:
                # Found a Main() method declaration
                # Check if opening brace is on the same line
                brace_pos = line.find('{', main_match.end())
                
                if brace_pos != -1:
                    # Opening brace on same line, start counting from here
                    brace_count = 1
                    start_line = i
                    j = i
                    pos = brace_pos + 1
                    
                    # First, count braces on the same line after the opening brace
                    while pos < len(line) and brace_count > 0:
                        if line[pos] == '{':
                            brace_count += 1
                        elif line[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    # If braces are balanced on the same line, skip this line entirely
                    if brace_count == 0:
                        i += 1
                        continue
                    
                    # Otherwise, continue to next lines
                    j += 1
                    while j < len(lines) and brace_count > 0:
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                        j += 1
                    
                    # Skip all lines from start_line to j (inclusive)
                    i = j
                    continue
                else:
                    # Opening brace on next line, find it
                    brace_count = 0
                    start_line = i
                    j = i
                    
                    # Look for opening brace
                    while j < len(lines) and brace_count == 0:
                        if '{' in lines[j]:
                            brace_count = 1
                            j += 1
                            break
                        j += 1
                    
                    # Now count braces until balanced
                    while j < len(lines) and brace_count > 0:
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                        j += 1
                    
                    # Skip all lines from start_line to j (inclusive)
                    i = j
                    continue
            else:
                # Not a Main() method, keep this line
                result_lines.append(line)
                i += 1
        
        result = '\n'.join(result_lines)
        
        # Clean up: remove multiple consecutive blank lines (more than 2)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Clean up: remove trailing blank lines before closing brace
        result = re.sub(r'\n+\s*}', '\n}', result)
        
        return result.strip()
    
    def run_agent1(self, instruction: str, task_context: Dict[str, Any], risks: list) -> Dict[str, Any]:
        """
        Run Agent #1: Natural language decomposition.
        
        Args:
            instruction: The agent's task instruction
            task_context: Context including environment and tools
            risks: List of risk categories
            
        Returns:
            Decomposition result with hierarchical requirements
        """
        system_prompt = AGENT1_SYSTEM_PROMPT
        
        user_message = AGENT1_DECOMPOSITION_PROMPT + f"""

[BEGIN_USER_INSTRUCTION]
{instruction}
[END_USER_INSTRUCTION]

[BEGIN_TASK_CONTEXT]
{json.dumps(task_context, indent=2)}
[END_TASK_CONTEXT]

[BEGIN_RISK_CATEGORIES]
{json.dumps(risks, indent=2)}
[END_RISK_CATEGORIES]
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self._call_llm(messages)
        if response is None:
            return {"error": "Agent 1 failed to respond"}
        
        result = self._extract_json(response)
        if result is None:
            return {"error": "Failed to parse Agent 1 response", "raw_response": response}
        
        result["raw_response"] = response
        return result
    
    def run_agent2(self, decomposition: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Agent #2: Formal language translation.
        
        Args:
            decomposition: Output from Agent 1
            task_context: Context including environment and tools
            
        Returns:
            Verification specification in the target language
        """
        system_prompt = get_agent2_system_prompt(self.language)
        user_template = get_agent2_user_prompt_template(self.language)
        
        # Use string replacement instead of .format() to avoid issues with braces in user content
        user_message = user_template.replace('{decomposition}', json.dumps(decomposition, indent=2)) \
                                    .replace('{task_context}', json.dumps(task_context, indent=2))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self._call_llm(messages)
        if response is None:
            return {"error": "Agent 2 failed to respond"}
        
        result = {
            "raw_response": response,
            "language": self.language
        }
        
        if self.language == "dafny":
            spec_code = self._extract_code_block(response, "dafny")
            # Remove any Main() methods to avoid collision with Agent 3's harness
            result["spec_code"] = self._remove_main_method(spec_code)
        elif self.language == "python":
            result["spec_code"] = self._extract_code_block(response, "python")
        else:  # natural_language
            # For NL, the spec is plain text guidelines (not JSON)
            result["spec_code"] = response
        
        return result
    
    def run_agent3(
        self,
        spec: Dict[str, Any],
        agent_trace: list,
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Agent #3: Trace abstraction and proof generation.
        
        Args:
            spec: Output from Agent 2
            agent_trace: The agent's reasoning trace (conversation history)
            task_context: Context including environment and tools
            
        Returns:
            Verification input with harness/script
        """
        system_prompt = get_agent3_system_prompt(self.language)
        user_template = get_agent3_user_prompt_template(self.language)
        
        # Use string replacement instead of .format() to avoid issues with braces in user content
        user_message = user_template.replace('{spec}', spec.get('spec_code', '')) \
                                    .replace('{trace}', json.dumps(agent_trace, indent=2)) \
                                    .replace('{task_context}', json.dumps(task_context, indent=2))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self._call_llm(messages)
        if response is None:
            return {"error": "Agent 3 failed to respond"}
        
        result = self._extract_json(response)
        if result is None:
            result = {"raw_response": response}
        else:
            result["raw_response"] = response
        
        result["language"] = self.language
        return result
    
    def run_full_pipeline(
        self,
        sample: Dict[str, Any],
        agent_output: list
    ) -> Dict[str, Any]:
        """
        Run the complete verification pipeline on a sample.
        
        Args:
            sample: The benchmark sample with instruction, environments, risks
            agent_output: The agent's output trace (conversation history)
            
        Returns:
            Complete verification result
        """
        # Prepare task context
        task_context = {
            "instruction": sample.get("instruction", ""),
            "environments": sample.get("environments", []),
            "risks": sample.get("risks", [])
        }
        
        # Stage 1: Agent #1 - Decomposition
        decomposition = self.run_agent1(
            instruction=sample.get("instruction", ""),
            task_context=task_context,
            risks=sample.get("risks", [])
        )
        
        if "error" in decomposition:
            return {
                "stage": "agent1",
                "error": decomposition["error"],
                "decomposition": decomposition
            }
        
        # Stage 2: Agent #2 - Formal translation
        spec = self.run_agent2(
            decomposition=decomposition,
            task_context=task_context
        )
        
        if "error" in spec:
            return {
                "stage": "agent2",
                "error": spec["error"],
                "decomposition": decomposition,
                "spec": spec
            }
        
        # Stage 3: Agent #3 - Trace abstraction
        verification_input = self.run_agent3(
            spec=spec,
            agent_trace=agent_output,
            task_context=task_context
        )
        
        if "error" in verification_input:
            return {
                "stage": "agent3",
                "error": verification_input["error"],
                "decomposition": decomposition,
                "spec": spec,
                "verification_input": verification_input
            }
        
        return {
            "stage": "complete",
            "decomposition": decomposition,
            "spec": spec,
            "verification_input": verification_input
        }

