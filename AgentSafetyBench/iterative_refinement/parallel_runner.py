"""
Ray-based parallel workers for iterative refinement.

Handles both API models and coordinates with vLLM servers.
"""
import ray
import os
import sys
import json
from typing import Dict, Any, List, Optional
from copy import deepcopy

from .config import ModelConfig, RefinementConfig
from .model_api import create_model_api, BaseModelAPI
from .trace_manager import VerificationResult


@ray.remote
class APIAgentWorker:
    """
    Ray worker for running agent evaluation with API models.

    Handles the full agent loop (tool calls, environment interaction).
    """

    def __init__(
        self,
        model_config_dict: Dict[str, Any],
        env_manager_module_path: str = "../environments",
        project_root: str = None
    ):
        """
        Initialize the worker.

        Args:
            model_config_dict: Model configuration as a dict (to avoid serialization issues)
            env_manager_module_path: Path to environment manager module
            project_root: Path to project root for imports
        """
        import sys

        # Add project root to path for imports
        if project_root:
            sys.path.insert(0, project_root)

        sys.path.insert(0, env_manager_module_path)
        from EnvManager import EnvManager

        # Import after path is set
        from iterative_refinement.model_api import create_model_api
        from iterative_refinement.config import ModelConfig

        # Reconstruct ModelConfig from dict
        model_config = ModelConfig(**model_config_dict)
        self.model_config = model_config
        self.api = create_model_api(model_config)
        self.env_manager = EnvManager()

    def _parse_envs(self, envs_info: List[Dict]) -> tuple:
        """Parse environment info and create tool descriptions."""
        envs = []
        tool_descs = []

        for env_info in envs_info:
            env_name = env_info.get('name', '')
            if not env_name:
                continue

            env_params = env_info.get('parameters') or None
            env = self.env_manager.init_env(env_name, env_params)
            if env is None:
                continue

            envs.append(env)

            tool_names = env_info.get('tools', [])
            tool_desc = env.get_tool_descs(tool_names)
            tool_descs.extend(tool_desc)

        # Format tool descriptions for API
        formatted_tools = []
        for item in tool_descs:
            item['type'] = "object"
            formatted_tools.append({'type': 'function', 'function': item})

        return envs, formatted_tools

    def run_agent(
        self,
        sample: Dict[str, Any],
        initial_messages: List[Dict[str, Any]],
        max_rounds: int = 50
    ) -> tuple:
        """
        Run the agent on a sample.

        Args:
            sample: The benchmark sample
            initial_messages: Initial conversation messages
            max_rounds: Maximum conversation rounds

        Returns:
            Tuple of (messages, success, error)
        """
        messages = deepcopy(initial_messages)
        envs = []
        tool_descs = []

        try:
            # Parse environments
            if sample.get('environments') and sample['environments'][0].get('name'):
                envs, tool_descs = self._parse_envs(sample['environments'])

            rounds = 0
            while rounds < max_rounds:
                rounds += 1

                # Generate response
                completion = self.api.generate_response(messages, tool_descs)

                if completion is None or 'type' not in completion:
                    continue
                if completion['type'] == 'error':
                    print(f"Sample {sample['id']}: {completion['message']}")
                    continue

                if completion['type'] == 'tool':
                    # Handle tool call
                    tool_call_id = completion['tool_call_id']
                    tool_name = completion['tool_name']
                    arguments = completion['arguments']

                    # Execute tool
                    tool_call_result = None
                    for env_idx, env in enumerate(envs):
                        provided_tools = sample['environments'][env_idx].get('tools', [])
                        if tool_name not in provided_tools:
                            continue
                        if not env.has_tool(tool_name):
                            continue
                        tool_call_result = env.call_tool(tool_name, deepcopy(arguments))
                        break

                    if tool_call_result is None:
                        tool_call_result = {
                            "success": False,
                            "message": f"Tool {tool_name} doesn't exist."
                        }

                    # Add to messages
                    tool_call_response = {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(arguments),
                                "name": tool_name
                            }
                        }]
                    }

                    tool_call_result_message = {
                        "role": "tool",
                        "content": json.dumps(tool_call_result),
                        "tool_call_id": tool_call_id,
                        "name": tool_name
                    }

                    messages.append(tool_call_response)
                    messages.append(tool_call_result_message)

                else:
                    # Content response - task complete
                    content = completion['content']
                    messages.append({
                        'role': 'assistant',
                        'content': content
                    })
                    return messages, True, None

            # Exceeded max rounds
            return messages, False, "Exceeded max rounds"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return messages, False, str(e)


@ray.remote
class VerificationWorker:
    """
    Ray worker for running verification on agent outputs.

    Uses the formal_verification pipeline.
    """

    def __init__(
        self,
        judge_model_config_dict: Dict[str, Any],
        language: str = "dafny",
        cache_dir: str = None,
        dafny_timeout: int = 120,
        project_root: str = None
    ):
        """
        Initialize the verification worker.

        Args:
            judge_model_config_dict: Configuration for the judge model as a dict
            language: Verification language (dafny, python, natural_language, direct, baseline, llm_cot, llm_fewshot)
            cache_dir: Directory for caching Dafny files
            dafny_timeout: Timeout for Dafny execution
            project_root: Path to project root for imports
        """
        import sys

        # Add project root to path for imports
        if project_root:
            sys.path.insert(0, project_root)
        else:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        from formal_verification.executors import (
            DafnyExecutor, PythonExecutor, NaturalLanguageExecutor,
            DirectExecutor, BaselineExecutor, LLMCoTExecutor, LLMFewShotExecutor
        )
        from iterative_refinement.config import ModelConfig

        # Reconstruct ModelConfig from dict
        judge_model_config = ModelConfig(**judge_model_config_dict)

        self.language = language
        self.cache_dir = cache_dir

        # Get API key for LLM-based executors
        api_key = os.getenv(judge_model_config.api_key_env) if judge_model_config.api_key_env else None

        # Initialize executor based on language
        if language == "dafny":
            from formal_verification.pipeline import VerificationPipeline
            self.pipeline = VerificationPipeline(
                model_name=judge_model_config.model_id,
                language=language,
                api_base=judge_model_config.api_base,
                api_key=api_key,
                generation_config=judge_model_config.generation_config
            )
            self.executor = DafnyExecutor(cache_dir=cache_dir, timeout=dafny_timeout)
        elif language == "python":
            from formal_verification.pipeline import VerificationPipeline
            self.pipeline = VerificationPipeline(
                model_name=judge_model_config.model_id,
                language=language,
                api_base=judge_model_config.api_base,
                api_key=api_key,
                generation_config=judge_model_config.generation_config
            )
            self.executor = PythonExecutor()
        elif language == "natural_language":
            from formal_verification.pipeline import VerificationPipeline
            self.pipeline = VerificationPipeline(
                model_name=judge_model_config.model_id,
                language=language,
                api_base=judge_model_config.api_base,
                api_key=api_key,
                generation_config=judge_model_config.generation_config
            )
            self.executor = NaturalLanguageExecutor()
        elif language == "direct":
            # Direct mode: no verification, always passes
            self.pipeline = None
            self.executor = DirectExecutor()
        elif language == "baseline":
            # Baseline mode: use shield model for evaluation
            self.pipeline = None
            self.executor = BaselineExecutor(project_root=self.project_root)
        elif language == "llm_cot":
            # LLM + Chain of Thought: single round with CoT prompting
            self.pipeline = None
            self.executor = LLMCoTExecutor(
                model_name=judge_model_config.model_id,
                api_base=judge_model_config.api_base,
                api_key=api_key,
                generation_config=judge_model_config.generation_config,
                project_root=self.project_root
            )
        elif language == "llm_fewshot":
            # LLM + Few-Shot: single round with few-shot examples
            self.pipeline = None
            self.executor = LLMFewShotExecutor(
                model_name=judge_model_config.model_id,
                api_base=judge_model_config.api_base,
                api_key=api_key,
                generation_config=judge_model_config.generation_config,
                project_root=self.project_root
            )
        else:
            raise ValueError(f"Unknown language: {language}")

    def verify(
        self,
        sample: Dict[str, Any],
        agent_output: List[Dict[str, Any]],
        sample_id: int
    ) -> Dict[str, Any]:
        """
        Run verification on agent output.

        Args:
            sample: The benchmark sample
            agent_output: The agent's output messages
            sample_id: Sample ID for caching

        Returns:
            Verification result dictionary
        """
        try:
            # Handle direct mode - no verification
            if self.language == "direct":
                passed, explanation = self.executor.execute("", "")
                return {
                    "passed": passed,
                    "explanation": explanation,
                    "language": self.language,
                    "decomposition": None,
                    "spec": None,
                    "verification_input": None,
                    "error": None
                }

            # Handle baseline mode - use shield model
            if self.language == "baseline":
                passed, explanation = self.executor.evaluate_with_shield(sample, agent_output)
                return {
                    "passed": passed,
                    "explanation": explanation,
                    "language": self.language,
                    "decomposition": None,
                    "spec": None,
                    "verification_input": None,
                    "error": None
                }

            # Handle LLM + CoT mode
            if self.language == "llm_cot":
                passed, explanation = self.executor.evaluate_with_cot(sample, agent_output)
                return {
                    "passed": passed,
                    "explanation": explanation,
                    "language": self.language,
                    "decomposition": None,
                    "spec": None,
                    "verification_input": None,
                    "error": None
                }

            # Handle LLM + Few-Shot mode
            if self.language == "llm_fewshot":
                passed, explanation = self.executor.evaluate_with_fewshot(sample, agent_output)
                return {
                    "passed": passed,
                    "explanation": explanation,
                    "language": self.language,
                    "decomposition": None,
                    "spec": None,
                    "verification_input": None,
                    "error": None
                }

            # Run full pipeline for formal verification modes
            result = self.pipeline.run_full_pipeline(sample, agent_output)

            # Execute verification
            if result.get("stage") == "complete":
                spec = result.get("spec", {})
                verification_input = result.get("verification_input", {})

                spec_code = spec.get("spec_code", "")

                if self.language == "dafny":
                    harness = verification_input.get("dafny_harness", "")
                    exec_result = self.executor.execute(spec_code, harness, sample_id=sample_id)

                    if len(exec_result) == 3:
                        passed, explanation, file_paths = exec_result
                    else:
                        passed, explanation = exec_result
                        file_paths = {}

                elif self.language == "python":
                    harness = verification_input.get("python_script", "")
                    passed, explanation = self.executor.execute(spec_code, harness)
                    file_paths = {}

                else:  # natural_language
                    harness = verification_input.get("raw_response", json.dumps(verification_input))
                    passed, explanation = self.executor.execute(spec_code, harness)
                    file_paths = {}

                return {
                    "passed": passed,
                    "explanation": explanation,
                    "language": self.language,
                    "decomposition": result.get("decomposition"),
                    "spec": spec,
                    "verification_input": verification_input,
                    "spec_code": spec_code,
                    "harness_code": harness if self.language == "dafny" else None,
                    "error": None
                }
            else:
                return {
                    "passed": False,
                    "explanation": result.get("error", "Pipeline failed"),
                    "language": self.language,
                    "decomposition": result.get("decomposition"),
                    "spec": result.get("spec"),
                    "verification_input": result.get("verification_input"),
                    "error": result.get("error")
                }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "passed": False,
                "explanation": str(e),
                "language": self.language,
                "error": str(e)
            }


class ParallelRefinementRunner:
    """
    Manages parallel execution of iterative refinement.

    Uses Ray for API models and coordinates with vLLM servers for local models.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        judge_config: ModelConfig,
        refinement_config: RefinementConfig
    ):
        """
        Initialize the runner.

        Args:
            model_config: Target model configuration
            judge_config: Judge model configuration
            refinement_config: Refinement configuration
        """
        self.model_config = model_config
        self.judge_config = judge_config
        self.config = refinement_config

        self.env_manager_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "environments"
        )

    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray_tmpdir = os.getenv('RAY_TMPDIR')
            if ray_tmpdir:
                os.makedirs(ray_tmpdir, exist_ok=True)
                os.environ['RAY_TMPDIR'] = ray_tmpdir

            # Ensure project root is in sys.path for all Ray workers
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Use runtime_env to ensure workers have the correct Python path
            existing_pythonpath = os.environ.get('PYTHONPATH', '')
            new_pythonpath = f"{project_root}:{existing_pythonpath}" if existing_pythonpath else project_root

            ray.init(
                num_cpus=self.config.num_workers + self.config.dafny_workers,
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {"PYTHONPATH": new_pythonpath}
                }
            )

    def _create_agent_workers(self, count: int = None) -> List:
        """Create agent worker pool."""
        from dataclasses import asdict
        n = count if count is not None else self.config.num_workers
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Convert ModelConfig to dict to avoid serialization issues
        model_config_dict = asdict(self.model_config)
        return [
            APIAgentWorker.remote(
                model_config_dict=model_config_dict,
                env_manager_module_path=self.env_manager_path,
                project_root=project_root
            )
            for _ in range(n)
        ]

    def _create_verification_workers(self, count: int = None) -> List:
        """Create verification worker pool."""
        from dataclasses import asdict
        n = count if count is not None else self.config.dafny_workers
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Convert ModelConfig to dict to avoid serialization issues
        judge_config_dict = asdict(self.judge_config)
        return [
            VerificationWorker.remote(
                judge_model_config_dict=judge_config_dict,
                language=self.config.verification_language,
                cache_dir=None,  # Will set per-run
                dafny_timeout=self.config.dafny_timeout,
                project_root=project_root
            )
            for _ in range(n)
        ]

    def shutdown(self):
        """Shutdown Ray."""
        if ray.is_initialized():
            ray.shutdown()
