"""
vLLM-specific pipeline for local models.

Uses vLLM server for inference instead of Ray workers.
"""
import os
import json
import ray
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datetime import datetime
from copy import deepcopy

from .config import ModelConfig, RefinementConfig, TARGET_MODELS, JUDGE_MODEL
from .model_api import VLLMServerAPI, create_model_api
from .trace_manager import TraceManager, VerificationResult
from .refinement_prompt import RefinementPromptGenerator
from .parallel_runner import VerificationWorker


class VLLMRefinementPipeline:
    """
    Refinement pipeline optimized for vLLM local models.

    Uses vLLM server for agent inference and Ray for verification.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        judge_config: ModelConfig = None,
        refinement_config: RefinementConfig = None,
        vllm_base_url: str = "http://localhost:8000/v1"
    ):
        """
        Initialize the vLLM pipeline.

        Args:
            model_config: Target model configuration
            judge_config: Judge model configuration
            refinement_config: Refinement configuration
            vllm_base_url: vLLM server URL
        """
        self.model_config = model_config
        self.judge_config = judge_config or JUDGE_MODEL
        self.config = refinement_config or RefinementConfig()
        self.vllm_base_url = vllm_base_url

        self.prompt_generator = RefinementPromptGenerator()

        # Create vLLM client
        self.vllm_api = VLLMServerAPI(
            model_id=model_config.model_id,
            base_url=vllm_base_url,
            generation_config=model_config.generation_config
        )

        # Environment manager
        self.env_manager = None
        self._init_env_manager()

    def _init_env_manager(self):
        """Initialize environment manager."""
        import sys
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "environments"
        )
        sys.path.insert(0, env_path)
        from EnvManager import EnvManager
        self.env_manager = EnvManager()

    def _init_ray(self):
        """Initialize Ray for verification workers."""
        if not ray.is_initialized():
            ray_tmpdir = os.getenv('RAY_TMPDIR')
            if ray_tmpdir:
                os.makedirs(ray_tmpdir, exist_ok=True)
            ray.init(num_cpus=self.config.dafny_workers, ignore_reinit_error=True)

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

        formatted_tools = []
        for item in tool_descs:
            item['type'] = "object"
            formatted_tools.append({'type': 'function', 'function': item})

        return envs, formatted_tools

    def _run_single_agent(
        self,
        sample: Dict[str, Any],
        initial_messages: List[Dict[str, Any]],
        max_rounds: int = 50
    ) -> tuple:
        """
        Run agent on a single sample using vLLM.

        Args:
            sample: Benchmark sample
            initial_messages: Initial messages
            max_rounds: Max conversation rounds

        Returns:
            Tuple of (messages, success, error)
        """
        messages = deepcopy(initial_messages)
        envs = []
        tool_descs = []

        try:
            if sample.get('environments') and sample['environments'][0].get('name'):
                envs, tool_descs = self._parse_envs(sample['environments'])

            rounds = 0
            while rounds < max_rounds:
                rounds += 1

                completion = self.vllm_api.generate_response(messages, tool_descs)

                if completion is None or 'type' not in completion:
                    continue
                if completion['type'] == 'error':
                    continue

                if completion['type'] == 'tool':
                    tool_call_id = completion['tool_call_id']
                    tool_name = completion['tool_name']
                    arguments = completion['arguments']

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

                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(arguments),
                                "name": tool_name
                            }
                        }]
                    })

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_call_result),
                        "tool_call_id": tool_call_id,
                        "name": tool_name
                    })

                else:
                    messages.append({
                        'role': 'assistant',
                        'content': completion['content']
                    })
                    return messages, True, None

            return messages, False, "Exceeded max rounds"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return messages, False, str(e)

    def run_batch_refinement(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        resume_from: str = None
    ) -> Dict[str, Any]:
        """
        Run iterative refinement on samples using vLLM.

        Args:
            samples: List of benchmark samples
            output_dir: Output directory
            resume_from: Optional path to resume from

        Returns:
            Summary of results
        """
        self._init_ray()

        trace_manager = TraceManager(output_dir, self.model_config.name)

        if resume_from and os.path.exists(resume_from):
            trace_manager.load_existing(resume_from)
            print(f"Resumed from {resume_from}")

        existing_ids = set(trace_manager.traces.keys())
        to_process = [s for s in samples if s['id'] not in existing_ids]

        if self.config.debug_samples:
            to_process = to_process[:self.config.debug_samples]

        print(f"\n{'='*60}")
        print(f"vLLM Iterative Refinement Pipeline")
        print(f"Model: {self.model_config.name}")
        print(f"vLLM Server: {self.vllm_base_url}")
        print(f"Verification Language: {self.config.verification_language}")
        print(f"Max Iterations: {self.config.max_iterations}")
        print(f"Samples to process: {len(to_process)}")
        print(f"{'='*60}\n")

        system_prompt = self.vllm_api.get_system_prompt()

        cache_dir = os.path.join(trace_manager.model_dir, "dafny_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Create verification workers (using Ray)
        # Convert ModelConfig to dict to avoid serialization issues
        from dataclasses import asdict
        judge_config_dict = asdict(self.judge_config)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        verification_workers = [
            VerificationWorker.remote(
                judge_model_config_dict=judge_config_dict,
                language=self.config.verification_language,
                cache_dir=cache_dir,
                dafny_timeout=self.config.dafny_timeout,
                project_root=project_root
            )
            for _ in range(self.config.dafny_workers)
        ]

        # Process samples sequentially (vLLM handles batching internally)
        for sample in tqdm(to_process, desc="Processing samples"):
            self._process_sample(
                sample,
                trace_manager,
                system_prompt,
                verification_workers
            )
            trace_manager.save_all()

        summary = trace_manager.save_all()

        print(f"\n{'='*60}")
        print(f"Refinement Complete")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Passed: {summary['passed_samples']}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"{'='*60}\n")

        ray.shutdown()
        return summary

    def _process_sample(
        self,
        sample: Dict[str, Any],
        trace_manager: TraceManager,
        system_prompt: str,
        verification_workers: List
    ):
        """Process a single sample through refinement iterations."""
        trace_manager.create_trace(sample)

        current_messages = self.prompt_generator.generate_initial_messages(
            sample, system_prompt
        )

        for iteration in range(self.config.max_iterations):
            # Run agent
            messages, success, error = self._run_single_agent(
                sample, current_messages
            )

            # Run verification
            if success and messages:
                worker = verification_workers[sample['id'] % len(verification_workers)]
                vr_data = ray.get(worker.verify.remote(sample, messages, sample['id']))
            else:
                vr_data = {
                    "passed": False,
                    "explanation": error or "Agent evaluation failed",
                    "error": error
                }

            vr = VerificationResult(
                passed=vr_data.get("passed", False),
                explanation=vr_data.get("explanation", ""),
                language=self.config.verification_language,
                error=vr_data.get("error"),
                spec_code=vr_data.get("spec_code"),
                harness_code=vr_data.get("harness_code"),
                decomposition=vr_data.get("decomposition"),
                spec=vr_data.get("spec"),
                verification_input=vr_data.get("verification_input")
            )

            # Generate refinement prompt
            _, refinement_prompt = self.prompt_generator.generate_refinement_messages(
                sample, messages, vr, system_prompt
            )

            # Add iteration
            trace_manager.add_iteration(
                sample['id'],
                iteration,
                messages,
                vr,
                refinement_prompt
            )
            trace_manager.save_iteration(sample['id'], iteration)

            if vr.passed:
                print(f"  Sample {sample['id']}: PASSED at iteration {iteration + 1}")
                break
            elif iteration < self.config.max_iterations - 1:
                # Prepare next iteration
                current_messages = [{"role": "system", "content": system_prompt}]
                current_messages.append({"role": "user", "content": refinement_prompt})
            else:
                print(f"  Sample {sample['id']}: FAILED after {iteration + 1} iterations")


def run_vllm_refinement(
    model_name: str,
    vllm_base_url: str = "http://localhost:8000/v1",
    data_path: str = None,
    output_dir: str = None,
    max_iterations: int = 3,
    verification_language: str = "dafny",
    debug_samples: int = None
) -> Dict[str, Any]:
    """
    Convenience function to run vLLM-based refinement.

    Args:
        model_name: Model name (from TARGET_MODELS)
        vllm_base_url: vLLM server URL
        data_path: Path to benchmark data
        output_dir: Output directory
        max_iterations: Maximum iterations
        verification_language: Verification language
        debug_samples: Number of debug samples

    Returns:
        Summary of results
    """
    if model_name not in TARGET_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_config = TARGET_MODELS[model_name]

    config = RefinementConfig(
        max_iterations=max_iterations,
        verification_language=verification_language,
        debug_samples=debug_samples
    )

    if data_path:
        config.data_path = data_path
    if output_dir:
        config.output_base_dir = output_dir

    with open(config.data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    pipeline = VLLMRefinementPipeline(
        model_config=model_config,
        judge_config=JUDGE_MODEL,
        refinement_config=config,
        vllm_base_url=vllm_base_url
    )

    return pipeline.run_batch_refinement(samples, config.output_base_dir)
