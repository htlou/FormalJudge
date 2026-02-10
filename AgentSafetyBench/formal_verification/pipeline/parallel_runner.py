"""
Parallel runner for large-scale verification using Ray.
"""
import json
import os
import ray
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

from .verification_pipeline import VerificationPipeline
from ..executors import DafnyExecutor, PythonExecutor, NaturalLanguageExecutor


@ray.remote
class VerificationWorker:
    """Ray worker for parallel verification."""

    def __init__(
        self,
        model_name: str,
        language: str,
        api_base: str,
        api_key: str,
        generation_config: Dict[str, Any],
        cache_dir: str = None,
        dafny_timeout: int = 120,
        project_root: str = None
    ):
        """Initialize the worker with pipeline configuration."""
        self.language = language
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Initialize executor based on language
        if language == "dafny":
            self.pipeline = VerificationPipeline(
                model_name=model_name,
                language=language,
                api_base=api_base,
                api_key=api_key,
                generation_config=generation_config
            )
            self.executor = DafnyExecutor(cache_dir=cache_dir, timeout=dafny_timeout)
        elif language == "python":
            self.pipeline = VerificationPipeline(
                model_name=model_name,
                language=language,
                api_base=api_base,
                api_key=api_key,
                generation_config=generation_config
            )
            self.executor = PythonExecutor()
        elif language == "natural_language":
            self.pipeline = VerificationPipeline(
                model_name=model_name,
                language=language,
                api_base=api_base,
                api_key=api_key,
                generation_config=generation_config
            )
            self.executor = NaturalLanguageExecutor()
        elif language == "direct":
            # Direct mode: no verification
            from ..executors import DirectExecutor
            self.pipeline = None
            self.executor = DirectExecutor()
        elif language == "baseline":
            # Baseline mode: use shield model
            from ..executors import BaselineExecutor
            self.pipeline = None
            self.executor = BaselineExecutor(project_root=self.project_root)
        elif language == "llm_cot":
            # LLM + Chain of Thought mode
            from ..executors import LLMCoTExecutor
            self.pipeline = None
            self.executor = LLMCoTExecutor(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                generation_config=generation_config,
                project_root=self.project_root
            )
        elif language == "llm_fewshot":
            # LLM + Few-Shot mode
            from ..executors import LLMFewShotExecutor
            self.pipeline = None
            self.executor = LLMFewShotExecutor(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                generation_config=generation_config,
                project_root=self.project_root
            )
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def run_stage1(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run Agent #1 decomposition for a sample."""
        task_context = {
            "instruction": sample.get("instruction", ""),
            "environments": sample.get("environments", []),
            "risks": sample.get("risks", [])
        }
        
        result = self.pipeline.run_agent1(
            instruction=sample.get("instruction", ""),
            task_context=task_context,
            risks=sample.get("risks", [])
        )
        
        return {
            "id": sample.get("id"),
            "decomposition": result
        }
    
    def run_stage2(self, sample: Dict[str, Any], decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """Run Agent #2 formal translation for a sample."""
        task_context = {
            "instruction": sample.get("instruction", ""),
            "environments": sample.get("environments", []),
            "risks": sample.get("risks", [])
        }
        
        result = self.pipeline.run_agent2(
            decomposition=decomposition,
            task_context=task_context
        )
        
        return {
            "id": sample.get("id"),
            "spec": result
        }
    
    def run_stage3(
        self,
        sample: Dict[str, Any],
        spec: Dict[str, Any],
        agent_output: list
    ) -> Dict[str, Any]:
        """Run Agent #3 trace abstraction for a sample."""
        task_context = {
            "instruction": sample.get("instruction", ""),
            "environments": sample.get("environments", []),
            "risks": sample.get("risks", [])
        }
        
        result = self.pipeline.run_agent3(
            spec=spec,
            agent_trace=agent_output,
            task_context=task_context
        )
        
        return {
            "id": sample.get("id"),
            "verification_input": result
        }
    
    def run_execution(
        self,
        sample_id: int,
        spec: Dict[str, Any],
        verification_input: Dict[str, Any],
        sample: Dict[str, Any] = None,
        agent_output: list = None
    ) -> Dict[str, Any]:
        """Run the verification execution for a sample."""
        try:
            # Handle direct mode - no verification
            if self.language == "direct":
                passed, explanation = self.executor.execute("", "")
                return {
                    "id": sample_id,
                    "passed": passed,
                    "explanation": explanation,
                    "error": None
                }

            # Handle baseline mode - use shield model
            if self.language == "baseline":
                if sample is None or agent_output is None:
                    return {
                        "id": sample_id,
                        "passed": False,
                        "explanation": "Baseline mode requires sample and agent_output",
                        "error": "Missing sample or agent_output for baseline mode"
                    }
                passed, explanation = self.executor.evaluate_with_shield(sample, agent_output)
                return {
                    "id": sample_id,
                    "passed": passed,
                    "explanation": explanation,
                    "error": None
                }

            # Handle LLM + CoT mode
            if self.language == "llm_cot":
                if sample is None or agent_output is None:
                    return {
                        "id": sample_id,
                        "passed": False,
                        "explanation": "LLM CoT mode requires sample and agent_output",
                        "error": "Missing sample or agent_output for LLM CoT mode"
                    }
                passed, explanation = self.executor.evaluate_with_cot(sample, agent_output)
                return {
                    "id": sample_id,
                    "passed": passed,
                    "explanation": explanation,
                    "error": None
                }

            # Handle LLM + Few-Shot mode
            if self.language == "llm_fewshot":
                if sample is None or agent_output is None:
                    return {
                        "id": sample_id,
                        "passed": False,
                        "explanation": "LLM Few-Shot mode requires sample and agent_output",
                        "error": "Missing sample or agent_output for LLM Few-Shot mode"
                    }
                passed, explanation = self.executor.evaluate_with_fewshot(sample, agent_output)
                return {
                    "id": sample_id,
                    "passed": passed,
                    "explanation": explanation,
                    "error": None
                }

            spec_code = spec.get("spec_code", "")

            # Get harness based on language
            if self.language == "dafny":
                harness = verification_input.get("dafny_harness", "")
                # DafnyExecutor returns (passed, explanation, file_paths)
                result = self.executor.execute(spec_code, harness, sample_id=sample_id)
                if len(result) == 3:
                    passed, explanation, file_paths = result
                else:
                    # Backward compatibility
                    passed, explanation = result
                    file_paths = {}
            elif self.language == "python":
                harness = verification_input.get("python_script", "")
                passed, explanation = self.executor.execute(spec_code, harness)
                file_paths = {}
            else:  # natural_language
                # For NL, pass the raw response directly (new simple format)
                # or fall back to JSON serialization (legacy format)
                harness = verification_input.get("raw_response", json.dumps(verification_input))
                passed, explanation = self.executor.execute(spec_code, harness)
                file_paths = {}

            result_dict = {
                "id": sample_id,
                "passed": passed,
                "explanation": explanation,
                "error": None
            }

            # Add file paths if available (for debugging)
            if file_paths:
                result_dict["debug_files"] = file_paths

            return result_dict
        except Exception as e:
            return {
                "id": sample_id,
                "passed": False,
                "explanation": "",
                "error": str(e)
            }


class ParallelVerificationRunner:
    """
    Manages parallel verification across multiple workers.
    Runs stages sequentially: all stage 1 -> all stage 2 -> all stage 3 -> execution.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        language: str = "dafny",
        num_workers: int = 4,
        api_base: str = None,
        api_key: str = None,
        generation_config: Dict[str, Any] = None,
        cache_dir: str = None,
        max_samples: int = None,
        dafny_timeout: int = 120,
        dafny_workers: int = None,
        project_root: str = None
    ):
        """
        Initialize the parallel runner.

        Args:
            model_name: Name of the LLM to use
            language: Verification language ('dafny', 'python', 'natural_language', 'direct', 'baseline', 'llm_cot', 'llm_fewshot')
            num_workers: Number of parallel workers for LLM API calls (stages 1-3)
            api_base: API base URL
            api_key: API key
            generation_config: LLM generation config
            cache_dir: Directory to cache intermediate files
            max_samples: Max samples to process (for debugging)
            dafny_timeout: Timeout in seconds for Dafny execution (default 120)
            dafny_workers: Number of parallel workers for Dafny execution (default: 4)
            project_root: Path to project root for imports
        """
        self.model_name = model_name
        self.language = language
        self.num_workers = num_workers
        self.dafny_workers = dafny_workers if dafny_workers is not None else 4
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api3.xhub.chat/v1")
        self.api_key = api_key or os.getenv("XHUB_API_KEY")
        self.generation_config = generation_config or {"temperature": 1.0, "max_tokens": 8192}
        self.max_samples = max_samples
        self.dafny_timeout = dafny_timeout
        self.project_root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up cache directory with timestamp subdirectory
        base_cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "cache"
        )
        self.cache_dir = os.path.join(base_cache_dir, self.timestamp)

        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            # Use RAY_TMPDIR from environment if set (by shell script)
            ray_tmpdir = os.getenv('RAY_TMPDIR')
            if ray_tmpdir:
                # Ensure the directory exists
                os.makedirs(ray_tmpdir, exist_ok=True)
                # Set it before initializing Ray
                os.environ['RAY_TMPDIR'] = ray_tmpdir
            ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
    
    def _create_workers(self, count: int = None) -> List:
        """Create worker pool for LLM API calls."""
        n = count if count is not None else self.num_workers
        return [
            VerificationWorker.remote(
                model_name=self.model_name,
                language=self.language,
                api_base=self.api_base,
                api_key=self.api_key,
                generation_config=self.generation_config,
                cache_dir=self.cache_dir,
                dafny_timeout=self.dafny_timeout,
                project_root=self.project_root
            )
            for _ in range(n)
        ]

    def _create_dafny_workers(self) -> List:
        """Create worker pool for Dafny execution (fewer workers to avoid CPU contention)."""
        return self._create_workers(self.dafny_workers)
    
    def run_stage1_parallel(
        self,
        samples: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Run Agent #1 for all samples in parallel.
        
        Args:
            samples: List of benchmark samples
            output_path: Path to save intermediate results
            
        Returns:
            List of decomposition results
        """
        self._init_ray()
        workers = self._create_workers()

        if not workers:
            raise RuntimeError(f"No workers created. num_workers={self.num_workers}")

        # Load existing results if any
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass
        
        # Filter samples that need processing
        to_process = [s for s in samples if s['id'] not in existing_results]
        print(f"Stage 1: Processing {len(to_process)} samples ({len(existing_results)} already done)")
        
        if not to_process:
            return list(existing_results.values())
        
        # Distribute work
        futures = []
        sample_map = {}
        for i, sample in enumerate(to_process):
            worker = workers[i % len(workers)]
            future = worker.run_stage1.remote(sample)
            futures.append(future)
            sample_map[id(future)] = sample
        
        # Collect results
        results = list(existing_results.values())
        for future in tqdm(futures, desc="Agent #1 - Decomposition"):
            try:
                result = ray.get(future)
                results.append(result)
                
                # Save intermediate results
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Worker error in stage 1: {e}")
        
        return results
    
    def run_stage2_parallel(
        self,
        samples: List[Dict[str, Any]],
        stage1_results: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Run Agent #2 for all samples in parallel.
        
        Args:
            samples: List of benchmark samples
            stage1_results: Results from stage 1
            output_path: Path to save intermediate results

        Returns:
            List of spec results
        """
        self._init_ray()
        workers = self._create_workers()

        if not workers:
            raise RuntimeError(f"No workers created. num_workers={self.num_workers}")

        # Create lookup for stage 1 results
        stage1_map = {r['id']: r['decomposition'] for r in stage1_results if 'decomposition' in r}
        
        # Load existing results
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass
        
        # Filter samples that need processing
        to_process = [
            (s, stage1_map.get(s['id']))
            for s in samples
            if s['id'] not in existing_results and s['id'] in stage1_map
        ]
        print(f"Stage 2: Processing {len(to_process)} samples ({len(existing_results)} already done)")
        
        if not to_process:
            return list(existing_results.values())
        
        # Distribute work
        futures = []
        for i, (sample, decomposition) in enumerate(to_process):
            worker = workers[i % len(workers)]
            future = worker.run_stage2.remote(sample, decomposition)
            futures.append(future)
        
        # Collect results
        results = list(existing_results.values())
        for future in tqdm(futures, desc="Agent #2 - Formal Translation"):
            try:
                result = ray.get(future)
                results.append(result)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Worker error in stage 2: {e}")
        
        return results
    
    def run_stage3_parallel(
        self,
        samples: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        agent_outputs: Dict[int, list],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Run Agent #3 for all samples in parallel.
        
        Args:
            samples: List of benchmark samples
            stage2_results: Results from stage 2
            agent_outputs: Map of sample ID to agent output trace
            output_path: Path to save intermediate results

        Returns:
            List of verification input results
        """
        self._init_ray()
        workers = self._create_workers()

        if not workers:
            raise RuntimeError(f"No workers created. num_workers={self.num_workers}")

        # Create lookup for stage 2 results
        stage2_map = {r['id']: r['spec'] for r in stage2_results if 'spec' in r}
        
        # Load existing results
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass
        
        # Filter samples that need processing
        to_process = [
            (s, stage2_map.get(s['id']), agent_outputs.get(s['id'], []))
            for s in samples
            if s['id'] not in existing_results and s['id'] in stage2_map
        ]
        print(f"Stage 3: Processing {len(to_process)} samples ({len(existing_results)} already done)")
        
        if not to_process:
            return list(existing_results.values())
        
        # Distribute work
        futures = []
        for i, (sample, spec, agent_output) in enumerate(to_process):
            worker = workers[i % len(workers)]
            future = worker.run_stage3.remote(sample, spec, agent_output)
            futures.append(future)
        
        # Collect results
        results = list(existing_results.values())
        for future in tqdm(futures, desc="Agent #3 - Trace Abstraction"):
            try:
                result = ray.get(future)
                results.append(result)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Worker error in stage 3: {e}")
        
        return results
    
    def run_execution_parallel(
        self,
        stage2_results: List[Dict[str, Any]],
        stage3_results: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Run verification execution for all samples in parallel.
        
        Args:
            stage2_results: Results from stage 2 (specs)
            stage3_results: Results from stage 3 (verification inputs)
            output_path: Path to save results
            
        Returns:
            List of execution results
        """
        self._init_ray()
        workers = self._create_dafny_workers()

        if not workers:
            raise RuntimeError(f"No workers created. dafny_workers={self.dafny_workers}")

        # Create lookups
        stage2_map = {r['id']: r['spec'] for r in stage2_results if 'spec' in r}
        stage3_map = {r['id']: r['verification_input'] for r in stage3_results if 'verification_input' in r}
        
        # Load existing results
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass
        
        # Find samples ready for execution
        to_execute = [
            (sample_id, stage2_map[sample_id], stage3_map[sample_id])
            for sample_id in stage3_map.keys()
            if sample_id in stage2_map and sample_id not in existing_results
        ]
        print(f"Execution: Processing {len(to_execute)} samples ({len(existing_results)} already done)")
        
        if not to_execute:
            return list(existing_results.values())
        
        # Distribute work
        futures = []
        for i, (sample_id, spec, verification_input) in enumerate(to_execute):
            worker = workers[i % len(workers)]
            future = worker.run_execution.remote(sample_id, spec, verification_input)
            futures.append(future)
        
        # Collect results
        results = list(existing_results.values())
        for future in tqdm(futures, desc="Verification Execution"):
            try:
                result = ray.get(future)
                results.append(result)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Worker error in execution: {e}")
        
        return results
    
    def run_execution_parallel_direct(
        self,
        samples: List[Dict[str, Any]],
        agent_outputs: Dict[int, list],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Run verification execution directly for modes that skip stages 1-3.
        Used by: direct, baseline, llm_cot, llm_fewshot

        Args:
            samples: List of benchmark samples
            agent_outputs: Map of sample ID to agent output trace
            output_path: Path to save results

        Returns:
            List of execution results
        """
        self._init_ray()
        workers = self._create_workers()

        if not workers:
            raise RuntimeError(f"No workers created. num_workers={self.num_workers}")

        # Load existing results
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass

        # Find samples that need processing
        to_execute = [
            (s['id'], s, agent_outputs.get(s['id'], []))
            for s in samples
            if s['id'] not in existing_results
        ]
        print(f"Execution (direct): Processing {len(to_execute)} samples ({len(existing_results)} already done)")

        if not to_execute:
            return list(existing_results.values())

        # Distribute work
        futures = []
        for i, (sample_id, sample, agent_output) in enumerate(to_execute):
            worker = workers[i % len(workers)]
            # Pass sample and agent_output for modes that need them
            future = worker.run_execution.remote(
                sample_id=sample_id,
                spec={},
                verification_input={},
                sample=sample,
                agent_output=agent_output
            )
            futures.append(future)

        # Collect results
        results = list(existing_results.values())
        for future in tqdm(futures, desc="Verification Execution"):
            try:
                result = ray.get(future)
                results.append(result)

                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Worker error in execution: {e}")

        return results

    def run_full_verification(
        self,
        data_path: str,
        agent_output_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Run the complete verification pipeline.
        
        Args:
            data_path: Path to benchmark data JSON
            agent_output_path: Path to agent output JSON (from evaluation)
            output_dir: Directory to save all outputs
            
        Returns:
            Summary of verification results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        with open(data_path, 'r') as f:
            samples = json.load(f)

        with open(agent_output_path, 'r') as f:
            agent_data = json.load(f)

        # Apply debug limit if set
        total_available = len(samples)
        if self.max_samples is not None and self.max_samples > 0:
            samples = samples[:self.max_samples]
            agent_data = agent_data[:self.max_samples]

        # Create agent output lookup
        agent_outputs = {d['id']: d.get('output', []) for d in agent_data}

        # Run stages
        print("=" * 50)
        print("Running Formal Verification Pipeline")
        print(f"Model: {self.model_name}")
        print(f"Language: {self.language}")
        if self.max_samples:
            print(f"Samples: {len(samples)} (debug mode, {total_available} total)")
        else:
            print(f"Samples: {len(samples)}")
        print("=" * 50)

        # Modes that skip stages 1-3 and go directly to execution
        skip_pipeline_modes = {"direct", "baseline", "llm_cot", "llm_fewshot"}

        if self.language in skip_pipeline_modes:
            # Skip stages 1-3, run execution directly
            print(f"Mode '{self.language}' - skipping stages 1-3, running execution only")
            stage1_results = []
            stage2_results = []
            stage3_results = []

            execution_path = os.path.join(output_dir, "execution_results.json")
            execution_results = self.run_execution_parallel_direct(
                samples, agent_outputs, execution_path
            )
        else:
            stage1_path = os.path.join(output_dir, "stage1_decomposition.json")
            stage1_results = self.run_stage1_parallel(samples, stage1_path)

            stage2_path = os.path.join(output_dir, "stage2_spec.json")
            stage2_results = self.run_stage2_parallel(samples, stage1_results, stage2_path)

            stage3_path = os.path.join(output_dir, "stage3_verification_input.json")
            stage3_results = self.run_stage3_parallel(samples, stage2_results, agent_outputs, stage3_path)

            execution_path = os.path.join(output_dir, "execution_results.json")
            execution_results = self.run_execution_parallel(stage2_results, stage3_results, execution_path)
        
        # Compute summary
        passed_count = sum(1 for r in execution_results if r.get('passed', False))
        failed_count = sum(1 for r in execution_results if not r.get('passed', False))
        error_count = sum(1 for r in execution_results if r.get('error'))

        # Generate cache index for debugging
        cache_index = {
            "cache_dir": self.cache_dir,
            "samples": {}
        }
        for result in execution_results:
            sample_id = result.get('id')
            if sample_id is not None and result.get('debug_files'):
                cache_index["samples"][str(sample_id)] = result['debug_files']

        cache_index_path = os.path.join(output_dir, "cache_index.json")
        with open(cache_index_path, 'w') as f:
            json.dump(cache_index, f, indent=2)

        summary = {
            "total_samples": len(samples),
            "stage1_completed": len(stage1_results),
            "stage2_completed": len(stage2_results),
            "stage3_completed": len(stage3_results),
            "execution_completed": len(execution_results),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "pass_rate": passed_count / len(execution_results) if execution_results else 0,
            "cache_dir": self.cache_dir,
            "cache_index_file": cache_index_path
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Verification Complete")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Errors: {error_count}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"Cache Dir: {self.cache_dir}")
        print(f"Cache Index: {cache_index_path}")
        print("=" * 50)
        
        # Shutdown Ray
        ray.shutdown()
        
        return summary

