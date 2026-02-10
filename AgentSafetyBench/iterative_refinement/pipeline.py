"""
Main iterative refinement pipeline.

Orchestrates the full refinement loop:
1. Initial agent evaluation
2. Formal verification (Dafny/Python/NL)
3. Shield model evaluation (determines PASSED/FAILED)
4. Feedback generation
5. Refinement iterations (all samples continue through all iterations)
"""
import os
import json
import ray
import time
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import ModelConfig, RefinementConfig, TARGET_MODELS, JUDGE_MODEL
from .model_api import create_model_api, BaseModelAPI
from .trace_manager import TraceManager, SampleTrace, VerificationResult
from .refinement_prompt import RefinementPromptGenerator, extract_agent_artifact
from .parallel_runner import APIAgentWorker, VerificationWorker
from .logging_config import setup_logger, get_logger
from .report_generator import ReportGenerator
from .shield_evaluator import ShieldEvaluator


class IterativeRefinementPipeline:
    """
    Main pipeline for iterative refinement with formal verification.

    Key behavior:
    - PASSED/FAILED is determined by Shield model (not verification)
    - All samples continue through ALL iterations (not just failed ones)
    - Alignment rate = agreement between verification and shield judgments
    """

    def __init__(
        self,
        model_config: ModelConfig,
        judge_config: ModelConfig = None,
        refinement_config: RefinementConfig = None
    ):
        """
        Initialize the pipeline.

        Args:
            model_config: Target model configuration
            judge_config: Judge model configuration (defaults to JUDGE_MODEL)
            refinement_config: Refinement configuration
        """
        self.model_config = model_config
        self.judge_config = judge_config or JUDGE_MODEL
        self.config = refinement_config or RefinementConfig()

        self.prompt_generator = RefinementPromptGenerator()

        # Environment manager path
        self.env_manager_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "environments"
        )

        # Project root for Ray workers
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Shield evaluator (lazy loaded)
        self._shield_evaluator = None

    @property
    def shield_evaluator(self) -> ShieldEvaluator:
        """Lazy load shield evaluator."""
        if self._shield_evaluator is None:
            self._shield_evaluator = ShieldEvaluator(
                model_path=self.config.shield_model_path,
                batch_size=self.config.shield_batch_size
            )
        return self._shield_evaluator

    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray_tmpdir = os.getenv('RAY_TMPDIR')
            if ray_tmpdir:
                os.makedirs(ray_tmpdir, exist_ok=True)
            ray.init(
                num_cpus=self.config.num_workers + self.config.dafny_workers,
                ignore_reinit_error=True
            )

    def _create_agent_workers(self, count: int = None) -> List:
        """Create agent worker pool."""
        from dataclasses import asdict
        n = count if count is not None else self.config.num_workers
        # Convert ModelConfig to dict to avoid serialization issues
        model_config_dict = asdict(self.model_config)
        return [
            APIAgentWorker.remote(
                model_config_dict=model_config_dict,
                env_manager_module_path=self.env_manager_path,
                project_root=self.project_root
            )
            for _ in range(n)
        ]

    def _create_verification_workers(self, cache_dir: str, count: int = None) -> List:
        """Create verification worker pool."""
        from dataclasses import asdict
        n = count if count is not None else self.config.dafny_workers
        # Convert ModelConfig to dict to avoid serialization issues
        judge_config_dict = asdict(self.judge_config)
        return [
            VerificationWorker.remote(
                judge_model_config_dict=judge_config_dict,
                language=self.config.verification_language,
                cache_dir=cache_dir,
                dafny_timeout=self.config.dafny_timeout,
                project_root=self.project_root
            )
            for _ in range(n)
        ]

    def run_batch_refinement(
        self,
        samples: List[Dict[str, Any]],
        output_dir: str,
        resume_from: str = None
    ) -> Dict[str, Any]:
        """
        Run iterative refinement on a batch of samples.

        Args:
            samples: List of benchmark samples
            output_dir: Directory to save results
            resume_from: Optional path to resume from previous run

        Returns:
            Summary of results
        """
        start_time = time.time()
        timing_data = {}

        self._init_ray()

        # Setup trace manager
        trace_manager = TraceManager(output_dir, self.model_config.name)

        # Setup logger
        logger = setup_logger(
            trace_manager.model_dir,
            log_level=self.config.log_level
        )
        logger.info("=" * 60)
        logger.info("Iterative Refinement Pipeline Started")
        logger.info("=" * 60)

        # Load existing traces if resuming
        if resume_from and os.path.exists(resume_from):
            trace_manager.load_existing(resume_from)
            logger.info(f"Resumed from {resume_from}, loaded {len(trace_manager.traces)} traces")

        # Filter samples that need processing
        existing_ids = set(trace_manager.traces.keys())
        to_process = [s for s in samples if s['id'] not in existing_ids]

        if self.config.debug_samples:
            to_process = to_process[:self.config.debug_samples]

        # Log configuration
        logger.info(f"Model: {self.model_config.name}")
        logger.info(f"Judge: {self.judge_config.name}")
        logger.info(f"Verification Language: {self.config.verification_language}")
        logger.info(f"Shield Model: {self.config.shield_model_path}")
        logger.info(f"Max Iterations: {self.config.max_iterations}")
        logger.info(f"Samples to process: {len(to_process)}")
        logger.info(f"Output Directory: {trace_manager.model_dir}")
        logger.debug(f"Batch Size: {self.config.batch_size}")
        logger.debug(f"Num Workers: {self.config.num_workers}")
        logger.debug(f"Dafny Workers: {self.config.dafny_workers}")

        # Get system prompt
        api = create_model_api(self.model_config)
        system_prompt = api.get_system_prompt()

        # Create cache directory for this run
        cache_dir = os.path.join(trace_manager.model_dir, "dafny_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Process samples in batches
        batch_start_time = time.time()
        for batch_start in range(0, len(to_process), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(to_process))
            batch = to_process[batch_start:batch_end]

            batch_num = batch_start // self.config.batch_size + 1
            logger.info(f"Processing batch {batch_num} ({batch_start+1}-{batch_end} of {len(to_process)})")

            # Run refinement on batch
            self._run_batch(
                batch,
                trace_manager,
                system_prompt,
                cache_dir,
                logger
            )

            # Save progress
            trace_manager.save_all()
            logger.debug(f"Batch {batch_num} completed, progress saved")

        timing_data['batch_processing'] = time.time() - batch_start_time

        # Final save and summary
        save_start = time.time()
        summary = trace_manager.save_all()
        timing_data['save_results'] = time.time() - save_start

        # Calculate total time
        timing_data['total_time'] = time.time() - start_time

        # Generate markdown report
        logger.info("Generating run report...")
        report_generator = ReportGenerator(
            trace_manager=trace_manager,
            model_name=self.model_config.name,
            judge_name=self.judge_config.name,
            verification_language=self.config.verification_language
        )

        config_details = {
            "max_iterations": self.config.max_iterations,
            "num_workers": self.config.num_workers,
            "dafny_workers": self.config.dafny_workers,
            "batch_size": self.config.batch_size,
            "verification_language": self.config.verification_language,
            "shield_model_path": self.config.shield_model_path
        }

        report_path = os.path.join(trace_manager.model_dir, "run_report.md")
        report_generator.save_report(
            output_path=report_path,
            timing_data=timing_data,
            config_details=config_details
        )
        logger.info(f"Report saved to: {report_path}")

        # Compute alignment for final logging
        alignment = trace_manager.compute_verification_shield_alignment()

        # Log final summary
        logger.info("=" * 60)
        logger.info("Refinement Complete")
        logger.info(f"Total Samples: {summary['total_samples']}")
        logger.info(f"Passed (Shield SAFE): {summary['passed_samples']}")
        logger.info(f"Safety Rate (Shield): {summary['pass_rate']:.2%}")
        logger.info(f"Alignment Rate (Verification vs Shield): {alignment['alignment_rate']:.2%}")
        logger.info(f"Average Iterations: {summary['avg_iterations']:.2f}")
        logger.info(f"Total Time: {timing_data['total_time']:.1f}s")
        logger.info(f"Output Directory: {summary['output_dir']}")
        logger.info("=" * 60)

        ray.shutdown()
        return summary

    def _run_batch(
        self,
        batch: List[Dict[str, Any]],
        trace_manager: TraceManager,
        system_prompt: str,
        cache_dir: str,
        logger=None
    ):
        """
        Run refinement on a batch of samples.

        All samples go through ALL iterations (not just failed ones).

        Args:
            batch: Batch of samples
            trace_manager: Trace manager instance
            system_prompt: System prompt for the agent
            cache_dir: Cache directory for Dafny files
            logger: Logger instance
        """
        if logger is None:
            logger = get_logger()

        # Create workers
        agent_workers = self._create_agent_workers()
        verification_workers = self._create_verification_workers(cache_dir)

        # Initialize traces for batch
        for sample in batch:
            trace_manager.create_trace(sample)

        # Track all samples - ALL samples go through ALL iterations
        all_samples = {s['id']: s for s in batch}
        iteration_messages = {s['id']: None for s in batch}

        for iteration in range(self.config.max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.config.max_iterations}, {len(all_samples)} samples")

            # Step 1: Run agent on ALL samples
            logger.debug(f"  Running agent evaluation on {len(all_samples)} samples")
            agent_results = self._run_agents_parallel(
                list(all_samples.values()),
                iteration_messages,
                system_prompt,
                agent_workers,
                iteration
            )

            # Step 2: Run verification on agent outputs
            logger.debug(f"  Running verification on agent outputs")
            verification_results = self._run_verification_parallel(
                all_samples,
                agent_results,
                verification_workers
            )

            # Step 3: Run shield evaluation on agent outputs
            logger.debug(f"  Running shield evaluation on agent outputs")
            shield_results = self._run_shield_evaluation(
                list(all_samples.values()),
                agent_results,
                logger
            )

            # Step 4: Process results and prepare next iteration for ALL samples
            for sample_id, (messages, success, error) in agent_results.items():
                vr_data = verification_results.get(sample_id, {})
                shield_data = shield_results.get(sample_id, {"passed": False, "pred_label": -1})

                # Create verification result object
                vr = VerificationResult(
                    passed=vr_data.get("passed", False),
                    explanation=vr_data.get("explanation", error or "Unknown error"),
                    language=self.config.verification_language,
                    error=vr_data.get("error"),
                    spec_code=vr_data.get("spec_code"),
                    harness_code=vr_data.get("harness_code"),
                    decomposition=vr_data.get("decomposition"),
                    spec=vr_data.get("spec"),
                    verification_input=vr_data.get("verification_input")
                )

                # Generate refinement prompt for next iteration
                sample = all_samples[sample_id]
                _, refinement_prompt = self.prompt_generator.generate_refinement_messages(
                    sample, messages, vr, system_prompt
                )

                # Add iteration to trace with both verification and shield results
                trace_manager.add_iteration(
                    sample_id,
                    iteration,
                    messages,
                    vr,
                    refinement_prompt,
                    shield_result=shield_data
                )
                trace_manager.save_iteration(sample_id, iteration)

                # Log status based on SHIELD result (not verification)
                shield_status = "SAFE" if shield_data.get("passed", False) else "UNSAFE"
                verif_status = "SAFE" if vr.passed else "UNSAFE"

                if shield_data.get("passed", False):
                    logger.info(f"    Sample {sample_id}: Shield={shield_status}, Verification={verif_status}")
                else:
                    logger.warning(f"    Sample {sample_id}: Shield={shield_status}, Verification={verif_status}")

                # Prepare for next iteration (ALL samples continue)
                if iteration < self.config.max_iterations - 1:
                    new_messages = [{"role": "system", "content": system_prompt}]
                    new_messages.append({"role": "user", "content": refinement_prompt})
                    iteration_messages[sample_id] = new_messages
                    logger.debug(f"    Sample {sample_id}: Preparing iteration {iteration + 2}")

    def _run_shield_evaluation(
        self,
        samples: List[Dict[str, Any]],
        agent_results: Dict[int, tuple],
        logger
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run shield model evaluation on agent outputs.

        Args:
            samples: List of samples
            agent_results: Results from agent evaluation
            logger: Logger instance

        Returns:
            Dict mapping sample_id to shield evaluation result
        """
        # Build messages dict for successful samples
        messages_dict = {}
        sample_list = []

        for sample in samples:
            sample_id = sample['id']
            result = agent_results.get(sample_id)
            if result:
                messages, success, error = result
                if success and messages:
                    messages_dict[sample_id] = messages
                    sample_list.append(sample)

        if not sample_list:
            return {}

        # Run shield evaluation
        logger.debug(f"  Evaluating {len(sample_list)} samples with shield model")
        shield_results = self.shield_evaluator.evaluate_batch(sample_list, messages_dict)

        return shield_results

    def _run_agents_parallel(
        self,
        samples: List[Dict[str, Any]],
        iteration_messages: Dict[int, Optional[List[Dict]]],
        system_prompt: str,
        workers: List,
        iteration: int
    ) -> Dict[int, tuple]:
        """
        Run agents in parallel on samples.

        Args:
            samples: List of samples
            iteration_messages: Pre-computed messages for each sample (None for initial)
            system_prompt: System prompt
            workers: List of Ray workers
            iteration: Current iteration number

        Returns:
            Dict mapping sample_id to (messages, success, error)
        """
        futures = []
        future_to_sample = {}

        for i, sample in enumerate(samples):
            worker = workers[i % len(workers)]

            # Get or create initial messages
            if iteration_messages.get(sample['id']) is not None:
                messages = iteration_messages[sample['id']]
            else:
                messages = self.prompt_generator.generate_initial_messages(
                    sample, system_prompt
                )

            future = worker.run_agent.remote(sample, messages)
            futures.append(future)
            future_to_sample[future] = sample

        # Collect results
        results = {}
        for future in tqdm(futures, desc=f"    Agent evaluation"):
            try:
                messages, success, error = ray.get(future)
                sample = future_to_sample[future]
                results[sample['id']] = (messages, success, error)
            except Exception as e:
                sample = future_to_sample[future]
                results[sample['id']] = ([], False, str(e))

        return results

    def _run_verification_parallel(
        self,
        samples_dict: Dict[int, Dict[str, Any]],
        agent_results: Dict[int, tuple],
        workers: List
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run verification in parallel on agent outputs.

        Args:
            samples_dict: Dict mapping sample_id to sample
            agent_results: Results from agent evaluation
            workers: List of verification workers

        Returns:
            Dict mapping sample_id to verification result
        """
        futures = []
        future_to_id = {}

        items = list(agent_results.items())
        for i, (sample_id, (messages, success, error)) in enumerate(items):
            if not success or not messages:
                continue

            worker = workers[i % len(workers)]
            sample = samples_dict[sample_id]

            future = worker.verify.remote(sample, messages, sample_id)
            futures.append(future)
            future_to_id[future] = sample_id

        # Collect results
        results = {}
        for future in tqdm(futures, desc=f"    Verification"):
            try:
                result = ray.get(future)
                sample_id = future_to_id[future]
                results[sample_id] = result
            except Exception as e:
                sample_id = future_to_id[future]
                results[sample_id] = {
                    "passed": False,
                    "explanation": str(e),
                    "error": str(e)
                }

        return results


def run_refinement(
    model_name: str,
    data_path: str = None,
    output_dir: str = None,
    max_iterations: int = 3,
    num_workers: int = 8,
    verification_language: str = "dafny",
    debug_samples: int = None
) -> Dict[str, Any]:
    """
    Convenience function to run iterative refinement.

    Args:
        model_name: Name of the target model (from TARGET_MODELS)
        data_path: Path to benchmark data
        output_dir: Output directory
        max_iterations: Maximum refinement iterations
        num_workers: Number of parallel workers
        verification_language: Verification language
        debug_samples: Number of samples for debugging

    Returns:
        Summary of results
    """
    if model_name not in TARGET_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(TARGET_MODELS.keys())}")

    model_config = TARGET_MODELS[model_name]

    config = RefinementConfig(
        max_iterations=max_iterations,
        num_workers=num_workers,
        verification_language=verification_language,
        debug_samples=debug_samples
    )

    if data_path:
        config.data_path = data_path
    if output_dir:
        config.output_base_dir = output_dir

    # Load samples
    with open(config.data_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Create pipeline and run
    pipeline = IterativeRefinementPipeline(
        model_config=model_config,
        judge_config=JUDGE_MODEL,
        refinement_config=config
    )

    return pipeline.run_batch_refinement(samples, config.output_base_dir)
