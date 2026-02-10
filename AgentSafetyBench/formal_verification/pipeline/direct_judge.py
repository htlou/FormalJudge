"""
Direct Judge: Single-round pure natural language safety judgment.
A simpler alternative to the 3-agent pipeline, using one LLM call per sample.
"""
import json
import os
import re
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
import ray
from tqdm import tqdm

from ..prompts.direct_judge_prompts import (
    DIRECT_JUDGE_PROMPT,
    format_direct_judge_input
)


class DirectJudge:
    """
    Single-round direct safety judgment using LLM.
    Bypasses the 3-agent hierarchical pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "claude-opus-4-5-20251101",
        api_base: str = None,
        api_key: str = None,
        generation_config: Dict[str, Any] = None
    ):
        """
        Initialize the direct judge.
        
        Args:
            model_name: Name of the LLM to use (default: Claude Opus)
            api_base: API base URL
            api_key: API key
            generation_config: Generation configuration
        """
        self.model_name = model_name
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api3.xhub.chat/v1")
        self.api_key = api_key or os.getenv("XHUB_API_KEY")
        
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=120
        )
        
        self.generation_config = generation_config or {
            "temperature": 0.0,
            "max_tokens": 4096
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
                time.sleep(2 ** attempt)  # Exponential backoff
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
    
    def judge(self, sample: Dict[str, Any], agent_output: list) -> Dict[str, Any]:
        """
        Judge a single sample.
        
        Args:
            sample: The benchmark sample with instruction, environments, risks
            agent_output: The agent's output trace
            
        Returns:
            Judgment result with verdict and explanation
        """
        # Format input
        user_content = format_direct_judge_input(
            instruction=sample.get("instruction", ""),
            risks=sample.get("risks", []),
            environments=sample.get("environments", []),
            trace=agent_output
        )
        
        messages = [
            {"role": "system", "content": DIRECT_JUDGE_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # Call LLM
        response = self._call_llm(messages)
        
        if response is None:
            return {
                "id": sample.get("id"),
                "verdict": "ERROR",
                "passed": False,
                "explanation": "Failed to get LLM response",
                "raw_response": None,
                "error": True
            }
        
        # Parse response
        result = self._extract_json(response)
        
        if result is None:
            # Try to extract verdict from text
            verdict = "UNSAFE"  # Default to unsafe if can't parse
            if "SAFE" in response.upper() and "UNSAFE" not in response.upper():
                verdict = "SAFE"
            
            return {
                "id": sample.get("id"),
                "verdict": verdict,
                "passed": verdict == "SAFE",
                "explanation": response,
                "raw_response": response,
                "error": False,
                "parse_error": True
            }
        
        verdict = result.get("verdict", "UNSAFE").upper()
        passed = verdict == "SAFE"
        
        return {
            "id": sample.get("id"),
            "verdict": verdict,
            "passed": passed,
            "confidence": result.get("confidence", "unknown"),
            "primary_concern": result.get("primary_concern", ""),
            "explanation": result.get("explanation", ""),
            "raw_response": response,
            "error": False
        }


@ray.remote
class DirectJudgeWorker:
    """Ray worker for parallel direct judgment."""
    
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        generation_config: Dict[str, Any]
    ):
        self.judge = DirectJudge(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            generation_config=generation_config
        )
    
    def judge_sample(self, sample: Dict[str, Any], agent_output: list) -> Dict[str, Any]:
        """Judge a single sample."""
        return self.judge.judge(sample, agent_output)


class ParallelDirectJudge:
    """
    Parallel runner for direct judgment using Ray.
    """

    def __init__(
        self,
        model_name: str = "claude-opus-4-5-20251101",
        num_workers: int = 4,
        api_base: str = None,
        api_key: str = None,
        generation_config: Dict[str, Any] = None,
        max_samples: int = None
    ):
        """
        Initialize the parallel direct judge.

        Args:
            model_name: Name of the LLM to use
            num_workers: Number of parallel workers
            api_base: API base URL
            api_key: API key
            generation_config: LLM generation config
            max_samples: Max samples to process (for debugging)
        """
        self.model_name = model_name
        self.num_workers = num_workers
        self.api_base = api_base or os.getenv("API_BASE_URL", "https://api3.xhub.chat/v1")
        self.api_key = api_key or os.getenv("XHUB_API_KEY")
        self.generation_config = generation_config or {"temperature": 0.0, "max_tokens": 4096}
        self.max_samples = max_samples
    
    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        # Check if Ray is already initialized and using wrong temp dir
        if ray.is_initialized():
            current_tmpdir = os.getenv('RAY_TMPDIR')
            if current_tmpdir:
                # If we want a specific temp dir but Ray is already initialized,
                # we need to shut down and reinitialize
                try:
                    ray.shutdown()
                except:
                    pass
        
        if not ray.is_initialized():
            # Use RAY_TMPDIR from environment if set (by shell script)
            ray_tmpdir = os.getenv('RAY_TMPDIR')
            if ray_tmpdir:
                # Ensure the directory exists
                os.makedirs(ray_tmpdir, exist_ok=True)
                # Set it before initializing Ray
                os.environ['RAY_TMPDIR'] = ray_tmpdir
            ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
    
    def _create_workers(self) -> List:
        """Create worker pool."""
        return [
            DirectJudgeWorker.remote(
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                generation_config=self.generation_config
            )
            for _ in range(self.num_workers)
        ]
    
    def run(
        self,
        data_path: str,
        agent_output_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Run direct judgment on all samples.

        Args:
            data_path: Path to benchmark data JSON
            agent_output_path: Path to agent output JSON
            output_path: Path to save results (can be a directory or file path)

        Returns:
            Summary of results
        """
        self._init_ray()
        workers = self._create_workers()

        if not workers:
            raise RuntimeError(f"No workers created. num_workers={self.num_workers}")

        # Determine if output_path is a directory or file
        if output_path.endswith('.json'):
            # Legacy mode: single file output
            output_dir = os.path.dirname(output_path)
            results_path = output_path
            summary_path = output_path.replace('.json', '_summary.json')
        else:
            # New mode: directory-based output like Dafny pipeline
            output_dir = output_path
            results_path = os.path.join(output_dir, "execution_results.json")
            summary_path = os.path.join(output_dir, "summary.json")

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

        # Create agent output lookup and sample lookup
        agent_outputs = {d['id']: d.get('output', []) for d in agent_data}
        sample_lookup = {s['id']: s for s in samples}

        # Load existing results if any
        existing_results = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    existing_data = json.load(f)
                existing_results = {r['id']: r for r in existing_data}
            except:
                pass

        # Filter samples that need processing
        to_process = [
            (s, agent_outputs.get(s['id'], []))
            for s in samples
            if s['id'] not in existing_results and s['id'] in agent_outputs
        ]

        print("=" * 50)
        print("Running Direct Judge (Pure NL)")
        print(f"Model: {self.model_name}")
        if self.max_samples:
            print(f"Total samples: {len(samples)} (debug mode, {total_available} total)")
        else:
            print(f"Total samples: {len(samples)}")
        print(f"Already processed: {len(existing_results)}")
        print(f"To process: {len(to_process)}")
        print(f"Output directory: {output_dir}")
        print("=" * 50)

        if not to_process:
            results = list(existing_results.values())
        else:
            # Distribute work
            futures = []
            for i, (sample, agent_output) in enumerate(to_process):
                worker = workers[i % len(workers)]
                future = worker.judge_sample.remote(sample, agent_output)
                futures.append(future)

            # Collect results
            results = list(existing_results.values())
            for future in tqdm(futures, desc="Direct Judge"):
                try:
                    result = ray.get(future)
                    results.append(result)

                    # Save intermediate results
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Worker error: {e}")

        # Compute summary
        passed_count = sum(1 for r in results if r.get('passed', False))
        failed_count = sum(1 for r in results if not r.get('passed', False) and not r.get('error', False))
        error_count = sum(1 for r in results if r.get('error', False))

        summary = {
            "total_samples": len(samples),
            "processed": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "errors": error_count,
            "pass_rate": passed_count / len(results) if results else 0,
            "model_name": self.model_name,
            "output_dir": output_dir
        }

        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results with original sample info and agent traces
        detailed_results_path = os.path.join(output_dir, "detailed_results.json")
        detailed_results = []
        for r in results:
            sample_id = r.get('id')
            detailed = {
                "id": sample_id,
                "verification_result": {
                    "passed": r.get('passed', False),
                    "verdict": r.get('verdict', 'UNKNOWN'),
                    "confidence": r.get('confidence', 'unknown'),
                    "primary_concern": r.get('primary_concern', ''),
                    "explanation": r.get('explanation', ''),
                    "error": r.get('error', False)
                },
                "raw_llm_response": r.get('raw_response', ''),
                "original_sample": sample_lookup.get(sample_id, {}),
                "agent_output": agent_outputs.get(sample_id, [])
            }
            detailed_results.append(detailed)

        with open(detailed_results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 50)
        print("Direct Judge Complete")
        print(f"Passed (SAFE): {passed_count}")
        print(f"Failed (UNSAFE): {failed_count}")
        print(f"Errors: {error_count}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"Results saved to: {output_dir}")
        print(f"  - execution_results.json")
        print(f"  - detailed_results.json")
        print(f"  - summary.json")
        print("=" * 50)

        # Shutdown Ray
        ray.shutdown()

        return summary

