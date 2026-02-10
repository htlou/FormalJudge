"""
Configuration for iterative refinement pipeline.
"""
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str  # Display name
    model_id: str  # Model ID for API calls
    backend: str  # 'api' for Ray-based API models, 'vllm' for local vLLM models
    api_base: str = None  # API base URL (for API models)
    api_key_env: str = None  # Environment variable for API key
    vllm_base_url: str = "http://localhost:8000/v1"  # vLLM server URL
    tensor_parallel_size: int = 1  # For vLLM
    generation_config: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "max_tokens": 8192
    })


# Target models configuration
TARGET_MODELS: Dict[str, ModelConfig] = {
    # API models (use Ray for parallelization)
    "claude-opus-4-5": ModelConfig(
        name="claude-opus-4-5",
        model_id="claude-opus-4-5-20251101",
        backend="api",
        api_base="https://api3.xhub.chat/v1",
        api_key_env="XHUB_API_KEY"
    ),
    "gpt-5": ModelConfig(
        name="gpt-5",
        model_id="gpt-5",
        backend="api",
        api_base="https://api3.xhub.chat/v1",
        api_key_env="XHUB_API_KEY"
    ),
    "gemini-3-pro-preview": ModelConfig(
        name="gemini-3-pro-preview",
        model_id="gemini-3-pro-preview",
        backend="api",
        api_base="https://api3.xhub.chat/v1",
        api_key_env="XHUB_API_KEY"
    ),
    # Local models (use vLLM)
    "qwen-2.5-7b": ModelConfig(
        name="qwen-2.5-7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        backend="vllm",
        tensor_parallel_size=1,
        generation_config={"temperature": 0.0, "max_tokens": 4096}
    ),
    "qwen-2.5-14b": ModelConfig(
        name="qwen-2.5-14b",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        backend="vllm",
        tensor_parallel_size=1,
        generation_config={"temperature": 0.0, "max_tokens": 4096}
    ),
    "qwen-2.5-32b": ModelConfig(
        name="qwen-2.5-32b",
        model_id="Qwen/Qwen2.5-32B-Instruct",
        backend="vllm",
        tensor_parallel_size=2,
        generation_config={"temperature": 0.0, "max_tokens": 4096}
    ),
    "qwen-2.5-72b": ModelConfig(
        name="qwen-2.5-72b",
        model_id="Qwen/Qwen2.5-72B-Instruct",
        backend="vllm",
        tensor_parallel_size=4,
        generation_config={"temperature": 0.0, "max_tokens": 4096}
    ),
}


# Verification judge model
JUDGE_MODEL = ModelConfig(
    name="claude-opus-4-5-20251101",
    model_id="claude-opus-4-5-20251101",
    backend="api",
    api_base="https://api3.xhub.chat/v1",
    api_key_env="XHUB_API_KEY"
)


@dataclass
class RefinementConfig:
    """Configuration for iterative refinement."""
    max_iterations: int = 3  # Maximum number of refinement iterations
    num_workers: int = 8  # Number of parallel workers for API calls
    dafny_workers: int = 4  # Number of parallel workers for Dafny execution
    dafny_timeout: int = 120  # Dafny execution timeout in seconds
    batch_size: int = 100  # Batch size for processing

    # Paths
    data_path: str = "../data/released_data.json"
    output_base_dir: str = "./refinement_results"

    # Verification settings
    verification_language: str = "dafny"  # dafny, python, natural_language, direct, baseline, llm_cot, llm_fewshot

    # Shield model settings
    shield_model_path: str = "/data/hantao/models/ShieldAgent"
    shield_batch_size: int = 16

    # Logging settings
    log_level: str = "INFO"  # Logging level for console output (DEBUG, INFO, WARNING, ERROR)

    # Debug settings
    debug_samples: int = None  # Process only first N samples (for debugging)
