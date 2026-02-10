"""
Iterative Refinement Pipeline for Agent Safety Benchmark.

This package provides tools for iterative refinement of agent responses
using formal verification feedback.

Main components:
- pipeline.py: Main iterative refinement pipeline (Ray for API models)
- vllm_pipeline.py: vLLM-optimized pipeline (for local models)
- trace_manager.py: Trace management compatible with evaluation/ and formal_verification/
- refinement_prompt.py: Feedback prompt generation
- logging_config.py: Logging configuration
- report_generator.py: Markdown report generation
- shield_evaluator.py: Shield model evaluation
"""

from .config import (
    ModelConfig,
    RefinementConfig,
    TARGET_MODELS,
    JUDGE_MODEL
)
from .pipeline import (
    IterativeRefinementPipeline,
    run_refinement
)
from .vllm_pipeline import (
    VLLMRefinementPipeline,
    run_vllm_refinement
)
from .trace_manager import (
    TraceManager,
    SampleTrace,
    VerificationResult
)
from .logging_config import (
    setup_logger,
    get_logger
)
from .report_generator import (
    ReportGenerator
)
from .shield_evaluator import (
    ShieldEvaluator
)

__all__ = [
    # Config
    'ModelConfig',
    'RefinementConfig',
    'TARGET_MODELS',
    'JUDGE_MODEL',
    # Main pipelines
    'IterativeRefinementPipeline',
    'run_refinement',
    # vLLM pipeline
    'VLLMRefinementPipeline',
    'run_vllm_refinement',
    # Trace management
    'TraceManager',
    'SampleTrace',
    'VerificationResult',
    # Logging
    'setup_logger',
    'get_logger',
    # Reporting
    'ReportGenerator',
    # Shield evaluation
    'ShieldEvaluator',
]
