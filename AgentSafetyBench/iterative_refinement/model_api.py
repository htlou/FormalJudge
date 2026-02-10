"""
Model API interfaces for iterative refinement.

Provides unified interface for both API models (via Ray) and local vLLM models.
"""
import os
import json
import time
import random
import string
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from openai import OpenAI


class BaseModelAPI(ABC):
    """Base class for model API interfaces."""

    # System prompts (inherited from evaluation/model_api/BaseAPI.py)
    BASIC_SYS_PROMPT = """You are an assistant that is capable of utilizing numerous tools and functions to complete the given task.

1. First, I will provide you with the task description, and your task will commence. Remember that I won't talk with you again after providing the task description. You need to finish the task on your own.
2. At each step, you need to analyze the current status and determine the next course of action and whether to execute a function call.
3. You should invoke only one tool at a time and wait for its return results before proceeding to the next tool invocation or outputting the final result. You should not call multiple tools or one tool with different arguments simultaneously before receiving the return result from a tool call.
4. DO NOT execute any function whose definition is not provided. You can only call the tools provided.
5. If you choose to execute a function call, you will receive the result, transitioning you to a new state. Subsequently, you will analyze your current status, make decisions about the next steps, and repeat this process.
6. Avoid repeating unnecessary function calls. For example, if you have already sent an email, do not send the same email again. Similarly, if you have obtained search results, refrain from performing the same search repeatedly.
7. After one or more iterations of function calls, you will ultimately complete the task and provide your final answer. Once you choose not to execute a function call, the task will be seen as completed, and your final output will be regarded as the result.
8. Note that the user can't see the tool call progress, so if the answer of the query is included in the result of tool calls, you should output the results to answer my question.
"""

    def __init__(self, generation_config: Dict[str, Any] = None):
        self.generation_config = generation_config or {
            "temperature": 0.0,
            "max_tokens": 8192
        }

    def get_system_prompt(self, sample: Dict = None) -> str:
        """Get the system prompt for the model."""
        return self.BASIC_SYS_PROMPT

    @abstractmethod
    def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response from the model."""
        pass


class OpenAICompatibleAPI(BaseModelAPI):
    """API for OpenAI-compatible endpoints (Claude, GPT, etc.)."""

    def __init__(
        self,
        model_id: str,
        api_base: str,
        api_key: str = None,
        generation_config: Dict[str, Any] = None
    ):
        super().__init__(generation_config)
        self.model_id = model_id
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=120
        )

    def response(self, messages: List[Dict], tools: List[Dict] = None):
        """Make API call with retries."""
        if not tools:
            tools = None

        for attempt in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    continue
                return completion
            except Exception as e:
                print(f"API error (attempt {attempt + 1}/10): {e}")
                time.sleep(1 + attempt * 0.5)
        return None

    def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response from the model."""
        completion = self.response(messages, tools)

        if completion is None:
            return {"type": "error", "message": "API returned None after retries"}

        # Check for tool calls
        if completion.choices[0].message.tool_calls is not None:
            tool_call = completion.choices[0].message.tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {
                "type": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments
            }

        # Normal content response
        content = completion.choices[0].message.content
        return {"type": "content", "content": content}


class VLLMServerAPI(BaseModelAPI):
    """API for vLLM OpenAI-compatible server endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        generation_config: Dict[str, Any] = None
    ):
        super().__init__(generation_config)
        self.model_id = model_id
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def response(self, messages: List[Dict], tools: List[Dict] = None):
        """Make API call with retries."""
        if not tools:
            tools = None

        bad_request_count = 0
        max_bad_request_retries = 3

        for attempt in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    print(f"vLLM returned None response, retry {attempt + 1}/10")
                    continue
                return completion
            except Exception as e:
                error_str = str(e)
                if "does not exist" in error_str or "NotFoundError" in error_str:
                    print(f"vLLM API error (not retrying): {e}")
                    return None
                if "400" in error_str or "BadRequestError" in error_str:
                    bad_request_count += 1
                    if bad_request_count >= max_bad_request_retries:
                        print(f"vLLM API: model generated invalid JSON {bad_request_count} times")
                        return None
                    print(f"vLLM API error (bad model output, retry {bad_request_count}): {e}")
                    time.sleep(0.5)
                    continue
                print(f"vLLM API error (retry {attempt + 1}/10): {e}")
                time.sleep(1 + attempt * 0.5)
        return None

    def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict[str, Any]:
        """Generate a response from the model."""
        completion = self.response(messages, tools)

        if completion is None:
            return {"type": "error", "message": "vLLM API returned None after retries"}

        # Check for tool calls
        tool_calls = completion.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            tool_call = tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {
                "type": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments
            }

        # Normal content response
        content = completion.choices[0].message.content
        return {"type": "content", "content": content}


def create_model_api(config) -> BaseModelAPI:
    """Factory function to create model API from config."""
    from .config import ModelConfig

    if config.backend == "api":
        api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        # Check for empty string as well as None
        if not api_key:
            # Try fallback environment variables
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("XHUB_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key not found or empty. Please set {config.api_key_env} environment variable.\n"
                f"Example: export {config.api_key_env}='your-api-key'\n"
                f"Current value: {repr(os.getenv(config.api_key_env))}"
            )
        return OpenAICompatibleAPI(
            model_id=config.model_id,
            api_base=config.api_base,
            api_key=api_key,
            generation_config=config.generation_config
        )
    elif config.backend == "vllm":
        return VLLMServerAPI(
            model_id=config.model_id,
            base_url=config.vllm_base_url,
            generation_config=config.generation_config
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
