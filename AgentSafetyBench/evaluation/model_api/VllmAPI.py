"""
vLLM API for fast inference.

Supports two modes:
1. Direct inference using vLLM's LLM class (offline)
2. OpenAI-compatible server mode (online)
"""
from openai import OpenAI
import time
import json
import random
import string
import sys
import os
sys.path.append('./model_api')
from BaseAPI import BaseAPI


class VllmAPI(BaseAPI):
    """vLLM API using OpenAI-compatible server endpoint."""

    def __init__(self, model_name, generation_config={}, base_url="http://localhost:8000/v1", api_key="EMPTY"):
        """
        Initialize vLLM API client.

        Args:
            model_name: Model name (must match the model served by vLLM)
            generation_config: Generation parameters
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: API key (default: EMPTY for local vLLM server)
        """
        super().__init__(generation_config)
        self.model_name = model_name
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.sys_prompt = self.without_strict_jsonformat_sys_prompt

    def response(self, messages, tools):
        if not tools:
            tools = None
        last_error = None
        bad_request_count = 0
        max_bad_request_retries = 3  # Limit retries for model-generated bad JSON

        for retry in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    print(f"vLLM returned None response, retry {retry + 1}/10")
                    continue
                return completion
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Don't retry on certain errors
                if "does not exist" in error_str or "NotFoundError" in error_str:
                    print(f"vLLM API error (not retrying): {e}")
                    return None
                # Limit retries for 400 errors (bad model output)
                if "400" in error_str or "BadRequestError" in error_str:
                    bad_request_count += 1
                    if bad_request_count >= max_bad_request_retries:
                        print(f"vLLM API: model generated invalid JSON {bad_request_count} times, giving up")
                        return None
                    print(f"vLLM API error (bad model output, retry {bad_request_count}/{max_bad_request_retries}): {e}")
                    time.sleep(0.5)
                    continue
                print(f"vLLM API error (retry {retry + 1}/10): {e}")
                time.sleep(1 + retry * 0.5)  # Increasing backoff
        print(f"vLLM API failed after 10 retries. Last error: {last_error}")
        return None

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        if completion is None:
            return {'type': 'error', 'message': 'vLLM API returned None after retries'}

        # Tool call part - check for non-empty tool_calls list
        tool_calls = completion.choices[0].message.tool_calls
        if tool_calls is not None and len(tool_calls) > 0:
            tool_call = tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}

        # Normal content part
        else:
            content = completion.choices[0].message.content
            return {'type': 'content', 'content': content}


class VllmOfflineAPI(BaseAPI):
    """vLLM API using direct offline inference (no server needed)."""

    def __init__(self, model_path, generation_config={}, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        """
        Initialize vLLM offline inference.

        Args:
            model_path: Path to the model or HuggingFace model name
            generation_config: Generation parameters (max_tokens, temperature, etc.)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        super().__init__(generation_config)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.model_path = model_path
        print(f'Loading vLLM model from {model_path}...')
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print('Finished loading vLLM model')

        # Set up sampling params from generation_config
        self.sampling_params = SamplingParams(
            max_tokens=generation_config.get('max_tokens', 2048),
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.95),
        )
        self.sys_prompt = self.basic_sys_prompt
        self.no_tool_sys_prompt = """You are a helpful assistant"""

    def get_system_prompt(self, d=None):
        if d is None:
            return self.sys_prompt
        else:
            if not d['environments'] or 'name' not in d['environments'][0] or d['environments'][0]['name'] == '':
                return self.no_tool_sys_prompt
            else:
                return self.sys_prompt

    def parse_tool_str(self, tool_str):
        try:
            if '{' not in tool_str or '}' not in tool_str:
                return {'type': 'error', 'message': f'Wrong tool call result: {tool_str}'}
            res = json.loads(tool_str)
            if 'name' not in res or ('arguments' not in res and 'parameters' not in res):
                return {'type': 'error', 'message': f'Wrong tool call result: {res}'}
            tool_call_id = ''.join(random.sample(string.ascii_letters + string.digits, 9))
            tool_name = res['name']
            if 'parameters' in res:
                res['arguments'] = res['parameters']
            arguments = json.loads(res['arguments']) if isinstance(res['arguments'], str) else res['arguments']
            return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}
        except Exception:
            return {'type': 'error', 'message': f'Wrong tool call result: {tool_str}'}

    def response(self, messages, tools):
        # Apply chat template
        if tools:
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        # Generate
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print(f'completion: {completion}')
        completion = completion.replace('```json', '').replace('```', '').strip('\n')

        # Tool call part
        if self.is_json(completion) and 'name' in self.parse_json(completion):
            return self.parse_tool_str(completion)

        # Try to extract JSON from content
        elif '{' in completion and '}' in completion:
            start = completion.index('{')
            end = completion.rindex('}')
            tool_str = completion[start:end + 1]
            tool_str = tool_str.replace("'", '"').replace('"{', '{').replace('}"', '}')
            if self.is_json(tool_str) and 'name' in self.parse_json(tool_str):
                return self.parse_tool_str(tool_str)
            else:
                return {'type': 'content', 'content': completion}
        else:
            return {'type': 'content', 'content': completion}
