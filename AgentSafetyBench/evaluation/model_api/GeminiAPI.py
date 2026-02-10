import os
import requests
import time
import json
import random
import string
import sys
import copy
sys.path.append("./model_api")
from BaseAPI import BaseAPI


class GeminiAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}):
        super().__init__(generation_config)
        self.model_name = model_name
        self.base_url="https://openrouter.ai/api/v1"
        self.api_key=os.getenv("OPENROUTER_API_KEY")
        if "temperature" not in self.generation_config:
            self.generation_config["temperature"] = 1.0
        self.sys_prompt = self.without_strict_jsonformat_sys_prompt

    def _clean_messages_for_api(self, messages):
        """
        Clean messages before sending to API.

        For Gemini 3 with thinking enabled:
        - Remove reasoning_content from messages (it's output-only, not for input)
        - Keep extra_content in tool_calls (API needs thought_signature for context)
        """
        cleaned = []
        for msg in messages:
            new_msg = {}
            for key, value in msg.items():
                # Skip reasoning_content - it's output-only
                if key == 'reasoning_content':
                    continue
                new_msg[key] = value
            cleaned.append(new_msg)
        return cleaned

    def response(self, messages, tools):
        """Make raw HTTP request to capture full response including thought_signature."""
        if not tools:
            tools = None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Clean messages before sending (removes reasoning_content but keeps extra_content)
        cleaned_messages = self._clean_messages_for_api(messages)

        payload = {
            "model": self.model_name,
            "messages": cleaned_messages,
            **self.generation_config
        }
        if tools:
            payload["tools"] = tools

        for _ in range(10):
            try:
                resp = requests.post(
                    # f"{self.base_url}/chat/completions",
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                if resp.status_code == 200:
                    # print(f"Response: {resp.json()}")
                    return resp.json()
                else:
                    print(f"API error: {resp.status_code} - {resp.text}")
                    print(f"In payload: {payload}")
                    time.sleep(1)
            except Exception as e:
                print(e)
                time.sleep(1)
        return None

    def generate_response(self, messages, tools):
        tool_call_id = None
        func_name = None
        for i, message in enumerate(messages):
            # Handle legacy "function_call" format (convert to tool_calls format)
            if "function_call" in message:
                tool_call_id = "".join(
                    random.sample(string.ascii_letters + string.digits, 9)
                )
                new_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": message["function_call"],
                        }
                    ],
                }
                messages[i] = new_message

            # Handle legacy "function" role (convert to "tool" role)
            elif message["role"] == "function":
                new_message = {
                    "role": "tool",
                    "content": message["content"],
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                }
                messages[i] = new_message

            # Track the function name for tool results that follow
            # But do NOT modify messages that already have tool_calls
            # (they may contain extra_content with thought_signature that must be preserved)
            if "tool_calls" in messages[i]:
                func_name = messages[i]["tool_calls"][0]["function"]["name"]

        completion = self.response(messages, tools)

        if completion is None:
            return None

        # print(f'completion: {completion}')

        try:
            choice = completion["choices"][0]
            message = choice["message"]

            # Check for tool calls
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_call_id = tool_call["id"]
                tool_name = tool_call["function"]["name"]
                arguments_str = tool_call["function"].get("arguments", "{}")
                if arguments_str:
                    arguments = json.loads(arguments_str)
                else:
                    arguments = {}

                # Build full assistant message preserving ALL fields from the response
                # CRITICAL: Use deep copy to preserve extra_content with thought_signature
                # This prevents any later modifications from affecting the original structure
                full_assistant_message = {
                    "role": "assistant",
                    "tool_calls": copy.deepcopy(message["tool_calls"])  # Deep copy preserves extra_content
                }

                # Copy content if present (empty string is valid)
                if "content" in message:
                    full_assistant_message["content"] = message["content"]

                # Copy reasoning_content if present (Gemini 3's thinking)
                # This will be stripped when sending back but useful for logging/debugging
                if message.get("reasoning_content"):
                    full_assistant_message["reasoning_content"] = message["reasoning_content"]

                # Debug: Log if extra_content is present (for Gemini 3)
                if "gemini-3" in self.model_name.lower():
                    has_extra = any(tc.get("extra_content") for tc in full_assistant_message["tool_calls"])
                    if not has_extra:
                        print(f"WARNING: No extra_content found in Gemini 3 response tool_calls")

                return {
                    "type": "tool",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "full_assistant_message": full_assistant_message
                }

            # Normal content response
            else:
                content = message.get("content", "")
                return {"type": "content", "content": content}

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing response: {e}")
            return None
