"""
Unified OSS Handler - Simplified Version

Basic Strategy:
- Leverage base_oss_handler's default behavior as much as possible
- Handle only obviously necessary special cases with minimal processing
- Eliminate complex guessing and learning features
- Prioritize safety (no eval usage)

Supported Patterns:
1. Standard JSON format (base_oss_handler default)
2. XML tool tag format (<tool_call>...</tool_call>)
3. JSON within Markdown code blocks
4. Basic special tags (<|python_tag|>, etc.)

Unsupported Cases:
- Fully custom system prompts (Llama 3.1, Hermes, etc.)
- Special tokenizers (DeepSeek-Coder, etc.)
- Custom roles (Gemma's "model" role, etc.)
- Complex reasoning formats (dedicated handlers recommended)

Usage Example:
In model_config.py, add a new model configuration like this:

    "organization/new-model": ModelConfig(
        model_name="organization/new-model",
        display_name="New Model (Prompt)",
        url="https://huggingface.co/organization/new-model",
        org="Organization",
        license="apache-2.0",
        model_handler=UnifiedOSSHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),

This handler is ideal for quickly testing new OSS models without creating
a dedicated handler, especially when the model follows common output conventions.
"""

import json
import re
from typing import Dict, List, Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class UnifiedOSSHandler(OSSHandler):
    """
    Unified OSS Handler (Simplified Version)

    Primarily uses base_oss_handler's default behavior,
    with additional support for common output formats only.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        print(f"[UnifiedOSSHandler] Initialization complete - Model: {model_name}")

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """
        Decode common output formats.
        No complex guessing - only handle obvious patterns.
        """
        if not result or not isinstance(result, str):
            return []

        result = result.strip()

        # Pattern 1: XML tool tags (Hermes-style)
        if "<tool_call>" in result and "</tool_call>" in result:
            return self._decode_xml_tool_tags(result)

        # Pattern 2: Basic special tags
        if "<|python_tag|>" in result:
            cleaned = result.replace("<|python_tag|>", "").strip()
            return super().decode_ast(cleaned, language, has_tool_call_tag)

        # Pattern 3: Markdown code blocks
        markdown_match = re.search(r"```(?:json|python)\s*\n(.*?)\n```", result, re.DOTALL)
        if markdown_match:
            content = markdown_match.group(1).strip()
            return self._try_json_decode(content)

        # Pattern 4: Code blocks (``` only, front and back)
        if result.startswith("```") and result.endswith("```"):
            content = result[3:-3].strip()
            # Remove json/python prefix
            if content.startswith("json\n"):
                content = content[5:]
            elif content.startswith("python\n"):
                content = content[7:]
            return self._try_json_decode(content)

        # Default: base_oss_handler behavior
        return super().decode_ast(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """
        Decode for execution.
        Basically the same logic as decode_ast.
        """
        if not result or not isinstance(result, str):
            return []

        # First try to convert with decode_ast
        decoded_ast = self.decode_ast(result, "Python", has_tool_call_tag)
        if decoded_ast:
            # Convert from AST to execution format
            execution_list = []
            for call in decoded_ast:
                for func_name, params in call.items():
                    param_str = ",".join([f"{k}={repr(v)}" for k, v in params.items()])
                    execution_list.append(f"{func_name}({param_str})")
            return execution_list

        # Default processing
        return super().decode_execute(result, has_tool_call_tag)

    def _decode_xml_tool_tags(self, result: str) -> List[Dict]:
        """Decode XML tool tags (Hermes-style)"""
        # Extract contents within <tool_call>...</tool_call>
        tool_call_pattern = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", result, re.DOTALL)

        func_calls = []
        for match in tool_call_pattern:
            try:
                # Try to parse as JSON
                tool_data = json.loads(match.strip())
                if isinstance(tool_data, dict) and "name" in tool_data:
                    func_name = tool_data["name"]
                    arguments = tool_data.get("arguments", {})
                    func_calls.append({func_name: arguments})
            except json.JSONDecodeError:
                print(f"[Decode] XML tool tag JSON parsing failed: {match[:100]}")
                continue

        return func_calls

    def _try_json_decode(self, content: str) -> List[Dict]:
        """Safely attempt JSON decoding"""
        try:
            parsed = json.loads(content)

            # Single object case
            if isinstance(parsed, dict):
                if "name" in parsed and "arguments" in parsed:
                    return [{parsed["name"]: parsed["arguments"]}]
                elif "name" in parsed and "parameters" in parsed:  # Llama-style
                    return [{parsed["name"]: parsed["parameters"]}]

            # Array case
            elif isinstance(parsed, list):
                func_calls = []
                for item in parsed:
                    if isinstance(item, dict):
                        if "name" in item and "arguments" in item:
                            func_calls.append({item["name"]: item["arguments"]})
                        elif "name" in item and "parameters" in item:  # Llama-style
                            func_calls.append({item["name"]: item["parameters"]})
                return func_calls

        except json.JSONDecodeError:
            print(f"[Decode] JSON parsing failed: {content[:100]}")

        return []

    @override
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        """
        Add execution results.
        Uses standard tool role (works with most models).
        """
        for execution_result in execution_results:
            inference_data["message"].append(
                {
                    "role": "tool",
                    "content": execution_result,
                }
            )

        return inference_data

    @override
    def _format_prompt(self, messages, function):
        """
        Use tokenizer's chat template for automatic formatting.
        This provides a generic solution for most models.
        """
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return formatted_prompt
