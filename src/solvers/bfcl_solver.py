"""
BFCL Solver - Solver for Function Calling Evaluation

Two modes supported:
1. Native Tool Calling (default) - for models that support tool calling
2. Text-based (prompt-based) - for models that don't support tool calling

Usage:
    # Native (default)
    solver="bfcl_solver"
    
    # Text-based (EXAONE, etc.)
    solver="bfcl_text_solver"
"""

import json
import re
from typing import Any, cast, get_args

from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    solver,
)
from inspect_ai.tool import ToolInfo, ToolParam, ToolParams
from inspect_ai.util import JSONType


# =============================================================================
# Tool Parsing Helper Functions
# =============================================================================

def _get_type(bfcl_type: str | None) -> JSONType | None:
    """Convert BFCL type to JSON type"""
    if bfcl_type is None:
        return None
    
    if bfcl_type == "dict":
        return "object"
    
    if bfcl_type == "float":
        bfcl_type = "number"
    
    if bfcl_type == "tuple":
        bfcl_type = "array"
    
    if bfcl_type in get_args(JSONType):
        return cast(JSONType, bfcl_type)
    return None


def _create_tool_param(param_dict: dict[str, Any] | None) -> ToolParam | None:
    """Create ToolParam instance recursively"""
    if param_dict is None:
        return None
    
    # Handle nested properties
    properties = None
    if param_dict.get("properties"):
        properties = {
            key: _create_tool_param(value)
            for key, value in param_dict["properties"].items()
            if value is not None
        }
    
    # Handle array items
    items = None
    if param_dict.get("items"):
        items = _create_tool_param(param_dict["items"])
    
    return ToolParam(
        type=_get_type(param_dict.get("type")),
        description=param_dict.get("description"),
        default=param_dict.get("default"),
        enum=param_dict.get("enum"),
        items=items,
        properties=properties,
        additionalProperties=param_dict.get("additionalProperties"),
        required=param_dict.get("required"),
    )


def _sanitize_function_name(name: str) -> str:
    """Normalize function name to OpenAI API compatible format"""
    sanitized = name.replace(".", "_")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    return sanitized


def _create_tool_info_from_dict(tool_dict: dict[str, Any]) -> ToolInfo:
    """Create ToolInfo instance from dictionary"""
    parameters = None
    if "parameters" in tool_dict:
        parameters = _create_tool_param(tool_dict["parameters"])
    
    if parameters is None or parameters.properties is None:
        tool_params = ToolParams(properties={}, required=[])
    else:
        tool_params = ToolParams(
            properties=parameters.properties,
            required=parameters.required or [],
        )
    
    raw_name = tool_dict.get("name", "unknown")
    sanitized_name = _sanitize_function_name(raw_name)
    
    return ToolInfo(
        name=sanitized_name,
        description=tool_dict.get("description", ""),
        parameters=tool_params,
    )


# =============================================================================
# Text-based Function Calling Prompt Generation
# =============================================================================

TEXT_BASED_SYSTEM_PROMPT = """You are a helpful assistant with access to the following functions. 
When you need to call a function, respond with ONLY a JSON object in this exact format:
{{"function": "function_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}

If no function call is needed or the question is irrelevant to the available functions, respond with:
{{"function": null, "arguments": null}}

IMPORTANT: 
- Respond with ONLY the JSON object, no other text.
- Use the exact function name as provided.
- Include all required arguments.

Available functions:
{functions_json}
"""


def _tools_to_text(tools_spec: list[dict]) -> str:
    """Convert Tools spec to text format"""
    functions = []
    for tool in tools_spec:
        func_info = {
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        functions.append(func_info)
    return json.dumps(functions, indent=2, ensure_ascii=False)


# =============================================================================
# Native Tool Calling Solver (default)
# =============================================================================

@solver
def bfcl_solver() -> Solver:
    """
    BFCL Native Tool Calling Solver
    
    For models that support tool calling (OpenAI, Claude, Gemini, etc.)
    Adds tools from metadata to state.tools and generates.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools_spec = state.metadata.get("tools", [])
        
        tool_infos: list[ToolInfo] = []
        for spec in tools_spec:
            try:
                tool_info = _create_tool_info_from_dict(spec)
                tool_infos.append(tool_info)
            except Exception as e:
                import logging
                logging.warning(f"Failed to create ToolInfo: {e}")
                continue
        
        state.tools.extend(tool_infos)
        # tool_calls="none": 모델이 tool_call 생성만 하고, 실행하지 않음
        # scorer에서 assistant 메시지의 tool_calls를 직접 파싱
        return await generate(state, tool_calls="none")
    
    return solve


# =============================================================================
# Text-based Function Calling Solver (prompt-based)
# =============================================================================

@solver
def bfcl_text_solver() -> Solver:
    """
    BFCL Text-based Function Calling Solver
    
    For models that don't support tool calling (EXAONE, some open-source models, etc.)
    Includes function definitions in the prompt and prompts for JSON output.
    
    Output format:
    {"function": "function_name", "arguments": {"arg1": "value1"}}
    
    Or for refusal:
    {"function": null, "arguments": null}
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools_spec = state.metadata.get("tools", [])
        
        # Convert function definitions to text
        functions_json = _tools_to_text(tools_spec)
        
        # Set system prompt
        system_prompt = TEXT_BASED_SYSTEM_PROMPT.format(functions_json=functions_json)
        
        # Add system prompt before existing messages
        from inspect_ai.model import ChatMessageSystem
        state.messages.insert(0, ChatMessageSystem(content=system_prompt))
        
        # Add text_based flag to metadata (for scorer to determine parsing method)
        state.metadata["_text_based_function_call"] = True
        
        # Normal generate (without tool_calls)
        return await generate(state)
    
    return solve
