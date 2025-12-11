"""
BFCL Solver - Function Calling 평가용 Solver

두 가지 모드 지원:
1. Native Tool Calling (기본) - tool calling 지원 모델용
2. Text-based (프롬프트 기반) - tool calling 미지원 모델용

사용 예:
    # Native (기본)
    solver="bfcl_solver"
    
    # Text-based (EXAONE 등)
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
# Tool 파싱 헬퍼 함수들
# =============================================================================

def _get_type(bfcl_type: str | None) -> JSONType | None:
    """BFCL 타입을 JSON 타입으로 변환"""
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
    """ToolParam 인스턴스 재귀적으로 생성"""
    if param_dict is None:
        return None
    
    # nested properties 처리
    properties = None
    if param_dict.get("properties"):
        properties = {
            key: _create_tool_param(value)
            for key, value in param_dict["properties"].items()
            if value is not None
        }
    
    # array items 처리
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
    """함수 이름을 OpenAI API 호환 형식으로 정규화"""
    sanitized = name.replace(".", "_")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    return sanitized


def _create_tool_info_from_dict(tool_dict: dict[str, Any]) -> ToolInfo:
    """딕셔너리에서 ToolInfo 인스턴스 생성"""
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
# Text-based Function Calling 프롬프트 생성
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
    """Tools 스펙을 텍스트 형식으로 변환"""
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
# Native Tool Calling Solver (기본)
# =============================================================================

@solver
def bfcl_solver() -> Solver:
    """
    BFCL Native Tool Calling Solver
    
    Tool calling을 지원하는 모델용 (OpenAI, Claude, Gemini 등)
    metadata의 tools를 state.tools에 추가하고 generate합니다.
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
        return await generate(state, tool_calls="none")
    
    return solve


# =============================================================================
# Text-based Function Calling Solver (프롬프트 기반)
# =============================================================================

@solver
def bfcl_text_solver() -> Solver:
    """
    BFCL Text-based Function Calling Solver
    
    Tool calling을 지원하지 않는 모델용 (EXAONE, 일부 오픈소스 등)
    프롬프트에 함수 정의를 포함하고 JSON 형식으로 출력하도록 유도합니다.
    
    출력 형식:
    {"function": "function_name", "arguments": {"arg1": "value1"}}
    
    또는 거부 시:
    {"function": null, "arguments": null}
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools_spec = state.metadata.get("tools", [])
        
        # 함수 정의를 텍스트로 변환
        functions_json = _tools_to_text(tools_spec)
        
        # 시스템 프롬프트 설정
        system_prompt = TEXT_BASED_SYSTEM_PROMPT.format(functions_json=functions_json)
        
        # 기존 메시지 앞에 시스템 프롬프트 추가
        from inspect_ai.model import ChatMessageSystem
        state.messages.insert(0, ChatMessageSystem(content=system_prompt))
        
        # metadata에 text_based 플래그 추가 (scorer에서 파싱 방식 결정)
        state.metadata["_text_based_function_call"] = True
        
        # 일반 generate (tool_calls 없이)
        return await generate(state)
    
    return solve
