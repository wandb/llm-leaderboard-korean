"""
BFCL Scorer - Function Calling 평가 Scorer

두 가지 모드 지원:
1. Native Tool Calling - tool_calls에서 직접 추출
2. Text-based - 텍스트 응답에서 JSON 파싱

평가 방식:
- 모델의 tool_calls/텍스트와 ground_truth 비교
- 함수명과 인자가 정확히 일치하면 정답
- irrelevance: tool_call이 없거나 {"function": null} 이면 정답

메트릭:
- accuracy: 전체 정확도
- category별 정확도
"""

import ast
import json
import re
from typing import Any

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import (
    Score,
    SampleScore,
    Scorer,
    Target,
    scorer,
    metric,
    Metric,
    accuracy,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


# =============================================================================
# Text-based 응답 파싱 (프롬프트 기반 모드용)
# =============================================================================

def _parse_text_response(text: str) -> dict[str, Any] | None:
    """
    텍스트 응답에서 JSON 함수 호출을 추출
    
    Expected format:
    {"function": "func_name", "arguments": {"arg1": "val1"}}
    
    또는 거부 시:
    {"function": null, "arguments": null}
    """
    if not text:
        return None
    
    try:
        # JSON 블록 추출 시도 (```json ... ``` 형식)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 그냥 JSON 찾기
            json_match = re.search(r'\{[^{}]*"function"[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 전체가 JSON인 경우
                json_str = text.strip()
        
        parsed = json.loads(json_str)
        
        # 유효성 검사
        if isinstance(parsed, dict) and "function" in parsed:
            func_name = parsed.get("function")
            arguments = parsed.get("arguments", {})
            
            # null 처리 (거부)
            if func_name is None:
                return {"function": None, "arguments": None}
            
            return {
                "function": str(func_name),
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
    except (json.JSONDecodeError, Exception):
        pass
    
    return None


def _parse_function_call(call_str: str) -> dict[str, Any] | None:
    """
    함수 호출 문자열을 파싱하여 {function, arguments} 딕셔너리로 변환
    
    Input: "calculate_area(width=10, height=5)"
    Output: {"function": "calculate_area", "arguments": {"width": 10, "height": 5}}
    """
    if not call_str or not isinstance(call_str, str):
        return None
    
    try:
        # AST로 파싱 시도
        tree = ast.parse(call_str, mode='eval')
        if not isinstance(tree.body, ast.Call):
            return None
        
        call = tree.body
        
        # 함수명 추출
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # triangle_properties.get 같은 경우
            parts = []
            node = call.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            func_name = ".".join(reversed(parts))
        else:
            return None
        
        # 인자 추출
        arguments = {}
        for kw in call.keywords:
            try:
                # ast.literal_eval로 값 평가
                arguments[kw.arg] = ast.literal_eval(kw.value)
            except:
                # 평가 실패시 문자열로
                arguments[kw.arg] = ast.unparse(kw.value)
        
        return {"function": func_name, "arguments": arguments}
    
    except Exception:
        return None


def _parse_dict_ground_truth(gt_dict: dict) -> dict[str, Any] | None:
    """
    딕셔너리 형식 ground_truth를 파싱
    
    Input: {'calculate_area': {'width': [10], 'height': [5]}}
    Output: {"function": "calculate_area", "arguments": {"width": 10, "height": 5}}
    """
    if not gt_dict or not isinstance(gt_dict, dict):
        return None
    
    try:
        func_name = list(gt_dict.keys())[0]
        params = gt_dict[func_name]
        
        arguments = {}
        for k, v in params.items():
            # 리스트면 첫 번째 값 사용
            if isinstance(v, list) and v:
                arguments[k] = v[0]
            else:
                arguments[k] = v
        
        return {"function": func_name, "arguments": arguments}
    
    except Exception:
        return None


def _normalize_arguments(args: dict) -> dict:
    """인자 정규화 (타입 통일, 빈 문자열 제거 등)"""
    normalized = {}
    for k, v in args.items():
        # 빈 문자열은 제외
        if v == "" or v == '':
            continue
        # 숫자 문자열은 숫자로 변환
        if isinstance(v, str):
            try:
                if '.' in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
        normalized[k] = v
    return normalized


def _sanitize_function_name(name: str) -> str:
    """함수 이름을 OpenAI API 호환 형식으로 정규화"""
    # .을 _로 대체
    sanitized = name.replace(".", "_")
    # 허용되지 않는 문자 제거
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    return sanitized


def _compare_tool_calls(
    predicted: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    예측 tool_call과 정답 비교
    
    Returns:
        (is_correct, explanation)
    """
    if predicted is None and expected is None:
        return True, "Both None"
    
    if predicted is None:
        return False, "No tool call predicted"
    
    if expected is None:
        return False, "No expected answer"
    
    # 함수명 비교 (정규화 후)
    pred_func = _sanitize_function_name(predicted.get("function", ""))
    exp_func = _sanitize_function_name(expected.get("function", ""))
    
    if pred_func != exp_func:
        return False, f"Function mismatch: {pred_func} != {exp_func}"
    
    # 인자 비교 (정규화 후)
    pred_args = _normalize_arguments(predicted.get("arguments", {}))
    exp_args = _normalize_arguments(expected.get("arguments", {}))
    
    if pred_args != exp_args:
        return False, f"Arguments mismatch: {pred_args} != {exp_args}"
    
    return True, "Exact match"


# =============================================================================
# Metrics
# =============================================================================

def _is_correct(score_value) -> bool:
    """Score value가 정답인지 확인 (문자열 또는 숫자 모두 처리)"""
    if isinstance(score_value, str):
        return score_value == CORRECT
    elif isinstance(score_value, (int, float)):
        return score_value >= 1.0
    return False


# 리더보드 기준 카테고리별 metric
@metric
def bfcl_simple_accuracy() -> Metric:
    """simple (= simple_python) 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "simple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_multiple_accuracy() -> Metric:
    """multiple 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "multiple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_irrelevance_accuracy() -> Metric:
    """irrelevance 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "irrelevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_java_accuracy() -> Metric:
    """java (= simple_java) 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "java"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_javascript_accuracy() -> Metric:
    """javascript (= simple_javascript) 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "javascript"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_simple_accuracy() -> Metric:
    """live_simple 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_simple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_multiple_accuracy() -> Metric:
    """live_multiple 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_multiple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_relevance_accuracy() -> Metric:
    """live_relevance 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_relevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_irrelevance_accuracy() -> Metric:
    """live_irrelevance 정확도"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_irrelevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn




# =============================================================================
# Scorer
# =============================================================================

@scorer(metrics=[
    accuracy(),
    # 기본 함수 호출
    bfcl_simple_accuracy(),
    bfcl_multiple_accuracy(),
    bfcl_irrelevance_accuracy(),
    # 언어별
    bfcl_java_accuracy(),
    bfcl_javascript_accuracy(),
    # 라이브 API
    bfcl_live_simple_accuracy(),
    bfcl_live_multiple_accuracy(),
    bfcl_live_relevance_accuracy(),
    bfcl_live_irrelevance_accuracy(),
])
def bfcl_scorer() -> Scorer:
    """
    BFCL Function Calling Scorer
    
    평가 방식:
    - 모델의 tool_calls와 ground_truth 비교
    - irrelevance: tool_call이 없으면 정답
    
    metadata에서 필요한 정보:
    - category: split 이름 (simple, multiple, exec_simple, exec_multiple, irrelevance)
    - ground_truth: 정답 (문자열 또는 딕셔너리)
    """
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        category = metadata.get("category", "unknown")
        is_text_based = metadata.get("_text_based_function_call", False)
        
        # 거부 카테고리들
        refusal_categories = {"irrelevance", "live_irrelevance", "live_relevance"}
        
        # =====================================================================
        # 1. 예측 추출 (Native vs Text-based)
        # =====================================================================
        predicted = None
        is_refusal = False
        
        if is_text_based:
            # Text-based 모드: 텍스트 응답에서 JSON 파싱
            assistant_messages = [
                m for m in state.messages if isinstance(m, ChatMessageAssistant)
            ]
            if assistant_messages:
                last_response = assistant_messages[-1].content
                if isinstance(last_response, str):
                    predicted = _parse_text_response(last_response)
                    
                    # {"function": null} = 거부
                    if predicted and predicted.get("function") is None:
                        is_refusal = True
                        predicted = None
        else:
            # Native 모드: tool_calls에서 직접 추출
            assistant_messages = [
                m for m in state.messages if isinstance(m, ChatMessageAssistant)
            ]
            
            tool_calls = []
            for msg in assistant_messages:
                if msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
            
            if tool_calls:
                tc = tool_calls[0]
                predicted = {
                    "function": tc.function,
                    "arguments": tc.arguments if isinstance(tc.arguments, dict) else {},
                }
            else:
                is_refusal = True
        
        # =====================================================================
        # 2. 거부 카테고리 처리
        # =====================================================================
        if category in refusal_categories:
            is_correct = is_refusal or predicted is None
            return Score(
                value=CORRECT if is_correct else INCORRECT,
                answer="Refused" if is_refusal else str(predicted),
                explanation="Correctly refused" if is_correct else "Should not call function",
                metadata={"category": category, "is_text_based": is_text_based},
            )
        
        # =====================================================================
        # 3. 일반 케이스: ground_truth와 비교
        # =====================================================================
        gt_raw = metadata.get("ground_truth", [])
        if isinstance(gt_raw, str):
            gt_raw = [gt_raw]
        
        expected = None
        if gt_raw:
            first_gt = gt_raw[0]
            if isinstance(first_gt, str):
                expected = _parse_function_call(first_gt)
            elif isinstance(first_gt, dict):
                expected = _parse_dict_ground_truth(first_gt)
        
        # 비교
        is_correct, explanation = _compare_tool_calls(predicted, expected)
        
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=str(predicted) if predicted else "No function call",
            explanation=explanation,
            metadata={
                "category": category,
                "predicted": predicted,
                "expected": expected,
                "is_text_based": is_text_based,
            },
        )
    
    return score

