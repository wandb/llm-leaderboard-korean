"""
BFCL Scorer - Function Calling Evaluation Scorer

Two modes supported:
1. Native Tool Calling - Extract directly from tool_calls
2. Text-based - Parse JSON from text response

Evaluation method:
- Compare model's tool_calls/text with ground_truth
- Correct if function name and arguments match exactly
- irrelevance: Correct if no tool_call or {"function": null}

Metrics:
- accuracy: Overall accuracy
- Category-specific accuracy
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
# Text-based response parsing (for prompt-based mode)
# =============================================================================

def _extract_text_from_content(content: Any) -> str:
    """
    Extract text from various content formats.
    
    Handles:
    - str: Return as-is
    - list: Extract text from content blocks (ContentText, dict with 'text', etc.)
    - Other: Convert to string
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif hasattr(item, 'text'):
                # ContentText or similar object with 'text' attribute
                text_parts.append(str(item.text))
            elif isinstance(item, dict):
                # Dict with 'text' or 'content' key
                if 'text' in item:
                    text_parts.append(str(item['text']))
                elif 'content' in item:
                    text_parts.append(str(item['content']))
            else:
                # Try to convert to string
                text_parts.append(str(item))
        return '\n'.join(text_parts)
    
    # Fallback: convert to string
    return str(content) if content else ''


def _extract_json_with_nested_braces(text: str) -> str | None:
    """
    Extract JSON object from text, handling nested braces properly.
    Finds the first '{' and matches to the corresponding closing '}'.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx:i+1]
    
    return None


def _remove_thinking_content(text: str) -> str:
    """
    Remove thinking/reasoning content from model response.
    Handles various formats: <thinking>...</thinking>, <think>...</think>, etc.
    """
    # Remove <thinking>...</thinking> tags (case insensitive)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove <reasoning>...</reasoning> tags
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown-style thinking blocks (```thinking ... ```)
    text = re.sub(r'```thinking.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove lines starting with "Thinking:" or similar
    text = re.sub(r'^(Thinking|Reasoning|Let me think).*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text.strip()


def _parse_text_response(text: str) -> dict[str, Any] | None:
    """
    Extract JSON function call from text response
    
    Expected format:
    {"function": "func_name", "arguments": {"arg1": "val1"}}
    
    Or for refusal:
    {"function": null, "arguments": null}
    
    Handles:
    - Thinking/reasoning content (removed before parsing)
    - Nested JSON objects in arguments
    - Code blocks (```json ... ```)
    """
    if not text:
        return None
    
    # Step 1: Remove thinking/reasoning content
    cleaned_text = _remove_thinking_content(text)
    
    try:
        # Step 2: Try to extract JSON block (```json ... ``` format)
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Step 3: Extract JSON with proper nested brace handling
            json_str = _extract_json_with_nested_braces(cleaned_text)
            
            if json_str is None:
                # Fallback: try entire text as JSON
                json_str = cleaned_text.strip()
        
        parsed = json.loads(json_str)
        
        # Validate
        if isinstance(parsed, dict) and "function" in parsed:
            func_name = parsed.get("function")
            arguments = parsed.get("arguments", {})
            
            # Handle null (refusal)
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
    Parse function call string to {function, arguments} dictionary
    
    Input: "calculate_area(width=10, height=5)"
    Output: {"function": "calculate_area", "arguments": {"width": 10, "height": 5}}
    """
    if not call_str or not isinstance(call_str, str):
        return None
    
    try:
        # Try AST parsing
        tree = ast.parse(call_str, mode='eval')
        if not isinstance(tree.body, ast.Call):
            return None
        
        call = tree.body
        
        # Extract function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Cases like triangle_properties.get
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
        
        # Extract arguments
        arguments = {}
        for kw in call.keywords:
            try:
                # Evaluate value with ast.literal_eval
                arguments[kw.arg] = ast.literal_eval(kw.value)
            except:
                # On evaluation failure, use as string
                arguments[kw.arg] = ast.unparse(kw.value)
        
        return {"function": func_name, "arguments": arguments}
    
    except Exception:
        return None


def _parse_dict_ground_truth(gt_dict: dict) -> dict[str, Any] | None:
    """
    Parse dictionary format ground_truth
    
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
            # Use first value if list
            if isinstance(v, list) and v:
                arguments[k] = v[0]
            else:
                arguments[k] = v
        
        return {"function": func_name, "arguments": arguments}
    
    except Exception:
        return None


def _normalize_arguments(args: dict) -> dict:
    """Normalize arguments (unify types, remove empty strings, etc.)"""
    normalized = {}
    for k, v in args.items():
        # Exclude empty strings
        if v == "" or v == '':
            continue
        # Convert numeric strings to numbers
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
    """Normalize function name to OpenAI API compatible format"""
    # Replace . with _
    sanitized = name.replace(".", "_")
    # Remove disallowed characters
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", sanitized)
    return sanitized


def _compare_tool_calls(
    predicted: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Compare predicted tool_call with expected
    
    Returns:
        (is_correct, explanation)
    """
    if predicted is None and expected is None:
        return True, "Both None"
    
    if predicted is None:
        return False, "No tool call predicted"
    
    if expected is None:
        return False, "No expected answer"
    
    # Compare function names (after normalization)
    pred_func = _sanitize_function_name(predicted.get("function", ""))
    exp_func = _sanitize_function_name(expected.get("function", ""))
    
    if pred_func != exp_func:
        return False, f"Function mismatch: {pred_func} != {exp_func}"
    
    # Compare arguments (after normalization)
    pred_args = _normalize_arguments(predicted.get("arguments", {}))
    exp_args = _normalize_arguments(expected.get("arguments", {}))
    
    if pred_args != exp_args:
        return False, f"Arguments mismatch: {pred_args} != {exp_args}"
    
    return True, "Exact match"


# =============================================================================
# Metrics
# =============================================================================

def _is_correct(score_value) -> bool:
    """Check if Score value is correct (handles both string and number)"""
    if isinstance(score_value, str):
        return score_value == CORRECT
    elif isinstance(score_value, (int, float)):
        return score_value >= 1.0
    return False


# Category-specific metrics (leaderboard standard)
@metric
def bfcl_simple_accuracy() -> Metric:
    """simple (= simple_python) accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "simple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_multiple_accuracy() -> Metric:
    """multiple accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "multiple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_irrelevance_accuracy() -> Metric:
    """irrelevance accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "irrelevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_java_accuracy() -> Metric:
    """java (= simple_java) accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "java"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_javascript_accuracy() -> Metric:
    """javascript (= simple_javascript) accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "javascript"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_simple_accuracy() -> Metric:
    """live_simple accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_simple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_multiple_accuracy() -> Metric:
    """live_multiple accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_multiple"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_relevance_accuracy() -> Metric:
    """live_relevance accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_relevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn


@metric
def bfcl_live_irrelevance_accuracy() -> Metric:
    """live_irrelevance accuracy"""
    def metric_fn(scores: list[SampleScore]) -> float:
        cat_scores = [s for s in scores if s.score.metadata and s.score.metadata.get("category") == "live_irrelevance"]
        return sum(1 for s in cat_scores if _is_correct(s.score.value)) / len(cat_scores) if cat_scores else 0.0
    return metric_fn




# =============================================================================
# Scorer
# =============================================================================

@scorer(metrics=[
    accuracy(),
    # Basic function calling
    bfcl_simple_accuracy(),
    bfcl_multiple_accuracy(),
    bfcl_irrelevance_accuracy(),
    # Language-specific
    bfcl_java_accuracy(),
    bfcl_javascript_accuracy(),
    # Live API
    bfcl_live_simple_accuracy(),
    bfcl_live_multiple_accuracy(),
    bfcl_live_relevance_accuracy(),
    bfcl_live_irrelevance_accuracy(),
])
def bfcl_scorer() -> Scorer:
    """
    BFCL Function Calling Scorer
    
    Evaluation method:
    - Compare model's tool_calls with ground_truth
    - irrelevance: Correct if no tool_call
    
    Required info from metadata:
    - category: split name (simple, multiple, exec_simple, exec_multiple, irrelevance)
    - ground_truth: Expected answer (string or dictionary)
    """
    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        category = metadata.get("category", "unknown")
        is_text_based = metadata.get("_text_based_function_call", False)
        
        # Refusal categories
        refusal_categories = {"irrelevance", "live_irrelevance", "live_relevance"}
        
        # =====================================================================
        # 1. Extract prediction (Native vs Text-based)
        # =====================================================================
        predicted = None
        is_refusal = False
        
        if is_text_based:
            # Text-based mode: Parse JSON from text response
            assistant_messages = [
                m for m in state.messages if isinstance(m, ChatMessageAssistant)
            ]
            if assistant_messages:
                last_response = assistant_messages[-1].content
                # Extract text from various content formats (str, list, etc.)
                response_text = _extract_text_from_content(last_response)
                
                if response_text:
                    predicted = _parse_text_response(response_text)
                    
                    # {"function": null} = refusal
                    if predicted and predicted.get("function") is None:
                        is_refusal = True
                        predicted = None
        else:
            # Native mode: Extract directly from tool_calls
            assistant_messages = [
                m for m in state.messages if isinstance(m, ChatMessageAssistant)
            ]
            
            tool_calls = []
            for msg in assistant_messages:
                if msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
            
            if tool_calls:
                tc = tool_calls[0]
                
                # Handle different tool_call formats:
                # 1. inspect-ai native: tc.function (str), tc.arguments (dict)
                # 2. vLLM/litellm raw: tc.function (dict with 'name', 'arguments')
                if hasattr(tc, 'function'):
                    func = tc.function
                    if isinstance(func, dict):
                        # vLLM/litellm format: {'name': '...', 'arguments': '...'}
                        func_name = func.get('name', '')
                        args_raw = func.get('arguments', '{}')
                        if isinstance(args_raw, str):
                            try:
                                args = json.loads(args_raw)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = args_raw if isinstance(args_raw, dict) else {}
                    else:
                        # inspect-ai native format
                        func_name = func
                        args = tc.arguments if isinstance(tc.arguments, dict) else {}
                else:
                    # Fallback: dict-like access
                    tc_dict = tc if isinstance(tc, dict) else vars(tc) if hasattr(tc, '__dict__') else {}
                    func = tc_dict.get('function', {})
                    if isinstance(func, dict):
                        func_name = func.get('name', '')
                        args_raw = func.get('arguments', '{}')
                        if isinstance(args_raw, str):
                            try:
                                args = json.loads(args_raw)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = args_raw if isinstance(args_raw, dict) else {}
                    else:
                        func_name = str(func) if func else ''
                        args = tc_dict.get('arguments', {})
                
                predicted = {
                    "function": func_name,
                    "arguments": args,
                }
            else:
                is_refusal = True
        
        # =====================================================================
        # 2. Handle refusal categories
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
        # 3. General case: Compare with ground_truth
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
        
        # Compare
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
