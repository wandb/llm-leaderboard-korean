"""
Grid Match Scorer for ARC-AGI style grid matching.

Compares the model's grid output with the expected grid.
The grids must match exactly (same dimensions and values).
"""

import re
from typing import Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState

# Pre-compiled regex patterns
_CODEFENCE_RE = re.compile(r"```([^`]*)```", flags=re.DOTALL)
_NON_DIGIT_RE = re.compile(r"[^0-9,\n ]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_WS_SPLIT_RE = re.compile(r"\s+")
_ROW_PATTERN = re.compile(r"^\s*\d+(,\d+)*\s*$")


def _clean_text(text: str) -> str:
    """Keep only digits, commas, newlines, and spaces."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CODEFENCE_RE.sub(r"\1", text)
    text = _NON_DIGIT_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n", text)
    return text.strip()


def _extract_grid_text(text: str) -> str:
    """
    Extract the last grid pattern from text.
    Returns the last grid found (most likely to be the final answer).
    """
    grids: list[list[str]] = []
    current_grid: list[str] = []
    
    for line in text.splitlines():
        line = line.strip()
        if _ROW_PATTERN.match(line):
            current_grid.append(line)
        else:
            if current_grid:
                grids.append(current_grid)
                current_grid = []
    
    if current_grid:
        grids.append(current_grid)
    
    if not grids:
        return ""
    
    return "\n".join(grids[-1])


def _parse_grid(text: str) -> list[list[int]]:
    """Parse cleaned text into a 2D grid of integers."""
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("empty_text")
    
    rows_raw = [r.strip() for r in cleaned.splitlines() if r.strip()]
    grid: list[list[int]] = []
    
    for r in rows_raw:
        if "," in r:
            cells = [c.strip() for c in r.split(",") if c.strip()]
        else:
            cells = [c.strip() for c in _WS_SPLIT_RE.split(r) if c.strip()]
        
        if not cells:
            raise ValueError("empty_row")
        if any(not c.isdigit() for c in cells):
            raise ValueError("non_digit_cell")
        
        grid.append([int(c) for c in cells])
    
    if not grid:
        raise ValueError("empty_grid")
    
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ragged_rows")
    
    return grid


def _exact_grid_match(pred_text: str, ref_text: str) -> dict[str, Any]:
    """Compare two grid texts for exact match."""
    try:
        pred = _parse_grid(pred_text)
        ref = _parse_grid(ref_text)
    except ValueError as e:
        return {"score": 0.0, "correct": False, "reason": f"parse_error:{e}"}
    
    if len(pred) != len(ref):
        return {
            "score": 0.0,
            "correct": False,
            "reason": f"shape_mismatch_rows (expected {len(ref)}, got {len(pred)})",
        }
    
    for i, (r1, r2) in enumerate(zip(pred, ref)):
        if len(r1) != len(r2):
            return {
                "score": 0.0,
                "correct": False,
                "reason": f"shape_mismatch_cols at row {i}",
            }
        if r1 != r2:
            return {
                "score": 0.0,
                "correct": False,
                "reason": f"value_mismatch at row {i}",
            }
    
    return {"score": 1.0, "correct": True, "reason": "exact_match"}


@scorer(metrics=[accuracy()])
def grid_match() -> Scorer:
    """
    Scorer for ARC-AGI style grid matching.
    
    Features:
    - Extracts the last grid from model response (handles chain-of-thought)
    - Supports comma-separated and space-separated formats
    - Removes code fences and non-digit characters
    - Validates grid shape (no ragged rows)
    """
    async def score(state: TaskState, target: Target) -> Score:
        model_output = state.output.completion
        ref_text = target.text
        
        pred_text = _extract_grid_text(_clean_text(model_output))
        
        if not pred_text:
            return Score(
                value=INCORRECT,
                answer=model_output,
                explanation="No grid pattern found in model response",
            )
        
        result = _exact_grid_match(pred_text, ref_text)
        
        return Score(
            value=CORRECT if result["correct"] else INCORRECT,
            answer=pred_text,
            explanation=result["reason"],
        )
    
    return score

