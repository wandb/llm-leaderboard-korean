import re
from typing import Any, Dict, List
from collections import Counter

from .base import BaseEvaluator
from . import register_evaluator
from llm_eval.utils.logging import get_logger
from tqdm import tqdm

logger = get_logger(name="grid_match", level="INFO")

# 전역 컴파일 패턴
_CODEFENCE_RE = re.compile(r"```([^`]*)```", flags=re.DOTALL)
_NON_DIGIT_RE = re.compile(r"[^0-9,\n ]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_WS_SPLIT_RE = re.compile(r"\s+")

def _clean_text(text: str) -> str:
    """
    숫자/콤마/개행/스페이스만 남김.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CODEFENCE_RE.sub(r"\1", text)
    text = _NON_DIGIT_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n", text)
    return text.strip()

def _parse_grid(text: str) -> List[List[int]]:
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("empty_text")

    rows_raw = [r.strip() for r in cleaned.splitlines() if r.strip()]
    grid: List[List[int]] = []
    for r in rows_raw:
        cells = [c.strip() for c in (r.split(",") if "," in r else _WS_SPLIT_RE.split(r)) if c]
        if not cells:
            raise ValueError("empty_row")
        if any(not c.isdigit() for c in cells):
            raise ValueError("non_digit_cell")
        grid.append([int(c) for c in cells])

    w = len(grid[0])
    if any(len(row) != w for row in grid):
        raise ValueError("ragged_rows")
    return grid

def _exact_grid_match(pred_text: str, ref_text: str) -> Dict[str, Any]:
    try:
        pred = _parse_grid(pred_text)
        ref = _parse_grid(ref_text)
    except Exception as e:
        return {"score": 0.0, "correct": False, "reason": f"parse_error:{e}"}

    if len(pred) != len(ref):
        return {"score": 0.0, "correct": False, "reason": "shape_mismatch_rows"}
    for r1, r2 in zip(pred, ref):
        if len(r1) != len(r2):
            return {"score": 0.0, "correct": False, "reason": "shape_mismatch_cols"}
        if r1 != r2:  # int 캐스팅 불필요
            return {"score": 0.0, "correct": False, "reason": "value_mismatch"}

    return {"score": 1.0, "correct": True, "reason": "exact_match"}

@register_evaluator("grid_match")
class GridMatchEvaluator(BaseEvaluator):
    """
    grid match 평가기 (격자 완전 일치)
    """

    name = "grid_match"
    requires_logits = False
    requires_chain_of_thought = False

    def prepare_prompt(self, input_text: str) -> str:
        return input_text

    def parse_prediction(self, raw_output: str) -> str:
        return _clean_text(raw_output)

    def evaluate_predictions(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        n_scored = 0
        n_correct = 0
        reason_counter = Counter()

        for s in tqdm(samples, desc="grid-match scoring", leave=False):
            ref = s.get("reference")
            # 안전망(선택): parse 누락 대비
            pred = self.parse_prediction(s.get("prediction", ""))

            if ref is None:
                s["score"] = None
                s["correct"] = None
                s["reason"] = "no_reference"
                continue

            res = _exact_grid_match(pred, ref)
            s["score"] = res["score"]
            s["correct"] = res["correct"]
            s["reason"] = res["reason"]
            reason_counter.update([res["reason"]])

            n_scored += 1
            if res["correct"]:
                n_correct += 1

        accuracy = (n_correct / n_scored) if n_scored > 0 else 0.0
        logger.info(
            f"accuracy={accuracy:.4f} (n_correct={n_correct}, n_scored={n_scored}) "
            f"reasons={dict(reason_counter)}"
        )

        return {
            "accuracy": accuracy,
            "n_scored": float(n_scored),
            "n_correct": float(n_correct),
            "reasons": dict(reason_counter),
        }
