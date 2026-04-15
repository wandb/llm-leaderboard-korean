"""
Weave trace retrieval for Horangi benchmark analysis.

Given a W&B run ID + benchmark name, finds the corresponding Weave
evaluation trace and returns sample-level data for error analysis.

Usage:
    from analyses.retrieve_eval import find_eval_call, get_eval_samples

    result = find_eval_call("horangi/horangi4", "jn7urcxl", "ko_moral")
    # → {'call_id': '019d8707-...', 'model_name': 'nemotron_3_nano...', ...}

    samples = get_eval_samples("horangi/horangi4", result["call_id"])
    # → [{'input': '...', 'completion': '...', 'is_correct': False, ...}, ...]
"""

from __future__ import annotations

import os
import re
from typing import Any

os.environ.setdefault("WANDB_SILENT", "true")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_env() -> None:
    """Load .env if WANDB_API_KEY is not already set."""
    if not os.environ.get("WANDB_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
        except ImportError:
            pass


def find_eval_call(
    entity_project: str,
    run_id: str,
    benchmark: str,
    *,
    limit: int = 1000,
) -> dict[str, Any] | None:
    """Find the Weave evaluation call for a (run, benchmark) pair.

    Fetches root-level Evaluation.evaluate calls from Weave (newest first),
    then matches by display_name == benchmark AND model name from the W&B run.

    Args:
        entity_project: "entity/project" (e.g. "horangi/horangi4").
        run_id: W&B run ID (e.g. "jn7urcxl").
        benchmark: Benchmark name as displayed in Weave
                   (e.g. "ko_moral", "korean_hate_speech").
        limit: Max root calls to scan.

    Returns:
        Dict with call_id, display_name, model_name, started_at, weave_url,
        or None if not found.
    """
    import wandb
    import weave

    _ensure_env()
    api = wandb.Api()
    run = api.run(f"{entity_project}/{run_id}")
    run_name_norm = run.name.lower().replace("-", "_")

    client = weave.init(entity_project)
    calls = client.get_calls(
        filter={"trace_roots_only": True},
        limit=limit,
        sort_by=[{"field": "started_at", "direction": "desc"}],
    )

    for c in calls:
        if (getattr(c, "display_name", "") or "") != benchmark:
            continue

        inp = c.inputs
        if not isinstance(inp, dict):
            continue

        model_obj = inp.get("model")
        model_name = ""
        if model_obj and hasattr(model_obj, "name"):
            model_name = model_obj.name
        elif isinstance(model_obj, dict):
            model_name = model_obj.get("name", "")

        model_name_norm = model_name.lower().replace("-", "_")
        if run_name_norm not in model_name_norm and model_name_norm not in run_name_norm:
            continue

        return {
            "call_id": c.id,
            "display_name": benchmark,
            "model_name": model_name,
            "started_at": str(getattr(c, "started_at", "")),
            "weave_url": (
                f"https://wandb.ai/{entity_project}/weave/traces"
                f"?filter=%7B%22parentId%22%3A%22{c.id}%22%7D"
            ),
        }

    return None


def get_eval_samples(
    entity_project: str,
    call_id: str,
) -> list[dict[str, Any]]:
    """Fetch all predict_and_score child calls for an evaluation.

    Args:
        entity_project: "entity/project" string.
        call_id: Parent evaluation call ID from find_eval_call().

    Returns:
        List of sample dicts, each containing:
          sample_id, input, completion, parsed_answer,
          is_correct, scores, display_name.
    """
    import weave

    _ensure_env()
    client = weave.init(entity_project)
    children = client.get_calls(
        filter={"parent_ids": [call_id]},
        limit=1000,
    )

    samples = []
    for c in children:
        op = getattr(c, "op_name", "") or ""
        if "predict_and_score" not in op:
            continue

        out = c.output
        inp = c.inputs
        display = getattr(c, "display_name", "") or ""

        if not isinstance(out, dict):
            continue

        scores = dict(out.get("scores", {}))

        model_out = out.get("output", out.get("model_output", ""))
        completion = ""
        if isinstance(model_out, str):
            completion = model_out
        elif isinstance(model_out, dict):
            completion = model_out.get("output", str(model_out))
        else:
            completion = str(model_out)

        is_correct = False
        choice_score = scores.get("choice")
        if choice_score is True:
            is_correct = True
        elif isinstance(choice_score, dict) and choice_score.get("value") in ("C", 1, True):
            is_correct = True

        input_text = ""
        if isinstance(inp, dict):
            inputs_dict = inp.get("inputs", {})
            if isinstance(inputs_dict, dict):
                input_text = str(inputs_dict.get("input", inputs_dict.get("question", "")))

        answer_match = re.search(
            r"(?:정답|답변|답|Answer|ANSWER)\s*[:：]\s*([A-Za-z])", completion
        )
        parsed_answer = answer_match.group(1).upper() if answer_match else ""

        samples.append({
            "sample_id": display,
            "input": input_text,
            "completion": completion,
            "parsed_answer": parsed_answer,
            "is_correct": is_correct,
            "scores": scores,
            "display_name": display,
        })

    return samples
