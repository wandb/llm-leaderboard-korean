"""
SWE-bench Server Scorer

Scores resolved status by calling scoring server API instead of Docker-based official harness.

References:
- Server implementation example: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/server/swebench_server.py
- Evaluator implementation example: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/evaluator/swe_bench.py

Environment variables:
- SWE_SERVER_URL: e.g., https://api.nejumi-swebench.org
- SWE_API_KEY: (optional) if server requires X-API-Key
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    """Remove code fences (```) and return inner content."""
    if not text:
        return ""
    # Remove ```diff ... ``` or ```patch ... ``` or ```...``` (extract inner content if possible)
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_patch(text: str) -> str:
    """
    Safely extract only the patch text from model output.
    
    Extracts only the diff portion even if description text is mixed in.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return ""

    # Case 1: diff --git header exists
    idx = cleaned.find("diff --git")
    if idx != -1:
        patch_text = cleaned[idx:]
        # Remove description text after diff (consider empty lines as separator)
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        patch_text = parts[0]
        # Find where patch ends (where diff/---/+++/@@ patterns end)
        lines = patch_text.split('\n')
        valid_lines = []
        for line in lines:
            # Patch line patterns
            if (line.startswith('diff --git') or 
                line.startswith('---') or 
                line.startswith('+++') or 
                line.startswith('@@') or
                line.startswith('+') or
                line.startswith('-') or
                line.startswith(' ') or
                line == ''):
                valid_lines.append(line)
            elif valid_lines:  # If outside pattern after patch started, stop
                # Could be context line, so simple check
                if not any(valid_lines[-1].startswith(p) for p in ['diff', '---', '+++', '@@', ' ', '+', '-']):
                    break
                valid_lines.append(line)
        return '\n'.join(valid_lines).strip() + '\n'

    # Case 2: unified diff starting with --- header
    m = re.search(r"(^---\s+[ab]?/?.+$)", cleaned, re.MULTILINE)
    if m:
        idx = m.start()
        patch_text = cleaned[idx:]
        # Remove description text after diff
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        return parts[0].strip() + '\n'

    # Case 3: Treat entire content as patch (last fallback)
    return cleaned.strip() + ('\n' if not cleaned.endswith('\n') else '')


def _http_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    """
    HTTP request helper (SSL verification bypass + retry logic)
    
    Reference: https://github.com/wandb/llm-leaderboard/blob/main/scripts/evaluator/swe_bench.py
    """
    data = None
    req_headers = dict(headers) if headers else {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    # Bypass SSL certificate verification (for development)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # No retry for POST (has side-effects)
    max_attempts = 1 if method.upper() == "POST" else 3
    attempt = 0
    backoff_sec = 2.0
    last_err: Optional[Exception] = None

    while attempt < max_attempts:
        attempt += 1
        try:
            req = Request(url=url, data=data, headers=req_headers, method=method.upper())
            with urlopen(req, timeout=timeout, context=ssl_context) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except (TimeoutError, URLError, HTTPError) as e:
            # Raise immediately for unrecoverable HTTP errors
            if isinstance(e, HTTPError) and e.code not in [524, 502, 503, 504]:
                raise
            last_err = e
            if attempt >= max_attempts:
                raise
            logger.warning(
                f"[API] {method} request failed (attempt {attempt}/{max_attempts}): {e}. "
                f"Retrying in {backoff_sec}s..."
            )
            time.sleep(backoff_sec)
            backoff_sec *= 2

    if last_err:
        raise last_err
    return {}


def _run_job_sync(
    *,
    server_url: str,
    api_key: Optional[str],
    instance_id: str,
    patch_diff: str,
    timeout_sec: int,
    poll_interval_sec: float,
    namespace: str,
    tag: str,
    model_name_or_path: str,
) -> dict[str, Any]:
    # Add header only if API key exists (empty dict otherwise)
    headers: Optional[dict[str, str]] = {"X-API-Key": api_key} if api_key else None

    # 1) create job
    create = _http_json(
        "POST",
        f"{server_url}/v1/jobs",
        headers=headers,
        body={
            "instance_id": instance_id,
            "patch_diff": patch_diff,
            "namespace": namespace,
            "tag": tag,
            "timeout_sec": timeout_sec,
            "model_name_or_path": model_name_or_path,
        },
        timeout=300,
    )
    job_id = create.get("job_id")
    if not job_id:
        raise RuntimeError(f"Invalid server response (missing job_id): {create}")

    # 2) poll status
    start = time.time()
    last = create
    while True:
        if time.time() - start > (timeout_sec + 120):
            raise TimeoutError(f"Job polling timed out: {job_id}")

        status = _http_json("GET", f"{server_url}/v1/jobs/{job_id}", headers=headers, timeout=300)
        last = status
        s = status.get("status")
        if s in ("finished", "failed"):
            break
        time.sleep(poll_interval_sec)

    # 3) fetch report (best-effort)
    report: Optional[dict] = None
    try:
        report = _http_json("GET", f"{server_url}/v1/jobs/{job_id}/report", headers=headers, timeout=300)
    except Exception:
        report = None

    return {"job_id": job_id, "status": last, "report": report}


@scorer(metrics=[accuracy()])
def swebench_server_scorer(
    server_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: int = 1800,
    poll_interval_sec: float = 2.0,
    namespace: str = "swebench",
    tag: str = "latest",
    model_name_or_path: str = "horangi",
) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        server = server_url or os.getenv("SWE_SERVER_URL", "https://api.nejumi-swebench.org")
        key = api_key or os.getenv("SWE_API_KEY")

        metadata = state.metadata or {}
        instance_id = (
            metadata.get("instance_id")
            or state.sample_id
            or metadata.get("id")
            or ""
        )

        response = state.output.completion if state.output else ""
        patch = _extract_patch(response)

        if not instance_id or not patch.strip():
            return Score(
                value=INCORRECT,
                answer=(response or "")[:200],
                explanation="Missing instance_id or empty patch",
                metadata={
                    "judgment": "ERROR",
                    "instance_id": instance_id,
                    "has_patch": bool(patch.strip()),
                },
            )

        # Offload network I/O to thread
        try:
            result = await asyncio.to_thread(
                _run_job_sync,
                server_url=server.rstrip("/"),
                api_key=key,
                instance_id=str(instance_id),
                patch_diff=patch,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
                namespace=namespace,
                tag=tag,
                model_name_or_path=model_name_or_path,
            )
        except HTTPError as e:
            return Score(
                value=INCORRECT,
                answer=patch[:200],
                explanation=f"HTTPError from server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )
        except URLError as e:
            return Score(
                value=INCORRECT,
                answer=patch[:200],
                explanation=f"URLError from server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )
        except Exception as e:
            return Score(
                value=INCORRECT,
                answer=patch[:200],
                explanation=f"Failed to score via server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )

        report = (result or {}).get("report") or {}
        resolved_ids = set(report.get("resolved_ids", []) or [])
        is_resolved = str(instance_id) in resolved_ids

        return Score(
            value=CORRECT if is_resolved else INCORRECT,
            answer=patch[:200],
            explanation="resolved" if is_resolved else "not resolved",
            metadata={
                "judgment": "CORRECT" if is_resolved else "INCORRECT",
                "instance_id": instance_id,
                "job_id": result.get("job_id"),
                "server_status": (result.get("status") or {}).get("status"),
                "report_summary": {
                    "resolved_ids_len": len(resolved_ids),
                    "unresolved_ids_len": len(report.get("unresolved_ids", []) or []),
                },
            },
        )

    return score
