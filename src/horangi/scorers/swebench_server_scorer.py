"""
SWE-bench Server Scorer

Docker 기반 공식 하네스 대신, 채점용 서버 API를 호출하여 해결(resolved) 여부를 채점합니다.

참고:
- 서버 구현 예시: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/server/swebench_server.py
- evaluator 구현 예시: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/evaluator/swe_bench.py

환경변수:
- SWE_SERVER_URL: 예) https://api.nejumi-swebench.org
- SWE_API_KEY: (옵션) 서버가 X-API-Key를 요구하는 경우
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
    """코드펜스(```)를 제거하고 내부 내용만 반환."""
    if not text:
        return ""
    # ```diff ... ``` or ```patch ... ``` or ```...``` 제거 (가능하면 내부만)
    m = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_patch(text: str) -> str:
    """
    모델 출력에서 patch 텍스트만 최대한 안전하게 추출.
    
    설명 텍스트가 섞여 있어도 diff 부분만 추출합니다.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return ""

    # 케이스 1: diff --git 헤더가 있는 경우
    idx = cleaned.find("diff --git")
    if idx != -1:
        patch_text = cleaned[idx:]
        # diff 이후의 설명 텍스트 제거 (빈 줄 2개 이후는 설명으로 간주)
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        patch_text = parts[0]
        # 패치가 끝나는 지점 찾기 (diff/---/+++/@@ 패턴이 끝나는 곳)
        lines = patch_text.split('\n')
        valid_lines = []
        for line in lines:
            # 패치 라인 패턴
            if (line.startswith('diff --git') or 
                line.startswith('---') or 
                line.startswith('+++') or 
                line.startswith('@@') or
                line.startswith('+') or
                line.startswith('-') or
                line.startswith(' ') or
                line == ''):
                valid_lines.append(line)
            elif valid_lines:  # 패치가 시작된 후 패턴 밖의 라인이면 종료
                # 컨텍스트 라인일 수 있으므로 간단한 체크
                if not any(valid_lines[-1].startswith(p) for p in ['diff', '---', '+++', '@@', ' ', '+', '-']):
                    break
                valid_lines.append(line)
        return '\n'.join(valid_lines).strip() + '\n'

    # 케이스 2: --- 헤더로 시작하는 unified diff
    m = re.search(r"(^---\s+[ab]?/?.+$)", cleaned, re.MULTILINE)
    if m:
        idx = m.start()
        patch_text = cleaned[idx:]
        # diff 이후의 설명 텍스트 제거
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        return parts[0].strip() + '\n'

    # 케이스 3: 그냥 전체를 patch로 취급 (마지막 fallback)
    return cleaned.strip() + ('\n' if not cleaned.endswith('\n') else '')


def _http_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    """
    HTTP 요청 helper (SSL 인증서 검증 우회 + 재시도 로직 포함)
    
    참고: https://github.com/wandb/llm-leaderboard/blob/main/scripts/evaluator/swe_bench.py
    """
    data = None
    req_headers = dict(headers) if headers else {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    # SSL 인증서 검증 우회 (개발 환경용)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # POST는 side-effect가 있어서 재시도 안 함
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
            # 복구 불가능한 HTTP 에러는 바로 raise
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
    # API 키가 있을 때만 헤더에 추가 (없으면 빈 dict)
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

        # 네트워크 I/O는 스레드로 오프로드
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

