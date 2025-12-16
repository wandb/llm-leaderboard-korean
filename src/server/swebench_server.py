#!/usr/bin/env python3
"""
SWE-bench Evaluation Server

Docker 기반으로 SWE-bench 패치를 평가하는 API 서버입니다.

Usage:
    # 서버 시작
    uv run python -m server.swebench_server --host 0.0.0.0 --port 8000

    # 또는
    uv run python src/server/swebench_server.py --host 0.0.0.0 --port 8000

    # 백그라운드 실행
    nohup python -m server.swebench_server --host 0.0.0.0 --port 8000 \
        >/tmp/swebench_server.out 2>&1 & disown

Reference:
    https://github.com/wandb/llm-leaderboard/blob/main/scripts/server/swebench_server.py
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("swebench_server")


# ============================================================================
# Configuration
# ============================================================================
class Settings:
    """서버 설정"""
    API_KEY: Optional[str] = os.getenv("SWE_API_KEY")
    MAX_CONCURRENT_JOBS: int = int(os.getenv("SWE_MAX_JOBS", "4"))
    JOB_TIMEOUT_SEC: int = int(os.getenv("SWE_JOB_TIMEOUT", "1800"))  # 30분
    PREBUILD_IMAGES: bool = os.getenv("SWE_PREBUILD_IMAGES", "true").lower() == "true"


settings = Settings()


# ============================================================================
# Data Models
# ============================================================================
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class JobCreateRequest(BaseModel):
    instance_id: str
    patch_diff: str
    namespace: str = "swebench"
    tag: str = "latest"
    timeout_sec: int = 1800
    model_name_or_path: str = "horangi"


class JobCreateResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    instance_id: str
    created_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None


class JobReportResponse(BaseModel):
    job_id: str
    instance_id: str
    resolved_ids: list[str]
    unresolved_ids: list[str]
    error_ids: list[str]


@dataclass
class Job:
    job_id: str
    instance_id: str
    patch_diff: str
    namespace: str
    tag: str
    timeout_sec: int
    model_name_or_path: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    report: Optional[dict] = None


# ============================================================================
# Job Store (In-Memory)
# ============================================================================
class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, req: JobCreateRequest) -> Job:
        async with self._lock:
            job_id = str(uuid.uuid4())
            job = Job(
                job_id=job_id,
                instance_id=req.instance_id,
                patch_diff=req.patch_diff,
                namespace=req.namespace,
                tag=req.tag,
                timeout_sec=req.timeout_sec,
                model_name_or_path=req.model_name_or_path,
            )
            self._jobs[job_id] = job
            return job

    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def update(self, job: Job):
        async with self._lock:
            self._jobs[job.job_id] = job


job_store = JobStore()


# ============================================================================
# SWE-bench Runner
# ============================================================================
class SWEBenchRunner:
    """
    swebench 라이브러리를 사용해서 패치를 평가합니다.
    
    Prerequisites:
        pip install swebench
    """
    
    def __init__(self, prebuild_images: bool = True):
        self.prebuild_images = prebuild_images
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)

    async def run_evaluation(self, job: Job) -> dict:
        """
        패치를 적용하고 테스트를 실행합니다.
        
        Returns:
            {
                "resolved_ids": [...],
                "unresolved_ids": [...],
                "error_ids": [...],
            }
        """
        async with self._semaphore:
            return await asyncio.to_thread(self._run_sync, job)

    def _run_sync(self, job: Job) -> dict:
        """동기적으로 swebench 평가 실행"""

        # 1) 패치를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as f:
            f.write(job.patch_diff)
            patch_file = f.name

        # 2) predictions JSON 생성
        predictions = [
            {
                "instance_id": job.instance_id,
                "model_name_or_path": job.model_name_or_path,
                "model_patch": job.patch_diff,
            }
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(predictions, f)
            predictions_file = f.name

        try:
            # 3) swebench harness 실행
            result = self._run_harness(
                predictions_path=predictions_file,
                instance_ids=[job.instance_id],
                timeout=job.timeout_sec,
            )
            return result
        finally:
            # 임시 파일 정리
            Path(patch_file).unlink(missing_ok=True)
            Path(predictions_file).unlink(missing_ok=True)

    def _run_harness(
        self,
        predictions_path: str,
        instance_ids: list[str],
        timeout: int,
    ) -> dict:
        """
        swebench harness 실행 (swebench 2.x 호환)
        
        llm-leaderboard 스타일로 subprocess 기반 실행.
        swebench 2.x와 4.x 모두 지원합니다.
        
        Note: swebench와 Docker가 설치되어 있어야 합니다.
        """
        import subprocess
        import shutil
        
        try:
            with tempfile.TemporaryDirectory() as run_dir:
                run_id = str(uuid.uuid4())[:8]
                log_dir = Path(run_dir) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # swebench CLI 실행 (2.x와 4.x 모두 지원)
                cmd = [
                    "python", "-m", "swebench.harness.run_evaluation",
                    "--predictions_path", predictions_path,
                    "--swe_bench_tasks", "princeton-nlp/SWE-bench_Verified",
                    "--log_dir", str(log_dir),
                    "--testbed", run_dir,
                    "--timeout", str(timeout),
                    "--run_id", run_id,
                    "--verbose",
                ]
                
                # instance_ids 추가
                if instance_ids:
                    cmd.extend(["--instance_ids"] + instance_ids)
                
                logger.info(f"Running swebench: {' '.join(cmd[:6])}...")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 300,  # CLI 타임아웃 여유
                    cwd=run_dir,
                )
                
                if result.returncode != 0:
                    logger.warning(f"swebench exit code: {result.returncode}")
                    if result.stderr:
                        logger.warning(f"stderr: {result.stderr[:500]}")
                
                # 결과 파싱
                resolved_ids = []
                unresolved_ids = []
                error_ids = []
                
                # 1. report.json 확인 (swebench가 생성)
                report_files = list(log_dir.glob("**/report.json"))
                if report_files:
                    with open(report_files[0]) as f:
                        report = json.load(f)
                    resolved_ids = report.get("resolved", [])
                    unresolved_ids = [
                        iid for iid in instance_ids 
                        if iid not in resolved_ids and iid not in error_ids
                    ]
                else:
                    # 2. 로그 파일에서 결과 파싱 (fallback)
                    for instance_id in instance_ids:
                        # 여러 가능한 로그 파일 위치 확인
                        log_patterns = [
                            log_dir / f"{instance_id}.log",
                            log_dir / run_id / f"{instance_id}.log",
                            *list(log_dir.glob(f"**/{instance_id}*.log")),
                        ]
                        
                        found = False
                        for log_file in log_patterns:
                            if isinstance(log_file, Path) and log_file.exists():
                                content = log_file.read_text()
                                if "PASSED" in content or "RESOLVED" in content.upper():
                                    resolved_ids.append(instance_id)
                                else:
                                    unresolved_ids.append(instance_id)
                                found = True
                                break
                        
                        if not found:
                            error_ids.append(instance_id)

                return {
                    "resolved_ids": resolved_ids,
                    "unresolved_ids": unresolved_ids,
                    "error_ids": error_ids,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"swebench timed out after {timeout + 300}s")
            return {
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": instance_ids,
            }
        except FileNotFoundError:
            logger.error("swebench not found. Run: pip install 'swebench>=2.0.0,<3.0.0'")
            return {
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": instance_ids,
            }
        except Exception as e:
            logger.exception(f"Harness execution failed: {e}")
            return {
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": instance_ids,
            }


runner = SWEBenchRunner(prebuild_images=settings.PREBUILD_IMAGES)


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="SWE-bench Evaluation Server",
    description="Docker 기반 SWE-bench 패치 평가 서버",
    version="1.0.0",
)


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """API 키 검증 (설정된 경우에만)"""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "ok"}


@app.post("/v1/jobs", response_model=JobCreateResponse)
async def create_job(
    req: JobCreateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
):
    """
    평가 작업 생성
    
    패치를 제출하고 백그라운드에서 평가를 시작합니다.
    """
    verify_api_key(x_api_key)
    
    job = await job_store.create(req)
    logger.info(f"Created job {job.job_id} for instance {job.instance_id}")
    
    # 백그라운드에서 평가 실행
    background_tasks.add_task(run_job_async, job.job_id)
    
    return JobCreateResponse(job_id=job.job_id, status=job.status.value)


async def run_job_async(job_id: str):
    """백그라운드에서 작업 실행"""
    job = await job_store.get(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    await job_store.update(job)
    logger.info(f"Starting job {job_id}")

    try:
        result = await runner.run_evaluation(job)
        job.report = result
        job.status = JobStatus.FINISHED
        logger.info(f"Job {job_id} finished: {result}")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        logger.exception(f"Job {job_id} failed: {e}")
    finally:
        job.finished_at = time.time()
        await job_store.update(job)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    x_api_key: Optional[str] = Header(None),
):
    """작업 상태 조회"""
    verify_api_key(x_api_key)
    
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        instance_id=job.instance_id,
        created_at=job.created_at,
        finished_at=job.finished_at,
        error=job.error,
    )


@app.get("/v1/jobs/{job_id}/report", response_model=JobReportResponse)
async def get_job_report(
    job_id: str,
    x_api_key: Optional[str] = Header(None),
):
    """작업 결과 조회"""
    verify_api_key(x_api_key)
    
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in (JobStatus.FINISHED, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail="Job not finished yet")
    
    report = job.report or {}
    return JobReportResponse(
        job_id=job.job_id,
        instance_id=job.instance_id,
        resolved_ids=report.get("resolved_ids", []),
        unresolved_ids=report.get("unresolved_ids", []),
        error_ids=report.get("error_ids", []),
    )


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SWE-bench Evaluation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logger.info(f"Starting SWE-bench server on {args.host}:{args.port}")
    logger.info(f"API Key: {'enabled' if settings.API_KEY else 'disabled'}")
    logger.info(f"Max concurrent jobs: {settings.MAX_CONCURRENT_JOBS}")
    logger.info(f"Prebuild images: {settings.PREBUILD_IMAGES}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()

