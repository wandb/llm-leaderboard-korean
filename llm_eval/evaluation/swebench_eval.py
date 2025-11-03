"""
SWE-bench Evaluation Method

This module provides evaluation logic for SWE-bench using an external API server
for running tests in Docker containers.
"""

import json
import logging
import os
import re
import ssl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from llm_eval.evaluation.base import BaseEvaluator
from llm_eval.evaluation import register_evaluator
from llm_eval.utils.logging import get_logger

logger = get_logger(name="swebench_eval", level=logging.INFO)


def _api_http_json(
    method: str,
    url: str,
    body_obj: Optional[Dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60
) -> Dict:
    """
    Helper function to make HTTP requests to the SWE-bench API server.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        body_obj: Optional JSON body for POST requests
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Dict: JSON response from server
    """
    data = json.dumps(body_obj).encode("utf-8") if body_obj else None
    attempt = 0
    backoff_sec = 2.0
    last_err = None

    # POST requests are not retried due to side effects
    max_attempts = 1 if method == "POST" else 3

    # SSL certificate verification bypass (for development environments)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    while attempt < max_attempts:
        attempt += 1
        try:
            req = Request(url=url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            with urlopen(req, timeout=timeout, context=ssl_context) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset)
                return json.loads(text) if text else {}
        except (TimeoutError, URLError, HTTPError) as e:
            # Retry transient errors (GET only)
            if isinstance(e, HTTPError) and e.code not in [524, 502, 503, 504]:
                raise  # Non-recoverable HTTP errors fail immediately
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


@register_evaluator("swebench")
class SWEBenchEvaluator(BaseEvaluator):
    """
    Evaluator for SWE-bench that uses an external API server for running tests.

    This evaluator:
    1. Extracts diff/patch from model outputs
    2. Submits patches to the API server
    3. Polls for evaluation results
    4. Computes metrics (resolved rate, pass rate, etc.)

    Args:
        api_endpoint (str): API server endpoint URL
        api_key (Optional[str]): API authentication key
        namespace (str): Docker image namespace (default: "swebench")
        tag (str): Docker image tag (default: "latest")
        timeout_sec (int): Timeout for each test execution (default: 1800)
        concurrency (int): Number of parallel jobs (default: 2)
        **kwargs: Additional parameters
    """

    name = "swebench"
    requires_logits = False
    requires_chain_of_thought = False

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        namespace: str = "swebench",
        tag: str = "latest",
        timeout_sec: int = 1800,
        concurrency: int = 2,
        **kwargs
    ):
        self.api_endpoint = api_endpoint or os.getenv("SWE_API_ENDPOINT", "http://127.0.0.1:8000")
        self.api_endpoint = self.api_endpoint.rstrip("/")
        self.api_key = api_key or os.getenv("SWE_API_KEY")
        self.namespace = namespace
        self.tag = tag
        self.timeout_sec = timeout_sec
        self.concurrency = max(1, min(concurrency, 32))
        self.kwargs = kwargs

    def parse_prediction(self, raw_output: str) -> str:
        """
        Extract the patch/diff from the model's raw output.

        Supports multiple formats:
        - <patch>...</patch> or <diff>...</diff> tags
        - ```diff ... ``` or ```patch ... ``` code blocks
        - Raw diff content

        Args:
            raw_output (str): Raw model output

        Returns:
            str: Extracted patch in unified diff format
        """
        if raw_output is None or not raw_output.strip():
            return ""

        # Try extracting from XML-style tags first
        diff_matches = []
        other_matches = []
        pattern = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)
        for code, match in pattern.findall(raw_output):
            if code in {"diff", "patch"}:
                diff_matches.append(match)
            else:
                other_matches.append(match)

        # Try extracting from markdown code blocks
        pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
        for code, match in pattern.findall(raw_output):
            if code in {"diff", "patch"}:
                diff_matches.append(match)
            else:
                other_matches.append(match)

        if diff_matches:
            return diff_matches[0].strip()
        if other_matches:
            return other_matches[0].strip()

        # Return raw output, remove </s> token if present
        return raw_output.split("</s>")[0].strip()

    def evaluate_predictions(
        self,
        subsets: Optional[List[str]],
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate patches by submitting them to the API server.

        Args:
            subsets (List[str]): Not used for SWE-bench
            samples (List[Dict[str, Any]]): Samples with predictions

        Returns:
            Dict[str, float]: Evaluation metrics including resolved_rate, etc.
        """
        if not samples:
            logger.warning("No samples to evaluate")
            return {"error": 1.0}

        logger.info(f"Evaluating {len(samples)} samples via API server: {self.api_endpoint}")

        # Prepare headers
        headers = {"X-API-Key": self.api_key} if self.api_key else {}

        # Track results
        resolved_ids = []
        unresolved_ids = []
        error_ids = []
        empty_patch_ids = []

        # Prepare instances for API submission
        instances = []
        for sample in samples:
            prediction = sample.get("prediction", "")
            instance_id = sample.get("instance_id", "unknown")

            if not prediction or not prediction.strip():
                empty_patch_ids.append(instance_id)
                logger.warning(f"Empty patch for {instance_id}")
                continue

            instances.append({
                "instance_id": instance_id,
                "model_patch": prediction,
                "sample": sample
            })

        if not instances:
            logger.warning("No valid patches to evaluate")
            return {
                "total_samples": len(samples),
                "empty_patches": len(empty_patch_ids),
                "resolved": 0,
                "resolved_rate": 0.0,
            }

        logger.info(f"Submitting {len(instances)} jobs to API server")

        # Simple bounded-concurrency loop
        in_flight = []  # list of dict(job_id, iid, start)
        pending = list(instances)

        def submit(inst):
            iid = inst["instance_id"]
            patch = inst.get("model_patch", "")
            sample = inst["sample"]

            payload = {
                "instance_id": iid,
                "patch_diff": patch,
                "namespace": self.namespace,
                "tag": self.tag,
                "model_name_or_path": sample.get("model_name", "unknown"),
                "timeout_sec": self.timeout_sec,
            }

            try:
                job = _api_http_json(
                    "POST",
                    f"{self.api_endpoint}/v1/jobs",
                    body_obj=payload,
                    headers=headers,
                    timeout=300
                )
                return {"job_id": job.get("job_id"), "iid": iid, "start": time.time()}
            except Exception as e:
                logger.error(f"Failed to submit job for {iid}: {e}")
                error_ids.append(iid)
                return None

        # Prime submit
        while pending and len(in_flight) < self.concurrency:
            result = submit(pending.pop(0))
            if result:
                in_flight.append(result)

        processed = 0
        while in_flight:
            time.sleep(2)
            # Poll all in-flight jobs
            new_in_flight = []
            for jf in in_flight:
                job_id = jf["job_id"]
                iid = jf["iid"]

                try:
                    j = _api_http_json(
                        "GET",
                        f"{self.api_endpoint}/v1/jobs/{job_id}",
                        headers=headers,
                        timeout=300
                    )
                except Exception as e:
                    logger.warning(f"Poll failed for {iid}: {e}")
                    new_in_flight.append(jf)
                    continue

                status = j.get("status")
                if status in {"finished", "failed"}:
                    final_class = status

                    if status == "finished":
                        try:
                            rep = _api_http_json(
                                "GET",
                                f"{self.api_endpoint}/v1/jobs/{job_id}/report",
                                headers=headers,
                                timeout=300
                            )

                            if rep.get("error_instances") == 1 or iid in (rep.get("error_ids") or []):
                                final_class = "error"
                            elif iid in (rep.get("resolved_ids") or []):
                                final_class = "resolved"
                            elif iid in (rep.get("unresolved_ids") or []):
                                final_class = "unresolved"
                            elif iid in (rep.get("empty_patch_ids") or []):
                                final_class = "empty_patch"
                            else:
                                final_class = "error"
                        except Exception as e:
                            logger.warning(f"Failed to get report for {iid}: {e}")
                            final_class = "error"

                    if final_class == "resolved":
                        resolved_ids.append(iid)
                    elif final_class == "unresolved":
                        unresolved_ids.append(iid)
                    elif final_class == "empty_patch":
                        empty_patch_ids.append(iid)
                    else:
                        error_ids.append(iid)

                    processed += 1
                    elapsed = time.time() - jf["start"]
                    logger.info(
                        f"[{processed}/{len(instances)}] {iid}: {final_class} "
                        f"(elapsed: {elapsed:.1f}s)"
                    )

                    # Submit next pending job
                    if pending:
                        result = submit(pending.pop(0))
                        if result:
                            new_in_flight.append(result)
                else:
                    # Still running
                    new_in_flight.append(jf)

            in_flight = new_in_flight

        # Compute metrics
        total = len(samples)
        resolved_count = len(resolved_ids)
        unresolved_count = len(unresolved_ids)
        error_count = len(error_ids)
        empty_count = len(empty_patch_ids)

        metrics = {
            "total_samples": total,
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "errors": error_count,
            "empty_patches": empty_count,
            "resolved_rate": resolved_count / total if total > 0 else 0.0,
            "unresolved_rate": unresolved_count / total if total > 0 else 0.0,
            "error_rate": error_count / total if total > 0 else 0.0,
            "empty_patch_rate": empty_count / total if total > 0 else 0.0,
        }

        logger.info(f"Evaluation complete: {metrics}")

        return metrics
