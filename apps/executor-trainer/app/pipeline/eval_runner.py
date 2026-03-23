from __future__ import annotations

import copy
import gc
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..adapters.reporter import Reporter
from ..bootstrap.schemas import JobConfig

logger = logging.getLogger(__name__)

VLLM_PYTHON_BIN = os.environ.get("VLLM_PYTHON_BIN", "/opt/vllm-venv/bin/python")


def _worker_script_path() -> str:
    return str(Path(__file__).with_name("vllm_eval_worker.py"))


def _cleanup_runtime(stage: str) -> None:
    logger.info("==> cleaning runtime after %s", stage)

    try:
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _build_worker_payload(cfg: JobConfig, training_result: Dict[str, Any]) -> Dict[str, Any]:
    ds_cfg = cfg.evaluation.dataset
    if ds_cfg is None:
        raise ValueError("evaluation.dataset is required when evaluation.enabled=true")

    return {
        "job_name": cfg.job_name,
        "model": json.loads(cfg.model.model_dump_json()),
        "dataset": json.loads(cfg.dataset.model_dump_json()),
        "lora": json.loads(cfg.lora.model_dump_json()),
        "outputs": json.loads(cfg.outputs.model_dump_json()),
        "evaluation": json.loads(cfg.evaluation.model_dump_json()),
        "training_result": training_result,
    }


def _build_eval_attempts(cfg: JobConfig) -> List[Dict[str, int | float]]:
    base_util = float(cfg.evaluation.gpu_memory_utilization or 0.9)
    base_seqs = int(cfg.evaluation.max_num_seqs or max(1, min(cfg.evaluation.batch_size, 8)))
    base_tokens = int(cfg.evaluation.max_num_batched_tokens or 8192)

    raw_attempts = [
        {
            "gpu_memory_utilization": base_util,
            "max_num_seqs": base_seqs,
            "max_num_batched_tokens": base_tokens,
        },
        {
            "gpu_memory_utilization": min(base_util, 0.85),
            "max_num_seqs": max(1, min(base_seqs, max(1, base_seqs // 2))),
            "max_num_batched_tokens": max(1024, min(base_tokens, max(1024, base_tokens // 2))),
        },
        {
            "gpu_memory_utilization": min(base_util, 0.80),
            "max_num_seqs": max(1, min(base_seqs, max(1, base_seqs // 2))),
            "max_num_batched_tokens": max(1024, min(base_tokens, max(1024, base_tokens // 4))),
        },
        {
            "gpu_memory_utilization": min(base_util, 0.75),
            "max_num_seqs": max(1, min(base_seqs, 2)),
            "max_num_batched_tokens": max(1024, min(base_tokens, 2048)),
        },
        {
            "gpu_memory_utilization": min(base_util, 0.70),
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
        },
    ]

    result: List[Dict[str, int | float]] = []
    seen = set()
    for attempt in raw_attempts:
        key = (
            round(float(attempt["gpu_memory_utilization"]), 4),
            int(attempt["max_num_seqs"]),
            int(attempt["max_num_batched_tokens"]),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(attempt)
    return result


def _is_retryable_vllm_failure(stderr: str, stdout: str, response_payload: Optional[Dict[str, Any]] = None) -> bool:
    text = "\n".join(
        part for part in [
            stderr or "",
            stdout or "",
            json.dumps(response_payload, ensure_ascii=False) if response_payload else "",
        ] if part
    ).lower()

    retry_markers = [
        "free memory on device",
        "desired gpu memory utilization",
        "cuda out of memory",
        "engine core initialization failed",
        "outofmemoryerror",
        "nccl",
    ]
    return any(marker in text for marker in retry_markers)


def _run_worker_once(
    request_path: Path,
    response_path: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        VLLM_PYTHON_BIN,
        _worker_script_path(),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]

    logger.info("==> starting evaluation worker: %s", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def run_evaluation(
    cfg: JobConfig,
    training_result: Dict[str, Any],
    reporter: Optional[Reporter] = None,
) -> Dict[str, Any]:
    if not cfg.evaluation.enabled:
        return {"enabled": False}

    ds_cfg = cfg.evaluation.dataset
    if ds_cfg is None:
        raise ValueError("evaluation.dataset is required when evaluation.enabled=true")

    output_dir = Path(cfg.outputs.eval_dir) / cfg.job_name
    output_dir.mkdir(parents=True, exist_ok=True)

    request_path = output_dir / "worker-request.json"
    response_path = output_dir / "worker-response.json"

    attempts = _build_eval_attempts(cfg)
    last_error: Optional[str] = None

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation_prepare",
            progress=0,
            message="preparing evaluation via vllm worker",
            extra={
                "engine": cfg.evaluation.engine,
                "task": ds_cfg.task,
                "batch_size": cfg.evaluation.batch_size,
                "max_num_seqs": cfg.evaluation.max_num_seqs,
                "max_num_batched_tokens": cfg.evaluation.max_num_batched_tokens,
                "worker_python": VLLM_PYTHON_BIN,
                "attempts": len(attempts),
            },
        )

    for attempt_index, attempt in enumerate(attempts, start=1):
        payload = copy.deepcopy(_build_worker_payload(cfg, training_result))
        payload["evaluation"]["gpu_memory_utilization"] = float(attempt["gpu_memory_utilization"])
        payload["evaluation"]["max_num_seqs"] = int(attempt["max_num_seqs"])
        payload["evaluation"]["max_num_batched_tokens"] = int(attempt["max_num_batched_tokens"])

        if response_path.exists():
            try:
                response_path.unlink()
            except Exception:
                pass

        with request_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        _cleanup_runtime(f"pre-evaluation-attempt-{attempt_index}")

        logger.info(
            "==> evaluation attempt %s/%s: gpu_memory_utilization=%s max_num_seqs=%s max_num_batched_tokens=%s",
            attempt_index,
            len(attempts),
            payload["evaluation"]["gpu_memory_utilization"],
            payload["evaluation"]["max_num_seqs"],
            payload["evaluation"]["max_num_batched_tokens"],
        )

        if reporter:
            reporter.report_status(
                "running",
                stage="evaluation",
                progress=round(((attempt_index - 1) / max(1, len(attempts))) * 10 + 1, 2),
                message="evaluation worker started",
                extra={
                    "attempt": attempt_index,
                    "attempts_total": len(attempts),
                    "engine": cfg.evaluation.engine,
                    "worker_python": VLLM_PYTHON_BIN,
                    "gpu_memory_utilization": payload["evaluation"]["gpu_memory_utilization"],
                    "max_num_seqs": payload["evaluation"]["max_num_seqs"],
                    "max_num_batched_tokens": payload["evaluation"]["max_num_batched_tokens"],
                },
            )

        process = _run_worker_once(
            request_path=request_path,
            response_path=response_path,
        )

        if process.stdout:
            logger.info("==> evaluation worker stdout:\n%s", process.stdout.strip())
        if process.stderr:
            logger.warning("==> evaluation worker stderr:\n%s", process.stderr.strip())

        response_payload: Optional[Dict[str, Any]] = None
        if response_path.exists():
            try:
                with response_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    response_payload = loaded
            except Exception:
                response_payload = None

        _cleanup_runtime(f"post-evaluation-attempt-{attempt_index}")

        if process.returncode == 0 and isinstance(response_payload, dict) and response_payload.get("status") != "failed":
            if reporter:
                reporter.report_status(
                    "running",
                    stage="evaluation_completed",
                    progress=100,
                    message="evaluation completed",
                    extra=response_payload.get("summary") or {},
                )
            return response_payload

        retryable = _is_retryable_vllm_failure(
            stderr=process.stderr or "",
            stdout=process.stdout or "",
            response_payload=response_payload,
        )

        if isinstance(response_payload, dict) and response_payload.get("status") == "failed":
            last_error = response_payload.get("error") or "Evaluation worker reported failure"
        else:
            last_error = (
                "Evaluation worker failed with exit code "
                f"{process.returncode}. stderr: {(process.stderr or '').strip() or '<empty>'}"
            )

        if retryable and attempt_index < len(attempts):
            logger.warning("==> retryable vLLM failure detected, retrying with smaller memory settings")
            if reporter:
                reporter.report_progress(
                    stage="evaluation",
                    progress=round((attempt_index / max(1, len(attempts))) * 50, 2),
                    message="retrying evaluation with lower vllm memory settings",
                    extra={
                        "attempt": attempt_index,
                        "attempts_total": len(attempts),
                        "error": last_error,
                    },
                )
            continue

        raise RuntimeError(last_error or "Evaluation worker failed")

    raise RuntimeError(last_error or "Evaluation worker failed")