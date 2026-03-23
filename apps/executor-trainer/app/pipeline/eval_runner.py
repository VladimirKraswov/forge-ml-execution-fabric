from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..adapters.reporter import Reporter
from ..bootstrap.schemas import JobConfig

logger = logging.getLogger(__name__)

VLLM_PYTHON_BIN = os.environ.get("VLLM_PYTHON_BIN", "/opt/vllm-venv/bin/python")


def _worker_script_path() -> str:
    return str(Path(__file__).with_name("vllm_eval_worker.py"))


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

    payload = _build_worker_payload(cfg, training_result)
    with request_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

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
            },
        )

    cmd = [
        VLLM_PYTHON_BIN,
        _worker_script_path(),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]

    logger.info("==> starting evaluation worker: %s", " ".join(cmd))
    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation",
            progress=1,
            message="evaluation worker started",
            extra={
                "engine": cfg.evaluation.engine,
                "worker_python": VLLM_PYTHON_BIN,
            },
        )

    try:
        process = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"vLLM python interpreter not found: {VLLM_PYTHON_BIN}. "
            "Make sure Docker image created /opt/vllm-venv."
        ) from exc

    if process.stdout:
        logger.info("==> evaluation worker stdout:\n%s", process.stdout.strip())
    if process.stderr:
        logger.warning("==> evaluation worker stderr:\n%s", process.stderr.strip())

    if process.returncode != 0:
        raise RuntimeError(
            "Evaluation worker failed with exit code "
            f"{process.returncode}. stderr: {process.stderr.strip() or '<empty>'}"
        )

    if not response_path.exists():
        raise RuntimeError(f"Evaluation worker did not create response file: {response_path}")

    with response_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if not isinstance(result, dict):
        raise RuntimeError("Evaluation worker returned invalid response payload")

    if result.get("status") == "failed":
        raise RuntimeError(result.get("error") or "Evaluation worker reported failure")

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation_completed",
            progress=100,
            message="evaluation completed",
            extra=result.get("summary") or {},
        )

    return result