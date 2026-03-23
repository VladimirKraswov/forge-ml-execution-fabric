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


def _cleanup_runtime(stage: str) -> None:
    import gc
    import time

    logger.info("==> cleaning runtime after %s", stage)

    try:
        gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    time.sleep(0.3)


def _is_retryable_vllm_error(stderr: str) -> bool:
    text = (stderr or "").lower()
    markers = [
        "free memory on device cuda",
        "decrease gpu memory utilization",
        "engine core initialization failed",
        "cuda out of memory",
        "out of memory",
    ]
    return any(marker in text for marker in markers)


def _attempt_overrides(cfg: JobConfig) -> list[dict]:
    base_max_len = cfg.evaluation.max_model_len or 1024

    return [
        {
            "gpu_memory_utilization": 0.72,
            "max_num_seqs": 2,
            "max_num_batched_tokens": min(1024, base_max_len),
            "max_model_len": min(base_max_len, 1024),
            "batch_size": 2,
        },
        {
            "gpu_memory_utilization": 0.68,
            "max_num_seqs": 2,
            "max_num_batched_tokens": min(768, base_max_len),
            "max_model_len": min(base_max_len, 768),
            "batch_size": 2,
        },
        {
            "gpu_memory_utilization": 0.64,
            "max_num_seqs": 1,
            "max_num_batched_tokens": min(512, base_max_len),
            "max_model_len": min(base_max_len, 512),
            "batch_size": 1,
        },
        {
            "gpu_memory_utilization": 0.58,
            "max_num_seqs": 1,
            "max_num_batched_tokens": min(384, base_max_len),
            "max_model_len": min(base_max_len, 384),
            "batch_size": 1,
        },
        {
            "gpu_memory_utilization": 0.52,
            "max_num_seqs": 1,
            "max_num_batched_tokens": min(256, base_max_len),
            "max_model_len": min(base_max_len, 256),
            "batch_size": 1,
        },
    ]


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

    if reporter:
        reporter.report_status(
            "running",
            stage="evaluation_prepare",
            progress=0,
            message="preparing evaluation via vllm worker",
            extra={
                "engine": cfg.evaluation.engine,
                "task": ds_cfg.task,
            },
        )

    _cleanup_runtime("pre-evaluation")

    attempts = _attempt_overrides(cfg)
    last_error = None

    for idx, override in enumerate(attempts, start=1):
        _cleanup_runtime(f"pre-evaluation-attempt-{idx}")

        eval_cfg = cfg.model_copy(deep=True)
        eval_cfg.evaluation.gpu_memory_utilization = override["gpu_memory_utilization"]
        eval_cfg.evaluation.max_num_seqs = override["max_num_seqs"]
        eval_cfg.evaluation.max_num_batched_tokens = override["max_num_batched_tokens"]
        eval_cfg.evaluation.max_model_len = override["max_model_len"]
        eval_cfg.evaluation.batch_size = override["batch_size"]

        payload = _build_worker_payload(eval_cfg, training_result)
        with request_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        cmd = [
            VLLM_PYTHON_BIN,
            _worker_script_path(),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]

        logger.info(
            "==> evaluation attempt %s/%s: gpu_memory_utilization=%s max_num_seqs=%s max_num_batched_tokens=%s max_model_len=%s",
            idx,
            len(attempts),
            override["gpu_memory_utilization"],
            override["max_num_seqs"],
            override["max_num_batched_tokens"],
            override["max_model_len"],
        )
        logger.info("==> starting evaluation worker: %s", " ".join(cmd))

        if reporter:
            reporter.report_status(
                "running",
                stage="evaluation",
                progress=1,
                message=f"evaluation attempt {idx}/{len(attempts)}",
                extra=override,
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

        _cleanup_runtime(f"post-evaluation-attempt-{idx}")

        if process.returncode == 0 and response_path.exists():
            with response_path.open("r", encoding="utf-8") as f:
                result = json.load(f)

            if not isinstance(result, dict):
                raise RuntimeError("Evaluation worker returned invalid response payload")

            if result.get("status") == "failed":
                last_error = result.get("error") or "Evaluation worker reported failure"
            else:
                if reporter:
                    reporter.report_status(
                        "running",
                        stage="evaluation_completed",
                        progress=100,
                        message="evaluation completed",
                        extra=result.get("summary") or {},
                    )
                return result
        else:
            last_error = (
                f"Evaluation worker failed with exit code {process.returncode}. "
                f"stderr: {process.stderr.strip() or '<empty>'}"
            )

        if idx < len(attempts) and _is_retryable_vllm_error(process.stderr):
            logger.warning("==> retryable vLLM failure detected, retrying with smaller memory settings")
            continue

        break

    raise RuntimeError(last_error or "Evaluation failed after all retry attempts")