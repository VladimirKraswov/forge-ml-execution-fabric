from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .pipeline.archiver import Archiver
from .pipeline.upload_runner import UploadRunner
from .pipeline.publish_runner import PublishRunner
from .pipeline.asset_manager import AssetManager
from .bootstrap.bootstrap_loader import load_remote_job_config
from .bootstrap.config_loader import load_config
from .adapters.hf_utils import try_hf_login
from .adapters.log_streamer import LogStreamer
from .adapters.reporter import Reporter

logger = logging.getLogger(__name__)


def setup_logging(
    logs_dir: str,
    job_id: Optional[str] = None,
    job_name: Optional[str] = None,
    logs_url: Optional[str] = None,
    logs_bearer_token: Optional[str] = None,
):
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "trainer.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]

    if logs_url and job_id and job_name:
        streamer = LogStreamer(
            logs_url=logs_url,
            job_id=job_id,
            job_name=job_name,
            bearer_token=logs_bearer_token,
        )
        streamer.setFormatter(formatter)
        handlers.append(streamer)

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )
    return log_file


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_file(file_path: Path, lines: int = 50) -> str:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return "Could not retrieve logs"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_config(args) -> Tuple[Any, str, Dict[str, Any]]:
    if args.job_config_url:
        cfg, meta = load_remote_job_config(args.job_config_url)
        return cfg, args.job_config_url, meta

    if args.config:
        cfg = load_config(args.config)
        return cfg, args.config, {}

    raise ValueError("Either --config or --job-config-url must be provided")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--job-config-url", required=False)
    args = parser.parse_args()

    try:
        cfg, config_source, bootstrap_meta = resolve_config(args)
    except Exception as exc:
        print(f"ERROR: failed to resolve config: {exc}")
        sys.exit(1)

    Path(cfg.outputs.base_dir).mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(
        cfg.outputs.logs_dir,
        job_id=cfg.job_id or cfg.job_name,
        job_name=cfg.job_name,
        logs_url=bootstrap_meta.get("logs_url") or cfg.reporting.logs.url,
        logs_bearer_token=(
            bootstrap_meta.get("callback_auth_token")
            or cfg.reporting.logs.auth.bearer_token
            or cfg.reporting.status.auth.bearer_token
            or cfg.reporting.progress.auth.bearer_token
            or cfg.reporting.final.auth.bearer_token
        ),
    )

    reporter = Reporter(cfg)
    asset_manager = AssetManager(cfg)
    archiver = Archiver()
    uploader = UploadRunner(cfg)
    publisher = PublishRunner(cfg)

    effective_config_path = Path(cfg.outputs.logs_dir) / "effective-job.json"
    result_path = Path(cfg.outputs.base_dir) / "job-result.json"

    started_at = utc_now_iso()

    # Heartbeat thread
    import threading
    import time
    def heartbeat_loop():
        while True:
            try:
                reporter.report_status(
                    "running",
                    message="Heartbeat",
                    stage="heartbeat",
                    extra={"heartbeat": True}
                )
            except Exception:
                pass
            time.sleep(30)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    try:
        reporter.report_status(
            "started",
            message="Training pipeline started",
            stage="bootstrap",
            progress=0,
        )

        write_json(effective_config_path, json.loads(cfg.model_dump_json()))
        logger.info("==> config loaded")
        logger.info("==> job_name: %s", cfg.job_name)
        logger.info("==> job_id: %s", cfg.job_id or cfg.job_name)
        logger.info("==> config source: %s", config_source)

        reporter.report_status(
            "running",
            message="Validating Hugging Face access",
            stage="hf_login",
            progress=2,
        )
        try_hf_login()
        publisher.ensure_hf_ready()

        pipeline = cfg.pipeline
        has_steps = pipeline and pipeline.steps

        if has_steps:
            logger.info("==> executing pipeline steps")
        else:
            logger.warning("==> pipeline.steps missing, falling back to legacy sequence")

        training_result = {}
        evaluation_result = None
        external_refs = []

        result: Dict[str, Any] = {
            "status": "success",
            "job_id": cfg.job_id or cfg.job_name,
            "job_name": cfg.job_name,
            "started_at": started_at,
            "finished_at": None,
            "config_source": config_source,
            "training": {},
            "evaluation": None,
            "artifacts": {
                "log_file": str(log_file),
                "effective_config_path": str(effective_config_path),
                "result_path": str(result_path),
            },
            "uploads": {},
            "upload_errors": {},
            "externalRefs": external_refs,
        }

        # Legacy sequence logic or Step-based execution
        # For simplicity in v1, we process steps in order from cfg.pipeline.steps
        # while maintaining compatibility with the runners.

        def run_step(step_key, step_kind):
            nonlocal training_result, evaluation_result

            if step_kind == "prepare_assets":
                logger.info("==> preparing assets")
                reporter.report_status("running", message="Preparing assets", stage="prepare_assets", progress=5)
                asset_manager.prepare_dataset(cfg)
                asset_manager.prepare_evaluation_dataset(cfg)

            elif step_kind == "training":
                from .pipeline.train_runner import run_training
                logger.info("==> starting training")
                training_result = run_training(cfg, reporter=reporter)
                logical_base_model_id = cfg.model.logical_base_model_id
                if logical_base_model_id and not training_result.get("base_model_id"):
                    training_result["base_model_id"] = logical_base_model_id
                result["training"] = training_result

            elif step_kind == "evaluation":
                from .pipeline.eval_runner import run_evaluation
                logger.info("==> starting evaluation")
                evaluation_result = run_evaluation(cfg, training_result, reporter=reporter)
                result["evaluation"] = evaluation_result

            elif step_kind == "publish_hf":
                reporter.report_status("running", message="Publishing to HF", stage="publish", progress=90)
                hf_model_uploads = publisher.upload_to_huggingface(training_result)
                if hf_model_uploads:
                    result["uploads"].update(hf_model_uploads)
                    # Gather external refs from publish
                    if hf_model_uploads.get("merged_model"):
                        plan = hf_model_uploads["merged_model"].get("plan", {})
                        external_refs.append({
                            "backend": "huggingface",
                            "repoId": plan.get("merged_repo"),
                            "repoType": "model",
                            "revision": plan.get("revision") or "main",
                        })

                hf_metadata_uploads = publisher.upload_hf_metadata(
                    log_file=str(log_file),
                    effective_config_path=str(effective_config_path),
                    result_path=str(result_path),
                    training_result=training_result,
                    eval_result=evaluation_result,
                )
                if hf_metadata_uploads:
                    result["uploads"].update(hf_metadata_uploads)

            elif step_kind == "upload_artifacts":
                reporter.report_status("running", message="Uploading artifacts", stage="upload", progress=95)
                extra_uploads, extra_upload_errors = uploader.upload_non_summary_artifacts(
                    log_file=str(log_file),
                    effective_config_path=str(effective_config_path),
                    training_result=training_result,
                    eval_result=evaluation_result,
                )
                if extra_uploads:
                    result["uploads"].update(extra_uploads)
                if extra_upload_errors:
                    result["upload_errors"].update(extra_upload_errors)

        if has_steps:
            for step in pipeline.steps:
                if step.enabled:
                    run_step(step.key, step.kind)
                else:
                    logger.info("==> step %s disabled, skipping", step.key)
        else:
            # Fallback legacy sequence
            if not pipeline or pipeline.prepare_assets.enabled:
                run_step("prepare_assets", "prepare_assets")

            if not pipeline or pipeline.training.enabled:
                run_step("training", "training")

            if (pipeline.evaluation.enabled if pipeline else cfg.evaluation.enabled):
                run_step("evaluation", "evaluation")

            if (pipeline.publish.enabled if pipeline else cfg.huggingface.enabled):
                run_step("publish", "publish_hf")

            if (pipeline.upload.enabled if pipeline else cfg.upload.enabled):
                run_step("upload", "upload_artifacts")

        result["finished_at"] = utc_now_iso()

        should_upload = pipeline.upload.enabled if pipeline else cfg.upload.enabled
        if should_upload and cfg.upload.target == "url" and cfg.upload.url_targets.summary_url:
            try:
                summary_upload = uploader.upload_summary(str(result_path))
                if summary_upload:
                    result["uploads"].update(summary_upload)
                    write_json(result_path, result)
            except Exception as exc:
                logger.exception("summary upload failed")
                result["upload_errors"]["summary"] = str(exc)
                write_json(result_path, result)

        if result["upload_errors"]:
            logger.warning("==> pipeline finished with upload warnings")
            for key, value in result["upload_errors"].items():
                logger.warning("==> upload warning [%s]: %s", key, value)

        logger.info("==> pipeline finished successfully")
        reporter.report_status(
            "finished",
            message=(
                "Training pipeline finished successfully"
                if not result["upload_errors"]
                else "Training completed, but some URL artifact uploads failed"
            ),
            stage="finished",
            progress=100,
        )
        reporter.report_final(result, status="finished")

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as exc:
        error_msg = str(exc)
        stack_trace = traceback.format_exc()

        logger.error("FATAL ERROR: %s", error_msg)
        logger.error(stack_trace)

        logs_tail = tail_file(log_file, 50)

        failed_result = {
            "status": "failed",
            "job_id": cfg.job_id or cfg.job_name,
            "job_name": cfg.job_name,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "config_source": config_source,
            "error": error_msg,
            "artifacts": {
                "log_file": str(log_file),
                "effective_config_path": str(effective_config_path),
                "result_path": str(result_path),
            },
        }
        write_json(result_path, failed_result)

        reporter.report_error(error_msg, logs=logs_tail)
        reporter.report_final(failed_result, status="failed")
        sys.exit(1)


if __name__ == "__main__":
    main()