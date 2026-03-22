from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from archiver import Archiver
from artifact_uploader import ArtifactUploader
from asset_manager import AssetManager
from bootstrap_loader import load_remote_job_config
from config_loader import load_config
from hf_utils import try_hf_login
from log_streamer import LogStreamer
from reporter import Reporter

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
    uploader = ArtifactUploader(cfg)

    effective_config_path = Path(cfg.outputs.logs_dir) / "effective-job.json"
    result_path = Path(cfg.outputs.base_dir) / "job-result.json"

    started_at = utc_now_iso()

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
        uploader.ensure_hf_ready()

        pipeline = cfg.pipeline

        if not pipeline or pipeline.prepare_assets.enabled:
            logger.info("==> preparing assets")
            reporter.report_status(
                "running",
                message="Preparing datasets and remote assets",
                stage="prepare_assets",
                progress=5,
            )
            asset_manager.prepare_dataset(cfg)
            asset_manager.prepare_evaluation_dataset(cfg)
        else:
            logger.info("==> skipping assets preparation")

        # Важно: Unsloth должен импортироваться до transformers/peft.
        from train_runner import run_training

        training_result = {}
        if not pipeline or pipeline.training.enabled:
            logger.info("==> starting training")
            training_result = run_training(cfg, reporter=reporter)

            logical_base_model_id = cfg.model.logical_base_model_id
            if logical_base_model_id and not training_result.get("base_model_id"):
                training_result["base_model_id"] = logical_base_model_id
                training_result["base_model_name_or_path"] = logical_base_model_id
                summary = training_result.get("summary")
                if isinstance(summary, dict):
                    summary.setdefault("base_model_id", logical_base_model_id)
                    summary.setdefault("base_model_name_or_path", logical_base_model_id)

            logger.info("==> training finished")
        else:
            logger.info("==> skipping training stage")
            training_result = {
                "status": "skipped",
                "message": "Training stage disabled in pipeline",
            }

        if cfg.postprocess.run_awq_quantization:
            raise NotImplementedError("AWQ quantization stage is not implemented in this service yet")

        evaluation_result = None
        should_eval = pipeline.evaluation.enabled if pipeline else cfg.evaluation.enabled

        if should_eval:
            from eval_runner import run_evaluation

            logger.info("==> starting evaluation")
            evaluation_result = run_evaluation(cfg, training_result, reporter=reporter)
            logger.info("==> evaluation finished")
        else:
            logger.info("==> skipping evaluation stage")

        result: Dict[str, Any] = {
            "status": "success",
            "job_id": cfg.job_id or cfg.job_name,
            "job_name": cfg.job_name,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "config_source": config_source,
            "training": training_result,
            "evaluation": evaluation_result,
            "artifacts": {
                "log_file": str(log_file),
                "effective_config_path": str(effective_config_path),
                "result_path": str(result_path),
            },
            "uploads": {},
            "upload_errors": {},
        }

        should_upload = pipeline.upload.enabled if pipeline else cfg.upload.enabled

        if should_upload:
            if cfg.upload.target == "url" and cfg.upload.upload_url:
                logger.info("==> legacy full archive upload")
                try:
                    archive_path = Path(cfg.outputs.base_dir) / f"{cfg.job_name}.tar.gz"
                    archiver.make_archive(cfg.outputs.base_dir, str(archive_path))
                    archiver.upload_archive(
                        str(archive_path),
                        cfg.upload.upload_url,
                        headers=uploader._headers(),
                        form_data={
                            "job_id": cfg.job_id or cfg.job_name,
                            "job_name": cfg.job_name,
                            "artifact_type": "full_archive",
                        },
                        timeout_sec=cfg.upload.timeout_sec,
                    )
                    result["uploads"]["legacy_full_archive"] = {
                        "url": cfg.upload.upload_url,
                        "archive_path": str(archive_path),
                    }
                except Exception as exc:
                    logger.exception("legacy full archive upload failed")
                    result["upload_errors"]["legacy_full_archive"] = str(exc)

            if cfg.upload.target == "url":
                reporter.report_status(
                    "running",
                    message="Uploading URL artifacts",
                    stage="upload_artifacts",
                    progress=97,
                )
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
        else:
            logger.info("==> skipping upload stage")

        write_json(result_path, result)

        should_publish = pipeline.publish.enabled if pipeline else cfg.huggingface.enabled

        if should_publish:
            reporter.report_status(
                "running",
                message="Publishing artifacts to Hugging Face",
                stage="publish_artifacts",
                progress=98,
            )

            hf_model_uploads = uploader.upload_to_huggingface(training_result)
            if hf_model_uploads:
                result["uploads"].update(hf_model_uploads)
                write_json(result_path, result)

            hf_metadata_uploads = uploader.upload_hf_metadata(
                log_file=str(log_file),
                effective_config_path=str(effective_config_path),
                result_path=str(result_path),
                training_result=training_result,
                eval_result=evaluation_result,
            )
            if hf_metadata_uploads:
                result["uploads"].update(hf_metadata_uploads)
                write_json(result_path, result)
        else:
            logger.info("==> skipping publish stage")

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