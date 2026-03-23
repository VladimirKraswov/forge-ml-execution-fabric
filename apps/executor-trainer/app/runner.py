from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

import torch

from .adapters.hf_utils import try_hf_login
from .adapters.log_streamer import LogStreamer
from .adapters.reporter import Reporter
from .bootstrap.bootstrap_loader import load_remote_job_config
from .bootstrap.config_loader import load_config, load_config_bundle
from .pipeline.archiver import Archiver
from .pipeline.asset_manager import AssetManager
from .pipeline.publish_runner import PublishRunner
from .pipeline.upload_runner import UploadRunner

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


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


def setup_logging(
    logs_dir: str,
    job_id: Optional[str] = None,
    job_name: Optional[str] = None,
    logs_url: Optional[str] = None,
    logs_bearer_token: Optional[str] = None,
) -> Tuple[Path, List[logging.Handler]]:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "trainer.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    handlers: List[logging.Handler] = [
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
    return log_file, handlers


def teardown_logging(handlers: List[logging.Handler]) -> None:
    root = logging.getLogger()
    for handler in handlers:
        try:
            root.removeHandler(handler)
        except Exception:
            pass
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass


def start_heartbeat(reporter: Reporter) -> Tuple[Event, Thread]:
    stop_event = Event()

    def heartbeat_loop():
        while not stop_event.wait(30):
            try:
                reporter.report_status(
                    "running",
                    message="Heartbeat",
                    stage="heartbeat",
                    extra={"heartbeat": True},
                )
            except Exception:
                pass

    thread = Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    return stop_event, thread


def stop_heartbeat(stop_event: Optional[Event], thread: Optional[Thread]) -> None:
    if stop_event:
        stop_event.set()
    if thread:
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass


def resolve_single_remote_config(args) -> Tuple[Any, str, Dict[str, Any]]:
    if args.job_config_url:
        cfg, meta = load_remote_job_config(args.job_config_url)
        return cfg, args.job_config_url, meta

    if args.config:
        cfg = load_config(args.config)
        return cfg, args.config, {}

    raise ValueError("Either --config or --job-config-url must be provided")


def resolve_config_list(args) -> Tuple[List[Any], str]:
    if args.job_config_url:
        cfg, _, _ = resolve_single_remote_config(args)
        return [cfg], args.job_config_url

    if args.config:
        cfgs = load_config_bundle(args.config)
        return cfgs, args.config

    raise ValueError("Either --config or --job-config-url must be provided")


def apply_run_output_paths(cfg) -> Tuple[str, Path]:
    run_name = f"{cfg.job_name}_{utc_compact_timestamp()}"
    base_root = Path(cfg.outputs.base_dir)
    run_root = base_root / run_name

    cfg.outputs.base_dir = str(run_root)
    cfg.outputs.logs_dir = str(run_root / "logs")
    cfg.outputs.lora_dir = str(run_root / "lora")
    cfg.outputs.checkpoints_dir = str(run_root / "checkpoints")
    cfg.outputs.metrics_dir = str(run_root / "metrics")
    cfg.outputs.merged_dir = str(run_root / "merged")
    cfg.outputs.quantized_dir = str(run_root / "quantized")
    cfg.outputs.eval_dir = str(run_root / "evaluation")
    cfg.outputs.downloads_dir = str(run_root / "downloads")

    return run_name, run_root


def cleanup_runtime() -> None:
    try:
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_single_job(cfg, config_source: str, bootstrap_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bootstrap_meta = bootstrap_meta or {}
    cfg = copy.deepcopy(cfg)

    run_name, run_root = apply_run_output_paths(cfg)
    Path(cfg.outputs.base_dir).mkdir(parents=True, exist_ok=True)

    log_file, handlers = setup_logging(
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

    heartbeat_stop, heartbeat_thread = start_heartbeat(reporter)

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
        logger.info("==> run_name: %s", run_name)
        logger.info("==> job_id: %s", cfg.job_id or cfg.job_name)
        logger.info("==> config source: %s", config_source)
        logger.info("==> run output dir: %s", run_root)

        reporter.report_status(
            "running",
            message="Validating Hugging Face access",
            stage="hf_login",
            progress=2,
        )
        try_hf_login()
        publisher.ensure_hf_ready()

        pipeline = cfg.pipeline
        has_steps = bool(pipeline and pipeline.steps)

        if has_steps:
            logger.info("==> executing pipeline steps")
        else:
            logger.warning("==> pipeline.steps missing, falling back to legacy sequence")

        training_result: Dict[str, Any] = {}
        evaluation_result: Optional[Dict[str, Any]] = None
        external_refs: List[Dict[str, Any]] = []

        result: Dict[str, Any] = {
            "status": "success",
            "job_id": cfg.job_id or cfg.job_name,
            "job_name": cfg.job_name,
            "run_name": run_name,
            "started_at": started_at,
            "finished_at": None,
            "config_source": config_source,
            "training": {},
            "evaluation": None,
            "artifacts": {
                "run_root": str(run_root),
                "log_file": str(log_file),
                "effective_config_path": str(effective_config_path),
                "result_path": str(result_path),
            },
            "uploads": {},
            "upload_errors": {},
            "externalRefs": external_refs,
        }

        def run_step(step_key: str, step_kind: str) -> None:
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
        write_json(result_path, result)

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
        return result

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
            "run_name": run_name,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "config_source": config_source,
            "error": error_msg,
            "artifacts": {
                "run_root": str(run_root),
                "log_file": str(log_file),
                "effective_config_path": str(effective_config_path),
                "result_path": str(result_path),
            },
        }
        write_json(result_path, failed_result)

        reporter.report_error(error_msg, logs=logs_tail)
        reporter.report_final(failed_result, status="failed")
        return failed_result

    finally:
        stop_heartbeat(heartbeat_stop, heartbeat_thread)
        try:
            reporter.close()
        except Exception:
            pass
        teardown_logging(handlers)
        cleanup_runtime()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--job-config-url", required=False)
    args = parser.parse_args()

    try:
        if args.job_config_url:
            cfg, config_source, bootstrap_meta = resolve_single_remote_config(args)
            results = [run_single_job(cfg, config_source, bootstrap_meta)]
            summary_root = Path(cfg.outputs.base_dir).parent if Path(cfg.outputs.base_dir).name else Path("/output")
        else:
            cfgs, config_source = resolve_config_list(args)
            if not cfgs:
                raise ValueError("No jobs found in config bundle")

            summary_root = Path(cfgs[0].outputs.base_dir)
            results = []
            for index, cfg in enumerate(cfgs, start=1):
                print(f"==> batch job {index}/{len(cfgs)}: {cfg.job_name}")
                results.append(run_single_job(cfg, config_source, {}))
    except Exception as exc:
        print(f"ERROR: failed to execute jobs: {exc}")
        sys.exit(1)

    batch_finished_at = utc_now_iso()
    batch_started_at = min((r.get("started_at") for r in results if r.get("started_at")), default=batch_finished_at)
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") != "success")

    batch_summary = {
        "status": "success" if failed_count == 0 else "partial_failed",
        "started_at": batch_started_at,
        "finished_at": batch_finished_at,
        "total_jobs": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results,
    }

    summary_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = summary_root / f"batch-summary_{utc_compact_timestamp()}.json"
    write_json(batch_summary_path, batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()