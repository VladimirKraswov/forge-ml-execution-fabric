import logging
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from .archiver import Archiver
from ..bootstrap.schemas import JobConfig

logger = logging.getLogger(__name__)

class UploadRunner:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg
        self.archiver = Archiver()
        self.session = self.archiver.session

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.cfg.upload.auth.headers:
            headers.update(self.cfg.upload.auth.headers)
        if self.cfg.upload.auth.bearer_token:
            headers["Authorization"] = f"Bearer {self.cfg.upload.auth.bearer_token}"
        return headers

    def _upload_file(self, file_path: str, upload_url: str, artifact_type: str) -> Dict[str, str]:
        path = Path(file_path)
        if not upload_url or not path.exists():
            return {}

        with path.open("rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            data = {
                "job_id": self.cfg.job_id or self.cfg.job_name,
                "job_name": self.cfg.job_name,
                "artifact_type": artifact_type,
            }
            response = self.session.post(
                upload_url,
                files=files,
                data=data,
                headers=self._headers(),
                timeout=(10, self.cfg.upload.timeout_sec),
            )
            response.raise_for_status()

        return {"url": upload_url, "path": str(path)}

    def _archive_and_upload_dir(
        self,
        dir_path: str,
        upload_url: str,
        archive_name: str,
        artifact_type: str,
    ) -> Dict[str, str]:
        path = Path(dir_path)
        if not upload_url or not path.exists():
            return {}

        archive_path = Path(self.cfg.outputs.base_dir) / archive_name
        self.archiver.make_archive(
            str(path),
            str(archive_path),
            exclude_names={"downloads", "__pycache__"},
        )
        self.archiver.upload_archive(
            str(archive_path),
            upload_url,
            headers=self._headers(),
            form_data={
                "job_id": self.cfg.job_id or self.cfg.job_name,
                "job_name": self.cfg.job_name,
                "artifact_type": artifact_type,
            },
            timeout_sec=self.cfg.upload.timeout_sec,
        )
        return {"url": upload_url, "archive_path": str(archive_path)}

    def _safe_upload(
        self,
        key: str,
        operation: Callable[[], Dict[str, str]],
        uploaded: Dict[str, Dict[str, str]],
        errors: Dict[str, str],
    ) -> None:
        try:
            result = operation()
            if result:
                uploaded[key] = result
        except Exception as exc:
            logger.exception("upload step failed: %s", key)
            errors[key] = str(exc)

    def upload_non_summary_artifacts(
        self,
        log_file: str,
        effective_config_path: str,
        training_result: Dict,
        eval_result: Dict = None,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        uploaded: Dict[str, Dict[str, str]] = {}
        errors: Dict[str, str] = {}

        if self.cfg.upload.enabled and self.cfg.upload.target == "url":
            targets = self.cfg.upload.url_targets

            if targets.logs_url:
                self._safe_upload(
                    "logs",
                    lambda: self._upload_file(log_file, targets.logs_url, "logs"),
                    uploaded,
                    errors,
                )

            if targets.effective_config_url:
                self._safe_upload(
                    "effective_config",
                    lambda: self._upload_file(
                        effective_config_path,
                        targets.effective_config_url,
                        "config",
                    ),
                    uploaded,
                    errors,
                )

            metrics_path = training_result.get("metrics_path")
            if targets.train_metrics_url and metrics_path:
                self._safe_upload(
                    "train_metrics",
                    lambda: self._upload_file(
                        metrics_path,
                        targets.train_metrics_url,
                        "train_metrics",
                    ),
                    uploaded,
                    errors,
                )

            history_path = training_result.get("history_path")
            if targets.train_history_url and history_path:
                self._safe_upload(
                    "train_history",
                    lambda: self._upload_file(
                        history_path,
                        targets.train_history_url,
                        "train_history",
                    ),
                    uploaded,
                    errors,
                )

            lora_dir = training_result.get("lora_dir")
            if targets.lora_archive_url and lora_dir:
                self._safe_upload(
                    "lora_archive",
                    lambda: self._archive_and_upload_dir(
                        lora_dir,
                        targets.lora_archive_url,
                        f"{self.cfg.job_name}.lora.tar.gz",
                        "lora_archive",
                    ),
                    uploaded,
                    errors,
                )

            merged_dir = training_result.get("merged_dir")
            if targets.merged_archive_url and merged_dir and Path(merged_dir).exists():
                self._safe_upload(
                    "merged_archive",
                    lambda: self._archive_and_upload_dir(
                        merged_dir,
                        targets.merged_archive_url,
                        f"{self.cfg.job_name}.merged.tar.gz",
                        "merged_archive",
                    ),
                    uploaded,
                    errors,
                )

            if targets.full_archive_url:
                self._safe_upload(
                    "full_archive",
                    lambda: self._archive_and_upload_dir(
                        self.cfg.outputs.base_dir,
                        targets.full_archive_url,
                        f"{self.cfg.job_name}.full.tar.gz",
                        "full_archive",
                    ),
                    uploaded,
                    errors,
                )

            if eval_result:
                if targets.eval_summary_url and eval_result.get("summary_json_path"):
                    self._safe_upload(
                        "eval_summary",
                        lambda: self._upload_file(
                            eval_result["summary_json_path"],
                            targets.eval_summary_url,
                            "evaluation_summary",
                        ),
                        uploaded,
                        errors,
                    )

                if targets.eval_details_url and eval_result.get("detailed_csv_path"):
                    self._safe_upload(
                        "eval_details",
                        lambda: self._upload_file(
                            eval_result["detailed_csv_path"],
                            targets.eval_details_url,
                            "evaluation_details",
                        ),
                        uploaded,
                        errors,
                    )

        return uploaded, errors

    def upload_summary(self, summary_path: str) -> Dict[str, Dict[str, str]]:
        uploaded: Dict[str, Dict[str, str]] = {}
        if (
            self.cfg.upload.enabled
            and self.cfg.upload.target == "url"
            and self.cfg.upload.url_targets.summary_url
        ):
            uploaded["summary"] = self._upload_file(
                summary_path,
                self.cfg.upload.url_targets.summary_url,
                "job_summary",
            )
        return uploaded