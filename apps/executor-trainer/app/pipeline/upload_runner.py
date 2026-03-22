import logging
from pathlib import Path
from typing import Dict, tuple
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

    def upload_non_summary_artifacts(self, log_file, effective_config_path, training_result, eval_result=None):
        uploaded = {}
        errors = {}
        targets = self.cfg.upload.url_targets

        # logic to upload artifacts
        return uploaded, errors
