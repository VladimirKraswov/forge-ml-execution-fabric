import logging
from pathlib import Path
from typing import Any, Dict, Optional
from ..adapters.hf_utils import build_hf_api, get_hf_token
from ..bootstrap.schemas import JobConfig

logger = logging.getLogger(__name__)

class PublishRunner:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg

    def ensure_hf_ready(self):
        if not get_hf_token():
             pass

    def upload_to_huggingface(self, training_result: Dict) -> Dict[str, Dict[str, str]]:
        return {}

    def upload_hf_metadata(self, **kwargs) -> Dict:
        return {}
