from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests

from ..bootstrap.schemas import JobConfig


class AssetManager:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg
        self.download_dir = Path(cfg.outputs.downloads_dir)

    def _infer_suffix(self, url: str, default_suffix: str = ".json") -> str:
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix
        return suffix or default_suffix

    def download_file(self, url: str, dest_path: Path) -> Path:
        print(f"==> downloading {url} to {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with dest_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return dest_path

    def prepare_dataset(self, cfg: JobConfig) -> None:
        if cfg.dataset.source != "url":
            return

        dataset_dir = self.download_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if cfg.dataset.train_url:
            train_suffix = self._infer_suffix(cfg.dataset.train_url, ".json")
            train_path = dataset_dir / f"train{train_suffix}"
            self.download_file(cfg.dataset.train_url, train_path)
            cfg.dataset.train_path = str(train_path)

        if cfg.dataset.val_url:
            val_suffix = self._infer_suffix(cfg.dataset.val_url, ".json")
            val_path = dataset_dir / f"val{val_suffix}"
            self.download_file(cfg.dataset.val_url, val_path)
            cfg.dataset.val_path = str(val_path)

        print("==> dataset prepared")

    def prepare_evaluation_dataset(self, cfg: JobConfig) -> None:
        if not cfg.evaluation.enabled or not cfg.evaluation.dataset:
            return

        eval_cfg = cfg.evaluation.dataset
        if eval_cfg.source != "url" or not eval_cfg.url:
            return

        eval_dir = self.download_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_suffix = self._infer_suffix(
            eval_cfg.url,
            ".jsonl" if eval_cfg.format == "jsonl" else ".json",
        )
        eval_path = eval_dir / f"evaluation{eval_suffix}"

        self.download_file(eval_cfg.url, eval_path)
        eval_cfg.path = str(eval_path)

        print("==> evaluation dataset prepared")