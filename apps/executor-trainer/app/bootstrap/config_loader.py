import copy
import json
from pathlib import Path
from typing import Any, Dict

import requests

from ..bootstrap.schemas import JobConfig


def _migrate_legacy_config(raw: Dict[str, Any], source_ref: str) -> Dict[str, Any]:
    data = copy.deepcopy(raw)

    if "base_model" in data and "model" not in data:
        base_model = data.get("base_model") or {}
        model_source = "local" if base_model.get("local_path") else "huggingface"
        data["model"] = {
            "source": model_source,
            "local_path": base_model.get("local_path"),
            "repo_id": base_model.get("repo_id"),
            "revision": base_model.get("revision", "main"),
            "trust_remote_code": base_model.get("trust_remote_code", False),
            "load_in_4bit": base_model.get("load_in_4bit"),
            "dtype": base_model.get("dtype", "auto"),
            "max_seq_length": base_model.get("max_seq_length", 4096),
        }

    if "artifacts" in data and "outputs" not in data:
        artifacts = data.get("artifacts") or {}
        training = data.get("training") or {}

        base_dir = artifacts.get("output_dir") or "/output"
        lora_subdir = artifacts.get("lora_subdir", "lora")
        merged_subdir = artifacts.get("merged_subdir", "merged")

        data["outputs"] = {
            "base_dir": base_dir,
            "logs_dir": f"{base_dir}/logs",
            "lora_dir": f"{base_dir}/{lora_subdir}",
            "checkpoints_dir": training.get("output_dir") or f"{base_dir}/checkpoints",
            "metrics_dir": f"{base_dir}/metrics",
            "merged_dir": f"{base_dir}/{merged_subdir}",
            "quantized_dir": f"{base_dir}/quantized",
            "eval_dir": f"{base_dir}/evaluation",
            "downloads_dir": f"{base_dir}/downloads",
        }

        postprocess = data.setdefault("postprocess", {})
        if "save_merged_16bit" in artifacts and "save_merged_16bit" not in postprocess:
            postprocess["save_merged_16bit"] = artifacts["save_merged_16bit"]
        if artifacts.get("save_merged_16bit") and "merge_lora" not in postprocess:
            postprocess["merge_lora"] = True

    if "report_url" in data and "reporting" not in data:
        report_url = data.get("report_url")
        if report_url:
            data["reporting"] = {
                "status": {"enabled": True, "url": report_url},
                "progress": {"enabled": True, "url": report_url},
                "final": {"enabled": True, "url": report_url},
            }

    if "mode" not in data:
        if source_ref.startswith("http://") or source_ref.startswith("https://"):
            data["mode"] = "remote"
        else:
            data["mode"] = "local"

    return data


def load_config(path: str) -> JobConfig:
    if path.startswith("http://") or path.startswith("https://"):
        print(f"==> loading remote config from {path}")
        response = requests.get(path, timeout=30)
        response.raise_for_status()
        raw = response.json()
    else:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

    migrated = _migrate_legacy_config(raw, path)
    return JobConfig.model_validate(migrated)