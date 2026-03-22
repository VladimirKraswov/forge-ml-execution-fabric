from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from ..pipeline.archiver import build_retry_session
from ..bootstrap.schemas import JobConfig


def _resolve_callback_url(payload: Dict[str, Any], key: str) -> str | None:
    explicit = payload.get(f"{key}_url")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    reporting = payload.get("reporting") or {}
    value = reporting.get(key)

    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()

    return None


def load_remote_job_config(
    job_config_url: str,
    timeout_sec: int = 30,
) -> Tuple[JobConfig, Dict[str, Any]]:
    resolved_url = str(job_config_url or "").strip()
    if not resolved_url:
        raise ValueError("JOB_CONFIG_URL is empty")

    session = build_retry_session(total_retries=3)

    print(f"==> loading remote job config from {resolved_url}")
    response = session.get(resolved_url, timeout=(10, timeout_sec))
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Remote job config response must be a JSON object")

    callback_auth_token = payload.get("callback_auth_token")

    if "config" in payload:
        raw_config = copy.deepcopy(payload.get("config") or {})
        if not raw_config.get("job_id") and payload.get("job_id"):
            raw_config["job_id"] = payload["job_id"]
        if not raw_config.get("job_name") and payload.get("job_name"):
            raw_config["job_name"] = payload["job_name"]
    else:
        raw_config = copy.deepcopy(payload)

    if not isinstance(raw_config, dict):
        raise ValueError("Remote config payload is invalid")

    raw_config["mode"] = "remote"

    reporting = raw_config.setdefault("reporting", {})
    for key in ("status", "progress", "final", "logs"):
        callback_cfg = copy.deepcopy(reporting.get(key) or {})
        explicit_url = _resolve_callback_url(payload, key)

        if explicit_url:
            callback_cfg["enabled"] = True
            callback_cfg["url"] = explicit_url

        auth_cfg = copy.deepcopy(callback_cfg.get("auth") or {})
        if callback_auth_token and not auth_cfg.get("bearer_token"):
            auth_cfg["bearer_token"] = callback_auth_token

        callback_cfg["auth"] = auth_cfg

        if callback_cfg.get("url"):
            callback_cfg["enabled"] = bool(callback_cfg.get("enabled", True))

        reporting[key] = callback_cfg

    raw_config["reporting"] = reporting

    cfg = JobConfig.model_validate(raw_config)

    logs_url = (
        _resolve_callback_url(payload, "logs")
        or _resolve_callback_url(raw_config, "logs")
    )

    meta = {
        "job_config_url": resolved_url,
        "logs_url": logs_url,
        "callback_auth_token": callback_auth_token,
    }
    return cfg, meta