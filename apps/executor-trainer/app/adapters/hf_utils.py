from __future__ import annotations

import os
from typing import Optional

from huggingface_hub import HfApi


def get_hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def build_hf_api() -> HfApi:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF upload requested but HF_TOKEN is not set")
    return HfApi(token=token)


def validate_hf_token() -> dict:
    api = build_hf_api()
    return api.whoami()


def try_hf_login() -> bool:
    token = get_hf_token()
    if not token:
        print("==> HF_TOKEN not set, skip HF auth check")
        return False

    info = HfApi(token=token).whoami()
    name = info.get("name") or info.get("fullname") or "unknown"
    print(f"==> Hugging Face auth success: {name}")
    return True