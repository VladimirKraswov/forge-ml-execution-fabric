from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_ALREADY_COMPRESSED_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
}

_LARGE_FILE_THRESHOLD = 512 * 1024 * 1024
_LARGE_TOTAL_THRESHOLD = 2 * 1024 * 1024 * 1024


def build_retry_session(total_retries: int = 3) -> requests.Session:
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        other=total_retries,
        backoff_factor=1.0,
        allowed_methods=None,
        status_forcelist=[408, 425, 429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class Archiver:
    def __init__(self, session: Optional[requests.Session] = None, retries: int = 3):
        self.session = session or build_retry_session(retries)

    def _iter_roots(self, source_path: Path, excluded: set[str]):
        for item in sorted(source_path.iterdir(), key=lambda p: p.name):
            if item.name in excluded:
                continue
            if item.is_file() and item.name.endswith((".tar", ".tar.gz", ".tgz")):
                continue
            yield item

    def _collect_stats(self, source_path: Path, excluded: set[str]) -> Dict[str, int | bool]:
        file_count = 0
        total_bytes = 0
        has_large_file = False
        has_already_compressed = False

        for root in self._iter_roots(source_path, excluded):
            if root.is_file():
                candidates = [root]
            else:
                candidates = [p for p in root.rglob("*") if p.is_file()]

            for file_path in candidates:
                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                file_count += 1
                total_bytes += stat.st_size

                if stat.st_size >= _LARGE_FILE_THRESHOLD:
                    has_large_file = True

                if file_path.suffix.lower() in _ALREADY_COMPRESSED_SUFFIXES:
                    has_already_compressed = True

        return {
            "file_count": file_count,
            "total_bytes": total_bytes,
            "has_large_file": has_large_file,
            "has_already_compressed": has_already_compressed,
        }

    def _should_compress(
        self,
        source_path: Path,
        excluded: set[str],
        compress: Optional[bool],
    ) -> bool:
        if compress is not None:
            return bool(compress)

        stats = self._collect_stats(source_path, excluded)

        if stats["has_already_compressed"]:
            return False
        if stats["has_large_file"]:
            return False
        if int(stats["total_bytes"]) >= _LARGE_TOTAL_THRESHOLD:
            return False

        return True

    @staticmethod
    def _normalize_output_path(output_path: Path, compress: bool) -> Path:
        name = output_path.name

        if compress:
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                return output_path
            if name.endswith(".tar"):
                return output_path.with_name(f"{name}.gz")
            return output_path.with_name(f"{name}.tar.gz")

        if name.endswith(".tar.gz"):
            return output_path.with_name(name[:-3])
        if name.endswith(".tgz"):
            return output_path.with_name(f"{name[:-4]}.tar")
        if name.endswith(".tar"):
            return output_path
        return output_path.with_name(f"{name}.tar")

    def make_archive(
        self,
        source_dir: str,
        output_filename: str,
        exclude_names: Optional[Iterable[str]] = None,
        compress: Optional[bool] = None,
    ) -> str:
        source_path = Path(source_dir).resolve()
        excluded = set(exclude_names or set())
        excluded.update({"downloads", "__pycache__"})

        requested_output_path = Path(output_filename).resolve()

        if requested_output_path.parent == source_path:
            excluded.add(requested_output_path.name)

        use_compression = self._should_compress(source_path, excluded, compress)
        output_path = self._normalize_output_path(requested_output_path, use_compression)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self._collect_stats(source_path, excluded)
        mode = "w:gz" if use_compression else "w"

        logger.info(
            "==> creating archive %s from %s (mode=%s, files=%s, size_bytes=%s)",
            output_path,
            source_path,
            mode,
            stats["file_count"],
            stats["total_bytes"],
        )
        print(f"==> creating archive {output_path} from {source_path}")

        open_kwargs = {"compresslevel": 1} if use_compression else {}
        with tarfile.open(output_path, mode, **open_kwargs) as tar:
            for item in self._iter_roots(source_path, excluded):
                tar.add(item, arcname=item.name)

        logger.info("==> archive ready: %s", output_path)
        return str(output_path)

    def upload_archive(
        self,
        archive_path: str,
        upload_url: str,
        headers: Optional[Dict[str, str]] = None,
        form_data: Optional[Dict[str, str]] = None,
        timeout_sec: int = 120,
    ) -> None:
        archive = Path(archive_path)
        logger.info("==> uploading archive to %s: %s", upload_url, archive)
        print(f"==> uploading archive to {upload_url}")

        content_type = "application/gzip" if archive.name.endswith(".gz") else "application/x-tar"

        with open(archive_path, "rb") as f:
            files = {"file": (archive.name, f, content_type)}
            response = self.session.post(
                upload_url,
                files=files,
                data=form_data or {},
                headers=headers or {},
                timeout=(10, timeout_sec),
            )
            response.raise_for_status()

        logger.info("==> upload successful: %s", archive)
        print("==> upload successful")