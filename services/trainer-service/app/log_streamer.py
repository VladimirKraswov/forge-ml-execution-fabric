from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Dict, Optional, Tuple

import requests


def _safe_timeout(timeout_sec: int | float | None, fallback: float = 3.0) -> Tuple[float, float]:
    try:
        value = float(timeout_sec or fallback)
    except Exception:
        value = fallback

    read_timeout = max(1.0, min(value, 5.0))
    connect_timeout = 2.0
    return connect_timeout, read_timeout


class LogStreamer(logging.Handler):
    def __init__(
        self,
        logs_url: str,
        job_id: str,
        job_name: str,
        bearer_token: Optional[str] = None,
        timeout_sec: int = 5,
    ):
        super().__init__()
        self.logs_url = str(logs_url or "").strip()
        self.job_id = job_id
        self.job_name = job_name
        self.timeout_sec = timeout_sec

        self.session = requests.Session()

        self.offset_lock = threading.Lock()
        self.offset = 0

        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if bearer_token:
            self.headers["Authorization"] = f"Bearer {bearer_token}"

        self._queue: queue.Queue[dict] = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"log-streamer-{self.job_id}",
            daemon=True,
        )
        self._worker.start()

    def _deliver(self, payload: dict) -> None:
        response = self.session.post(
            self.logs_url,
            json=payload,
            headers=self.headers,
            timeout=_safe_timeout(self.timeout_sec, 3.0),
        )
        response.raise_for_status()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._deliver(payload)
            except Exception as exc:
                try:
                    sys.stderr.write(f"[log-streamer] failed to send log chunk: {exc}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if not message:
                return

            chunk = f"{message}\n"
            encoded = chunk.encode("utf-8", errors="replace")

            with self.offset_lock:
                current_offset = self.offset
                self.offset += len(encoded)

            payload = {
                "job_id": self.job_id,
                "job_name": self.job_name,
                "offset": current_offset,
                "chunk": chunk,
            }

            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                try:
                    sys.stderr.write("[log-streamer] queue is full, dropping log chunk\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        except Exception as exc:
            try:
                sys.stderr.write(f"[log-streamer] emit failed: {exc}\n")
                sys.stderr.flush()
            except Exception:
                pass

    def flush(self, timeout_sec: float = 3.0) -> None:
        deadline = time.time() + max(0.0, timeout_sec)
        while time.time() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.05)

    def close(self) -> None:
        try:
            self.flush(timeout_sec=2.0)
            self._stop_event.set()
            self._worker.join(timeout=2.0)
        finally:
            try:
                self.session.close()
            except Exception:
                pass
            super().close()