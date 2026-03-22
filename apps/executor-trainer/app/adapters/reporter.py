from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Tuple

import requests

from ..bootstrap.schemas import CallbackConfig, JobConfig

logger = logging.getLogger(__name__)


def _safe_timeout(timeout_sec: int | float | None, fallback: float = 3.0) -> Tuple[float, float]:
    try:
        value = float(timeout_sec or fallback)
    except Exception:
        value = fallback

    read_timeout = max(1.0, min(value, 5.0))
    connect_timeout = 2.0
    return connect_timeout, read_timeout


class Reporter:
    def __init__(self, cfg: JobConfig):
        self.cfg = cfg
        self.job_name = cfg.job_name
        self.job_id = cfg.job_id or cfg.job_name

        self.session = requests.Session()

        self._queue: queue.Queue[Tuple[str, CallbackConfig, Dict[str, Any]]] = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"reporter-{self.job_id}",
            daemon=True,
        )
        self._worker.start()

    def _build_headers(self, callback: CallbackConfig) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if callback.auth.headers:
            headers.update(callback.auth.headers)
        if callback.auth.bearer_token:
            headers["Authorization"] = f"Bearer {callback.auth.bearer_token}"
        return headers

    def _callbacks_for(self, event_kind: str) -> List[CallbackConfig]:
        callbacks: List[CallbackConfig] = []

        if event_kind == "status" and self.cfg.reporting.status.active:
            callbacks.append(self.cfg.reporting.status)
        elif event_kind == "progress" and self.cfg.reporting.progress.active:
            callbacks.append(self.cfg.reporting.progress)
        elif event_kind == "final" and self.cfg.reporting.final.active:
            callbacks.append(self.cfg.reporting.final)

        if not callbacks and self.cfg.report_url:
            callbacks.append(
                CallbackConfig(
                    enabled=True,
                    url=self.cfg.report_url,
                    timeout_sec=5,
                )
            )

        return callbacks

    def _deliver(self, event_kind: str, callback: CallbackConfig, payload: Dict[str, Any]) -> None:
        logger.info("==> reporting %s to %s", event_kind, callback.url)
        response = self.session.post(
            callback.url,
            json=payload,
            headers=self._build_headers(callback),
            timeout=_safe_timeout(callback.timeout_sec, 3.0),
        )
        response.raise_for_status()

    def _enqueue(self, event_kind: str, callback: CallbackConfig, payload: Dict[str, Any]) -> None:
        item = (event_kind, callback, copy.deepcopy(payload))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("reporter queue is full, dropping %s callback to %s", event_kind, callback.url)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                event_kind, callback, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._deliver(event_kind, callback, payload)
            except Exception as exc:
                logger.warning("failed to report %s: %s", event_kind, exc)
            finally:
                self._queue.task_done()

    def _emit(
        self,
        event_kind: str,
        payload: Dict[str, Any],
        *,
        synchronous: bool = False,
    ) -> None:
        callbacks = self._callbacks_for(event_kind)
        if not callbacks:
            return

        payload = {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "event": event_kind,
            "timestamp": time.time(),
            **payload,
        }

        for callback in callbacks:
            if synchronous:
                try:
                    self._deliver(event_kind, callback, payload)
                except Exception as exc:
                    logger.warning("failed to report %s: %s", event_kind, exc)
            else:
                self._enqueue(event_kind, callback, payload)

    def report_status(
        self,
        status: str,
        message: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        extra: Dict[str, Any] | None = None,
        logs: str | None = None,
    ) -> None:
        self._emit(
            "status",
            {
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
                "logs": logs,
                "extra": extra or {},
            },
            synchronous=False,
        )

    def report_progress(
        self,
        stage: str,
        progress: float | None = None,
        message: str | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "progress",
            {
                "status": "running",
                "stage": stage,
                "progress": progress,
                "message": message,
                "extra": extra or {},
            },
            synchronous=False,
        )

    def report_final(self, result: Dict[str, Any], status: str = "finished") -> None:
        self._emit(
            "final",
            {
                "status": status,
                "result": result,
            },
            synchronous=True,
        )

    def report_error(
        self,
        error_message: str,
        logs: str | None = None,
        stage: str = "failed",
        extra: Dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "status",
            {
                "status": "failed",
                "stage": stage,
                "progress": 100,
                "message": error_message,
                "logs": logs,
                "extra": extra or {},
            },
            synchronous=True,
        )

    def flush(self, timeout_sec: float = 5.0) -> None:
        deadline = time.time() + max(0.0, timeout_sec)
        while time.time() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.05)

    def close(self) -> None:
        try:
            self.flush(timeout_sec=3.0)
            self._stop_event.set()
            self._worker.join(timeout=2.0)
        finally:
            try:
                self.session.close()
            except Exception:
                pass