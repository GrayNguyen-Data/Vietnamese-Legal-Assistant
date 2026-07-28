"""Monitoring hooks — Buổi 7, Section 4 (LangFuse).

Tối thiểu: 1 trace cho mỗi lần gọi pipeline.answer*(), gắn question/answer/
latency/lỗi. KHÔNG bọc qua LangChain — dùng langfuse SDK trực tiếp.

Mặc định tắt (MONITORING_ENABLED=false) nên khi chưa điền LANGFUSE_* trong
.env, toàn bộ hàm ở đây là no-op — không ai bắt buộc phải cài/kích hoạt
LangFuse để chạy phần còn lại của codebase.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import settings


def _get_langfuse():
    """Lazy import + lazy client — tránh phụ thuộc cứng vào package `langfuse`
    khi MONITORING_ENABLED=false, và tránh tạo client ở import-time."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


@contextmanager
def trace_answer(name: str, question: str, metadata: dict[str, Any] | None = None):
    """Bọc quanh 1 lần gọi pipeline (answer/answer_stream/answer_structured).

    Dùng như:
        with trace_answer("answer", question) as t:
            result = ...
            t["output"] = result
    """
    if not settings.monitoring_enabled:
        yield {}
        return

    langfuse = _get_langfuse()
    start = time.perf_counter()
    span = langfuse.start_observation(name=name, input=question, metadata=metadata or {})
    box: dict[str, Any] = {}
    try:
        yield box
    except Exception as exc:
        span.update(level="ERROR", status_message=str(exc))
        raise
    finally:
        span.update(output=box.get("output"), metadata={"latency_s": time.perf_counter() - start})
        span.end()
        langfuse.flush()


def trace_stream(name: str, question: str, tokens: Iterator[str]) -> Iterator[str]:
    """Bọc quanh answer_stream(): trace toàn bộ output ghép lại sau khi stream
    kết thúc (không thể tạo span "giữa chừng" cho streaming)."""
    if not settings.monitoring_enabled:
        yield from tokens
        return

    langfuse = _get_langfuse()
    start = time.perf_counter()
    chunks: list[str] = []
    try:
        for token in tokens:
            chunks.append(token)
            yield token
    finally:
        span = langfuse.start_observation(
            name=name,
            input=question,
            output="".join(chunks),
            metadata={"latency_s": time.perf_counter() - start, "streamed": True},
        )
        span.end()
        langfuse.flush()
