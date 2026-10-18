"""``decimalai.flush()`` exports what the OTel pipeline ``otel.instrument()`` built still
holds, bounded, so a framework traced only through OTel has a trace to send."""
from __future__ import annotations

import time

import decimalai
from decimalai import otel


class _Provider:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls = []
        self.delay = delay

    def force_flush(self, timeout_millis: int) -> bool:
        time.sleep(self.delay)
        self.calls.append(timeout_millis)
        return True


def test_flush_exports_the_installed_providers(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(otel, "_installed_providers", [provider])
    decimalai._flush_otel_pipelines()
    assert len(provider.calls) == 1


def test_a_slow_export_is_bounded(monkeypatch):
    provider = _Provider(delay=2.0)
    monkeypatch.setattr(otel, "_installed_providers", [provider])
    started = time.monotonic()
    assert otel.flush_installed(timeout_s=0.2) is False
    assert time.monotonic() - started < 1.5


def test_nothing_installed_is_a_no_op(monkeypatch):
    monkeypatch.setattr(otel, "_installed_providers", [])
    assert otel.flush_installed() is True
