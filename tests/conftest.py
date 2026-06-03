"""Shared pytest fixtures.

The most important fixture here is an autouse network guard: unit tests must
NEVER touch the real network. All grounding/verification/resolution network
calls go through ``httpx`` (via ``oracle.tools``), so any real outbound socket
during a test indicates a code path that should have been dependency-injected
or mocked. We block non-loopback socket connections to make that failure loud
and deterministic instead of flaky/slow.
"""

from __future__ import annotations

import socket

import pytest


class _NetworkBlocked(RuntimeError):
    """Raised when a unit test attempts a real outbound network connection."""


_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _ban_network(monkeypatch):
    """Fail any test that opens a non-loopback socket connection."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = None
        if isinstance(address, tuple) and address:
            host = address[0]
        if host not in _ALLOWED_HOSTS:
            raise _NetworkBlocked(
                f"Real network access is banned in unit tests (attempted {address!r}). "
                "Inject a mock provider/grounding/verifier instead."
            )
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
