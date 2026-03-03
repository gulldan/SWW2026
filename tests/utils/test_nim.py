"""Tests for shared NIM health-check utilities."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from dualdocking.utils.nim import nim_url_ready


class _HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler with configurable readiness payload."""

    ready_payload: bytes = b"true"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/health/ready":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(self.ready_payload)
            return
        self.send_error(404)

    def log_message(self, *_args: object) -> None:
        """Suppress test server logs."""


def test_nim_url_ready_true_response() -> None:
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    _HealthHandler.ready_payload = b"true"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert nim_url_ready(url) is True
    finally:
        server.shutdown()


def test_nim_url_ready_false_response() -> None:
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    _HealthHandler.ready_payload = b"false"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert nim_url_ready(url) is False
    finally:
        server.shutdown()
