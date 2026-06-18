#!/usr/bin/env python3
"""HTTP sidecar for the OpenAI Privacy Filter (opf).

Keeps the OPF model resident in memory and exposes a small JSON API so that an
external process (e.g. the model proxy) can redact PII out of text and later
restore it.

Endpoints
---------
GET  /health
    -> {"status": "ok", "device": "...", "output_mode": "...", "model_loaded": true}

POST /redact
    body {"texts": ["...", "..."]}
    -> {
         "redacted": ["...", "..."],   # same order as input, PII swapped for sentinels
         "mapping": {"\u27e6PII:0\u27e7": "alice@x.com", ...},  # sentinel -> original
         "span_count": <int>
       }

Sentinels are unique across the whole batch (not just per text), so the returned
mapping is self-consistent for one request. Placeholders from opf alone are NOT
reversible (two emails both render as <PRIVATE_EMAIL>), which is why we assign a
unique sentinel per detected span here.

Usage
-----
    source ~/dev/ai/bin/activate
    OPF_MOE_TRITON=0 python serve.py --device cpu --port 8799
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time as time_module
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Sentinel brackets are rare unicode so they are extremely unlikely to occur in
# real input and survive JSON round-trips. Format: ⟦PII:<n>⟧
_SENTINEL_OPEN = "\u27e6PII:"
_SENTINEL_CLOSE = "\u27e7"


def _make_sentinel(index: int) -> str:
    return f"{_SENTINEL_OPEN}{index}{_SENTINEL_CLOSE}"


def _resolve_device_candidates(requested: str) -> list[str]:
    """Return an ordered list of devices to attempt, always ending in cpu.

    ``requested == "auto"`` probes for mps (Apple Silicon) then cuda via torch,
    falling back to cpu. An explicit device is tried first, then cpu as a
    fallback if that device fails to load.
    """
    import torch

    if requested != "auto":
        candidates = [requested]
        if requested != "cpu":
            candidates.append("cpu")
        return candidates

    candidates = []
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        candidates.append("mps")
    if torch.cuda.is_available():
        candidates.append("cuda")
    candidates.append("cpu")
    return candidates


class _Redactor:
    """Thread-safe wrapper around a single resident OPF instance."""

    def __init__(self, *, device: str, checkpoint: str | None, output_mode: str) -> None:
        from opf._api import OPF

        self._output_mode = output_mode
        # OPF inference is not guaranteed thread-safe; serialize calls.
        self._lock = threading.Lock()

        last_error: Exception | None = None
        for candidate in _resolve_device_candidates(device):
            try:
                print(f"Trying device '{candidate}'...", flush=True)
                # output_text_only=False so we get structured detected_spans back.
                opf = OPF(
                    model=checkpoint,
                    device=candidate,  # type: ignore[arg-type]
                    output_mode=output_mode,  # type: ignore[arg-type]
                    output_text_only=False,
                )
                # Warm the model so the first real request isn't slow / racy.
                # This also exercises the device so an unusable backend fails here.
                opf.redact("warmup")
                self._opf = opf
                self._device = candidate
                print(f"Loaded OPF on device '{candidate}'.", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - try next device
                last_error = exc
                print(f"Device '{candidate}' failed: {exc}", flush=True)

        raise RuntimeError(
            f"Could not load OPF on any device ({device}); last error: {last_error}"
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def output_mode(self) -> str:
        return self._output_mode

    def redact_batch(self, texts: list[str]) -> dict[str, Any]:
        """Redact a batch of texts, returning sentinel-substituted text + mapping.

        Sentinel indices are unique across the entire batch.
        """
        from opf._api import RedactionResult

        redacted: list[str] = []
        mapping: dict[str, str] = {}
        next_index = 0
        span_count = 0

        with self._lock:
            for text in texts:
                if not isinstance(text, str) or not text:
                    redacted.append(text if isinstance(text, str) else "")
                    continue
                result = self._opf.redact(text)
                if not isinstance(result, RedactionResult):
                    # output_text_only must stay False; defensive guard.
                    redacted.append(text)
                    continue
                spans = sorted(result.detected_spans, key=lambda s: s.start)
                pieces: list[str] = []
                cursor = 0
                for span in spans:
                    if span.start < cursor or span.end <= span.start:
                        # overlapping / empty span — skip to keep output coherent
                        continue
                    sentinel = _make_sentinel(next_index)
                    next_index += 1
                    span_count += 1
                    mapping[sentinel] = text[span.start : span.end]
                    pieces.append(text[cursor : span.start])
                    pieces.append(sentinel)
                    cursor = span.end
                pieces.append(text[cursor:])
                redacted.append("".join(pieces))

        return {"redacted": redacted, "mapping": mapping, "span_count": span_count}


def _build_handler(redactor: _Redactor, timeout_s: float | None) -> type[BaseHTTPRequestHandler]:
    # Single worker so inference stays serialized; the future lets us bound how
    # long a request waits. A stuck inference cannot be killed (Python threads
    # are not cancellable), but the client still gets a prompt 504 instead of
    # hanging until its own socket timeout.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opf-redact")

    class Handler(BaseHTTPRequestHandler):
        # Silence default per-request stderr logging; keep it minimal.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
            pass

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "device": redactor.device,
                        "output_mode": redactor.output_mode,
                        "model_loaded": True,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/redact":
                self._send_json(404, {"error": "not found"})
                print(f"POST {self.path}  status=404  error=not found", flush=True)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                print(f"POST /redact  status=400  error=invalid Content-Length", flush=True)
                return
            raw = self.rfile.read(length) if length > 0 else b""
            input_len = len(raw)
            try:
                parsed = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"invalid JSON: {exc}"})
                print(f"POST /redact  status=400  input={input_len}B  error=invalid JSON", flush=True)
                return
            texts = parsed.get("texts")
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                self._send_json(400, {"error": "body must be {\"texts\": [string, ...]}"})
                print(f"POST /redact  status=400  input={input_len}B  error=invalid body", flush=True)
                return
            future = executor.submit(redactor.redact_batch, texts)
            try:
                result = future.result(timeout=timeout_s)
            except FutureTimeoutError:
                future.cancel()
                self._send_json(
                    504,
                    {"error": f"redaction timed out after {timeout_s:g}s"},
                )
                print(f"POST /redact  status=504  input={input_len}B  error=timeout", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - surface as 500 to caller
                self._send_json(500, {"error": f"redaction failed: {exc}"})
                print(f"POST /redact  status=500  input={input_len}B  error={exc}", flush=True)
                return
            self._send_json(200, result)
            response_body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            output_len = len(response_body)
            print(
                f"POST /redact  status=200  input={input_len}B  output={output_len}B  "
                f"texts={len(texts)}  spans={result.get('span_count', 0)}",
                flush=True,
            )

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OPF privacy-filter HTTP sidecar.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8799, help="Bind port (default 8799).")
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Inference device for OPF. 'auto' (default) tries mps then cuda then cpu. "
            "An explicit value (cpu/cuda/mps) is tried first, falling back to cpu."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint dir; defaults to OPF_CHECKPOINT or ~/.opf/privacy_filter.",
    )
    parser.add_argument(
        "--output-mode",
        default="typed",
        choices=("typed", "redacted"),
        help="OPF output mode (default typed).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request redaction timeout in seconds (default 30). 0 disables it.",
    )
    args = parser.parse_args(argv)

    print(f"Loading OPF model (device={args.device}, output_mode={args.output_mode})...", flush=True)
    redactor: _Redactor = _Redactor(
        device=args.device,
        checkpoint=args.checkpoint,
        output_mode=args.output_mode,
    )
    timeout_s = args.timeout if args.timeout and args.timeout > 0 else None
    handler = _build_handler(redactor, timeout_s)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"OPF sidecar listening on http://{args.host}:{args.port} "
        f"(request timeout: {f'{timeout_s:g}s' if timeout_s else 'disabled'})",
        flush=True,
    )

    def _warmup() -> None:
        try:
            body = json.dumps({"texts": ["test@abc.com, Street No.123, LA"]}).encode("utf-8")
            req = urllib.request.Request(
                f"http://{args.host}:{args.port}/redact",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time_module.perf_counter()
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
            elapsed = time_module.perf_counter() - t0
            output = json.dumps(result, ensure_ascii=False)
            print(
                f"Warmup OK  input={len(body)}B  output={len(output)}B  "
                f"spans={result.get('span_count', 0)}  elapsed={elapsed:.3f}s",
                flush=True,
            )
            print(f"Warmup response: {output}", flush=True)
        except Exception as exc:
            print(f"Warmup failed: {exc}", flush=True)

    threading.Thread(target=_warmup, name="opf-warmup", daemon=True).start()
    print("Warmup request: curl -s -X POST http://127.0.0.1:8799/redact -H \"Content-Type: application/json\" -d '{\"texts\": [\"test@abc.com, Street No.123, LA\"]}'", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down OPF sidecar.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
