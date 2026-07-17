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
         "redacted": ["...", "..."],   # same order as input, secrets swapped for sentinels
         "mapping": {"\u27e6HASH:0\u27e7": "<api key>", "\u27e6PII:1\u27e7": "alice@x.com", ...},
         "span_count":  <int>,         # total spans (hash + PII)
         "hash_count":  <int>,         # spans from the hash detector
         "pii_count":   <int>,         # spans from the OPF model
       }

Sentinels are unique across the whole batch (not just per text), so the returned
mapping is self-consistent for one request. Placeholders from opf alone are NOT
reversible (two emails both render as <PRIVATE_EMAIL>), which is why we assign a
unique sentinel per detected span here.

Hash / key detection
--------------------
In addition to OPF, the sidecar runs a fast hex-entropy scan (``hash_detect.py``)
to catch cryptographic-hash-shaped secrets (API keys, tokens, signed URLs) that
sequence-labelling models tend to miss. Hash spans are emitted with sentinel
prefix ``HASH:``; OPF spans use ``PII:``. Hash spans win on overlap (HASH_HIGH
> HASH_LOW > OPF), so the most reliable signal always wins. Disable with
``--no-hash-detect`` if your input contains many false-positive hex strings
(e.g. SHA-256s of public artifacts that you want to keep).

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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import hash_detect
from hash_detect import HASH_HIGH, HASH_LOW, find_hash_spans

# Sentinel brackets are rare unicode so they are extremely unlikely to occur in
# real input and survive JSON round-trips. Format: ⟦<KIND>:<n>⟧
_SENTINEL_OPEN = "\u27e6"
_SENTINEL_CLOSE = "\u27e7"
_HASH_SENTINEL_PREFIX = "HASH:"
_PII_SENTINEL_PREFIX = "PII:"

# Span priority: hash HIGH > hash LOW > OPF. Lower rank = wins on overlap.
_SPAN_PRIORITY_RANK = {HASH_HIGH: 0, HASH_LOW: 1, "MODEL": 2}


def _make_sentinel(kind: str, index: int) -> str:
    """Build a sentinel of the form ``⟦<kind>:<n>⟧``.

    ``kind`` is either ``"hash"`` (-> prefix ``HASH:``) or ``"pii"`` (-> ``PII:``).
    The prefix lets callers tell apart secrets found by the hash detector from
    PII found by the OPF model, while keeping the overall response shape stable.
    """
    prefix = _HASH_SENTINEL_PREFIX if kind == "hash" else _PII_SENTINEL_PREFIX
    return f"{_SENTINEL_OPEN}{prefix}{index}{_SENTINEL_CLOSE}"


@dataclass(frozen=True)
class _MergedSpan:
    """Internal union of hash + OPF spans for the merge pass."""

    start: int
    end: int
    priority: str  # HASH_HIGH, HASH_LOW, or "MODEL"
    kind: str  # "hash" or "pii"


def _merge_spans_with_priority(spans: list[_MergedSpan]) -> list[_MergedSpan]:
    """Drop overlapping spans, keeping the highest-priority (then longest) one.

    We process spans in priority order so a HASH_HIGH span always wins over an
    OPF span even if it starts later. Within one priority tier, longer spans
    win so e.g. an OPF "PERSON" span (0, 20) is dropped in favour of a HASH_HIGH
    span (10, 30) that fully contains it.
    """
    if not spans:
        return []
    accepted: list[_MergedSpan] = []
    for priority in (HASH_HIGH, HASH_LOW, "MODEL"):
        group = [s for s in spans if s.priority == priority]
        # Sort by end-descending (then start-ascending) so a span that extends
        # further right is preferred on overlap. e.g. for (0,20) and (10,30)
        # both HASH_HIGH, we want (10,30) to win because it captures more of
        # the secret. Equal-end ties are broken by earlier start.
        group.sort(key=lambda s: (-s.end, s.start))
        for span in group:
            if any(span.start < acc.end and span.end > acc.start for acc in accepted):
                continue
            accepted.append(span)
    accepted.sort(key=lambda s: s.start)
    return accepted


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

    def __init__(
        self,
        *,
        device: str,
        checkpoint: str | None,
        output_mode: str,
        detect_hashes: bool = True,
    ) -> None:
        from opf._api import OPF

        self._output_mode = output_mode
        self._detect_hashes = detect_hashes
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

    @property
    def detect_hashes(self) -> bool:
        return self._detect_hashes

    def redact_batch(self, texts: list[str]) -> dict[str, Any]:
        """Redact a batch of texts, returning sentinel-substituted text + mapping.

        Sentinel indices are unique across the entire batch. Two kinds of
        sentinels are emitted:

        * ``⟦HASH:n⟧`` — span found by the entropy/whitelist hash detector.
        * ``⟦PII:n⟧``  — span found by the OPF model.

        On overlap the higher-priority span wins (HASH_HIGH > HASH_LOW > MODEL).
        """
        from opf._api import RedactionResult

        redacted: list[str] = []
        mapping: dict[str, str] = {}
        next_index = 0
        hash_count = 0
        pii_count = 0

        with self._lock:
            for text in texts:
                if not isinstance(text, str) or not text:
                    redacted.append(text if isinstance(text, str) else "")
                    continue

                # 1) Hash detection — fast, lock-free, no model involved.
                hash_match: list[_MergedSpan] = []
                if self._detect_hashes:
                    hash_match = [
                        _MergedSpan(hs.start, hs.end, hs.priority, "hash")
                        for hs in find_hash_spans(text)
                    ]

                # 2) OPF detection.
                result = self._opf.redact(text)
                opf_match: list[_MergedSpan] = []
                if isinstance(result, RedactionResult):
                    opf_match = [
                        _MergedSpan(s.start, s.end, "MODEL", "pii")
                        for s in sorted(result.detected_spans, key=lambda x: x.start)
                    ]
                else:
                    # output_text_only must stay False; defensive guard.
                    # If OPF returned something unexpected we still honour any
                    # hash spans we found above rather than leaking them.
                    pass

                # 3) Merge with priority: hash HIGH > hash LOW > MODEL.
                merged = _merge_spans_with_priority(hash_match + opf_match)

                # 4) Apply merged spans.
                pieces: list[str] = []
                cursor = 0
                for span in merged:
                    if span.start < cursor or span.end <= span.start:
                        # overlapping / empty span — skip to keep output coherent
                        continue
                    sentinel = _make_sentinel(span.kind, next_index)
                    next_index += 1
                    if span.kind == "hash":
                        hash_count += 1
                    else:
                        pii_count += 1
                    mapping[sentinel] = text[span.start : span.end]
                    pieces.append(text[cursor : span.start])
                    pieces.append(sentinel)
                    cursor = span.end
                pieces.append(text[cursor:])
                redacted.append("".join(pieces))

        return {
            "redacted": redacted,
            "mapping": mapping,
            "span_count": hash_count + pii_count,
            "hash_count": hash_count,
            "pii_count": pii_count,
        }


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
                f"texts={len(texts)}  spans={result.get('span_count', 0)}  "
                f"hashes={result.get('hash_count', 0)}  pii={result.get('pii_count', 0)}",
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
    parser.add_argument(
        "--no-hash-detect",
        dest="detect_hashes",
        action="store_false",
        help=(
            "Disable the hex-entropy hash/key detector. By default the sidecar "
            "runs hash_detect.find_hash_spans on every text in addition to OPF, "
            "to catch API keys and other hash-shaped secrets that the model "
            "tends to miss. Use this flag if your input contains too many "
            "false-positive hex strings (e.g. SHA-256s of public artifacts)."
        ),
    )
    parser.set_defaults(detect_hashes=True)
    args = parser.parse_args(argv)

    print(
        f"Loading OPF model (device={args.device}, output_mode={args.output_mode}, "
        f"hash_detect={args.detect_hashes})...",
        flush=True,
    )
    redactor: _Redactor = _Redactor(
        device=args.device,
        checkpoint=args.checkpoint,
        output_mode=args.output_mode,
        detect_hashes=args.detect_hashes,
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
