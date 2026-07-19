#!/usr/bin/env python3
"""
Convert sidecar `/redact` predictions into OPF `train`-compatible JSONL.

Input format (one record per line, JSON):
    {
      "text": "raw text...",
      "predicted": [
        {"label": "private_person", "start": 0, "end": 2},
        {"label": "private_email",  "start": 7, "end": 25},
        ...
      ]
    }

Output format (one record per line, JSON):
    {
      "text": "...",
      "spans": {
        "private_person: 王伟":      [[0, 2]],
        "private_email: wangwei@…":  [[7, 25]]
      },
      "info": {"id": "review_00042", "source": "review"}
    }

This matches the `spans` schema in opf/_eval/preprocess.py::parse_record.

Usage:
    python scripts/predictions_to_jsonl.py \\
        --input predictions.jsonl \\
        --output to_review.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def convert(records: Iterable[dict], *, id_prefix: str = "review") -> Iterable[dict]:
    """Convert an iterable of prediction records into OPF-format JSONL records."""
    for idx, rec in enumerate(records):
        text: str = str(rec.get("text", ""))
        predicted = rec.get("predicted", []) or []
        spans: dict[str, list[list[int]]] = {}
        for ent in predicted:
            label = str(ent.get("label", "")).strip()
            if not label:
                continue
            try:
                start = int(ent["start"])
                end = int(ent["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= start < end <= len(text)):
                continue
            surface = text[start:end]
            key = f"{label}: {surface}"
            spans.setdefault(key, []).append([start, end])
        yield {
            "text":  text,
            "spans": spans,
            "info":  {"id": f"{id_prefix}_{idx:05d}", "source": "review"},
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, type=Path,
                        help="Path to predictions JSONL (text + predicted[]).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write OPF-format JSONL.")
    parser.add_argument("--id-prefix", default="review",
                        help="Prefix for generated example ids.")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with args.input.open() as fin, args.output.open("w", encoding="utf-8") as fout:
        records = (json.loads(line) for line in fin if line.strip())
        for out_rec in convert(records, id_prefix=args.id_prefix):
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_in += 1
            n_out += 1

    print(f"converted {n_in} prediction records → {n_out} JSONL records at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())