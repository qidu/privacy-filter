#!/usr/bin/env python3
"""
Run `opf eval` against one or more test JSONL files, parse its output, and
write a markdown report summarizing precision/recall/F1 broken down by:
  * label category
  * record (eval split)
  * region (CN vs ZH-YT) — only if `info.region` is populated

Usage:
    python scripts/eval_to_markdown.py \\
        --eval-dir /path/to/eval_slices/ \\
        --checkpoint ./opf_mix_ft \\
        --out reports/eval_report.md

Each JSONL file under --eval-dir is treated as one eval split. Their
filenames (without .jsonl) become section headers.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


# -----------------------------------------------------------------------------
# Parse `opf eval` stdout
# -----------------------------------------------------------------------------

EVAL_HEADER_RE = re.compile(r"^##\s*Evaluation result on \d+ examples$")
LABEL_METRIC_RE = re.compile(
    r"^(?P<label>\S+)\s+(?P<examples>\d+)\s+(?P<correct>\d+)\s+(?P<predicted>\d+)\s+(?P<real>\d+)\s+(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+(?P<f1>[\d.]+)\s*$"
)
TOTAL_METRIC_RE = re.compile(
    r"^(?P<label>\*\*ALL\*\*)\s+(?P<examples>\d+)\s+(?P<correct>\d+)\s+(?P<predicted>\d+)\s+(?P<real>\d+)\s+(?P<precision>[\d.]+)\s+(?P<recall>[\d.]+)\s+(?P<f1>[\d.]+)\s*$"
)


def parse_eval_output(stdout: str) -> list[dict]:
    """Parse label-level rows from `opf eval` output.

    Returns a list of dicts with keys:
        label, examples, correct, predicted, real, precision, recall, f1
    """
    rows = []
    for line in stdout.splitlines():
        m = TOTAL_METRIC_RE.match(line)
        if m:
            rows.append({k: m.group(k) for k in
                         ("label", "examples", "correct", "predicted",
                          "real", "precision", "recall", "f1")})
            continue
        m = LABEL_METRIC_RE.match(line)
        if m:
            rows.append({k: m.group(k) for k in
                         ("label", "examples", "correct", "predicted",
                          "real", "precision", "recall", "f1")})
    return rows


# -----------------------------------------------------------------------------
# Per-region breakdown from the JSONL files themselves
# -----------------------------------------------------------------------------

def breakdown_by_region(jsonl_path: Path) -> Counter:
    """Return Counter[region] for the records in jsonl_path. 'unknown' if no info.region."""
    regions = Counter()
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            regions[r.get("info", {}).get("region", "unknown")] += 1
    return regions


def breakdown_by_label(jsonl_path: Path) -> Counter:
    labels = Counter()
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            for k in r.get("spans", {}):
                labels[k.split(": ", 1)[0]] += 1
    return labels


# -----------------------------------------------------------------------------
# Markdown generation
# -----------------------------------------------------------------------------

def fmt_pct(v: str) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (ValueError, TypeError):
        return v


def render_report(
    split_name: str,
    rows: list[dict],
    jsonl_path: Path,
) -> str:
    """Render one split section."""
    md = [f"## Split: `{split_name}`", ""]
    md.append(f"- Path: `{jsonl_path}`")
    md.append(f"- Records: **{len(open(jsonl_path).readlines())}**")

    # Label distribution in this split (ground truth).
    label_dist = breakdown_by_label(jsonl_path)
    md.append("- Ground-truth labels:")
    for k, n in sorted(label_dist.items(), key=lambda x: -x[1]):
        md.append(f"  - `{k}`: {n}")
    md.append("")

    # Region split if available.
    region_dist = breakdown_by_region(jsonl_path)
    if any(r != "unknown" for r in region_dist):
        md.append("- Regions:")
        for r, n in sorted(region_dist.items(), key=lambda x: -x[1]):
            md.append(f"  - `{r}`: {n}")
        md.append("")

    if not rows:
        md.append("> ⚠️ No eval rows parsed. Check the opf eval output above.")
        md.append("")
        return "\n".join(md)

    # Predicted-vs-real table.
    md.append("### Metrics (per label)")
    md.append("")
    md.append("| Label | Examples | Correct | Predicted | Real | Precision | Recall | F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| `{r['label']}` | {r['examples']} | {r['correct']} | "
            f"{r['predicted']} | {r['real']} | "
            f"{fmt_pct(r['precision'])} | {fmt_pct(r['recall'])} | {fmt_pct(r['f1'])} |"
        )
    md.append("")
    return "\n".join(md)


def render_combined_table(all_results: list[tuple[str, str, list[dict]]]) -> str:
    """A single F1 comparison table across all splits."""
    lines = []
    lines.append("## At a glance — F1 by label and split")
    lines.append("")
    lines.append("| Label | " + " | ".join(s for s, _, _ in all_results) + " |")
    lines.append("|---" + "|---:" * len(all_results) + "|")

    # Union of all labels.
    labels = set()
    for _, _, rows in all_results:
        for r in rows:
            labels.add(r["label"])
    # Put **ALL** first, then others alphabetically.
    ordered = sorted(labels, key=lambda l: (0 if l == "**ALL**" else 1, l))
    for label in ordered:
        cells = []
        for _, _, rows in all_results:
            match = next((r for r in rows if r["label"] == label), None)
            cells.append(fmt_pct(match["f1"]) if match else "—")
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_eval(jsonl_path: Path, checkpoint: Path, device: str = "cpu") -> str:
    """Invoke `opf eval` and return stdout."""
    cmd = [
        sys.executable, "-m", "opf", "eval", str(jsonl_path),
        "--checkpoint", str(checkpoint),
        "--device", device,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(f"⚠️ opf eval failed for {jsonl_path} (exit {proc.returncode})\n")
        sys.stderr.write(out + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True, type=Path,
                   help="Directory containing *.jsonl files to evaluate.")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to OPF checkpoint directory.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output markdown report path.")
    p.add_argument("--device", default="cpu", help="Device for opf eval.")
    args = p.parse_args(argv)

    jsonls = sorted(args.eval_dir.glob("*.jsonl"))
    if not jsonls:
        sys.stderr.write(f"No JSONL files found in {args.eval_dir}\n")
        return 1

    md_parts: list[str] = []
    md_parts.append("# OPF Evaluation Report")
    md_parts.append("")
    md_parts.append(f"- **Checkpoint:** `{args.checkpoint}`")
    md_parts.append(f"- **Device:** `{args.device}`")
    md_parts.append(f"- **Eval dir:** `{args.eval_dir}`")
    md_parts.append("")

    all_results: list[tuple[str, str, list[dict]]] = []

    for jsonl in jsonls:
        split_name = jsonl.stem
        print(f"=== Evaluating {jsonl} ===")
        stdout = run_eval(jsonl, args.checkpoint, device=args.device)
        rows = parse_eval_output(stdout)
        all_results.append((split_name, stdout, rows))
        md_parts.append(f"```\n{stdout.strip()}\n```\n")
        md_parts.append(render_report(split_name, rows, jsonl))

    md_parts.append(render_combined_table(all_results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n📝 Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
