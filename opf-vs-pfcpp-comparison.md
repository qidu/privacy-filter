# OPF (`opf`) vs privacy-filter.cpp (`pf-cli`) Comparison

Both tools run the **same model architecture** (OpenAI privacy-filter, MoE transformer, 8 layers,
640 d_model, 128 experts, ROPE+YaRN) on the same labels, but through different inference engines.

## Output: Identical

On the demo document (`demo/document.txt`, 730 chars), both tools produce exactly the same 4 PII
spans at the same byte offsets:

| Label | Text | Start | End |
|-------|------|-------|-----|
| `private_person` | John Doe | 224 | 232 |
| `private_phone` | +1 555-0112 | 361 | 372 |
| `private_date` | 2026-05-12 | 494 | 504 |
| `private_email` | jane.roe@northside-clinic.org | 612 | 641 |

Redacted output (identical):

```
Thanks for getting back to me so quickly about the prior-authorization request. I want to make sure everything is in order before the review board meets next week, since the last batch got held up over a missing signature. The patient, <PRIVATE_PERSON>, has been with our practice for years and is in good health, so we expect this one to be routine. If anything looks incomplete on your end, please call our billing office at <PRIVATE_PHONE> rather than replying here. I have attached the updated history, and the relevant appointment was on <PRIVATE_DATE>. If you still need the original signed consent form, you can email me directly at <PRIVATE_EMAIL>. Appreciate you helping get claim 4471 across the line before the deadline.
```

Note: `claim 4471` (an account number) was NOT flagged by either tool. The model does not
classify it as PII at the default threshold.

## Shared Label Taxonomy (8 user-facing categories)

`account_number`, `private_address`, `private_date`, `private_email`, `private_person`,
`private_phone`, `private_url`, `secret`

## Speed Benchmarks

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel i7-13700H CPU.

OPF measured via Python API (model kept warm, no reload). pf-cli measured via `pf-bench`
(pure forward-pass, model preloaded). Times in milliseconds — lower is better.

### GPU (CUDA)

| Text Length | OPF (ms) | pf-cli (ms) | Winner |
|------------:|---------:|------------:|--------|
| 730 char    |     55.3 |        21.9 | pf-cli **2.5×** |
| 3,650 char  |     82.7 |       101.5 | OPF **1.2×** |
| 7,300 char  |    108.3 |       252.4 | OPF **2.3×** |
| 14,600 char |    206.5 |       635.0 | OPF **3.1×** |
| 36,500 char |    545.9 |     1,989.0 | OPF **3.6×** |

**Pattern**: pf-cli wins at short text; OPF (PyTorch) wins at longer text. This is the
opposite of the README's claim (7.4× speedup at 8k tokens) because:

- pf-cli uses **banded / truncated self-attention** (near-linear in sequence length), which
  should theoretically win at length.
- OPF's PyTorch implementation uses **full self-attention** (O(n²)), which should lose at
  length.
- However: `pf-cli` reloads the model per call in `--classify` mode (~1.4s overhead), and
  even `pf-bench` shows higher per-token latency on this GPU (RTX 4060 laptop, 8 GB vs the
  README's RTX 5070 Ti desktop). The banded attention may be less efficient on this GPU, or
  the model is small enough that full attention fits easily in VRAM and runs fast in PyTorch's
  highly optimized CUDA kernels.

### CPU

OPF CPU is impractically slow (single inference of 730 chars did not complete within a
reasonable timeout). pf-cli CPU is usable but much slower than GPU:

| Tool | Time (ms) |
|------|----------:|
| pf-cli CPU, 730 char | ~390 |
| pf-cli CPU, 36,500 char | ~80,000 (est.) |

## Feature Comparison

| Feature | `opf` | `pf-cli` |
|---------|-------|----------|
| **Engine** | PyTorch (bf16/fp32) | ggml (q8/f16, C++) |
| **Model size on disk** | ~3 GB (bf16) | ~1.6 GB (q8) |
| **Default decode** | Viterbi CRF | Viterbi CRF |
| **Confidence scores** | No | Yes (per-entity `score` field) |
| **GPU backends** | CUDA | CUDA, Vulkan |
| **CPU** | Yes (slow) | Yes (fast, SIMD) |
| **Long-document support** | Windowed (overlapping chunks) | Windowed with halo (banded attn) |
| **Max context window** | 2048 tokens (model default) | Configurable via `--window` |
| **JSON output** | Rich schema + metadata | Flat entity list |
| **Interactive mode** | Yes (`opf` with no args) | No |
| **Eval against ground truth** | Built-in (`opf eval`) | Separate Python scripts |
| **Fine-tuning** | Experimental (`opf train`) | No (GGUF is inference-only) |
| **C API for embedding** | No (Python library) | Yes (`include/pf.h`) |
| **CLI model loading** | Once (daemon/API) | Every call (`--classify` reloads) |
| **Fuzz testing** | No | Yes (libFuzzer) |

## Summary

- **Same model, same labels, same output** — interchangeable at the functional level.
- **pf-cli** wins on: model size (1.6 GB vs 3 GB), multi-backend (CUDA+Vulkan+CPU), confidence
  scores, C API for embedding, and long-document handling with configurable windows.
- **OPF** wins on: full-featured CLI (interactive, eval, train), Python API with rich output
  (JSON schema, metadata), and faster inference at longer text lengths on this particular GPU
  (potentially due to PyTorch's optimized CUDA kernels for small models).
