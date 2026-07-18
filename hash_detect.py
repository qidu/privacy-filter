"""Detect likely-cryptographic-hash strings in text.

Identifies contiguous hex strings whose length and entropy profile match
real cryptographic digests (MD5, SHA-1, SHA-256, their truncations, Git
short hashes), while filtering out:

  * Hexspeak magic numbers (``deadbeef``, ``cafebabe``, ...).
  * Repetitive / ordered sequences (``ffffffff``, ``1234567890abcdef``).
  * English dictionary words that happen to be valid hex (``fabaceae``).

The original use case is pre-redacting API keys, tokens and other
hex-shaped secrets before forwarding text to a downstream LLM, because
sequence-labeling PII models tend to miss them.

Public API
----------
* :class:`HashSpan`            — frozen (start, end, token, priority).
* :func:`shannon_entropy`      — bits/symbol; theoretical max for hex is 4.0.
* :func:`detect_hash_priority` — classify a single token ("HIGH" | "LOW" | "NO").
* :func:`find_hash_spans`      — find all hash spans in a string.

Constants
---------
* :data:`HASH_HIGH`, :data:`HASH_LOW`, :data:`HASH_NO`
* :data:`HEX_WORDS_WHITELIST`

User overrides
--------------
Set ``OPF_HASH_WHITELIST_FILE=/path/to/whitelist.txt`` to extend or trim the
built-in whitelist without editing this file. Format (one entry per line)::

    # Comments start with '#'.
    badcafe          # add this token to the whitelist
    -fabaceae        # remove this built-in from the whitelist

Tokens are lowercased and whitespace-stripped. Entries shorter than 8 chars
are dropped (the length filter is applied uniformly to additions and removals).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "HashSpan",
    "shannon_entropy",
    "detect_hash_priority",
    "find_hash_spans",
    "build_whitelist",
    "HEX_WORDS_WHITELIST",
    "HASH_HIGH",
    "HASH_LOW",
    "HASH_NO",
]

# ---- Classification levels ------------------------------------------------
HASH_HIGH = "HIGH"  # 16..256 chars AND a multiple of 8, high entropy.
HASH_LOW = "LOW"    # > 8 chars, high entropy, but not HIGH-shaped.
HASH_NO = "NO"      # not a hash (non-hex, whitelisted, low entropy, ...).

# Minimum length for a token to be considered for whitelist lookup. Anything
# shorter is filtered by the length check in :func:`detect_hash_priority`.
_MIN_WHITELIST_LEN = 8

# ---- Built-in whitelist (immutable; merged with user overrides at load) ---
_BUILTIN_HEX_WORDS_WHITELIST: frozenset[str] = frozenset(
    w.lower()
    for w in {
        # ---- Classic hexspeak magic numbers ----
        "deadbeef",   # invalid-memory marker
        "cafebabe",   # Java .class magic
        "decafbad",   # common placeholder
        "feedface",   # debugger marker
        "baadfeed",   # bad-block marker
        "feedbeef",   # variant
        "beefdead",   # variant
        "defacade",   # architecture / art term
        "addbedface", # 11-char variant
        # ---- Hexspeak with 0 / 1 digits ----
        "c0edbabe",   # 8 chars
        "c001d00d",   # 8 chars
        "baadf00d",   # 8 chars
        # ---- Real English dictionary words (pure A-F) ----
        "fabaceae",   # the bean family
    }
    if len(w) >= _MIN_WHITELIST_LEN
)


def _load_user_whitelist_overrides(path: str) -> tuple[set[str], set[str]]:
    """Parse a whitelist-override file into ``(additions, removals)`` sets.

    File format (see module docstring):
      * One entry per line.
      * ``#`` starts a comment to end of line.
      * ``-token`` removes ``token`` from the built-in whitelist.
      * Otherwise the token is added.
      * Tokens are lowercased and whitespace-stripped.
      * Entries shorter than :data:`_MIN_WHITELIST_LEN` are silently dropped.

    Raises:
        OSError: If ``path`` cannot be opened.
    """
    additions: set[str] = set()
    removals: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.split("#", 1)[0].strip().lower()
            if not stripped:
                continue
            if stripped.startswith("-"):
                candidate = stripped[1:].strip()
                if len(candidate) >= _MIN_WHITELIST_LEN:
                    removals.add(candidate)
            else:
                if len(stripped) >= _MIN_WHITELIST_LEN:
                    additions.add(stripped)
    return additions, removals


def _resolve_whitelist() -> frozenset[str]:
    """Build the final whitelist = built-ins +/- user-file overrides."""
    merged = set(_BUILTIN_HEX_WORDS_WHITELIST)
    override_path = os.environ.get("OPF_HASH_WHITELIST_FILE")
    if override_path:
        additions, removals = _load_user_whitelist_overrides(override_path)
        merged -= removals
        merged |= additions
    return frozenset(merged)


def build_whitelist(
    add: list[str] | None = None,
    remove: list[str] | None = None,
    whitelist_file: str | None = None,
) -> frozenset[str]:
    """Build a whitelist from built-ins plus caller-supplied additions/removals.

    This is the programmatic equivalent of the ``OPF_HASH_WHITELIST_FILE``
    env-var mechanism used at import time: callers (e.g. ``serve.py``) pass
    additions and removals gathered from CLI flags instead of a file.

    Args:
        add: Tokens to add on top of the built-ins (lowercased, length-filtered).
        remove: Tokens to remove from the built-ins (lowercased, length-filtered).
        whitelist_file: Optional path to an override file (same format as
            ``OPF_HASH_WHITELIST_FILE``). Applied after ``add``/``remove``.
    """
    merged = set(_BUILTIN_HEX_WORDS_WHITELIST)
    for token in (add or []):
        t = token.strip().lower()
        if len(t) >= _MIN_WHITELIST_LEN:
            merged.add(t)
    for token in (remove or []):
        t = token.strip().lower()
        merged.discard(t)
    if whitelist_file:
        file_add, file_remove = _load_user_whitelist_overrides(whitelist_file)
        merged -= file_remove
        merged |= file_add
    return frozenset(merged)


# Public whitelist — computed at import time from the built-ins plus any
# overrides in $OPF_HASH_WHITELIST_FILE. See module docstring for format.
HEX_WORDS_WHITELIST: frozenset[str] = _resolve_whitelist()


def _build_token_re(min_len: int) -> re.Pattern[str]:
    """Return a compiled regex that captures hex runs of at least ``min_len`` chars."""
    return re.compile(rf"(?<![a-fA-F0-9])[a-fA-F0-9]{{{min_len},}}(?![a-fA-F0-9])")


# Default token regex (min_len == 8).  Callers that need a different min_len
# should call find_hash_spans with an explicit min_len.
_TOKEN_RE = _build_token_re(_MIN_WHITELIST_LEN)

# Reference ordering for sequence detection. Strings whose sorted chars
# are a permutation of the first N entries of this alphabet (e.g.
# "1234567890abcdef", "fedcba9876543210", "0123456789abcdef0123") are
# treated as deliberate sequences, not hashes.
_HEX_ALPHABET = "0123456789abcdef"


@dataclass(frozen=True)
class HashSpan:
    """A span of text classified as a likely cryptographic hash."""

    start: int
    end: int
    token: str
    priority: str  # one of HASH_HIGH, HASH_LOW


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits/symbol (theoretical max for hex = log2(16) = 4.0)."""
    if not s:
        return 0.0
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in Counter(s).values())


def _is_ordered_hex_sequence(token: str) -> bool:
    """True if ``token`` is a permutation of the first ``len(token)`` hex digits.

    Catches things like ``"1234567890abcdef"``, ``"fedcba9876543210"`` and
    ``"0123456789abcdef0123"`` which have max entropy (4.0) but are clearly
    not hashes. For a random hex token of length N the probability of this
    matching is N! / 16**N, vanishingly small for N >= 9.
    """
    return sorted(token.lower()) == list(_HEX_ALPHABET[: len(token)])


def detect_hash_priority(
    token: str,
    entropy_threshold: float = 3.0,
    whitelist: frozenset[str] | None = None,
    min_len: int = _MIN_WHITELIST_LEN,
) -> str:
    """Classify a single token as a likely hash.

    Returns one of :data:`HASH_HIGH`, :data:`HASH_LOW`, :data:`HASH_NO`.

    Args:
        token: The hex string to classify.
        entropy_threshold: Shannon entropy cutoff (default 3.0).
        whitelist: Token whitelist to use; defaults to :data:`HEX_WORDS_WHITELIST`.
        min_len: Minimum token length to consider (default 8).
    """
    if not re.fullmatch(r"[a-fA-F0-9]+", token):
        return HASH_NO
    effective_whitelist = HEX_WORDS_WHITELIST if whitelist is None else whitelist
    if token.lower() in effective_whitelist:
        return HASH_NO
    length = len(token)
    if length < min_len:
        return HASH_NO
    if _is_ordered_hex_sequence(token):
        return HASH_NO
    if shannon_entropy(token) < entropy_threshold:
        return HASH_NO
    if 16 <= length <= 256 and length % 8 == 0:
        return HASH_HIGH
    return HASH_LOW


def find_hash_spans(
    text: str,
    entropy_threshold: float = 3.0,
    whitelist: frozenset[str] | None = None,
    min_len: int = _MIN_WHITELIST_LEN,
) -> list[HashSpan]:
    """Find hash-shaped spans in ``text``.

    Spans are returned in left-to-right order. ``HASH_NO`` candidates are
    omitted; only ``HASH_HIGH`` and ``HASH_LOW`` spans are returned.

    Args:
        text: Input string to scan.
        entropy_threshold: Shannon entropy cutoff (default 3.0).
        whitelist: Token whitelist to use; defaults to :data:`HEX_WORDS_WHITELIST`.
        min_len: Minimum hex token length to classify (default 8).
    """
    if not text:
        return []
    token_re = _TOKEN_RE if min_len == _MIN_WHITELIST_LEN else _build_token_re(min_len)
    spans: list[HashSpan] = []
    for match in token_re.finditer(text):
        token = match.group(0)
        priority = detect_hash_priority(
            token,
            entropy_threshold=entropy_threshold,
            whitelist=whitelist,
            min_len=min_len,
        )
        if priority == HASH_NO:
            continue
        spans.append(HashSpan(match.start(), match.end(), token, priority))
    return spans


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases: list[tuple[str, str, str]] = [
        # (token, expected, description)
        # ---- Whitelist hits (must return NO) ----
        ("deadbeef", HASH_NO, "hexspeak"),
        ("CAFEBABE", HASH_NO, "hexspeak upper"),
        ("Fabaceae", HASH_NO, "dictionary word"),
        ("c0edbabe", HASH_NO, "hexspeak w/ digit"),
        ("addbedface", HASH_NO, "11-char hexspeak"),
        # ---- Real hashes (must return HIGH) ----
        ("5d41402abc4b2a76b9719d911017c592", HASH_HIGH, "MD5 32"),
        ("A665A45920422F9D417E4867EFDC4FB8", HASH_HIGH, "MD5 32 upper"),
        ("5d41402abc4b2a76", HASH_HIGH, "16 chars / 8-mult"),
        # ---- Real-but-irregular lengths (must return LOW) ----
        ("5d41402abc4b2a76b", HASH_LOW, "17 chars"),
        ("a1b2c3d4e5f6789", HASH_LOW, "15 chars"),
        # ---- Definite non-hashes (must return NO) ----
        ("ffffffffffffffff", HASH_NO, "16 f's (low entropy)"),
        ("1234567890abcdef", HASH_NO, "ascending (low entropy)"),
        ("abc123", HASH_NO, "len <= 8"),
        ("HelloWorld", HASH_NO, "non-hex chars"),
    ]

    print("=" * 72)
    failures = 0
    for token, expected, desc in test_cases:
        got = detect_hash_priority(token)
        ok = "OK  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"[{ok}] {desc:<30} | {token:<35} | {got}")
    print("=" * 72)
    print(f"find_hash_spans demo on mixed input:")
    sample = (
        "API key=5d41402abc4b2a76b9719d911017c592 was leaked. "
        "Magic=deadbeef. UUID=5d41402a-bc4b-2a76-b971-9d911017c592. "
        "Truncated=5d41402abc4b2a76b. Random=a1b2c3d4e5f6789."
    )
    for span in find_hash_spans(sample):
        print(f"  [{span.priority}] {span.start:>3}..{span.end:<3} {span.token!r}")
    print("=" * 72)
    print("User whitelist override demo (OPF_HASH_WHITELIST_FILE):")
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as override_file:
        override_file.write("# add a custom placeholder, remove a built-in\n")
        override_file.write("badcafe12\n")
        override_file.write("-fabaceae\n")
        override_path = override_file.name
    saved = os.environ.get("OPF_HASH_WHITELIST_FILE")
    os.environ["OPF_HASH_WHITELIST_FILE"] = override_path
    try:
        added, removed = _load_user_whitelist_overrides(override_path)
        print(f"  additions parsed: {sorted(added)}")
        print(f"  removals parsed:  {sorted(removed)}")
    finally:
        os.environ.pop("OPF_HASH_WHITELIST_FILE", None)
        if saved is not None:
            os.environ["OPF_HASH_WHITELIST_FILE"] = saved
        os.unlink(override_path)
    print("=" * 72)
    raise SystemExit(0 if failures == 0 else 1)
