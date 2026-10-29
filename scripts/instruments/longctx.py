#!/usr/bin/env python3
"""Controlled long-context padding for the Polyglot tasks.

Section 12.5 measured how much context a card can *hold*. This measures how much
the model can still *use*: the task is held fixed and only the surrounding
context grows, so a failure at depth is attributable to depth alone.

Padding is placed before the instructions, leaving them last. Recency therefore
favours success, which makes any degradation found a lower bound rather than an
adversarial worst case.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from scripts import ROOT

CORPORA = {"rust": ROOT / "corpora" / "rust.txt", "javascript": ROOT / "corpora" / "js-ts.txt"}

# Padding is drawn from the L1 corpus repos, which the leakage test already proves
# disjoint from the task pool. This is the second line of defence: a chunk that
# happens to name the exercise would hand over the answer.
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# A leak used to abort the whole sweep, so one unlucky offset cost every task
# behind it. Retrying is bounded: exhausting it still fails loudly.
_OFFSET_ATTEMPTS = 24


def _tokenize(text: str, base_url: str, timeout: float = 120.0) -> int:
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        json.dumps({"content": text}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return len(json.loads(response.read())["tokens"])


def forbidden_terms(task: dict) -> set[str]:
    """Identifiers whose appearance in padding would leak the exercise."""

    slug = str(task["task_id"]).split("/", 1)[-1]
    terms = {slug, slug.replace("-", "_"), slug.replace("-", "")}
    return {t.lower() for t in terms if len(t) > 3}


def build_padding(task: dict, target_tokens: int, base_url: str) -> tuple[str, int]:
    """Return padding of ~target_tokens for this task, and its exact token count.

    The offset is derived from the task id, so padding varies across tasks but is
    identical for the same task across quants and reruns -- the comparison stays
    paired.
    """

    if target_tokens <= 0:
        return "", 0
    language = str(task.get("language", "")).lower()
    corpus = CORPORA.get("rust" if language in {"rust", "rs"} else "javascript")
    text = corpus.read_text(encoding="utf-8")
    banned = forbidden_terms(task)

    # Offsets are tried in a fixed order so a task that needs the third one still
    # gets that same padding on every quant and rerun; the pairing survives.
    leaked: set[str] = set()
    for attempt in range(_OFFSET_ATTEMPTS):
        seed = task["task_id"] if attempt == 0 else f"{task['task_id']}:{attempt}"
        digest = hashlib.sha256(seed.encode()).hexdigest()
        start = int(digest[:12], 16) % max(1, len(text) - 1)

        # ~3.2 bytes/token on this corpus; overshoot then trim by real token count.
        window = text[start:] + text[:start]
        chunk = window[: int(target_tokens * 4.0)]
        found = banned & {m.group(0).lower() for m in _WORD.finditer(chunk)}
        if not found:
            break
        leaked |= found
    else:
        raise SystemExit(
            f"PADDING LEAK for {task['task_id']}: every offset mentions {sorted(leaked)}. "
            "Use a different corpus; do not measure with this padding."
        )

    count = _tokenize(chunk, base_url)
    while count > target_tokens and len(chunk) > 1000:
        chunk = chunk[: int(len(chunk) * target_tokens / count)]
        count = _tokenize(chunk, base_url)
    return chunk, count


def wrap_prompt(prompt: str, padding: str) -> str:
    if not padding:
        return prompt
    return (
        "The following is unrelated reference material from other projects. "
        "It is provided as background and is not part of your task.\n\n"
        "<reference>\n" + padding + "\n</reference>\n\n" + prompt
    )
