# -*- coding: utf-8 -*-
"""Text cleaning and formatting utilities."""

import re, unicodedata


def short_preview(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return (t[:n] + "\u2026") if len(t) > n else t


def clean_keep_ascii_marks(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u00AD", "").replace("\u200B", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def escape_markdown(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>])", r"\\\1", s)


def normalize_plain(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    return s


# ─── Response post-processing ────────────────────────────────────────────────

_FILLER_PATTERNS = [
    re.compile(r'^(?:As an AI(?: language model)?|Como (?:modelo|inteligencia artificial))[,.]?\s*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(?:Sure|Claro|Por supuesto)[,!.]?\s*(?:here|aquí|acá)?[,.]?\s*', re.IGNORECASE),
]


def postprocess_response(text: str) -> str:
    """Clean up LLM response: strip filler, fix whitespace, ensure final punctuation."""
    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove filler preamble patterns
    for pat in _FILLER_PATTERNS:
        text = pat.sub('', text)
    text = text.strip()

    # Collapse multiple blank lines to single
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Ensure response ends with proper punctuation
    if text and text[-1] not in '.!?:;…"\'»)]\u201d':
        text += '.'

    return text
