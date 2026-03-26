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
