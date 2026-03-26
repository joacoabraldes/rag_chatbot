# -*- coding: utf-8 -*-
"""Scoring helpers: similarity, recency, keyword boosting."""

import math, datetime
from typing import List, Optional


def recency_score(d: Optional[datetime.date], today: datetime.date, half_life_days: int) -> float:
    """1.0 = today, exponential decay with *half_life_days*."""
    if not d:
        return 0.0
    age_days = max(0, (today - d).days)
    return math.exp(-age_days / max(1, half_life_days))


def sim_from_distance(dist: float) -> float:
    """Convert distance (lower = better) to similarity ~[0..1)."""
    return 1.0 / (1.0 + dist)


def combined_score(
    dist: float,
    date_: Optional[datetime.date],
    today: datetime.date,
    weight: float,
    half_life_days: int,
) -> float:
    """Final score = (1-w)*similarity + w*recency (higher is better)."""
    s = sim_from_distance(dist)
    r = recency_score(date_, today, half_life_days)
    w = max(0.0, min(1.0, weight))
    return (1.0 - w) * s + w * r


KEYWORDS_BOOST = ["mep", "ccl", "mulc", "tipo de cambio", "dólar", "cotización", "oficial"]


def boost_keywords(text: str, keywords: Optional[List[str]] = None) -> float:
    kws = keywords if keywords is not None else KEYWORDS_BOOST
    tl = text.lower()
    return 0.10 * sum(kw in tl for kw in kws)
