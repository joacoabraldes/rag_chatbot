# -*- coding: utf-8 -*-
"""Pydantic request/response models for the API."""

import os
from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    k: int = 10
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    embed_model: Optional[str] = None
    collections: Optional[List[str]] = None
    recency_weight: float = 0.45
    half_life_days: int = 30
    use_llm_expansion: bool = False
    use_simple_expansion: bool = False
    history: Optional[List[dict]] = None
