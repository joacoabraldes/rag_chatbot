# -*- coding: utf-8 -*-
"""Pydantic request/response models for the API."""

from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    show_sources: bool = True
    history: Optional[List[dict]] = None
