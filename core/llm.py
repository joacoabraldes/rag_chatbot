# -*- coding: utf-8 -*-
"""LLM streaming and topic extraction utilities."""

import json
from typing import AsyncIterator, List

from openai import AsyncOpenAI, OpenAI

from core.config import OPENAI_MODEL, OPENAI_MODEL_FAST

_async_client = AsyncOpenAI(timeout=60.0)
_sync_client = OpenAI(timeout=30.0)


async def stream_chat(messages: list, model: str | None = None) -> AsyncIterator[str]:
    """Yield text chunks from an OpenAI streaming response."""
    model_name = model or OPENAI_MODEL
    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }
    # Reasoning models (gpt-5*) silence the stream during internal reasoning;
    # minimal effort keeps time-to-first-token low for a chat UX.
    if model_name.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "minimal"

    stream = await _async_client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if (
            chunk.choices
            and chunk.choices[0].delta
            and chunk.choices[0].delta.content
        ):
            yield chunk.choices[0].delta.content


def extract_topic_tags_batch(
    chunks_texts: List[str],
    model: str | None = None,
) -> List[str]:
    """Extract topic tags for multiple chunks via batched LLM calls.

    Returns a list of comma-separated tag strings, one per input chunk.
    """
    llm_model = model or OPENAI_MODEL_FAST
    all_tags: List[str] = []
    batch_size = 10

    for i in range(0, len(chunks_texts), batch_size):
        batch = chunks_texts[i : i + batch_size]
        numbered = "\n".join(
            f"[{j + 1}] {text[:300]}" for j, text in enumerate(batch)
        )
        prompt = (
            "Analizá los siguientes fragmentos de informes económicos y extraé "
            "2-4 etiquetas temáticas para cada uno.\n"
            "Las etiquetas deben ser términos concisos en español (ej: inflación, "
            "tipo de cambio, PBI, política monetaria, mercado laboral).\n\n"
            f"Fragmentos:\n{numbered}\n\n"
            "Respondé SOLO con un array JSON donde cada elemento es un array de "
            "strings con las etiquetas del fragmento correspondiente.\n"
            'Ejemplo: [["inflación", "precios"], ["tipo de cambio", "dólar"]]'
        )
        try:
            resp = _sync_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            tags_list = json.loads(raw)
            if isinstance(tags_list, list):
                for tags in tags_list:
                    if isinstance(tags, list):
                        all_tags.append(",".join(str(t) for t in tags))
                    else:
                        all_tags.append("")
                # Pad if LLM returned fewer items than expected
                while len(all_tags) < i + len(batch):
                    all_tags.append("")
            else:
                all_tags.extend([""] * len(batch))
        except Exception as e:
            print(f"  Warning: topic extraction failed for batch: {e}")
            all_tags.extend([""] * len(batch))

    return all_tags
