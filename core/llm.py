# -*- coding: utf-8 -*-
"""LLM streaming + topic extraction + tool calling.

stream_chat() loops on tool calls: when the model wants to call SQL tools
it emits delta.tool_calls (no content). We accumulate the partial deltas,
execute the tools, append a 'tool' role message, and re-stream. When the
model finally has the data it streams normal content tokens — only those
get yielded back to the caller for SSE.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI

from core.config import OPENAI_MODEL, OPENAI_MODEL_FAST
from core.sql_tools import execute_tool, tool_result_message
from core.taxonomy import canonicalize_topics, taxonomy_for_prompt

log = logging.getLogger("rag.llm")

_async_client = AsyncOpenAI(timeout=60.0)
_sync_client = OpenAI(timeout=30.0)

# Cap how many tool-call rounds we allow per request before giving up.
# Each round is a separate LLM call — keeps a misbehaving model from
# looping forever (or running up the bill).
_MAX_TOOL_ROUNDS = 4


def _build_completion_kwargs(
    model_name: str,
    messages: list,
    tools: Optional[list],
    stream: bool,
) -> dict:
    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }
    # Reasoning models (gpt-5*) silence the stream during internal reasoning;
    # minimal effort keeps time-to-first-token low for a chat UX.
    if model_name.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "minimal"
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return kwargs


def _accumulate_tool_call_delta(
    buffer: list[dict],
    tc_delta,
) -> None:
    """OpenAI streams tool calls in partial deltas: ``id`` and ``name``
    arrive on the first chunk for each call, then ``arguments`` streams in
    pieces. We accumulate by ``index``."""
    idx = tc_delta.index
    while len(buffer) <= idx:
        buffer.append({"id": None, "name": None, "arguments": ""})
    slot = buffer[idx]
    if tc_delta.id:
        slot["id"] = tc_delta.id
    fn = tc_delta.function
    if fn:
        if fn.name:
            slot["name"] = fn.name
        if fn.arguments:
            slot["arguments"] += fn.arguments


def _materialize_assistant_tool_message(tool_calls_buffer: list[dict]) -> dict:
    """Reconstruct the assistant message with the tool calls so the next
    round of the conversation has the context."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc["id"] or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"] or "unknown",
                    "arguments": tc["arguments"] or "{}",
                },
            }
            for i, tc in enumerate(tool_calls_buffer)
        ],
    }


async def stream_chat(
    messages: list,
    model: str | None = None,
    tools: Optional[list] = None,
) -> AsyncIterator[str]:
    """Yield text tokens from an OpenAI streaming response.

    When ``tools`` is provided the model can request SQL tool calls. We
    execute them transparently and re-stream until the model produces a
    plain content response. Only content tokens are yielded — the caller
    sees a single uninterrupted stream of text.
    """
    model_name = model or OPENAI_MODEL
    # Mutable copy — we'll append assistant + tool messages as we loop.
    convo = list(messages)

    for round_idx in range(_MAX_TOOL_ROUNDS + 1):
        kwargs = _build_completion_kwargs(model_name, convo, tools, stream=True)
        stream = await _async_client.chat.completions.create(**kwargs)

        tool_calls_buffer: list[dict] = []
        had_content = False

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    _accumulate_tool_call_delta(tool_calls_buffer, tc_delta)
                continue

            if delta.content:
                had_content = True
                yield delta.content

        # Stream finished. Decide what to do next.
        if had_content and not tool_calls_buffer:
            # Pure content response — we're done.
            return

        if not tool_calls_buffer:
            # Empty stream (rare). Don't loop forever.
            return

        # Cap the loop in case the model keeps asking for tools.
        if round_idx >= _MAX_TOOL_ROUNDS:
            log.warning(
                "Tool-call loop hit %d rounds — aborting and asking model to summarize",
                _MAX_TOOL_ROUNDS,
            )
            convo.append(_materialize_assistant_tool_message(tool_calls_buffer))
            for tc in tool_calls_buffer:
                convo.append(
                    tool_result_message(
                        tc["id"] or "stop",
                        {"error": f"límite de {_MAX_TOOL_ROUNDS} rondas alcanzado; respondé con lo que tengas"},
                    )
                )
            # One last call WITHOUT tools so it has to produce content.
            kwargs = _build_completion_kwargs(model_name, convo, tools=None, stream=True)
            final = await _async_client.chat.completions.create(**kwargs)
            async for chunk in final:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        # Append the assistant message that requested the tools, then run
        # each tool and append a 'tool' role message with the result.
        convo.append(_materialize_assistant_tool_message(tool_calls_buffer))
        for tc in tool_calls_buffer:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError as e:
                args = {}
                log.warning("Invalid JSON args from model for %s: %s", tc["name"], e)
            log.info(
                "\033[33m[TOOL]\033[0m %s(%s)",
                tc["name"],
                json.dumps(args, ensure_ascii=False)[:120],
            )
            payload = await execute_tool(tc["name"] or "", args)
            n_rows = payload.get("n_rows")
            if "error" in payload:
                log.info("\033[33m[TOOL]\033[0m   ⮕ error: %s", payload["error"])
            else:
                log.info("\033[33m[TOOL]\033[0m   ⮕ %d filas", n_rows or 0)
            convo.append(tool_result_message(tc["id"] or "no-id", payload))

        # Loop again — model now sees the tool results and either streams
        # the final answer or requests more tools.


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
            "Analizá los siguientes fragmentos de informes económicos y asigná "
            "1-4 temas por fragmento usando SOLO las keys de esta taxonomía:\n"
            f"{taxonomy_for_prompt()}\n\n"
            f"Fragmentos:\n{numbered}\n\n"
            "Respondé SOLO con un array JSON donde cada elemento es un array de "
            "strings (keys de taxonomía) para el fragmento correspondiente.\n"
            'Ejemplo: [["inflacion", "tipo_cambio"], ["tasas"]]'
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
                        keys = canonicalize_topics([str(t) for t in tags])
                        all_tags.append(",".join(keys))
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
