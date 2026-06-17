"""Helpers for streaming Perceptron reasoning and output to the UI."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from perceptron import question
from perceptron.dsl.perceive import PerceiveResult, _perceive_result_from_response
from perceptron.errors import TimeoutError as PerceptronTimeoutError


@dataclass
class FlowUpdate:
    """Incremental UI state for a Gradio analysis flow."""

    reasoning: str = ""
    output: str = ""
    trace_id: str = ""
    pdf_path: str | None = None


def format_progress(
    *,
    status: str,
    reasoning: str = "",
    output_preview: str = "",
) -> str:
    sections = [f"**{status}**"]
    if reasoning:
        sections.append(f"### Reasoning\n\n{reasoning}")
    if output_preview:
        sections.append(f"### Output (streaming)\n\n```\n{output_preview}\n```")
    return "\n\n".join(sections)


def _is_stream_transport_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "streamreset",
        "remote_reset",
        "connection reset",
        "connection error",
        "read operation timed out",
        "timeout",
        "broken pipe",
        "incomplete chunked read",
    )
    return any(marker in message for marker in markers)


def _non_streaming_question(media, prompt: str, **kwargs: Any) -> PerceiveResult:
    kwargs = {**kwargs, "stream": False}
    return question(media, prompt, **kwargs)


def consume_question_stream(
    media,
    prompt: str,
    *,
    status: str = "Analyzing video with Perceptron Mk1…",
    **kwargs: Any,
) -> Iterator[tuple[str, PerceiveResult | None]]:
    """Yield ``(progress_markdown, None)`` while streaming; final yield includes the result."""
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    current_status = status
    stream_kwargs = {**kwargs, "stream": True}

    yield format_progress(status=current_status), None

    try:
        stream = question(media, prompt, **stream_kwargs)
    except PerceptronTimeoutError:
        raise
    except Exception as exc:
        if _is_stream_transport_error(exc):
            yield from _fallback_non_stream(
                media,
                prompt,
                stream_kwargs,
                status="Live stream unavailable — waiting for final response…",
            )
            return
        raise RuntimeError(str(exc)) from exc

    final_result: PerceiveResult | None = None

    try:
        for event in stream:
            event_type = event.get("type")

            if event_type == "reasoning.delta":
                reasoning_parts.append(event.get("chunk") or "")
                yield format_progress(
                    status=current_status,
                    reasoning="".join(reasoning_parts),
                    output_preview=_preview_text(text_parts),
                ), None
                continue

            if event_type == "text.delta":
                text_parts.append(event.get("chunk") or "")
                current_status = "Generating structured output…"
                yield format_progress(
                    status=current_status,
                    reasoning="".join(reasoning_parts),
                    output_preview=_preview_text(text_parts),
                ), None
                continue

            if event_type == "error":
                message = str(event.get("message") or "Perceptron stream error")
                if message == "timeout":
                    raise PerceptronTimeoutError("request timed out")
                if _is_stream_transport_error(RuntimeError(message)):
                    yield from _fallback_non_stream(
                        media,
                        prompt,
                        stream_kwargs,
                        status="Live stream interrupted — waiting for final response…",
                    )
                    return
                raise RuntimeError(message)

            if event_type == "final":
                payload = event.get("result") or {}
                issues = payload.get("errors") or []
                final_result = _perceive_result_from_response(payload, issues)
                yield format_progress(
                    status="Finalizing response…",
                    reasoning="".join(reasoning_parts),
                    output_preview=_preview_text(text_parts),
                ), final_result
    except PerceptronTimeoutError:
        raise
    except Exception as exc:
        if _is_stream_transport_error(exc):
            yield from _fallback_non_stream(
                media,
                prompt,
                stream_kwargs,
                status="Live stream interrupted — waiting for final response…",
            )
            return
        raise RuntimeError(str(exc)) from exc

    if final_result is None:
        yield from _fallback_non_stream(
            media,
            prompt,
            stream_kwargs,
            status="Stream ended early — waiting for final response…",
        )


def _fallback_non_stream(
    media,
    prompt: str,
    stream_kwargs: dict[str, Any],
    *,
    status: str,
) -> Iterator[tuple[str, PerceiveResult | None]]:
    yield format_progress(status=status), None
    try:
        result = _non_streaming_question(media, prompt, **stream_kwargs)
    except PerceptronTimeoutError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    reasoning = getattr(result, "reasoning", None) or ""
    yield format_progress(status="Complete.", reasoning=reasoning), result


def _preview_text(parts: list[str], limit: int = 4000) -> str:
    text = "".join(parts)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…"
