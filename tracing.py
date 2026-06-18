"""Langfuse observability following langfuse/skills best practices.

Call ``init_tracing()`` after ``load_dotenv()`` and before importing modules that
create the Langfuse client.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langfuse import get_client, propagate_attributes

TRACE_ID_KEY = "langfuse_trace_id"
SCORE_NAME_USER_RATING = "user-rating"
MODEL_ID = "perceptron-mk1"


def init_tracing() -> None:
    """Initialize Langfuse after environment variables are loaded."""
    base_url = os.environ.get("LANGFUSE_BASE_URL")
    if base_url and not os.environ.get("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url

    if not is_configured():
        return

    try:
        client = get_client()
        if client.auth_check():
            return
    except Exception:
        pass


def is_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def video_metadata(path: Path) -> dict[str, Any]:
    """Safe trace input: filename and size only (no absolute paths)."""
    size_mb = round(path.stat().st_size / (1024 * 1024), 2) if path.is_file() else None
    return {
        "filename": path.name,
        "size_mb": size_mb,
        "suffix": path.suffix.lower(),
    }


def trace_context(*, flow: str, session_id: str | None, tags: list[str]):
    """Propagate session and feature tags to all nested observations."""
    session = session_id or "anonymous"
    return propagate_attributes(
        session_id=session,
        tags=["safety-analytics", f"flow-{flow.lower()}", *tags],
        metadata={"flow": flow, "model": MODEL_ID},
    )


def set_span_io(*, input_data: dict[str, Any] | None = None, output_data: dict[str, Any] | None = None) -> None:
    if not is_configured():
        return
    try:
        client = get_client()
        if client.get_current_trace_id() is None:
            return
        if input_data is not None:
            client.update_current_span(input=input_data)
        if output_data is not None:
            client.update_current_span(output=output_data)
    except Exception:
        pass


def set_generation_io(
    *,
    prompt: str,
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    if not is_configured():
        return
    try:
        client = get_client()
        if client.get_current_trace_id() is None:
            return
        client.update_current_generation(
            model=MODEL_ID,
            input={
                "prompt": prompt,
                "expects": kwargs.get("expects"),
                "reasoning": kwargs.get("reasoning"),
                "structured_output": "response_format" in kwargs,
            },
        )
        client.update_current_generation(
            output={
                "text_preview": (result.text or "")[:500] or None,
                "clip_count": len(result.clips or []),
                "error_count": len(result.errors or []),
            },
        )
    except Exception:
        pass


def structured_output_for_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Log evaluation-friendly fields without huge reasoning blobs on the span."""
    trace_output = {k: v for k, v in payload.items() if k != "reasoning"}
    if "reasoning" in payload:
        trace_output["has_reasoning"] = True
    return trace_output


def attach_trace_id(payload: dict[str, Any]) -> dict[str, Any]:
    if not is_configured():
        return payload
    try:
        trace_id = get_client().get_current_trace_id()
        if trace_id:
            payload[TRACE_ID_KEY] = trace_id
    except Exception:
        pass
    return payload


def pop_trace_id(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    trace_id = str(payload.pop(TRACE_ID_KEY, "") or "")
    return payload, trace_id


def submit_user_feedback(
    trace_id: str,
    *,
    helpful: bool,
    comment: str = "",
    flow: str = "",
) -> str:
    if not trace_id:
        return "Run an analysis first so feedback can be linked to a trace."
    if not is_configured():
        return "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."

    metadata: dict[str, str] = {}
    if flow:
        metadata["flow"] = flow

    try:
        get_client().create_score(
            trace_id=trace_id,
            name=SCORE_NAME_USER_RATING,
            value=1.0 if helpful else 0.0,
            data_type="BOOLEAN",
            comment=comment.strip() or None,
            metadata=metadata or None,
        )
        flush_tracing()
    except Exception as exc:
        return f"Failed to record feedback: {exc}"

    label = "positive" if helpful else "negative"
    return f"Recorded {label} feedback (`{SCORE_NAME_USER_RATING}`) for trace `{trace_id[:8]}…`."


def flush_tracing() -> None:
    if not is_configured():
        return
    try:
        get_client().flush()
    except Exception:
        pass


class FlowTrace:
    """Explicit Langfuse span for Gradio generators (no OTEL context across yields)."""

    def __init__(self) -> None:
        self.trace_id = ""
        self._root = None

    @classmethod
    def start(
        cls,
        name: str,
        *,
        flow: str,
        session_id: str | None,
        tags: list[str],
    ) -> FlowTrace:
        trace = cls()
        if not is_configured():
            return trace

        client = get_client()
        trace._root = client.start_observation(
            name=name,
            as_type="span",
            metadata={
                "flow": flow,
                "model": MODEL_ID,
                "session_id": session_id or "anonymous",
                "tags": ["safety-analytics", f"flow-{flow.lower()}", *tags],
            },
        )
        trace.trace_id = str(trace._root.trace_id or "")
        return trace

    def set_input(self, data: dict[str, Any]) -> None:
        if self._root is not None:
            self._root.update(input=data)

    def set_output(self, data: dict[str, Any]) -> None:
        if self._root is not None:
            self._root.update(output=data)

    def record_generation(self, *, prompt: str, kwargs: dict[str, Any], result: Any) -> None:
        if self._root is None:
            return
        generation = self._root.start_observation(
            name=MODEL_ID,
            as_type="generation",
            model=MODEL_ID,
            input={
                "prompt": prompt,
                "expects": kwargs.get("expects"),
                "reasoning": kwargs.get("reasoning"),
                "structured_output": "response_format" in kwargs,
            },
        )
        generation.update(
            output={
                "text_preview": (getattr(result, "text", None) or "")[:500] or None,
                "clip_count": len(getattr(result, "clips", None) or []),
                "error_count": len(getattr(result, "errors", None) or []),
            }
        )
        generation.end()

    def end(self) -> None:
        if self._root is not None:
            self._root.end()
            self._root = None
