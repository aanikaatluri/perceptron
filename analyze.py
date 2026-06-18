"""Safety video analysis: structured incident review"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import perceptron
from langfuse import observe
from perceptron import pydantic_format, question, video
from perceptron.errors import TimeoutError as PerceptronTimeoutError

from models import SafetyReport
from incident_report import fill_incident_report
from compress_video import VideoPreparationError, prepare_video_for_upload
from streaming import FlowUpdate, consume_question_stream, format_progress
from tracing import (
    FlowTrace,
    attach_trace_id,
    set_generation_io,
    set_span_io,
    structured_output_for_trace,
    trace_context,
    video_metadata,
)

MAX_VIDEO_MB = 15
PERCEPTRON_TIMEOUT = float(os.environ.get("PERCEPTRON_TIMEOUT", "300"))

TIMEOUT_ERROR = (
    "Perceptron request timed out after {timeout:.0f}s. "
    "Try a shorter clip (under ~2 minutes): "
    "python compress_video.py your_clip.mov --trim 128"
)

INCIDENT_REVIEW_PROMPT = """\
Review this workplace security/safety camera footage.

Identify every distinct safety event, near-miss, policy violation, or hazardous state change \
visible in the video. For each event, cite only what is directly observable on camera.

Use MM:SS timestamps aligned to the uploaded clip. If no incidents are visible, return an empty \
events list and explain why in overall_summary.
"""


def configure_perceptron(api_key: str) -> None:
    perceptron.configure(
        provider="perceptron",
        api_key=api_key,
        model="perceptron-mk1",
        timeout=PERCEPTRON_TIMEOUT,
    )


def _validate_video(path: Path) -> dict | None:
    if not path.is_file():
        return {"error": f"Video not found: {path}"}

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        return {
            "error": (
                f"Video is {size_mb:.1f} MB after preparation. "
                f"Try a shorter clip (under ~{MAX_VIDEO_MB} MB)."
            )
        }
    return None


def _prepare_video(path: Path, *, trim_seconds: float | None = None) -> tuple[Path, dict | None]:
    try:
        return prepare_video_for_upload(
            path,
            max_mb=MAX_VIDEO_MB,
            trim_seconds=trim_seconds,
        ), None
    except VideoPreparationError as exc:
        return path, {"error": str(exc)}
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return path, {"error": str(exc)}


def _timeout_payload() -> dict:
    return {"error": TIMEOUT_ERROR.format(timeout=PERCEPTRON_TIMEOUT)}


def _report_to_api_shape(report: SafetyReport) -> dict:
    return {
        "summary": report.overall_summary,
        "events": [event.model_dump() for event in report.events],
        "requires_human_review": report.requires_human_review,
    }


def _payload_from_result(result) -> tuple[dict | None, dict | None]:
    """Return ``(success_payload, error_payload)``; exactly one side is non-None."""
    if result.errors:
        return None, {"error": "Model returned validation warnings.", "details": result.errors}

    try:
        report = SafetyReport.model_validate_json(result.text)
    except Exception as exc:
        return None, {
            "error": "Failed to parse structured safety report.",
            "raw_text": result.text,
            "details": str(exc),
        }

    payload = _report_to_api_shape(report)
    if result.reasoning:
        payload["reasoning"] = result.reasoning
    payload["structured_output_schema"] = SafetyReport.__name__
    payload["event_count"] = len(report.events)
    return payload, None


def _display_json(payload: dict) -> str:
    display = {k: v for k, v in payload.items() if k not in ("incident_report_pdf", "reasoning")}
    return format_json(display)


def _attach_incident_pdf(
    payload: dict,
    *,
    video_path: Path,
    source_video_path: Path,
    record_model_call,
) -> str | None:
    try:
        pdf_path = str(
            fill_incident_report(
                payload,
                video_path=video_path,
                source_video_path=source_video_path,
                record_model_call=record_model_call,
            )
        )
        payload["incident_report_pdf"] = pdf_path
        return pdf_path
    except Exception as exc:
        payload["pdf_error"] = f"Could not generate incident report PDF: {exc}"
        return None


@observe(
    name="perceptron-mk1",
    as_type="generation",
    capture_input=False,
    capture_output=False,
)
def _perceptron_question(media, prompt: str, **kwargs):
    result = question(media, prompt, **kwargs)
    set_generation_io(prompt=prompt, kwargs=kwargs, result=result)
    return result


def _flow_error_update(payload: dict, *, trace: FlowTrace) -> FlowUpdate:
    trace.set_output(payload)
    return FlowUpdate(
        reasoning=format_progress(status="Analysis failed."),
        output=format_json(payload),
        trace_id=trace.trace_id,
    )


def _flow_success_update(
    payload: dict,
    *,
    trace: FlowTrace,
    reasoning_md: str,
    pdf_path: str | None = None,
    generation: tuple[str, dict[str, Any], Any] | None = None,
) -> FlowUpdate:
    if generation is not None:
        prompt, kwargs, result = generation
        trace.record_generation(prompt=prompt, kwargs=kwargs, result=result)
    trace.set_output(structured_output_for_trace(payload))
    return FlowUpdate(
        reasoning=reasoning_md,
        output=_display_json(payload),
        trace_id=trace.trace_id,
        pdf_path=pdf_path,
    )


@observe(name="flow-a-incident-review", capture_input=False, capture_output=False)
def incident_review(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> dict:
    """Flow A: structured safety incident report."""
    source_path = Path(video_path)
    path = source_path

    with trace_context(flow="A", session_id=session_id, tags=["incident-review", "structured-output"]):
        path, prep_error = _prepare_video(path)
        set_span_io(
            input_data={
                "video": video_metadata(path),
                "flow": "incident_review",
                "prepared": prep_error is None and path.suffix.lower() == ".mp4",
            }
        )

        if prep_error:
            set_span_io(output_data=prep_error)
            return attach_trace_id(prep_error)

        validation_error = _validate_video(path)
        if validation_error:
            set_span_io(output_data=validation_error)
            return attach_trace_id(validation_error)

        try:
            result = _perceptron_question(
                video(str(path)),
                INCIDENT_REVIEW_PROMPT,
                reasoning=True,
                response_format=pydantic_format(SafetyReport, strict=True),
            )
        except PerceptronTimeoutError:
            payload = _timeout_payload()
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        payload, error = _payload_from_result(result)
        if error:
            set_span_io(output_data=error)
            return attach_trace_id(error)

        _attach_incident_pdf(
            payload,
            video_path=path,
            source_video_path=source_path,
            record_model_call=lambda prompt, kwargs, result: set_generation_io(
                prompt=prompt,
                kwargs=kwargs,
                result=result,
            ),
        )

        set_span_io(output_data=structured_output_for_trace(payload))
        return attach_trace_id(payload)


def incident_review_stream(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> Iterator[FlowUpdate]:
    """Flow A with live reasoning streamed to the UI."""
    source_path = Path(video_path)
    path = source_path
    trace = FlowTrace.start(
        "flow-a-incident-review",
        flow="A",
        session_id=session_id,
        tags=["incident-review", "structured-output"],
    )
    gen_kwargs = {
        "reasoning": True,
        "response_format": pydantic_format(SafetyReport, strict=True),
    }
    prompt = INCIDENT_REVIEW_PROMPT

    try:
        yield FlowUpdate(reasoning=format_progress(status="Preparing video…"), trace_id=trace.trace_id)

        path, prep_error = _prepare_video(path)
        if prep_error:
            trace.set_input({"flow": "incident_review"})
            yield _flow_error_update(prep_error, trace=trace)
            return

        trace.set_input({"video": video_metadata(path), "flow": "incident_review"})

        validation_error = _validate_video(path)
        if validation_error:
            yield _flow_error_update(validation_error, trace=trace)
            return

        try:
            result = None
            for progress, stream_output, partial in consume_question_stream(
                video(str(path)),
                prompt,
                **gen_kwargs,
            ):
                if partial is None:
                    yield FlowUpdate(
                        reasoning=progress,
                        output=stream_output,
                        trace_id=trace.trace_id,
                    )
                else:
                    result = partial
        except PerceptronTimeoutError:
            yield _flow_error_update(_timeout_payload(), trace=trace)
            return
        except RuntimeError as exc:
            yield _flow_error_update({"error": str(exc)}, trace=trace)
            return

        payload, error = _payload_from_result(result)
        if error:
            yield _flow_error_update(error, trace=trace)
            return

        yield FlowUpdate(
            reasoning=format_progress(
                status="Safety report ready. Compiling incident report…",
                reasoning=result.reasoning or "",
            ),
            output=_display_json(payload),
            trace_id=trace.trace_id,
        )

        pdf_path = _attach_incident_pdf(
            payload,
            video_path=path,
            source_video_path=source_path,
            record_model_call=trace.record_generation,
        )

        reasoning_md = format_progress(status="Complete.", reasoning=result.reasoning or "")
        if pdf_path is None and payload.get("pdf_error"):
            reasoning_md += f"\n\n**Incident report PDF:** {payload['pdf_error']}"

        yield _flow_success_update(
            payload,
            trace=trace,
            reasoning_md=reasoning_md,
            pdf_path=pdf_path,
            generation=(prompt, gen_kwargs, result),
        )
    finally:
        trace.end()


def format_json(data: dict) -> str:
    return json.dumps(data, indent=2)
