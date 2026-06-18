"""Safety video analysis: structured incident review and visual search."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import perceptron
from langfuse import observe
from perceptron import pydantic_format, question, video
from perceptron.errors import TimeoutError as PerceptronTimeoutError

from models import (
    ClipMatch,
    OccupationalInjuryExtraction,
    SafetyReport,
    VisualSearchResult,
    WorkplaceIncidentReport,
)
from form5020 import fill_form5020
from incident_report import fill_incident_report
from compress_video import DEFAULT_TRIM_SECONDS, VideoPreparationError, prepare_video_for_upload
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

DEFAULT_SEARCH_QUERY = (
    "Find all moments where a worker is too close to moving equipment."
)

INJURY_REPORT_PROMPT = """\
Review this workplace security camera footage and extract information for an occupational \
injury incident (OSHA Form 5020).

Focus only on the incident visible in the video. Populate these fields when supported by \
direct visual evidence; otherwise leave them blank (empty string, empty list, or blank \
evidence_clip):

- apparent_body_part_affected (Form Q19): body part(s) injured or affected
- location_or_camera_area (Form Q20): where the event occurred
- other_workers_injured (Form Q23): "yes", "no", or blank if other workers' involvement is not visible
- equipment_or_materials_involved (Form Q24): equipment/materials that contributed to the incident
- activity_being_performed (Form Q25): action being performed at the time of the incident
- sequence_of_events (Form Q26): brief chronological description of the incident
- evidence_clip: MM:SS timestamp or range where the incident occurs (e.g. 00:12–00:18)

Use MM:SS timestamps aligned to the uploaded clip whenever an incident is described. \
If an incident is visible, evidence_clip must not be empty.

Do not invent employee names, medical diagnoses, calendar dates, or facts not visible on camera. \
If no incident is shown, leave incident fields blank and set requires_human_review to true.
"""

INCIDENT_CLIP_PROMPT = """\
Find the video segment that best shows the workplace injury or incident described below.

{context}

Return the clearest temporal clip where this incident occurs. Be precise with MM:SS timestamps \
aligned to the uploaded clip.
"""

WORKPLACE_INCIDENT_PROMPT = """\
Review this workplace security camera footage and answer the workplace incident questions below.

1. Did a safety incident occur that resulted in harm or injury to a person? Set incident_occurred \
to true only if harm or injury to a person is directly visible or clearly indicated. Near-misses \
without harm, unsafe acts without injury, or policy violations alone are not sufficient.

2. If incident_occurred is false, leave all other fields empty and stop.

3. If incident_occurred is true, populate:
   - timestamp: MM:SS time or range when the incident occurred (e.g. 00:12–00:18)
   - incident_description: who was involved and what happened
   - previous_activity: what the employee was doing just before the incident
   - potential_injuries: potential injuries sustained by involved parties
   - involved_equipment: objects or substances that caused or contributed to harm

Use only what is directly visible on camera. Do not invent names, diagnoses, or facts.
"""

WORKPLACE_INCIDENT_CLIP_PROMPT = """\
Find the video segment showing the workplace safety incident that resulted in harm or injury \
to a person.

{context}

Return the clearest temporal clip. Use MM:SS timestamps aligned to the uploaded clip.
"""

NO_INCIDENT_MESSAGE = "no incidents observed."

SEARCH_PROMPT_TEMPLATE = """\
{query}

Return every matching moment in the video as temporal clips. Be exhaustive: include repeated \
occurrences, near-misses, and borderline cases. Describe what visible evidence supports each match.
"""


def configure_perceptron(api_key: str) -> None:
    perceptron.configure(
        provider="perceptron",
        api_key=api_key,
        model="perceptron-mk1",
        timeout=PERCEPTRON_TIMEOUT,
    )


def seconds_to_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_evidence_clip_range(start_seconds: float, end_seconds: float | None) -> str:
    start = seconds_to_timestamp(start_seconds)
    if end_seconds is not None:
        return f"{start}–{seconds_to_timestamp(end_seconds)}"
    return start


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


def _incident_detected(extraction: OccupationalInjuryExtraction) -> bool:
    return bool(
        extraction.sequence_of_events.strip()
        or extraction.apparent_body_part_affected.strip()
        or extraction.activity_being_performed.strip()
        or extraction.equipment_or_materials_involved
    )


def _evidence_clip_missing(extraction: OccupationalInjuryExtraction) -> bool:
    return not extraction.evidence_clip.strip()


def _incident_clip_prompt(extraction: OccupationalInjuryExtraction) -> str:
    context_lines: list[str] = []
    if extraction.sequence_of_events.strip():
        context_lines.append(f"Incident: {extraction.sequence_of_events.strip()}")
    if extraction.activity_being_performed.strip():
        context_lines.append(f"Activity: {extraction.activity_being_performed.strip()}")
    if extraction.apparent_body_part_affected.strip():
        context_lines.append(f"Body part affected: {extraction.apparent_body_part_affected.strip()}")
    equipment = ", ".join(extraction.equipment_or_materials_involved)
    if equipment:
        context_lines.append(f"Equipment involved: {equipment}")
    context = "\n".join(context_lines) or (
        "Identify the workplace injury, near-miss, or hazardous incident shown in this footage."
    )
    return INCIDENT_CLIP_PROMPT.format(context=context)


def _ground_incident_evidence(
    path: Path,
    extraction: OccupationalInjuryExtraction,
) -> OccupationalInjuryExtraction:
    """Use clip grounding (like Flow B) when structured JSON omits evidence timestamps."""
    if not _incident_detected(extraction) or not _evidence_clip_missing(extraction):
        return extraction

    result = _perceptron_question(
        video(str(path)),
        _incident_clip_prompt(extraction),
        reasoning=True,
        expects="clip",
    )
    if not result.clips:
        return extraction

    clip = result.clips[0]
    return extraction.model_copy(
        update={
            "evidence_clip": _format_evidence_clip_range(
                clip.timestamp.at,
                clip.timestamp.until,
            ),
        }
    )


def _extraction_to_api_shape(report: OccupationalInjuryExtraction) -> dict:
    return report.model_dump()


def _workplace_incident_to_api_shape(report: WorkplaceIncidentReport) -> dict:
    if not report.incident_occurred:
        return {
            "incident_occurred": False,
            "timestamp": "",
            "incident_description": NO_INCIDENT_MESSAGE,
            "previous_activity": "",
            "potential_injuries": "",
            "involved_equipment": [],
        }
    return {
        "incident_occurred": True,
        "timestamp": report.timestamp,
        "incident_description": report.incident_description,
        "previous_activity": report.previous_activity,
        "potential_injuries": report.potential_injuries,
        "involved_equipment": report.involved_equipment,
    }


def _workplace_incident_clip_prompt(report: WorkplaceIncidentReport) -> str:
    context = report.incident_description.strip() or (
        "Identify the workplace safety incident that resulted in harm or injury to a person."
    )
    return WORKPLACE_INCIDENT_CLIP_PROMPT.format(context=context)


def _ground_workplace_incident_timestamp(
    path: Path,
    report: WorkplaceIncidentReport,
) -> WorkplaceIncidentReport:
    if not report.incident_occurred or report.timestamp.strip():
        return report

    result = _perceptron_question(
        video(str(path)),
        _workplace_incident_clip_prompt(report),
        reasoning=True,
        expects="clip",
    )
    if not result.clips:
        return report

    clip = result.clips[0]
    return report.model_copy(
        update={
            "timestamp": _format_evidence_clip_range(
                clip.timestamp.at,
                clip.timestamp.until,
            ),
        }
    )


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


def _stream_perceptron_question(media, prompt: str, **kwargs) -> Iterator[tuple[str, object | None]]:
    last_result = None
    for progress, result in consume_question_stream(media, prompt, **kwargs):
        if result is not None:
            last_result = result
        yield progress, result
    if last_result is None:
        raise RuntimeError("Perceptron stream ended without a final result.")


def _flow_error_update(payload: dict, *, trace: FlowTrace) -> FlowUpdate:
    trace.set_output(payload)
    display = dict(payload)
    return FlowUpdate(
        reasoning=format_progress(status="Analysis failed."),
        output=format_json(display),
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
    display = dict(payload)
    display.pop("form_5020_pdf", None)
    display.pop("incident_report_pdf", None)
    display.pop("reasoning", None)
    return FlowUpdate(
        reasoning=reasoning_md,
        output=format_json(display),
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
    path = Path(video_path)

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

        if result.errors:
            payload = {"error": "Model returned validation warnings.", "details": result.errors}
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        try:
            report = SafetyReport.model_validate_json(result.text)
        except Exception as exc:
            payload = {
                "error": "Failed to parse structured safety report.",
                "raw_text": result.text,
                "details": str(exc),
            }
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        payload = _report_to_api_shape(report)
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        payload["structured_output_schema"] = SafetyReport.__name__
        payload["event_count"] = len(report.events)

        try:
            pdf_path = fill_incident_report(
                payload,
                video_path=path,
                record_model_call=lambda prompt, kwargs, result: set_generation_io(
                    prompt=prompt,
                    kwargs=kwargs,
                    result=result,
                ),
            )
            payload["incident_report_pdf"] = str(pdf_path)
        except Exception as exc:
            payload["pdf_error"] = f"Could not generate incident report PDF: {exc}"

        set_span_io(output_data=structured_output_for_trace(payload))
        return attach_trace_id(payload)


def incident_review_stream(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> Iterator[FlowUpdate]:
    """Flow A with live reasoning streamed to the UI."""
    path = Path(video_path)
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
            for progress, partial in _stream_perceptron_question(
                video(str(path)),
                prompt,
                **gen_kwargs,
            ):
                if partial is None:
                    yield FlowUpdate(reasoning=progress, trace_id=trace.trace_id)
                else:
                    result = partial
        except PerceptronTimeoutError:
            yield _flow_error_update(_timeout_payload(), trace=trace)
            return
        except RuntimeError as exc:
            yield _flow_error_update({"error": str(exc)}, trace=trace)
            return

        if result.errors:
            yield _flow_error_update(
                {"error": "Model returned validation warnings.", "details": result.errors},
                trace=trace,
            )
            return

        try:
            report = SafetyReport.model_validate_json(result.text)
        except Exception as exc:
            yield _flow_error_update(
                {
                    "error": "Failed to parse structured safety report.",
                    "raw_text": result.text,
                    "details": str(exc),
                },
                trace=trace,
            )
            return

        payload = _report_to_api_shape(report)
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        payload["structured_output_schema"] = SafetyReport.__name__
        payload["event_count"] = len(report.events)

        yield FlowUpdate(
            reasoning=format_progress(status="Enriching incident report fields…"),
            trace_id=trace.trace_id,
        )

        pdf_path: str | None = None
        try:
            pdf_path = str(
                fill_incident_report(
                    payload,
                    video_path=path,
                    record_model_call=trace.record_generation,
                )
            )
            payload["incident_report_pdf"] = pdf_path
        except Exception as exc:
            payload["pdf_error"] = f"Could not generate incident report PDF: {exc}"

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


def visual_search_stream(
    video_path: str | Path,
    query: str = DEFAULT_SEARCH_QUERY,
    *,
    session_id: str | None = None,
) -> Iterator[FlowUpdate]:
    """Flow B with live reasoning streamed to the UI."""
    path = Path(video_path)
    cleaned_query = query.strip() or DEFAULT_SEARCH_QUERY
    prompt = SEARCH_PROMPT_TEMPLATE.format(query=cleaned_query)
    trace = FlowTrace.start(
        "flow-b-visual-search",
        flow="B",
        session_id=session_id,
        tags=["visual-search", "video-clipping"],
    )
    gen_kwargs = {"reasoning": True, "expects": "clip"}

    try:
        yield FlowUpdate(reasoning=format_progress(status="Preparing video…"), trace_id=trace.trace_id)

        path, prep_error = _prepare_video(path)
        if prep_error:
            yield _flow_error_update(prep_error, trace=trace)
            return

        trace.set_input(
            {
                "video": video_metadata(path),
                "query": cleaned_query,
                "flow": "visual_search",
            }
        )

        validation_error = _validate_video(path)
        if validation_error:
            yield _flow_error_update(validation_error, trace=trace)
            return

        try:
            result = None
            for progress, partial in _stream_perceptron_question(
                video(str(path)),
                prompt,
                **gen_kwargs,
            ):
                if partial is None:
                    yield FlowUpdate(reasoning=progress, trace_id=trace.trace_id)
                else:
                    result = partial
        except PerceptronTimeoutError:
            yield _flow_error_update(_timeout_payload(), trace=trace)
            return
        except RuntimeError as exc:
            yield _flow_error_update({"error": str(exc)}, trace=trace)
            return

        matches: list[ClipMatch] = []
        for clip in result.clips or []:
            end = clip.timestamp.until
            matches.append(
                ClipMatch(
                    start_time=seconds_to_timestamp(clip.timestamp.at),
                    end_time=seconds_to_timestamp(end) if end is not None else None,
                    label=clip.mention or "match",
                    description=clip.mention or "Matching segment identified in video.",
                )
            )

        if not matches and result.text:
            matches.append(
                ClipMatch(
                    start_time="00:00",
                    end_time=None,
                    label="analysis",
                    description=result.text.strip(),
                )
            )

        search_result = VisualSearchResult(
            query=cleaned_query,
            summary=(
                f"Found {len(matches)} matching segment(s) for the query."
                if matches
                else "No matching segments were identified."
            ),
            match_count=len(matches),
            matches=matches,
        )

        payload = search_result.model_dump()
        payload["structured_output_schema"] = VisualSearchResult.__name__
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        if result.text:
            payload["narrative"] = result.text
        if result.errors:
            payload["warnings"] = result.errors
        yield _flow_success_update(
            payload,
            trace=trace,
            reasoning_md=format_progress(status="Complete.", reasoning=result.reasoning or ""),
            generation=(prompt, gen_kwargs, result),
        )
    finally:
        trace.end()


def occupational_injury_report_stream(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> Iterator[FlowUpdate]:
    """Flow C with live reasoning streamed to the UI."""
    path = Path(video_path)
    trace = FlowTrace.start(
        "flow-c-injury-report",
        flow="C",
        session_id=session_id,
        tags=["injury-report", "form-5020"],
    )
    gen_kwargs = {
        "reasoning": True,
        "response_format": pydantic_format(OccupationalInjuryExtraction, strict=True),
    }
    prompt = INJURY_REPORT_PROMPT

    try:
        yield FlowUpdate(reasoning=format_progress(status="Preparing video…"), trace_id=trace.trace_id)

        path, prep_error = _prepare_video(path, trim_seconds=DEFAULT_TRIM_SECONDS)
        if prep_error:
            yield _flow_error_update(prep_error, trace=trace)
            return

        trace.set_input(
            {
                "video": video_metadata(path),
                "flow": "occupational_injury_report",
            }
        )

        validation_error = _validate_video(path)
        if validation_error:
            yield _flow_error_update(validation_error, trace=trace)
            return

        try:
            result = None
            for progress, partial in _stream_perceptron_question(
                video(str(path)),
                prompt,
                **gen_kwargs,
            ):
                if partial is None:
                    yield FlowUpdate(reasoning=progress, trace_id=trace.trace_id)
                else:
                    result = partial
        except PerceptronTimeoutError:
            yield _flow_error_update(_timeout_payload(), trace=trace)
            return
        except RuntimeError as exc:
            yield _flow_error_update({"error": str(exc)}, trace=trace)
            return

        if result.errors:
            yield _flow_error_update(
                {"error": "Model returned validation warnings.", "details": result.errors},
                trace=trace,
            )
            return

        try:
            extraction = OccupationalInjuryExtraction.model_validate_json(result.text)
        except Exception as exc:
            yield _flow_error_update(
                {
                    "error": "Failed to parse occupational injury extraction.",
                    "raw_text": result.text,
                    "details": str(exc),
                },
                trace=trace,
            )
            return

        if _incident_detected(extraction) and _evidence_clip_missing(extraction):
            yield FlowUpdate(
                reasoning=format_progress(status="Locating incident timestamps…"),
                trace_id=trace.trace_id,
            )
            extraction = _ground_incident_evidence(path, extraction)

        payload = _extraction_to_api_shape(extraction)
        payload["structured_output_schema"] = OccupationalInjuryExtraction.__name__

        pdf_path: str | None = None
        try:
            pdf_path = str(fill_form5020(payload))
            payload["form_5020_pdf"] = pdf_path
        except Exception as exc:
            payload["pdf_error"] = f"Could not generate Form 5020 PDF: {exc}"

        if result.reasoning:
            payload["reasoning"] = result.reasoning
        yield _flow_success_update(
            payload,
            trace=trace,
            reasoning_md=format_progress(status="Complete.", reasoning=result.reasoning or ""),
            pdf_path=pdf_path,
            generation=(prompt, gen_kwargs, result),
        )
    finally:
        trace.end()


@observe(name="flow-b-visual-search", capture_input=False, capture_output=False)
def visual_search(
    video_path: str | Path,
    query: str = DEFAULT_SEARCH_QUERY,
    *,
    session_id: str | None = None,
) -> dict:
    """Flow B: natural-language visual search with timestamped clip matches."""
    path = Path(video_path)
    cleaned_query = query.strip() or DEFAULT_SEARCH_QUERY
    prompt = SEARCH_PROMPT_TEMPLATE.format(query=cleaned_query)

    with trace_context(flow="B", session_id=session_id, tags=["visual-search", "video-clipping"]):
        path, prep_error = _prepare_video(path)
        set_span_io(
            input_data={
                "video": video_metadata(path),
                "query": cleaned_query,
                "flow": "visual_search",
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
                prompt,
                reasoning=True,
                expects="clip",
            )
        except PerceptronTimeoutError:
            payload = _timeout_payload()
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        matches: list[ClipMatch] = []
        for clip in result.clips or []:
            end = clip.timestamp.until
            matches.append(
                ClipMatch(
                    start_time=seconds_to_timestamp(clip.timestamp.at),
                    end_time=seconds_to_timestamp(end) if end is not None else None,
                    label=clip.mention or "match",
                    description=clip.mention or "Matching segment identified in video.",
                )
            )

        if not matches and result.text:
            matches.append(
                ClipMatch(
                    start_time="00:00",
                    end_time=None,
                    label="analysis",
                    description=result.text.strip(),
                )
            )

        search_result = VisualSearchResult(
            query=cleaned_query,
            summary=(
                f"Found {len(matches)} matching segment(s) for the query."
                if matches
                else "No matching segments were identified."
            ),
            match_count=len(matches),
            matches=matches,
        )

        payload = search_result.model_dump()
        payload["structured_output_schema"] = VisualSearchResult.__name__
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        if result.text:
            payload["narrative"] = result.text
        if result.errors:
            payload["warnings"] = result.errors
        set_span_io(output_data=structured_output_for_trace(payload))
        return attach_trace_id(payload)


@observe(name="flow-c-injury-report", capture_input=False, capture_output=False)
def occupational_injury_report(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> dict:
    """Flow C: extract Form 5020 fields from video and produce a downloadable PDF."""
    path = Path(video_path)

    with trace_context(flow="C", session_id=session_id, tags=["injury-report", "form-5020"]):
        path, prep_error = _prepare_video(path, trim_seconds=DEFAULT_TRIM_SECONDS)
        set_span_io(
            input_data={
                "video": video_metadata(path),
                "flow": "occupational_injury_report",
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
                INJURY_REPORT_PROMPT,
                reasoning=True,
                response_format=pydantic_format(OccupationalInjuryExtraction, strict=True),
            )
        except PerceptronTimeoutError:
            payload = _timeout_payload()
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        if result.errors:
            payload = {"error": "Model returned validation warnings.", "details": result.errors}
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        try:
            extraction = OccupationalInjuryExtraction.model_validate_json(result.text)
        except Exception as exc:
            payload = {
                "error": "Failed to parse occupational injury extraction.",
                "raw_text": result.text,
                "details": str(exc),
            }
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        if _incident_detected(extraction) and _evidence_clip_missing(extraction):
            extraction = _ground_incident_evidence(path, extraction)

        payload = _extraction_to_api_shape(extraction)
        payload["structured_output_schema"] = OccupationalInjuryExtraction.__name__

        try:
            pdf_path = fill_form5020(payload)
            payload["form_5020_pdf"] = str(pdf_path)
        except Exception as exc:
            payload["pdf_error"] = f"Could not generate Form 5020 PDF: {exc}"

        if result.reasoning:
            payload["reasoning"] = result.reasoning
        set_span_io(output_data=structured_output_for_trace(payload))
        return attach_trace_id(payload)


def workplace_incident_report_stream(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> Iterator[FlowUpdate]:
    """Flow D: workplace incident Q&A with live reasoning streamed to the UI."""
    path = Path(video_path)
    trace = FlowTrace.start(
        "flow-d-workplace-incident",
        flow="D",
        session_id=session_id,
        tags=["workplace-incident", "structured-output"],
    )
    gen_kwargs = {
        "reasoning": True,
        "response_format": pydantic_format(WorkplaceIncidentReport, strict=True),
    }
    prompt = WORKPLACE_INCIDENT_PROMPT

    try:
        yield FlowUpdate(reasoning=format_progress(status="Preparing video…"), trace_id=trace.trace_id)

        path, prep_error = _prepare_video(path)
        if prep_error:
            yield _flow_error_update(prep_error, trace=trace)
            return

        trace.set_input({"video": video_metadata(path), "flow": "workplace_incident_report"})

        validation_error = _validate_video(path)
        if validation_error:
            yield _flow_error_update(validation_error, trace=trace)
            return

        try:
            result = None
            for progress, partial in _stream_perceptron_question(
                video(str(path)),
                prompt,
                **gen_kwargs,
            ):
                if partial is None:
                    yield FlowUpdate(reasoning=progress, trace_id=trace.trace_id)
                else:
                    result = partial
        except PerceptronTimeoutError:
            yield _flow_error_update(_timeout_payload(), trace=trace)
            return
        except RuntimeError as exc:
            yield _flow_error_update({"error": str(exc)}, trace=trace)
            return

        if result.errors:
            yield _flow_error_update(
                {"error": "Model returned validation warnings.", "details": result.errors},
                trace=trace,
            )
            return

        try:
            report = WorkplaceIncidentReport.model_validate_json(result.text)
        except Exception as exc:
            yield _flow_error_update(
                {
                    "error": "Failed to parse workplace incident report.",
                    "raw_text": result.text,
                    "details": str(exc),
                },
                trace=trace,
            )
            return

        if not report.incident_occurred:
            payload = _workplace_incident_to_api_shape(report)
            payload["structured_output_schema"] = WorkplaceIncidentReport.__name__
            yield _flow_success_update(
                payload,
                trace=trace,
                reasoning_md=format_progress(status="Complete."),
                generation=(prompt, gen_kwargs, result),
            )
            return

        if not report.timestamp.strip():
            yield FlowUpdate(
                reasoning=format_progress(status="Locating incident timestamp…"),
                trace_id=trace.trace_id,
            )
            report = _ground_workplace_incident_timestamp(path, report)

        payload = _workplace_incident_to_api_shape(report)
        payload["structured_output_schema"] = WorkplaceIncidentReport.__name__
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        yield _flow_success_update(
            payload,
            trace=trace,
            reasoning_md=format_progress(status="Complete.", reasoning=result.reasoning or ""),
            generation=(prompt, gen_kwargs, result),
        )
    finally:
        trace.end()


@observe(name="flow-d-workplace-incident", capture_input=False, capture_output=False)
def workplace_incident_report(
    video_path: str | Path,
    *,
    session_id: str | None = None,
) -> dict:
    """Flow D: workplace incident Q&A from security footage."""
    path = Path(video_path)

    with trace_context(flow="D", session_id=session_id, tags=["workplace-incident", "structured-output"]):
        path, prep_error = _prepare_video(path)
        set_span_io(
            input_data={
                "video": video_metadata(path),
                "flow": "workplace_incident_report",
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
                WORKPLACE_INCIDENT_PROMPT,
                reasoning=True,
                response_format=pydantic_format(WorkplaceIncidentReport, strict=True),
            )
        except PerceptronTimeoutError:
            payload = _timeout_payload()
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        if result.errors:
            payload = {"error": "Model returned validation warnings.", "details": result.errors}
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        try:
            report = WorkplaceIncidentReport.model_validate_json(result.text)
        except Exception as exc:
            payload = {
                "error": "Failed to parse workplace incident report.",
                "raw_text": result.text,
                "details": str(exc),
            }
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        if not report.incident_occurred:
            payload = _workplace_incident_to_api_shape(report)
            payload["structured_output_schema"] = WorkplaceIncidentReport.__name__
            set_span_io(output_data=payload)
            return attach_trace_id(payload)

        if not report.timestamp.strip():
            report = _ground_workplace_incident_timestamp(path, report)

        payload = _workplace_incident_to_api_shape(report)
        payload["structured_output_schema"] = WorkplaceIncidentReport.__name__
        if result.reasoning:
            payload["reasoning"] = result.reasoning
        set_span_io(output_data=structured_output_for_trace(payload))
        return attach_trace_id(payload)


def format_json(data: dict) -> str:
    return json.dumps(data, indent=2)
