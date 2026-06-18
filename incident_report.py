"""Generate and fill a Workplace Incident Report PDF from Flow A safety data."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from perceptron import question, video
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

ModelCallRecorder = Callable[[str, dict[str, Any], Any], None] | None

INJURY_INFERENCE_PROMPT = (
    "Look at the clip {timestamp_range} of this incident where {description}. "
    "Infer the most likely potential injuries that could have been sustained from this incident based on the clip. Be concise and only include the most likely injuries. Always begin your response with 'According to the video footage, the most likely injuries are:'"
)

INCIDENT_DATETIME_PROMPT = (
    "Look for an on-screen timestamp in the video during the clip {timestamp_range}. "
    "If visible, return the date and time on one line in this exact format: MM/DD/YY HH:MM "
    "(example: 03/15/26 14:32). If no on-screen date and time are visible, return an empty string."
)

FIELD_DATE_OF_INCIDENT = "date_of_incident"
FIELD_TIME_EMPLOYEE_BEGAN_WORK = "time_employee_began_work"
FIELD_LOCATION = "location"
FIELD_TIME_OF_INCIDENT = "time_of_incident"
FIELD_ACTIVITY_BEFORE = "activity_before_incident"
FIELD_INCIDENT_DESCRIPTION = "incident_description"
FIELD_INJURIES_YES = "injuries_yes"
FIELD_INJURIES_NO = "injuries_no"
FIELD_INJURY_DESCRIPTION = "injury_or_illness_description"
FIELD_OBJECTS_SUBSTANCES = "objects_or_substances_that_harmed_employee"

_INJURY_HINTS = re.compile(
    r"\b("
    r"injur(?:y|ies|ed)|ill(?:ness)?|harm|hurt|wound|lacerat|fractur|burn|"
    r"crush|amputat|strain|sprain|contusion|bleed|unconscious|"
    r"slip|trip|fall|struck|hit by|caught in|pinch"
    r")\b",
    re.IGNORECASE,
)

_EQUIPMENT_HINTS = re.compile(
    r"\b("
    r"forklift|pallet jack|crane|hoist|conveyor|ladder|scaffold|vehicle|truck|"
    r"machine|press|saw|drill|grinder|chemical|solvent|acid|caustic|"
    r"box|crate|barrel|drum|tool|rack|shelf|door|gate|barrier"
    r")\b",
    re.IGNORECASE,
)

_INCIDENT_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
_INCIDENT_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")


def _normalize_incident_date(date_str: str) -> str:
    month, day, year = date_str.split("/")
    if len(year) == 4:
        year = year[-2:]
    return f"{int(month):02d}/{int(day):02d}/{year}"


def _normalize_incident_time(time_str: str) -> str:
    hour, minute, *_rest = time_str.split(":")
    return f"{int(hour):02d}:{minute[:2]}"


def _parse_incident_datetime_response(text: str) -> tuple[str, str]:
    """Extract MM/DD/YY and HH:MM from the enrichment model response."""
    cleaned = text.strip()
    if not cleaned:
        return "", ""

    date_match = _INCIDENT_DATE_RE.search(cleaned)
    time_match = _INCIDENT_TIME_RE.search(cleaned)
    incident_date = _normalize_incident_date(date_match.group(1)) if date_match else ""
    incident_time = _normalize_incident_time(time_match.group(1)) if time_match else ""
    return incident_date, incident_time


def _severity_rank(event: dict) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(event.get("severity", "")).lower(), 3)


def _primary_event(events: list[dict]) -> dict | None:
    if not events:
        return None
    return min(events, key=_severity_rank)


def _text_suggests_injury(text: str) -> bool:
    return bool(_INJURY_HINTS.search(text))


def _event_text_blob(event: dict) -> str:
    return " ".join(
        str(event.get(key, "") or "").replace("_", " ")
        for key in ("event_type", "description", "visual_evidence")
    )


def _suggests_injury(events: list[dict]) -> bool:
    if not events:
        return False
    for event in events:
        blob = _event_text_blob(event)
        if _text_suggests_injury(blob):
            return True
        if event.get("severity") == "high" and event.get("event_type"):
            return True
    return False


def _format_timestamp(event: dict) -> str:
    start = (event.get("start_time") or "").strip()
    end = (event.get("end_time") or "").strip()
    if start and end and end != start:
        return f"{start}–{end}"
    return start


def _prompt_timestamp_range(event: dict) -> str:
    return _format_timestamp(event).replace("\u2013", "-").replace("\u2014", "-")


def _incident_context(report: dict) -> tuple[str, str] | None:
    """Return (timestamp_range, description) for the primary safety event."""
    events = list(report.get("events") or [])
    primary = _primary_event(events)
    if not primary:
        return None

    timestamp_range = _prompt_timestamp_range(primary)
    if not timestamp_range:
        return None

    description = (primary.get("description") or "").strip()
    if not description:
        description = (report.get("summary") or "").strip()
    if not description:
        return None

    return timestamp_range, description


def _perceptron_text_question(
    video_path: Path,
    prompt: str,
    *,
    label: str,
    record_model_call: ModelCallRecorder = None,
    warn_if_empty: bool = True,
) -> str:
    kwargs = {"reasoning": True}
    result = question(video(str(video_path)), prompt, **kwargs)

    if record_model_call is not None:
        try:
            record_model_call(prompt=prompt, kwargs=kwargs, result=result)
        except Exception as exc:
            logger.warning(
                "Incident report enrichment (%s) tracing failed: %s",
                label,
                exc,
            )

    if result.errors:
        logger.warning(
            "Incident report enrichment (%s) returned validation issues: %s",
            label,
            result.errors,
        )

    text = (result.text or "").strip()
    if not text and warn_if_empty:
        logger.warning(
            "Incident report enrichment (%s) returned an empty response.",
            label,
        )
    return text


def _infer_injuries_from_video(
    video_path: Path,
    timestamp_range: str,
    description: str,
    *,
    record_model_call: ModelCallRecorder = None,
) -> str:
    prompt = INJURY_INFERENCE_PROMPT.format(
        timestamp_range=timestamp_range,
        description=description,
    )
    return _perceptron_text_question(
        video_path,
        prompt,
        label="injury description",
        record_model_call=record_model_call,
    )


def _infer_incident_datetime_from_video(
    video_path: Path,
    timestamp_range: str,
    *,
    record_model_call: ModelCallRecorder = None,
) -> str:
    prompt = INCIDENT_DATETIME_PROMPT.format(timestamp_range=timestamp_range)
    return _perceptron_text_question(
        video_path,
        prompt,
        label="incident date/time",
        record_model_call=record_model_call,
        warn_if_empty=False,
    )


def enrich_incident_report_fields(
    report: dict,
    video_path: str | Path,
    values: dict[str, str | bool],
    *,
    record_model_call: ModelCallRecorder = None,
) -> dict[str, str | bool]:
    """Second-pass Perceptron calls for injury description and incident date/time."""
    context = _incident_context(report)
    if context is None:
        logger.warning(
            "Incident report enrichment skipped: no primary safety event with a timestamp "
            "and description available from the safety report."
        )
        return values

    timestamp_range, description = context
    path = Path(video_path)
    if not path.is_file():
        logger.warning(
            "Incident report enrichment skipped: video not found at %s",
            path,
        )
        return values

    enriched = dict(values)

    try:
        enriched[FIELD_INJURY_DESCRIPTION] = _infer_injuries_from_video(
            path,
            timestamp_range,
            description,
            record_model_call=record_model_call,
        )
    except Exception as exc:
        enriched[FIELD_INJURY_DESCRIPTION] = ""
        logger.warning(
            "Incident report injury enrichment failed for clip %s: %s",
            timestamp_range,
            exc,
            exc_info=True,
        )

    try:
        datetime_text = _infer_incident_datetime_from_video(
            path,
            timestamp_range,
            record_model_call=record_model_call,
        )
        incident_date, incident_time = _parse_incident_datetime_response(datetime_text)
        enriched[FIELD_TIME_OF_INCIDENT] = incident_time
        if incident_date:
            enriched[FIELD_DATE_OF_INCIDENT] = incident_date
    except Exception as exc:
        enriched[FIELD_TIME_OF_INCIDENT] = ""
        logger.warning(
            "Incident report date/time enrichment failed for clip %s: %s",
            timestamp_range,
            exc,
            exc_info=True,
        )

    return enriched


def _format_previous_activity(
    body: str,
    video_path: str | Path | None,
) -> str:
    content = body.strip()
    if not content:
        return ""
    filename = Path(video_path).name if video_path else "the uploaded clip"
    return f"According to the video footage from {filename}, {content}"


def _previous_activity(
    events: list[dict],
    summary: str,
    video_path: str | Path | None = None,
) -> str:
    primary = _primary_event(events)
    if primary:
        description = (primary.get("description") or "").strip()
        evidence = (primary.get("visual_evidence") or "").strip()
        if description and evidence:
            body = f"{description}\n\nObserved cues: {evidence}"
        else:
            body = description or evidence
        return _format_previous_activity(body, video_path)
    return _format_previous_activity(summary, video_path)


def _incident_description(events: list[dict], summary: str) -> str:
    if not events:
        return summary or "No safety incidents observed in the reviewed clip."

    lines: list[str] = []
    if summary:
        lines.append(summary)

    for index, event in enumerate(events, start=1):
        window = _format_timestamp(event)
        event_type = (event.get("event_type") or "event").replace("_", " ")
        severity = (event.get("severity") or "unknown").upper()
        description = (event.get("description") or "").strip()
        prefix = f"Event {index}"
        if window:
            prefix += f" [{window}]"
        lines.append(f"{prefix} ({event_type}, {severity} severity): {description}")
        evidence = (event.get("visual_evidence") or "").strip()
        if evidence:
            lines.append(f"Visual evidence: {evidence}")

    return "\n".join(line for line in lines if line)


def _involved_equipment(events: list[dict]) -> str:
    found: list[str] = []
    seen: set[str] = set()
    for event in events:
        blob = " ".join(
            str(event.get(key, "") or "")
            for key in ("description", "visual_evidence")
        )
        for match in _EQUIPMENT_HINTS.finditer(blob):
            term = match.group(0).lower()
            if term not in seen:
                seen.add(term)
                found.append(match.group(0))
    return ", ".join(found)


def safety_report_to_form_values(
    report: dict,
    *,
    video_path: str | Path | None = None,
) -> dict[str, str | bool]:
    """Map Flow A safety report JSON to workplace incident form values."""
    events = list(report.get("events") or [])
    summary = (report.get("summary") or "").strip()
    injuries = _suggests_injury(events)

    values: dict[str, str | bool] = {
        FIELD_DATE_OF_INCIDENT: "",
        FIELD_TIME_EMPLOYEE_BEGAN_WORK: "",
        FIELD_LOCATION: "",
        FIELD_TIME_OF_INCIDENT: "",
        FIELD_INJURY_DESCRIPTION: "",
        FIELD_INJURIES_YES: injuries,
        FIELD_INJURIES_NO: not injuries,
    }

    previous_activity = _previous_activity(events, summary, video_path)
    if previous_activity:
        values[FIELD_ACTIVITY_BEFORE] = previous_activity

    incident_description = _incident_description(events, summary)
    if incident_description:
        values[FIELD_INCIDENT_DESCRIPTION] = incident_description

    equipment = _involved_equipment(events)
    if equipment:
        values[FIELD_OBJECTS_SUBSTANCES] = equipment

    return values


def _sanitize_pdf_text(text: str) -> str:
    """ReportLab AcroForm appearance streams only tolerate Latin-1 safely."""
    if not text:
        return ""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _field_text(values: dict[str, str | bool], name: str) -> str:
    value = values.get(name, "")
    if not isinstance(value, str):
        return ""
    return _sanitize_pdf_text(value)


def _render_incident_report_pdf(values: dict[str, str | bool]) -> bytes:
    """Create a fillable workplace incident report PDF populated with values."""
    buffer = BytesIO()
    page_width, page_height = letter
    pdf = canvas.Canvas(buffer, pagesize=letter)
    form = pdf.acroForm

    left = 54
    right = page_width - 54
    width = right - left
    y = page_height - 54

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(page_width / 2, y, "WORKPLACE INCIDENT REPORT")
    y -= 10
    pdf.setLineWidth(1.5)
    pdf.line(left, y, right, y)
    y -= 28

    def section_title(title: str) -> None:
        nonlocal y
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, y, title)
        y -= 22

    def single_line_field(label: str, name: str, label_width: float = 170) -> None:
        nonlocal y
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left, y, label)
        form.textfield(
            name=name,
            x=left + label_width,
            y=y - 3,
            width=width - label_width,
            height=16,
            value=_field_text(values, name),
            borderStyle="underlined",
            fontSize=10,
            maxlen=4096,
        )
        y -= 26

    def multiline_field(label: str, name: str, height: float, lines_after: float = 8) -> None:
        nonlocal y
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left, y, label)
        y -= 16
        form.textfield(
            name=name,
            x=left,
            y=y - height + 14,
            width=width,
            height=height,
            value=_field_text(values, name),
            borderStyle="inset",
            fieldFlags="multiline",
            fontSize=9,
            maxlen=8192,
        )
        y -= height + lines_after

    section_title("Incident Details")
    single_line_field("Date of Incident:", FIELD_DATE_OF_INCIDENT)
    single_line_field("Time Employee Began Work:", FIELD_TIME_EMPLOYEE_BEGAN_WORK)
    single_line_field("Location:", FIELD_LOCATION)
    single_line_field("Time of Incident:", FIELD_TIME_OF_INCIDENT)
    multiline_field(
        "What was the employee doing just before the incident occurred?",
        FIELD_ACTIVITY_BEFORE,
        height=54,
    )
    multiline_field("Describe the incident:", FIELD_INCIDENT_DESCRIPTION, height=72)

    section_title("Injuries and Illnesses")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, y, "Were there any injuries or illnesses?")
    checkbox_y = y - 2
    pdf.drawString(left + 220, y, "Yes")
    form.checkbox(
        name=FIELD_INJURIES_YES,
        x=left + 200,
        y=checkbox_y,
        size=12,
        checked=bool(values.get(FIELD_INJURIES_YES)),
        buttonStyle="check",
    )
    pdf.drawString(left + 280, y, "No")
    form.checkbox(
        name=FIELD_INJURIES_NO,
        x=left + 260,
        y=checkbox_y,
        size=12,
        checked=bool(values.get(FIELD_INJURIES_NO)),
        buttonStyle="check",
    )
    y -= 28

    multiline_field(
        "Describe the injuries or illnesses:",
        FIELD_INJURY_DESCRIPTION,
        height=54,
    )
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, y, "What objects or substances directly harm the employee?")
    y -= 16
    form.textfield(
        name=FIELD_OBJECTS_SUBSTANCES,
        x=left,
        y=y - 2,
        width=width,
        height=16,
        value=_field_text(values, FIELD_OBJECTS_SUBSTANCES),
        borderStyle="underlined",
        fontSize=10,
        maxlen=4096,
    )

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _sanitize_video_stem(stem: str) -> str:
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_stem = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^\w\-]+", "_", ascii_stem).strip("_")
    return safe or "clip"


def _incident_report_output_path(source_video_path: str | Path | None) -> Path:
    stem = _sanitize_video_stem(
        Path(source_video_path).stem if source_video_path else "clip"
    )
    filename = f"perceptron_{stem}_incident_report.pdf"
    output_path = Path(tempfile.gettempdir()) / filename
    if output_path.exists():
        fd, tmp_name = tempfile.mkstemp(
            suffix=f"_{stem}_incident_report.pdf",
            prefix="perceptron_",
        )
        os.close(fd)
        return Path(tmp_name)
    return output_path


def fill_incident_report(
    safety_report: dict,
    *,
    video_path: str | Path | None = None,
    source_video_path: str | Path | None = None,
    template_path: Path | None = None,
    record_model_call: ModelCallRecorder = None,
) -> Path:
    """Return path to a filled, editable Workplace Incident Report PDF."""
    del template_path  # Template is generated programmatically.

    values = safety_report_to_form_values(safety_report, video_path=video_path)
    if video_path is not None:
        values = enrich_incident_report_fields(
            safety_report,
            video_path,
            values,
            record_model_call=record_model_call,
        )
    else:
        logger.warning(
            "Incident report enrichment skipped: no video path was provided for the "
            "second model call."
        )

    pdf_bytes = _render_incident_report_pdf(values)

    output_path = _incident_report_output_path(source_video_path or video_path)
    output_path.write_bytes(pdf_bytes)
    return output_path
