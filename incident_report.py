"""Generate and fill a Workplace Incident Report PDF from Flow A safety data."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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


def _extract_location(summary: str, events: list[dict]) -> str:
    for event in events:
        evidence = (event.get("visual_evidence") or "").strip()
        for pattern in (
            r"(?:loading dock|warehouse|aisle|bay|zone|dock|floor|room|area|camera)\s+[\w\s\-#]+",
            r"(?:near|at|in)\s+the\s+[\w\s\-]+",
        ):
            match = re.search(pattern, evidence, re.IGNORECASE)
            if match:
                return match.group(0).strip().rstrip(".,;")
    if summary:
        return "Per reviewed security footage (see incident description)."
    return ""


def _previous_activity(events: list[dict], summary: str) -> str:
    primary = _primary_event(events)
    if primary:
        description = (primary.get("description") or "").strip()
        evidence = (primary.get("visual_evidence") or "").strip()
        if description and evidence:
            return f"{description}\n\nObserved cues: {evidence}"
        return description or evidence
    return summary


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
        action = (event.get("recommended_action") or "").strip()
        if action:
            lines.append(f"Recommended action: {action}")

    return "\n".join(line for line in lines if line)


def _injuries_description(events: list[dict]) -> str:
    lines: list[str] = []
    for event in events:
        if event.get("severity") not in ("high", "medium"):
            continue
        evidence = (event.get("visual_evidence") or "").strip()
        description = (event.get("description") or "").strip()
        if evidence:
            lines.append(evidence)
        elif description:
            lines.append(description)
    if lines:
        return "\n".join(lines)
    return "Potential injuries or illnesses should be verified by qualified medical personnel."


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


def safety_report_to_form_values(report: dict) -> dict[str, str | bool]:
    """Map Flow A safety report JSON to workplace incident form values."""
    events = list(report.get("events") or [])
    summary = (report.get("summary") or "").strip()
    primary = _primary_event(events)
    injuries = _suggests_injury(events)

    values: dict[str, str | bool] = {
        FIELD_DATE_OF_INCIDENT: "Per reviewed security footage",
        FIELD_TIME_EMPLOYEE_BEGAN_WORK: "",
        FIELD_INJURIES_YES: injuries,
        FIELD_INJURIES_NO: not injuries,
    }

    if primary:
        stamp = _format_timestamp(primary)
        if stamp:
            values[FIELD_TIME_OF_INCIDENT] = stamp

    location = _extract_location(summary, events)
    if location:
        values[FIELD_LOCATION] = location

    previous_activity = _previous_activity(events, summary)
    if previous_activity:
        values[FIELD_ACTIVITY_BEFORE] = previous_activity

    incident_description = _incident_description(events, summary)
    if incident_description:
        values[FIELD_INCIDENT_DESCRIPTION] = incident_description

    if injuries:
        injury_description = _injuries_description(events)
        if injury_description:
            values[FIELD_INJURY_DESCRIPTION] = injury_description

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


def fill_incident_report(
    safety_report: dict,
    *,
    template_path: Path | None = None,
) -> Path:
    """Return path to a filled, editable Workplace Incident Report PDF."""
    del template_path  # Template is generated programmatically.

    values = safety_report_to_form_values(safety_report)
    pdf_bytes = _render_incident_report_pdf(values)

    fd, tmp_name = tempfile.mkstemp(suffix="_incident_report.pdf", prefix="perceptron_")
    os.close(fd)
    output_path = Path(tmp_name)
    output_path.write_bytes(pdf_bytes)
    return output_path
