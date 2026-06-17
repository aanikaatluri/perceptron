"""Fill OSHA Form 5020 PDF with video-extracted injury report fields."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

FORM_TEMPLATE = Path(__file__).resolve().parent / "assets" / "form5020.pdf"

# AcroForm field names for Form 5020 incident section (questions 19–20, 23–26).
FORM_Q19_INJURY_BODY = "19_SPECIFIC_INJURYILLNESS"
FORM_Q20_LOCATION = "20_LOCATION_WHERE_EVENT_O"
FORM_Q24_EQUIPMENT = "24_EQUIPMENT_MATERIALS_AN"
FORM_Q25_ACTIVITY = "25_SPECIFIC_ACTIVITY_THE"
FORM_Q26_SEQUENCE = "26_HOW_INJURY_ILLNESS"


def _format_evidence_window(evidence: str | dict) -> str:
    if isinstance(evidence, str):
        clip = evidence.strip()
        if clip:
            return f"Video evidence: {clip}."
        return ""

    start = (evidence.get("start_time") or "").strip()
    end = (evidence.get("end_time") or "").strip()
    if start and end:
        return f"Video evidence: {start}–{end}."
    if start:
        return f"Video evidence at {start}."
    return ""


def _format_q23_note(value: str) -> str:
    answer = value.strip().lower()
    if answer == "yes":
        return "Other workers injured or ill in this event: Yes."
    if answer == "no":
        return "Other workers injured or ill in this event: No."
    return ""


def extraction_to_pdf_fields(extraction: dict) -> dict[str, str]:
    """Map extracted JSON to Form 5020 fields Q19, Q20, Q24, Q25, Q26."""
    fields: dict[str, str] = {}

    body_part = (extraction.get("apparent_body_part_affected") or "").strip()
    if body_part:
        fields[FORM_Q19_INJURY_BODY] = body_part

    location = (extraction.get("location_or_camera_area") or "").strip()
    if location:
        fields[FORM_Q20_LOCATION] = location

    equipment = extraction.get("equipment_or_materials_involved") or []
    equipment_text = ", ".join(item.strip() for item in equipment if str(item).strip())
    if equipment_text:
        fields[FORM_Q24_EQUIPMENT] = equipment_text

    activity = (extraction.get("activity_being_performed") or "").strip()
    if activity:
        fields[FORM_Q25_ACTIVITY] = activity

    sequence_parts: list[str] = []
    sequence = (extraction.get("sequence_of_events") or "").strip()
    if sequence:
        sequence_parts.append(sequence)

    q23_note = _format_q23_note(extraction.get("other_workers_injured") or "")
    if q23_note:
        sequence_parts.append(q23_note)

    evidence_note = _format_evidence_window(extraction.get("evidence_clip") or "")
    if evidence_note:
        sequence_parts.append(evidence_note)

    if sequence_parts:
        fields[FORM_Q26_SEQUENCE] = " ".join(sequence_parts)

    return fields


def fill_form5020(
    extraction: dict,
    *,
    template_path: Path | None = None,
) -> Path:
    """Return path to a filled PDF with only the provided fields populated."""
    template = template_path or FORM_TEMPLATE
    if not template.is_file():
        raise FileNotFoundError(f"Form 5020 template not found: {template}")

    reader = PdfReader(template)
    writer = PdfWriter()
    writer.append(reader)

    pdf_fields = extraction_to_pdf_fields(extraction)
    if pdf_fields:
        writer.update_page_form_field_values(
            writer.pages[0],
            pdf_fields,
            auto_regenerate=False,
        )

    fd, tmp_name = tempfile.mkstemp(suffix="_form5020.pdf", prefix="perceptron_")
    os.close(fd)
    output_path = Path(tmp_name)
    with output_path.open("wb") as handle:
        writer.write(handle)

    return output_path
