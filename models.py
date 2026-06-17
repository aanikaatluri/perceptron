"""Pydantic schemas for structured safety analysis outputs."""

from typing import List, Literal

from pydantic import BaseModel, Field


class SafetyEvent(BaseModel):
    event_type: str = Field(
        description=(
            "Short machine-readable slug for the event, e.g. forklift_near_miss, "
            "ppe_violation, slip_trip_fall, pinch_point_exposure, loading_zone_conflict."
        )
    )
    severity: Literal["low", "medium", "high"] = Field(
        description=(
            "Risk level from visible cues: low=minor policy deviation; "
            "medium=unsafe act with moderate injury risk; "
            "high=imminent danger or near-miss with serious injury potential."
        )
    )
    start_time: str = Field(
        description="Start timestamp in MM:SS when the unsafe condition first becomes visible."
    )
    end_time: str = Field(
        description="End timestamp in MM:SS when the unsafe condition resolves or the clip ends."
    )
    description: str = Field(
        description="Objective narrative of what workers, vehicles, or equipment did during this event."
    )
    visual_evidence: str = Field(
        description=(
            "Concrete visual cues: relative positions, distances, PPE worn or missing, "
            "equipment motion state, barriers, signage, lighting, and worker body mechanics."
        )
    )
    recommended_action: str = Field(
        description="Specific corrective action a safety manager should take for this event."
    )


class SafetyReport(BaseModel):
    overall_summary: str = Field(
        description="One-paragraph executive summary of the most important safety findings in the clip."
    )
    events: List[SafetyEvent] = Field(
        description="Ordered list of distinct safety events; use an empty list if none are observed."
    )
    requires_human_review: bool = Field(
        description=(
            "True if any high-severity event occurs, the scene is ambiguous, multiple workers "
            "are at risk, or confidence is insufficient for automated triage."
        )
    )


class ClipMatch(BaseModel):
    start_time: str = Field(description="Match start timestamp in MM:SS.")
    end_time: str | None = Field(
        description="Match end timestamp in MM:SS, or null for a single-frame moment."
    )
    label: str = Field(description="Short label for what was found in this segment.")
    description: str = Field(
        description="What visible actions, positions, or hazards justify this match."
    )


class VisualSearchResult(BaseModel):
    query: str
    summary: str = Field(
        description="Brief overview of how many matches were found and overall pattern."
    )
    match_count: int
    matches: List[ClipMatch]


class OccupationalInjuryExtraction(BaseModel):
    """Video-extractable incident fields for OSHA Form 5020 questions 19–20 and 23–26."""

    apparent_body_part_affected: str = Field(
        default="",
        description=(
            "Form Q19 — Part(s) of the body injured or affected during the incident, only if "
            "directly visible or clearly indicated by posture, movement, or guarding. "
            "Leave blank if not observable."
        ),
    )
    location_or_camera_area: str = Field(
        default="",
        description=(
            "Form Q20 — Where the event occurred: observable work area, zone, aisle, room, or "
            "camera viewpoint (e.g. loading dock aisle B, warehouse camera 3). "
            "Leave blank if the location cannot be determined from the video."
        ),
    )
    other_workers_injured: str = Field(
        default="",
        description=(
            "Form Q23 — Were other workers injured or ill in this event? "
            "Answer 'yes', 'no', or leave blank if not observable from the video."
        ),
    )
    equipment_or_materials_involved: List[str] = Field(
        default_factory=list,
        description=(
            "Form Q24 — Equipment, machinery, vehicles, tools, or materials shown in the video "
            "that contributed to or were involved in the incident. Empty list if none are visible."
        ),
    )
    activity_being_performed: str = Field(
        default="",
        description=(
            "Form Q25 — Brief description of the action the worker was performing at the time "
            "of the incident, based only on visible behavior. Leave blank if unclear."
        ),
    )
    sequence_of_events: str = Field(
        default="",
        description=(
            "Form Q26 — Brief chronological description of the incident and what occurred in "
            "that timeframe, citing only visible actions and equipment states. "
            "Leave blank if no incident is shown."
        ),
    )
    evidence_clip: str = Field(
        default="",
        description=(
            "MM:SS timestamp or range for the incident segment in the uploaded clip "
            "(e.g. '00:12–00:18' or '00:12'). Leave blank only if no incident is shown."
        ),
    )
    requires_human_review: bool = Field(
        default=True,
        description=(
            "True if any field was inferred with low confidence, no clear incident is shown, "
            "the scene is ambiguous, or the form should not be filed without human verification."
        ),
    )
