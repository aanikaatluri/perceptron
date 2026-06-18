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
