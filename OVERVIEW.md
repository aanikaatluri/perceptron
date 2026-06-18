## Intro

**Incident video review** is the backbone of workplace safety investigations — determining what happened, who or what was involved, whether policy was violated, and what should go on the official report. Doing it manually is slow, expensive, and inconsistent. Investigators may scrub hours of CCTV to find a few seconds of relevance, then transcribe observations into forms that insurers, OSHA, and internal safety teams all expect in a standard shape.

This project demonstrates how a **vision-language model (VLM)** with native video understanding can compress that loop into a short, auditable workflow: upload a security clip and Perceptron's **Mk1** returns a timestamped safety analysis — events, severity, visual evidence, recommended actions — plus a **fillable Workplace Incident Report PDF** populated from the same findings.

The build composes several of Mk1's core multimodal primitives in one end-to-end flow:

- **Native video Q&A with chain-of-thought reasoning** (`question(..., reasoning=True)`)
- **Schema-constrained structured output** via `pydantic_format(SafetyReport, strict=True)` — typed events with MM:SS timestamps, severity, and a `requires_human_review` triage flag
- **Focused follow-up calls** on the highest-severity event for injury inference and on-screen date/time extraction


## Why I chose this problem

Every serious workplace — warehouses, construction sites, offices — already has cameras. What they lack is a fast path from **footage → documented incident**. Today that path is almost entirely human:

1. A supervisor or safety officer gets alerted.
2. Someone searches DVR/NVR footage for the relevant window.
3. They watch repeatedly, take notes, and fill out a paper or PDF incident form.
4. Compliance, insurance, and operations each want the same facts in slightly different packaging.

The existing tooling splits into two camps and misses the middle:

- **Traditional VMS / CCTV platforms** — Milestone, Genetec, Verkada, etc. Excellent at storage, playback, and motion triggers. They detect *that something moved* but do not reason about *what safety violation occurred* or draft narrative fields for a report.
- **Manual review + generic form software** — SharePoint forms, PDF templates, EHS suites. The human still does all visual interpretation and wording.

Neither answers a natural-language question or produces a structured, timestamped event list tied to visual evidence. This gap is where a **VLM with semantic video understanding and structured output** can make a meaningful impact.

The audience spans three tiers with the same underlying primitive:

| Tier | Who | Utility |
|------|-----|--------|
| **Operational** | Warehouse, manufacturing, and construction safety teams | Faster first-pass triage after near-misses and injuries |
| **Compliance** | EHS, risk, and insurance investigators | Consistent, evidence-linked documentation for audits and claims |
| **Platform** | VMS vendors, integrators, and enterprise AI teams | A model layer on top of existing camera infrastructure |

Worker safety, regulatory exposure, and insurance outcomes all depend on getting incident documentation right. Automating the *review and draft* step while keeping humans in the loop for sign-off improves efficiency while maintaining accuracy.


## Architecture

```mermaid
graph TD
    U["User uploads security footage"]
    UI["Gradio Web App"]

    U --> UI

    UI --> API["Perceptron Mk1"]

    API --> Report["Structured Incident Report"]
    API --> Clips["Timestamped Evidence Clips"]

    Report --> PDF["Auto-filled Injury Report PDF"]
    Clips --> PDF

    UI --> FB["User Feedback"]

    API -. "traces" .-> LF["Langfuse"]
    FB -. "ratings" .-> LF
```

**One user-facing flow (Safety Incident Review)**, up to **three Mk1 call sites** per successful run:

1. **Primary review** — full-clip structured `SafetyReport` with reasoning.
2. **Injury enrichment** — targeted pass on the highest-severity event's time window for additional insights.
3. **Datetime enrichment** — reads on-screen camera timestamps when present for incident report population.

All model calls are wrapped in **Langfuse generation spans** capturing prompts, structured-output flags, text previews, and error counts. 

### Repository layout

| Module | Responsibility |
|--------|----------------|
| `app.py` | Gradio UI — upload, sample clips, streaming results, PDF download, feedback |
| `analyze.py` | Core Flow A orchestration (prep → model → PDF attach) |
| `models.py` | Pydantic `SafetyReport` / `SafetyEvent` schemas |
| `streaming.py` | SSE consumption, transport-error fallback, `FlowUpdate` dataclass |
| `incident_report.py` | JSON → form mapping, enrichment prompts, ReportLab PDF generation |
| `compress_video.py` | ffmpeg/ffprobe convert, compress, trim for API limits |
| `tracing.py` | Langfuse init, spans, session propagation, user scores |
| `safety_check.py` | Headless CLI entry point |

Deploy artifacts: `requirements.txt`, `apt.txt` (ffmpeg on Hugging Face Spaces), `.env.example`.

## Methodology

**1. Define the output contract first.** Before touching UI code, the `SafetyReport` Pydantic schema locked down what "done" looks like: `overall_summary`, a list of `SafetyEvent` objects (type, severity, MM:SS window, description, visual evidence, recommended action), and `requires_human_review` for triage. Field descriptions double as prompt guidance for Mk1 via `pydantic_format(..., strict=True)`.

**2. Compression and conversion pipeline.** Raw security footage is often MOV from a phone, AVI from a DVR export, or hundreds of megabytes long. Mk1's request body cap is 20 MB; this app targets **15 MB** with headroom. `compress_video.py` wraps ffmpeg to:

- Convert non-MP4 / non-H.264 sources to MP4 (H.264/AAC)
- Progressively raise CRF and downscale (720p → 480p) until under the cap
- Auto-trim to **~128 seconds** when clips are long and oversized — aligned with how much footage Mk1 meaningfully samples.

The Gradio app calls `prepare_video_for_upload` automatically on Analyze; the CLI script is for manual prep.

**3. Structured incident review.** One focused prompt asks for every observable safety event, near-miss, or policy violation — with an explicit constraint to **cite only what is directly visible on camera**. Chain-of-thought reasoning is enabled so reviewers (and Langfuse traces) can audit *why* the model flagged each event.

**4. PDF workflow as a second layer** 
The app maps structured JSON to workplace form fields heuristically:
- Runs **two narrow follow-up video questions** on the primary (highest-severity) event for fields that need additional verification: likely injuries and on-screen date/time
- Renders a **fillable AcroForm PDF** programmatically with ReportLab so safety managers can edit before filing

**6. LLMOps observability and evaluation loop.** Langfuse captures:

- **Flow span** (`flow-a-incident-review`) — video metadata only (no raw bytes), structured output, event counts
- **Generation spans** (`perceptron-mk1`) — per model call
- **Session grouping** via Gradio `session_hash`
- **User feedback** — boolean `user-rating` score with optional comment, linked to the same trace ID

This enables satisfaction dashboards, low-rated trace review, annotation queues, and future regression datasets when prompts or schemas change.

## What the app produces

For each analyzed clip, a reviewer gets:

**Structured JSON safety report**

```json
{
  "summary": "…",
  "events": [
    {
      "event_type": "forklift_near_miss",
      "severity": "high",
      "start_time": "00:42",
      "end_time": "00:51",
      "description": "…",
      "visual_evidence": "…",
      "recommended_action": "…"
    }
  ],
  "requires_human_review": true
}
```

**Workplace Incident Report PDF** — editable fields including date/time of incident, activity before incident, incident narrative, injury checkboxes, injury description, and objects/substances involved.

## Who it is useful for
The primary audience is workplace safety teams responsible for investigating incidents and documenting workplace injuries. Additional users include insurance investigators reviewing claims and compliance teams searching large video archives for safety risks. By transforming raw surveillance footage into structured, searchable incident reports, the system reduces manual review effort while making safety investigations faster, more consistent, and easier to audit for the following groups:

- Warehouse and logistics operators
- Manufacturing facilities
- Construction companies
- Workplace safety and EHS teams
- Insurance investigators
- Compliance and risk management organizations

## Current limitations

- **Upload size** — must end up as MP4 under ~15 MB; longer sources are trimmed to ~2 minutes.
- **Single-clip scope** — no multi-camera sync or hours-long archive search

## Next steps

The most valuable extensions move from analyzing a single uploaded clip to operating over a facility's footage backlog.

**1. Temporal clip export at event timestamps.** The structured report already carries MM:SS windows per event. Pair Mk1's analysis with an **ffmpeg harness** (or MCP server) to cut evidentiary subclips for each flagged event and attach them to the PDF or case file — the same composition pattern as grounded `<clip>` tags, but driven off schema timestamps.

```mermaid
graph LR
    Clip[Uploaded security clip]
    Mk1[Mk1 SafetyReport]
    Clip --> Mk1
    Mk1 --> Events[Event list with MM:SS windows]
    Events --> FF[ffmpeg extract subclips]
    FF --> Evidence[Per-event video evidence files]
    Evidence --> PDF[Incident packet · PDF + clips]
```

**2. Agentic search over long archives.** An orchestrator could decompose searches into parallel Mk1 calls over candidate segments, verify reasoning against returned events, and escalate only high-severity hits to human review — turning the current single-clip demo into a **semantic search layer** on top of NVR exports.

**3. VMS and EHS integrations.** Webhook or MCP adapters that pull clips from Verkada/Genetec/Milestone by camera ID and time range, push completed reports into ServiceNow, Enablon, or insurer portals.

**5. Jurisdiction-specific form variants.** Support the filing of additional forms, such as OSHA 301, WSIB, and state workers' comp layouts. The same `SafetyReport` intermediate representation could be used to populate different PDF templates.
