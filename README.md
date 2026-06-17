---
title: Workplace Safety Video Analytics
emoji: 🦺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
license: mit
---

# Workplace Safety Video Analytics

Enterprise-style safety analytics over short security clips, powered by [Perceptron Mk1](https://docs.perceptron.inc).

## Flows

### Flow A — Safety Incident Review
Upload a security clip → structured JSON report with timestamped events, severity, visual evidence, and recommended actions. Output is constrained to a Pydantic `SafetyReport` schema via `pydantic_format()`. A fillable **Workplace Incident Report** PDF is generated from the same analysis (JSON output is unchanged).

### Flow B — Visual Safety Search
Upload a clip + natural-language query (e.g. *"Find all moments where a worker is too close to moving equipment."*) → timestamped clip matches via Perceptron video clipping (`expects="clip"`).

### Flow C — Occupational Injury Report (Form 5020)
Upload a security clip → extract incident fields for Form 5020 **Q19–Q20, Q23–Q26** (body part injured, location, other workers injured, equipment, activity, sequence of events, evidence timestamps) → download a partially filled PDF. Unobservable fields stay blank.

### Flow D — Workplace Incident Report
Upload a security clip → structured JSON answering whether a safety incident caused harm or injury, the timestamp, who was involved and what happened, prior activity, potential injuries, and objects or substances that caused harm. If no incident is observed, returns `incident_occurred: false` with `incident_description: "no incidents observed."`

## Hugging Face Spaces setup

1. Create a new **Gradio** Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Push this repository (`app.py`, `analyze.py`, `models.py`, `requirements.txt`, `README.md`).
3. In **Settings → Repository secrets**, add `PERCEPTRON_API_KEY`.
4. Open the **App** tab after the build finishes.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your PERCEPTRON_API_KEY
python app.py          # Gradio UI at http://127.0.0.1:7860
python safety_check.py /path/to/video.mp4   # Flow A CLI
python compress_video.py /path/to/clip.mov    # convert + compress for upload
```

## Limits

- Video must end up as **MP4 under ~15 MB** (API request body cap is 20 MB). MOV and other formats are **auto-converted when you click Analyze/Search** (requires ffmpeg; included on HF Spaces via `apt.txt`).
- Perceptron meaningfully samples the first **~2 minutes** of each clip; oversized uploads are trimmed automatically when compressing.
- CLI prep: `python compress_video.py your_clip.mov` (requires [ffmpeg](https://ffmpeg.org/)).

## Langfuse observability

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_BASE_URL` in `.env` or Space secrets.

Each analysis run traces:
- **Flow span** (`flow-a-incident-review` / `flow-b-visual-search` / `flow-c-injury-report` / `flow-d-workplace-incident`) — explicit trace I/O (video metadata + query only, no raw bytes), structured JSON output, event/match counts
- **Generation span** (`perceptron-mk1`) — nested model call with prompt and clip/error summaries
- **Session grouping** — Gradio `session_hash` propagated via `propagate_attributes`
- **User feedback scores** — thumbs up/down (`user-thumbs` boolean score) with optional comments

Use Langfuse to filter low-rated traces, build annotation queues, and export datasets for regression testing.
