
# Workplace Injury Reviewer — built on [Perceptron Mk1](https://docs.perceptron.inc)

Workplace injuries often require safety teams to review video footage, determine what happened, identify contributing factors, and complete incident reports. This process is important for worker safety, regulatory compliance, and insurance investigations. Manual video review is slow and time-consuming. Investigators may need to search through hours of footage to find the incident, identify equipment involved, and document the sequence of events.

This project demonstrates how foundation models can serve as a practical layer of intelligence on top of workplace safety footage, reducing review time while keeping claims tied to what is actually visible on camera.

This application uses **Perceptron Mk1** to turn workplace safety footage into structured incident reports. Users upload a clip and receive a grounded, timestamped analysis of the incident, equipment involved, worker activity, and key events.

The system follows a workplace reporting workflow: it automatically fills report fields using only visually verified information and links findings to supporting video evidence.

https://github.com/user-attachments/assets/bd4ac4e0-85fc-4857-b93e-1b2e182fbecd


## What it does

1. **Upload a security clip** (MP4, WebM, MOV, and other common formats; auto-converted on analyze).
2. **Run structured incident review** — Perceptron Mk1 returns a JSON safety report with:
   - Timestamped events (MM:SS aligned to the clip)
   - Severity ratings (low / medium / high)
   - Visual evidence and recommended corrective actions
   - An executive summary and human-review flag
3. **Download a Workplace Incident Report PDF** — a fillable form populated from the analysis, with an additional model pass for injury description and on-screen date/time when visible in the footage.
4. **Track performance with Langfuse** — each run is traced end-to-end (latency, structured outputs, session grouping), and reviewers can submit  feedback so teams can measure quality, spot failures, and improve the workflow over time.

Output is constrained to a Pydantic `SafetyReport` schema, keeping responses consistent and machine-readable.


## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your PERCEPTRON_API_KEY
python app.py          # Gradio UI at http://127.0.0.1:7860
python safety_check.py /path/to/video.mp4   # CLI
python compress_video.py /path/to/clip.mov    # convert + compress videos for upload
```

## Langfuse observability

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_BASE_URL` in `.env` or Space secrets.

Each analysis run traces:

- **Flow span** (`flow-a-incident-review`) — video metadata (no raw bytes), structured JSON output, event counts
- **Generation span** (`perceptron-mk1`) — model call with prompt and clip/error summaries
- **Session grouping** — Gradio `session_hash` propagated via `propagate_attributes`
- **User feedback scores** — thumbs up/down (`user-rating` boolean score) with optional comments
