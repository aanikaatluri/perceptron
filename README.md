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
Upload a security clip → structured JSON report with timestamped events, severity, visual evidence, and recommended actions. Output is constrained to a Pydantic `SafetyReport` schema via `pydantic_format()`.

### Flow B — Visual Safety Search
Upload a clip + natural-language query (e.g. *"Find all moments where a worker is too close to moving equipment."*) → timestamped clip matches via Perceptron video clipping (`expects="clip"`).

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
```

## Limits

- Video must be **MP4 or WebM**, under **~15 MB** (API request body cap is 20 MB).
- Perceptron meaningfully samples the first **~2 minutes** of each clip.
