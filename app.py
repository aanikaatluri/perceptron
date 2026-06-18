"""Gradio app: workplace safety video analytics."""

import atexit
import os
import shutil
import subprocess
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from analyze import configure_perceptron, incident_review_stream
from tracing import flush_tracing, init_tracing, is_configured, submit_user_feedback

init_tracing()
atexit.register(flush_tracing)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
THUMBNAIL_DIR = ASSETS_DIR / "thumbnails"
VIDEO_SUFFIXES = {".mov", ".mp4", ".webm", ".avi", ".mkv"}

api_key = os.environ.get("PERCEPTRON_API_KEY")
if not api_key:
    raise RuntimeError(
        "PERCEPTRON_API_KEY is not set. Add it to Space Secrets on Hugging Face "
        "or to a local .env file."
    )

configure_perceptron(api_key)

LANGFUSE_STATUS = (
    "Langfuse tracing is **enabled**: sessions, latency, structured outputs, and user feedback is logged."
    if is_configured()
    else "Langfuse tracing is **disabled**. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable."
)


def _session_id(request: gr.Request | None) -> str | None:
    if request is None:
        return None
    return getattr(request, "session_hash", None)


def _discover_sample_videos() -> list[Path]:
    if not ASSETS_DIR.is_dir():
        return []
    return [
        path
        for path in sorted(ASSETS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]


def _video_thumbnail(video_path: Path) -> str | None:
    if not shutil.which("ffmpeg"):
        return None

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMBNAIL_DIR / f"{video_path.stem}.jpg"
    if thumb_path.exists() and thumb_path.stat().st_mtime >= video_path.stat().st_mtime:
        return str(thumb_path)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-vf",
                "scale=96:96:force_original_aspect_ratio=increase,crop=96:96",
                str(thumb_path),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return str(thumb_path)


def _sample_gallery_items() -> tuple[list[tuple[str, str]], list[str]]:
    items: list[tuple[str, str]] = []
    paths: list[str] = []
    for video in _discover_sample_videos():
        thumb = _video_thumbnail(video)
        if thumb is None:
            continue
        items.append((thumb, video.stem))
        paths.append(str(video.resolve()))
    return items, paths


SAMPLE_GALLERY_ITEMS, SAMPLE_VIDEO_PATHS = _sample_gallery_items()


def select_sample_video(evt: gr.SelectData) -> str:
    return SAMPLE_VIDEO_PATHS[evt.index]


def _empty_review_state():
    return "", "", "", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def run_incident_review(uploaded_video, request: gr.Request):
    if uploaded_video is None:
        yield _empty_review_state()
        return
    report_loading = True
    pdf_loading = True
    for update in incident_review_stream(uploaded_video, session_id=_session_id(request)):
        if update.output:
            report_loading = False
        if update.pdf_path is not None:
            pdf_loading = False
        yield (
            update.reasoning,
            update.output if update.output else gr.skip(),
            update.trace_id if update.trace_id else gr.skip(),
            update.pdf_path if update.pdf_path is not None else gr.skip(),
            gr.update(visible=True),
            gr.update(visible=report_loading),
            gr.update(visible=pdf_loading),
        )
    flush_tracing()


def start_incident_review(uploaded_video):
    if uploaded_video is None:
        return _empty_review_state()
    return "", "", "", None, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def _feedback_status_message(message: str) -> str:
    return f"*{message}*"


def feedback_helpful(trace_id: str, comment: str) -> tuple[bool, str]:
    return True, _feedback_status_message(
        submit_user_feedback(trace_id, helpful=True, comment=comment, flow="A")
    )


def feedback_not_helpful(trace_id: str, comment: str) -> tuple[bool, str]:
    return False, _feedback_status_message(
        submit_user_feedback(trace_id, helpful=False, comment=comment, flow="A")
    )


def submit_feedback_comment(trace_id: str, rating: bool | None, comment: str) -> tuple[str, str | object]:
    if rating is None:
        return _feedback_status_message(
            "Select 👍 Helpful or 👎 Not helpful before submitting a comment."
        ), gr.skip()
    if not comment.strip():
        return _feedback_status_message("Enter a comment before submitting."), gr.skip()
    result = submit_user_feedback(trace_id, helpful=rating, comment=comment, flow="A")
    if result.startswith(("Failed", "Run an analysis", "Langfuse is not")):
        return _feedback_status_message(result), gr.skip()
    return (
        f"{_feedback_status_message(result)}\n\n*Feedback comment submitted successfully.*",
        "",
    )


with gr.Blocks(
    title="Workplace Safety Analytics and Incident Reporting",
    css="""
    .loading-panel {
        position: relative;
    }

    .result-panel-label p {
        margin: 0 0 0.35rem 0;
        font-size: var(--text-sm, 14px);
        font-weight: 600;
        color: var(--block-label-text-color, var(--body-text-color));
        line-height: 1.2;
    }

    .pdf-loading-panel {
        min-height: 7.5rem;
    }

    .panel-loading-overlay {
        position: absolute;
        inset: 0;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        background: radial-gradient(circle at center, rgba(255, 255, 255, 0.92) 0%, rgba(255, 255, 255, 0.82) 38%, rgba(255, 255, 255, 0) 72%);
        border-radius: var(--radius-lg, 12px);
    }

    .dark .panel-loading-overlay {
        background: radial-gradient(circle at center, rgba(17, 24, 39, 0.88) 0%, rgba(17, 24, 39, 0.76) 38%, rgba(17, 24, 39, 0) 72%);
    }

    .orange-spinner {
        width: 32px;
        height: 32px;
        border-radius: 9999px;
        border: 4px solid rgba(249, 115, 22, 0.22);
        border-top-color: #f97316;
        animation: panel-spin 0.8s linear infinite;
        box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.08);
    }

    @keyframes panel-spin {
        to {
            transform: rotate(360deg);
        }
    }

    .safety-report-code .cm-scroller,
    .safety-report-code textarea {
        overflow: auto !important;
    }

    .progress-markdown {
        display: flex;
        flex-direction: column-reverse;
    }

    .sample-video-picker {
        align-items: flex-start !important;
        gap: 0.75rem !important;
        flex-wrap: nowrap !important;
        padding: 6px 0 8px 0;
        overflow: visible !important;
    }

    .sample-video-picker .sample-video-label {
        flex: 0 0 auto;
        width: auto !important;
        min-width: max-content;
        padding: 0 0.25rem 0 0;
        margin: 0;
        align-self: flex-start;
    }

    .sample-video-picker .sample-video-label p {
        margin: 0;
        white-space: nowrap;
        line-height: 1.2;
    }

    .sample-video-picker .sample-video-gallery {
        flex: 0 0 auto !important;
        width: auto !important;
        max-width: none;
        padding: 4px 0 !important;
        margin-top: 0 !important;
        overflow: visible !important;
    }

    .sample-video-gallery .grid-wrap {
        width: auto !important;
        max-width: none;
        padding: 4px 0 !important;
        margin-top: 0 !important;
        overflow: visible !important;
    }

    .sample-video-gallery .grid {
        display: flex !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        gap: 0.5rem !important;
        width: max-content !important;
        max-width: none !important;
        grid-template-columns: none !important;
        overflow: visible !important;
        padding: 2px 0;
    }

    .sample-video-gallery .thumbnail-item {
        width: 96px !important;
        height: 96px !important;
        flex: 0 0 96px !important;
        margin: 0 !important;
        border-radius: 8px;
        overflow: visible !important;
    }

    .sample-video-gallery .thumbnail-item img {
        width: 96px !important;
        height: 96px !important;
        object-fit: cover;
        border-radius: 8px;
        display: block;
    }

    .sample-video-gallery,
    .sample-video-gallery > .wrap,
    .sample-video-gallery .block {
        overflow: visible !important;
    }

    .sample-video-gallery .selected {
        outline: 2px solid #f97316;
        outline-offset: 2px;
        border-radius: 8px;
    }
    """,
) as demo:
    gr.Markdown(
        """
        # Workplace Safety Video Analytics and Incident Reporting
        Powered by [Perceptron Mk1](https://docs.perceptron.inc) — structured safety incident review
        over short security clips (MP4, WebM, MOV).
        """
    )
    gr.Markdown(LANGFUSE_STATUS)

    gr.Markdown(
        "Upload a security clip or select one of the sample videos and receive a structured JSON safety report with "
        "timestamped events, severity, visual evidence, and recommended actions. "
        "Download a populated and modifiable Workplace Incident Report PDF derived from the analysis."
    )
    review_trace_id = gr.State("")
    review_feedback_rating = gr.State(None)
    with gr.Row():
        with gr.Column():
            review_video = gr.Video(
                label="Security clip",
                sources=["upload"],
            )
            if SAMPLE_GALLERY_ITEMS:
                with gr.Row(elem_classes=["sample-video-picker"]):
                    gr.Markdown(
                        "Sample Clips:",
                        elem_classes=["sample-video-label"],
                    )
                    sample_gallery = gr.Gallery(
                        value=SAMPLE_GALLERY_ITEMS,
                        show_label=False,
                        container=False,
                        columns=len(SAMPLE_GALLERY_ITEMS),
                        rows=1,
                        height=108,
                        object_fit="cover",
                        allow_preview=False,
                        show_download_button=False,
                        elem_classes=["sample-video-gallery"],
                        min_width=0,
                    )
            review_btn = gr.Button("Analyze incidents", variant="primary")
            review_reasoning = gr.Markdown(value="", elem_classes=["progress-markdown"])
            review_feedback_status = gr.Markdown()
        with gr.Column(visible=False) as results_column:
            gr.Markdown("Structured safety report", elem_classes=["result-panel-label"])
            with gr.Group(elem_classes=["loading-panel"]):
                review_output = gr.Code(
                    show_label=False,
                    container=False,
                    language="json",
                    value="",
                    lines=18,
                    max_lines=18,
                    elem_classes=["safety-report-code"],
                )
                review_output_loading = gr.HTML(
                    '<div class="orange-spinner" aria-label="Loading structured safety report"></div>',
                    visible=False,
                    elem_classes=["panel-loading-overlay"],
                )
            gr.Markdown("Download Workplace Incident Report", elem_classes=["result-panel-label"])
            with gr.Group(elem_classes=["loading-panel", "pdf-loading-panel"]):
                review_pdf = gr.File(
                    show_label=False,
                    container=False,
                    interactive=False,
                )
                review_pdf_loading = gr.HTML(
                    '<div class="orange-spinner" aria-label="Loading workplace incident report"></div>',
                    visible=False,
                    elem_classes=["panel-loading-overlay"],
                )
    with gr.Row():
        review_helpful_btn = gr.Button("👍 Helpful")
        review_not_helpful_btn = gr.Button("👎 Not helpful")
    review_feedback_comment = gr.Textbox(
        label="Feedback comment (optional)",
        placeholder="What was accurate or missing?",
        lines=2,
        submit_btn="Submit",
    )
    review_btn.click(
        start_incident_review,
        inputs=review_video,
        outputs=[
            review_reasoning,
            review_output,
            review_trace_id,
            review_pdf,
            results_column,
            review_output_loading,
            review_pdf_loading,
        ],
    ).then(
        run_incident_review,
        inputs=review_video,
        outputs=[
            review_reasoning,
            review_output,
            review_trace_id,
            review_pdf,
            results_column,
            review_output_loading,
            review_pdf_loading,
        ],
    )
    if SAMPLE_GALLERY_ITEMS:
        sample_gallery.select(select_sample_video, outputs=review_video)
    review_helpful_btn.click(
        feedback_helpful,
        inputs=[review_trace_id, review_feedback_comment],
        outputs=[review_feedback_rating, review_feedback_status],
    )
    review_not_helpful_btn.click(
        feedback_not_helpful,
        inputs=[review_trace_id, review_feedback_comment],
        outputs=[review_feedback_rating, review_feedback_status],
    )
    review_feedback_comment.submit(
        submit_feedback_comment,
        inputs=[review_trace_id, review_feedback_rating, review_feedback_comment],
        outputs=[review_feedback_status, review_feedback_comment],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="127.0.0.1")
