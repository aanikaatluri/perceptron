"""Gradio app for Hugging Face Spaces — workplace safety video analytics."""

import atexit
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from analyze import configure_perceptron, incident_review_stream
from models import SafetyReport
from tracing import flush_tracing, init_tracing, is_configured, submit_user_feedback

init_tracing()
atexit.register(flush_tracing)

api_key = os.environ.get("PERCEPTRON_API_KEY")
if not api_key:
    raise RuntimeError(
        "PERCEPTRON_API_KEY is not set. Add it to Space Secrets on Hugging Face "
        "or to a local .env file."
    )

configure_perceptron(api_key)

LANGFUSE_STATUS = (
    "Langfuse tracing is **enabled** — sessions, latency, structured outputs, and "
    "`user-thumbs` scores are logged."
    if is_configured()
    else "Langfuse tracing is **disabled**. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable."
)


def _session_id(request: gr.Request | None) -> str | None:
    if request is None:
        return None
    return getattr(request, "session_hash", None)


def run_incident_review(uploaded_video, request: gr.Request):
    if uploaded_video is None:
        yield "", "", "", None
        return
    for update in incident_review_stream(uploaded_video, session_id=_session_id(request)):
        yield update.reasoning, update.output, update.trace_id, update.pdf_path
    flush_tracing()


def feedback_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=True, comment=comment, flow="A")


def feedback_not_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=False, comment=comment, flow="A")


with gr.Blocks(title="Workplace Safety Video Analytics") as demo:
    gr.Markdown(
        """
        # Workplace Safety Video Analytics
        Powered by [Perceptron Mk1](https://docs.perceptron.inc) — structured safety incident review
        over short security clips (MP4, WebM, MOV, and more — auto-converted on analyze).
        """
    )
    gr.Markdown(LANGFUSE_STATUS)

    gr.Markdown(
        "Upload a security clip and receive a structured JSON safety report with "
        "timestamped events, severity, visual evidence, and recommended actions. "
        "Download a partially filled Workplace Incident Report PDF derived from the analysis."
    )
    review_trace_id = gr.State("")
    with gr.Row():
        with gr.Column():
            review_video = gr.Video(
                label="Security clip",
                sources=["upload"],
            )
            review_btn = gr.Button("Analyze incidents", variant="primary")
        with gr.Column():
            review_reasoning = gr.Markdown(value="")
            review_output = gr.Code(
                label="Structured safety report",
                language="json",
                value="",
                lines=18,
            )
            review_pdf = gr.File(
                label="Download Workplace Incident Report (partial)",
                interactive=False,
            )
    with gr.Row():
        review_helpful_btn = gr.Button("👍 Helpful")
        review_not_helpful_btn = gr.Button("👎 Not helpful")
    review_feedback_comment = gr.Textbox(
        label="Feedback comment (optional)",
        placeholder="What was accurate or missing?",
        lines=2,
    )
    review_feedback_status = gr.Markdown()
    review_btn.click(
        run_incident_review,
        inputs=review_video,
        outputs=[review_reasoning, review_output, review_trace_id, review_pdf],
    )
    review_helpful_btn.click(
        feedback_helpful,
        inputs=[review_trace_id, review_feedback_comment],
        outputs=review_feedback_status,
    )
    review_not_helpful_btn.click(
        feedback_not_helpful,
        inputs=[review_trace_id, review_feedback_comment],
        outputs=review_feedback_status,
    )

    gr.Markdown(
        f"""
        ### Structured output schema
        Flow A validates against `{SafetyReport.__name__}` via Perceptron constrained decoding and generates a downloadable fillable Workplace Incident Report PDF.

        ### Observability (Langfuse)
        Traces use nested spans (`flow-a-incident-review` → `perceptron-mk1` generation), Gradio
        `session_id`, feature tags, explicit trace I/O (no raw video bytes), and `user-thumbs`
        boolean scores for regression testing.
        """
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="127.0.0.1")
