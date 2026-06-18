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
    "Langfuse tracing is **enabled**: sessions, latency, structured outputs, and user feedback is logged."
    if is_configured()
    else "Langfuse tracing is **disabled**. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable."
)


def _session_id(request: gr.Request | None) -> str | None:
    if request is None:
        return None
    return getattr(request, "session_hash", None)


def run_incident_review(uploaded_video, request: gr.Request):
    if uploaded_video is None:
        yield "", "", "", None, gr.update(visible=False), gr.update(visible=False)
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
            gr.update(visible=report_loading),
            gr.update(visible=pdf_loading),
        )
    flush_tracing()


def start_incident_review(uploaded_video):
    if uploaded_video is None:
        return "", "", "", None, gr.update(visible=False), gr.update(visible=False)
    return "", "", "", None, gr.update(visible=True), gr.update(visible=True)


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

    .panel-loading-overlay {
        position: absolute;
        inset: 0;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        background: radial-gradient(circle at center, rgba(255, 255, 255, 0.92) 0%, rgba(255, 255, 255, 0.82) 38%, rgba(255, 255, 255, 0) 72%);
        border-radius: 12px;
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
        "Upload a security clip and receive a structured JSON safety report with "
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
            review_btn = gr.Button("Analyze incidents", variant="primary")
            review_reasoning = gr.Markdown(value="", elem_classes=["progress-markdown"])
            review_feedback_status = gr.Markdown()
        with gr.Column():
            with gr.Group(elem_classes=["loading-panel"]):
                review_output = gr.Code(
                    label="Structured safety report",
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
            with gr.Group(elem_classes=["loading-panel"]):
                review_pdf = gr.File(
                    label="Download Workplace Incident Report (partial)",
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
            review_output_loading,
            review_pdf_loading,
        ],
    )
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

    gr.Markdown(
        f"""
        ### Structured output schema
        Flow A validates against `{SafetyReport.__name__}` via Perceptron constrained decoding and generates a downloadable fillable Workplace Incident Report PDF.

        ### Observability (Langfuse)
        Traces use nested spans (`flow-a-incident-review` → `perceptron-mk1` generation), Gradio
        `session_id`, feature tags, explicit trace I/O (no raw video bytes), and `user-rating`
        boolean scores for regression testing.
        """
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="127.0.0.1")
