"""Gradio app for Hugging Face Spaces — dual-flow safety video analytics."""

import atexit
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from analyze import (
    DEFAULT_SEARCH_QUERY,
    configure_perceptron,
    incident_review_stream,
    occupational_injury_report_stream,
    visual_search_stream,
)
from models import OccupationalInjuryExtraction, SafetyReport
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
        yield "", "", ""
        return
    for update in incident_review_stream(uploaded_video, session_id=_session_id(request)):
        yield update.reasoning, update.output, update.trace_id
    flush_tracing()


def run_visual_search(uploaded_video, query: str, request: gr.Request):
    if uploaded_video is None:
        yield "", "", ""
        return
    for update in visual_search_stream(uploaded_video, query, session_id=_session_id(request)):
        yield update.reasoning, update.output, update.trace_id
    flush_tracing()


def run_injury_report(uploaded_video, request: gr.Request):
    if uploaded_video is None:
        yield "", "", "", None
        return
    for update in occupational_injury_report_stream(uploaded_video, session_id=_session_id(request)):
        yield update.reasoning, update.output, update.trace_id, update.pdf_path
    flush_tracing()


def feedback_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=True, comment=comment, flow="A")


def feedback_not_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=False, comment=comment, flow="A")


def search_feedback_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=True, comment=comment, flow="B")


def search_feedback_not_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=False, comment=comment, flow="B")


def injury_feedback_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=True, comment=comment, flow="C")


def injury_feedback_not_helpful(trace_id: str, comment: str) -> str:
    return submit_user_feedback(trace_id, helpful=False, comment=comment, flow="C")


with gr.Blocks(title="Workplace Safety Video Analytics") as demo:
    gr.Markdown(
        """
        # Workplace Safety Video Analytics
        Powered by [Perceptron Mk1](https://docs.perceptron.inc) — structured incident review and
        visual safety search over short security clips (MP4, WebM, MOV, and more — auto-converted on analyze).
        """
    )
    gr.Markdown(LANGFUSE_STATUS)

    with gr.Tabs():
        with gr.Tab("Flow A — Safety Incident Review"):
            gr.Markdown(
                "Upload a security clip and receive a structured JSON safety report with "
                "timestamped events, severity, visual evidence, and recommended actions."
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
                outputs=[review_reasoning, review_output, review_trace_id],
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

        with gr.Tab("Flow B — Visual Safety Search"):
            gr.Markdown(
                "Upload a clip and describe what to find. Perceptron returns timestamped "
                "segments matching your query — useful for search-time analytics and event detection."
            )
            search_trace_id = gr.State("")
            with gr.Row():
                with gr.Column():
                    search_video = gr.Video(
                        label="Security clip",
                        sources=["upload"],
                    )
                    search_query = gr.Textbox(
                        label="Visual search query",
                        value=DEFAULT_SEARCH_QUERY,
                        lines=3,
                    )
                    search_btn = gr.Button("Search video", variant="primary")
                with gr.Column():
                    search_reasoning = gr.Markdown(value="")
                    search_output = gr.Code(
                        label="Timestamped matches",
                        language="json",
                        value="",
                        lines=18,
                    )
            with gr.Row():
                search_helpful_btn = gr.Button("👍 Helpful")
                search_not_helpful_btn = gr.Button("👎 Not helpful")
            search_feedback_comment = gr.Textbox(
                label="Feedback comment (optional)",
                placeholder="Were the timestamps and matches correct?",
                lines=2,
            )
            search_feedback_status = gr.Markdown()
            search_btn.click(
                run_visual_search,
                inputs=[search_video, search_query],
                outputs=[search_reasoning, search_output, search_trace_id],
            )
            search_helpful_btn.click(
                search_feedback_helpful,
                inputs=[search_trace_id, search_feedback_comment],
                outputs=search_feedback_status,
            )
            search_not_helpful_btn.click(
                search_feedback_not_helpful,
                inputs=[search_trace_id, search_feedback_comment],
                outputs=search_feedback_status,
            )

        with gr.Tab("Flow C — Occupational Injury Report (Form 5020)"):
            gr.Markdown(
                "Upload a security clip to extract incident details for OSHA Form 5020 "
                "(questions 19–20, 23–26: injury/body part, location, equipment, activity, "
                "sequence of events, and evidence timestamps). Download the partially filled PDF."
            )
            injury_trace_id = gr.State("")
            with gr.Row():
                with gr.Column():
                    injury_video = gr.Video(
                        label="Security clip",
                        sources=["upload"],
                    )
                    injury_btn = gr.Button("Generate injury report", variant="primary")
                with gr.Column():
                    injury_reasoning = gr.Markdown(value="")
                    injury_output = gr.Code(
                        label="Extracted Form 5020 fields",
                        language="json",
                        value="",
                        lines=16,
                    )
                    injury_pdf = gr.File(
                        label="Download Form 5020 (partial)",
                        interactive=False,
                    )
            with gr.Row():
                injury_helpful_btn = gr.Button("👍 Helpful")
                injury_not_helpful_btn = gr.Button("👎 Not helpful")
            injury_feedback_comment = gr.Textbox(
                label="Feedback comment (optional)",
                placeholder="Were the extracted fields accurate?",
                lines=2,
            )
            injury_feedback_status = gr.Markdown()
            injury_btn.click(
                run_injury_report,
                inputs=injury_video,
                outputs=[injury_reasoning, injury_output, injury_trace_id, injury_pdf],
            )
            injury_helpful_btn.click(
                injury_feedback_helpful,
                inputs=[injury_trace_id, injury_feedback_comment],
                outputs=injury_feedback_status,
            )
            injury_not_helpful_btn.click(
                injury_feedback_not_helpful,
                inputs=[injury_trace_id, injury_feedback_comment],
                outputs=injury_feedback_status,
            )

    gr.Markdown(
        f"""
        ### Structured output schema
        Flow A validates against `{SafetyReport.__name__}` via Perceptron constrained decoding.
        Flow B uses clip grounding (`expects="clip"`) for timestamped segment retrieval.
        Flow C validates against `{OccupationalInjuryExtraction.__name__}` and fills `assets/form5020.pdf`.

        ### Observability (Langfuse)
        Traces use nested spans (`flow-a-*` / `flow-b-*` / `flow-c-*` → `perceptron-mk1` generation), Gradio
        `session_id`, feature tags, explicit trace I/O (no raw video bytes), and `user-thumbs`
        boolean scores for regression testing.
        """
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="127.0.0.1")
