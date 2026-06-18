"""CLI for Flow A structured safety incident review.

Usage::

    python safety_check.py
    python safety_check.py /path/to/video.mp4
"""

import atexit
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from analyze import configure_perceptron, incident_review
from tracing import flush_tracing, init_tracing, pop_trace_id

init_tracing()
atexit.register(flush_tracing)

DEFAULT_VIDEO = Path.home() / "Downloads" / "example_vid.mp4"

api_key = os.environ.get("PERCEPTRON_API_KEY")
if not api_key:
    sys.exit(
        "PERCEPTRON_API_KEY is not set. Copy .env.example to .env and add your key, "
        "or export PERCEPTRON_API_KEY in your shell."
    )

configure_perceptron(api_key)


def main() -> None:
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    payload, _trace_id = pop_trace_id(incident_review(video_path, session_id="cli"))
    flush_tracing()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
