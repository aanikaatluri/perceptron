"""Perceptron Mk1 quickstart: configure credentials and send a first request.

The API key is loaded from a local .env file (or the PERCEPTRON_API_KEY
environment variable) so it is never committed to source control. Copy
.env.example to .env, drop in your key, then run:

    python quickstart.py [path/to/image.png]
"""

import os
import sys

from dotenv import load_dotenv

import perceptron
from perceptron import perceive, image, text

load_dotenv()

api_key = os.environ.get("PERCEPTRON_API_KEY")
if not api_key:
    sys.exit(
        "PERCEPTRON_API_KEY is not set. Copy .env.example to .env and add your key, "
        "or export PERCEPTRON_API_KEY in your shell."
    )

perceptron.configure(
    provider="perceptron",  # "perceptron" or "fal"
    api_key=api_key,
)


@perceive(model="perceptron-mk1")
def caption(image_path):
    return image(image_path) + text("Describe the primary object in this photo.")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "drone.png"
    result = caption(image_path)
    print(result.text)
    if result.errors:
        print("Errors:", result.errors)


if __name__ == "__main__":
    main()
