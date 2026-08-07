"""Gradio UI, and a --cli flag for running one query in a terminal.

    python app.py                       launch the UI (this is what a Space runs)
    python app.py --cli "glioma MRI"    one headless run

Set HF_TOKEN in Space settings under Variables and secrets.
"""

import argparse
import os
import sys

from agent import build_agent
from lesions import LESIONS

DESCRIPTION = f"""
Finds **open** MRI datasets for a lesion type, then checks what image formats they
actually contain: `.jpg`/`.png` you can open with PIL, NIfTI/DICOM that need
`nibabel`, or images hidden inside archives.

Supported lesion types: {', '.join(LESIONS)}

*Research tooling only. This does not interpret scans and gives no medical advice.*
"""

EXAMPLES = [
    "Open datasets of multiple sclerosis lesions with jpg images",
    "Brain tumour / glioma MRI datasets I can load with PIL",
    "Prostate lesion MRI datasets with a permissive licence",
]


def build_ui():
    import gradio as gr
    from smolagents import GradioUI

    if not os.environ.get("HF_TOKEN", "").strip():
        print("WARNING: HF_TOKEN is not set — model calls will fail.")

    ui = GradioUI(build_agent(verbosity_level=1))

    return gr.ChatInterface(
        fn=ui._stream_response,
        title="MRI Lesion Dataset Finder",
        description=DESCRIPTION,
        examples=EXAMPLES,
        # Spaces sets GRADIO_CACHE_EXAMPLES=true, which would run every example
        # at startup and kill the container if one fails.
        cache_examples=False,
        chatbot=gr.Chatbot(label="Agent", height=520),
    )


def main():
    parser = argparse.ArgumentParser(description="Find open MRI lesion datasets.")
    parser.add_argument("--cli", metavar="QUERY", help="Run one query and print the report.")
    args = parser.parse_args()

    if args.cli:
        print(build_agent().run(args.cli))
        return 0

    build_ui().queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=7860, show_error=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
