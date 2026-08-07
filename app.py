#!/usr/bin/env python3
"""Single entry point — Gradio UI by default, CLI on request.

    python app.py                                  # launch the UI (what a Space runs)
    python app.py --cli "MS white matter lesions"  # one headless run, no Gradio needed
    python app.py --cli "glioma MRI" --policy photographic_or_volumetric
    python app.py --list-lesions

The UI defaults to a CodeAgent, because it works with any chat model. The
ToolCallingAgent is preferable on a public Space — it executes no generated
code — but it needs a model whose provider supports the OpenAI `tools`
parameter, and many Hugging Face Inference Provider routes do not (they answer
422 UNSUPPORTED_OPENAI_PARAMS). Set MRI_AGENT_TYPE=tool_calling together with a
tools-capable MRI_AGENT_MODEL if you have one.

Space secrets (Settings -> Variables and secrets):
    HF_TOKEN          required — your Inference Providers token
    MRI_AGENT_MODEL   optional — defaults to Qwen/Qwen2.5-Coder-32B-Instruct
    MRI_AGENT_TYPE    optional — 'code' (default) or 'tool_calling'
"""

from __future__ import annotations

import argparse
import os
import sys

from lesion_finder.agent import build_agent, env
from lesion_finder.ontology import supported_lesion_keys

# gradio is imported lazily inside build_ui() so --cli works without it installed.

AGENT_TYPE = env("MRI_AGENT_TYPE", "code") or "code"

TITLE = "MRI Lesion Dataset Finder"

DESCRIPTION = f"""
Finds **open** MRI datasets for a lesion type, then verifies what image formats they
actually contain — `.jpg`/`.png` you can load with PIL, versus NIfTI/DICOM that need
`nibabel`/`pydicom`, versus images hidden inside archives.

Searches Hugging Face Hub, Zenodo and OpenNeuro. Running as a **{AGENT_TYPE}** agent.
A full run takes 30–60 seconds.

**Supported lesion types:** {', '.join(supported_lesion_keys())}

*Research tooling only — this app does not interpret scans and gives no medical advice.
Verify a dataset's licence and de-identification status before using it.*
"""

EXAMPLES = [
    "Open datasets of multiple sclerosis white matter lesions with jpg images",
    "Glioma / brain tumour MRI datasets I can load with PIL",
    "Ischaemic stroke lesion segmentation datasets, NIfTI is fine",
    "Prostate lesion MRI datasets with a permissive licence",
]

POLICIES = ["photographic_only", "photographic_or_volumetric", "any"]


def _task(query: str, policy: str) -> str:
    return f"{query}\n\nUse image_policy='{policy}' when shortlisting."


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #

def build_ui():
    import gradio as gr
    from smolagents import GradioUI

    raw = os.environ.get("HF_TOKEN", "")
    if not raw.strip():
        # Warn at startup rather than failing cryptically on the first query.
        print("WARNING: HF_TOKEN is not set — model calls will fail. "
              "Set it in Space settings under Variables and secrets.")
    elif raw != raw.strip():
        print("NOTE: HF_TOKEN had surrounding whitespace (probably a trailing "
              "newline from the settings box) — stripped. Re-paste it without "
              "the newline to silence this.")

    ui = GradioUI(build_agent(agent_type=AGENT_TYPE, verbosity_level=1))

    # Reuse smolagents' streaming callback, but wrap it in our own ChatInterface
    # so we get a title, description and examples. If a future smolagents renames
    # the private method, fall back to its stock app rather than crashing.
    stream = getattr(ui, "_stream_response", None)
    if stream is None:  # pragma: no cover
        return ui.create_app()

    type_kwarg = {"type": "messages"} if gr.__version__.startswith("5") else {}

    return gr.ChatInterface(
        fn=stream,
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
        # MUST stay False. HF Spaces sets GRADIO_CACHE_EXAMPLES=true, which makes
        # Gradio execute every example at startup — four full agent runs before
        # the app can serve a single request. That burns minutes and inference
        # credits on every boot, and if anything raises (e.g. HF_TOKEN missing)
        # the startup-events endpoint 500s and the whole Space dies with exit 1.
        cache_examples=False,
        chatbot=gr.Chatbot(label="Agent", height=520, **type_kwarg),
        save_history=True,
        **type_kwarg,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find open MRI lesion datasets. With no arguments, launches the UI.",
    )
    parser.add_argument("--cli", metavar="QUERY",
                        help="Run one query headlessly and print the report.")
    parser.add_argument("--policy", default="photographic_only", choices=POLICIES,
                        help="Which image formats count as usable.")
    parser.add_argument("--agent-type", default=None, choices=["code", "tool_calling"],
                        help="Overrides MRI_AGENT_TYPE. CLI defaults to 'code'.")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-lesions", action="store_true",
                        help="Print the supported lesion keys and exit.")
    args = parser.parse_args(argv)

    if args.list_lesions:
        print("\n".join(supported_lesion_keys()))
        return 0

    if args.cli:
        agent = build_agent(
            agent_type=args.agent_type or AGENT_TYPE,
            max_steps=args.max_steps,
            verbosity_level=0 if args.quiet else 2,
        )
        print(agent.run(_task(args.cli, args.policy)))
        return 0

    # queue() matters: a run takes 30-60s and a Space multiplexes visitors.
    # show_error surfaces the actual exception in the chat window; Gradio's
    # default hides it behind a bare "Error", which is useless to debug from.
    build_ui().queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
