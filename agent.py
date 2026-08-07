"""Builds the agent.

A CodeAgent calls tools by writing Python. smolagents puts each tool's
signature and docstring into the system prompt, so nothing depends on the
model's provider supporting the OpenAI "tools" parameter.
"""

import os

from smolagents import CodeAgent, InferenceClientModel
from smolagents.models import REMOVE_PARAMETER

from tools import TOOLS

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

INSTRUCTIONS = """
You find open, public MRI lesion datasets that contain .jpg, .jpeg or .png
images. You never interpret anyone's scan and never give medical advice.

Follow these steps in order:
  1. normalize_lesion_query(query) to get a lesion_key
  2. search_datasets(lesion_key) to get candidates
  3. inspect_dataset(dataset_id) on the 3-5 most promising ones
  4. final_answer() with a short report

A dataset name mentioning a lesion proves nothing. Only inspect_dataset shows
what is inside. Recommend a dataset only if its has_images is true, and say
which of the inspected ones had none.

If a tool raises an error, read the message and fix your call rather than
retrying the same arguments.

The report should give, per recommended dataset: name, URL, licence, image
count and formats. Remind the reader to check the licence before using any of it.
""".strip()


def build_agent(model=None, max_steps=12, verbosity_level=2):
    if model is None:
        model = InferenceClientModel(
            model_id=os.environ.get("MRI_AGENT_MODEL", DEFAULT_MODEL).strip(),
            # .strip() because a token pasted into HF Space settings often keeps
            # a trailing newline, which makes an illegal HTTP header.
            token=os.environ.get("HF_TOKEN", "").strip() or None,
            # Some HF providers reject this parameter outright; omit it.
            tool_choice=REMOVE_PARAMETER,
        )

    return CodeAgent(
        tools=TOOLS,
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        instructions=INSTRUCTIONS,
        additional_authorized_imports=["json"],
    )
