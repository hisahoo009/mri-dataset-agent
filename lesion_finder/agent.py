"""Assembles the multi-step agent.

A `CodeAgent` does the tool calling: smolagents renders each tool's name,
signature and docstring into the system prompt, and the model orchestrates them
by writing Python. Nothing about the OpenAI `tools` parameter is involved, so
this works with any chat model — including provider routes that answer
`422 UNSUPPORTED_OPENAI_PARAMS` to a `ToolCallingAgent`.

The trade-off is that generated code gets executed. smolagents runs it through a
restricted interpreter rather than `exec`, and `additional_authorized_imports`
below keeps the reachable surface to three parsing modules.
"""

from __future__ import annotations

import os

from smolagents import CodeAgent, InferenceClientModel, Model
from smolagents.models import REMOVE_PARAMETER

from .tools import build_tools

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


def env(name: str, default: str = "") -> str:
    """Read an environment variable, stripped of surrounding whitespace.

    Secrets pasted into a Space settings box very often keep a trailing
    newline. An HF_TOKEN ending in "\\n" produces the header
    `Bearer hf_...\\n`, which httpx rejects outright with

        LocalProtocolError: Illegal header value b'Bearer hf_******\\n'

    — long after startup, on the first model call, and nowhere near the
    actual cause. Stripping on read makes that impossible.
    """
    return os.environ.get(name, default).strip()


def _tool_choice():
    """Decide what to do with smolagents' `tool_choice` parameter.

    smolagents defaults to `tool_choice="required"`. Hugging Face Inference
    Providers disagree about this parameter in two different ways:

        400 INVALID_TOOL_CHOICE        only "auto"/"none" accepted
        422 UNSUPPORTED_OPENAI_PARAMS  the parameter isn't supported at all

    A `CodeAgent` sends no tools, so it should never need this — but a model
    kwarg is attached to *every* request, so a stray value would ride along on
    calls that have no business carrying it. Omitting is the safe default.
    Set MRI_TOOL_CHOICE to "auto", "none" or "required" to send it explicitly.
    """
    value = env("MRI_TOOL_CHOICE", "omit").lower()
    return REMOVE_PARAMETER if value in ("omit", "remove", "") else value


INSTRUCTIONS = """
You find OPEN, PUBLIC MRI lesion datasets. You never interpret anyone's scan and
never give medical advice; if asked, say so and stop.

Work through the pipeline in order — do not skip a step, do not invent results:

  1. normalize_lesion_query(query)      -> lesion_key + search_terms
  2. search_open_datasets(lesion_key)   -> candidate datasets (contents UNVERIFIED)
  3. inspect_dataset_files(...)         -> once per promising candidate (3-6 of them),
                                           to see which image formats it really has
  4. shortlist_image_datasets(...)      -> final ranking

Every tool returns a JSON string. Parse it with json.loads() before using it, and
keep the parsed dicts in variables so you can pass them to the next step.

Rules:
- A dataset name mentioning a lesion proves nothing. Only inspect_dataset_files
  tells you what is inside. Never claim a dataset has .jpg images without inspecting.
- If a tool raises a validation error, read the message and fix the call. Do not
  retry the same arguments.
- If a source fails, continue with the others and say so in your answer.
- final_answer() must be a short report: dataset name, URL, licence, image formats
  and counts, and one line on how to load it. Flag any dataset whose licence is
  unknown, and remind the user to check the licence and de-identification status
  before any real use.
""".strip()


def build_agent(
    model: Model | None = None,
    *,
    max_steps: int = 12,
    verbosity_level: int = 2,
) -> CodeAgent:
    """Create the agent.

    `max_steps` is itself a guardrail: the pipeline needs ~8 steps, so 12 leaves
    slack for one or two self-corrections without letting a confused agent loop.
    """
    if model is None:
        model = InferenceClientModel(
            model_id=env("MRI_AGENT_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
            # `or None` so an empty/whitespace secret falls back to whatever
            # token the environment is logged in with, rather than sending "".
            token=env("HF_TOKEN") or None,
            # Model kwargs have the highest priority in smolagents' completion
            # kwargs, so this overrides its "required" default. See _tool_choice.
            tool_choice=_tool_choice(),
        )

    return CodeAgent(
        tools=build_tools(),
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        instructions=INSTRUCTIONS,
        planning_interval=4,
        # Narrow allowlist: enough to parse tool output, not enough to fetch
        # anything the tools did not sanction.
        additional_authorized_imports=["json", "re", "statistics"],
    )
