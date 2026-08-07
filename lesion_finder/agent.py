"""Assembles the multi-step agent.

Two flavours, same tools and same guardrails:

  agent_type="code"          CodeAgent — the model writes Python to orchestrate
                             the tools. Better reasoning, but it executes
                             generated code. Use locally.
  agent_type="tool_calling"  ToolCallingAgent — the model emits structured tool
                             calls, no code execution at all. Use for a public
                             deployment (e.g. an HF Space) where the input box
                             is a prompt-injection surface.
"""

from __future__ import annotations

import os
from typing import Literal

from smolagents import CodeAgent, InferenceClientModel, Model, MultiStepAgent, ToolCallingAgent
from smolagents.models import REMOVE_PARAMETER

from .tools import build_tools

AgentType = Literal["code", "tool_calling"]

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
    """Work around smolagents' default of `tool_choice="required"`.

    Several Hugging Face Inference Providers accept only "auto" or "none" and
    reject anything else with a 400 INVALID_TOOL_CHOICE, which surfaces as an
    AgentGenerationError on the very first step. "auto" is the safe default:
    the model may still call tools, it just isn't forced to.

    Set MRI_TOOL_CHOICE to "required" if your provider supports it, or to
    "omit" to leave the parameter out of the request entirely.
    """
    value = env("MRI_TOOL_CHOICE", "auto").lower()
    return REMOVE_PARAMETER if value in ("omit", "remove", "") else value

_SHARED_INSTRUCTIONS = """
You find OPEN, PUBLIC MRI lesion datasets. You never interpret anyone's scan and
never give medical advice; if asked, say so and stop.

Work through the pipeline in order — do not skip a step, do not invent results:

  1. normalize_lesion_query(query)      -> lesion_key + search_terms
  2. search_open_datasets(lesion_key)   -> candidate datasets (contents UNVERIFIED)
  3. inspect_dataset_files(...)         -> once per promising candidate (3-6 of them),
                                           to see which image formats it really has
  4. shortlist_image_datasets(...)      -> final ranking
""".strip()

_CODE_STATE_HINT = """
Every tool returns a JSON string. Parse it with json.loads() before using it, and
keep the parsed dicts in variables so you can pass them to the next step.
""".strip()

_TOOL_CALLING_STATE_HINT = """
Every tool returns a JSON string. Read the values you need out of it. When you
call shortlist_image_datasets, pass `inspections` as a list containing the exact
JSON strings that inspect_dataset_files returned — the tool parses them for you.
""".strip()

_RULES = """
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


def instructions_for(agent_type: AgentType) -> str:
    hint = _CODE_STATE_HINT if agent_type == "code" else _TOOL_CALLING_STATE_HINT
    return f"{_SHARED_INSTRUCTIONS}\n\n{hint}\n\n{_RULES}"


def build_agent(
    model: Model | None = None,
    *,
    agent_type: AgentType = "code",
    max_steps: int = 12,
    verbosity_level: int = 2,
) -> MultiStepAgent:
    """Create the agent.

    `max_steps` is itself a guardrail: the pipeline needs ~8 steps, so 12 leaves
    slack for one or two self-corrections without letting a confused agent loop.
    """
    if agent_type not in ("code", "tool_calling"):
        raise ValueError(f"agent_type must be 'code' or 'tool_calling', got {agent_type!r}")

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

    common = dict(
        tools=build_tools(),
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        instructions=instructions_for(agent_type),
        planning_interval=4,
    )

    if agent_type == "tool_calling":
        return ToolCallingAgent(**common)

    return CodeAgent(
        **common,
        # Narrow allowlist: enough to parse tool output, not enough to fetch
        # anything the tools did not sanction.
        additional_authorized_imports=["json", "re", "statistics"],
    )
