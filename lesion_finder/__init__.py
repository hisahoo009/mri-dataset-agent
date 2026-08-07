"""Multi-step smolagents agent for finding open MRI lesion datasets.

    ontology.py   controlled vocabulary of lesion types  (what counts as a query)
    schemas.py    pydantic In/Out models + image-format policy
    tools.py      ValidatedTool guardrail + the four pipeline tools
    sources.py    HTTP, format classification, one adapter per repository
    agent.py      assembly — the CodeAgent that orchestrates the tools
"""

from .agent import build_agent
from .ontology import LESION_ONTOLOGY, supported_lesion_keys
from .tools import ToolInputError, ToolOutputError, ValidatedTool, build_tools

__version__ = "1.1.0"

__all__ = [
    "build_agent", "build_tools", "ValidatedTool",
    "ToolInputError", "ToolOutputError",
    "LESION_ONTOLOGY", "supported_lesion_keys",
]
