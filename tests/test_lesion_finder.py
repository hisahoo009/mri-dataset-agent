"""Offline test suite — no network, no LLM. Source adapters are stubbed.

Covers: the lesion ontology, the input guardrail, the output guardrail,
file-format classification, ranking, source degradation, and the two agent
flavours.
"""

from __future__ import annotations

import json

import pytest
from smolagents import CodeAgent, Model, ToolCallingAgent
from smolagents.models import ChatMessage

from lesion_finder.agent import build_agent, instructions_for
from lesion_finder.ontology import detect_clinical_intent, resolve_lesion_type
from lesion_finder.sources import categorize, file_extension, summarize_files
from lesion_finder.tools import (
    InspectDatasetFilesTool,
    NormalizeLesionQueryTool,
    SearchOpenDatasetsTool,
    ShortlistImageDatasetsTool,
    ToolInputError,
    ToolOutputError,
    ValidatedTool,
)


# --------------------------------------------------------------------------- #
# ontology
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("multiple sclerosis lesions", "ms_lesion"),
    ("MS white matter plaques on MRI", "ms_lesion"),
    ("glioblastoma segmentation data", "glioma"),
    ("ischemic stroke infarct", "stroke_lesion"),
    ("prostate PI-RADS lesions", "prostate_lesion"),
])
def test_resolve_lesion_type(text, expected):
    assert resolve_lesion_type(text).key == expected


def test_resolve_rejects_unrelated():
    assert resolve_lesion_type("cat photos dataset") is None


def test_clinical_intent_detected():
    assert detect_clinical_intent("what does my MRI show, do I have cancer")


# --------------------------------------------------------------------------- #
# input guardrail
# --------------------------------------------------------------------------- #

def test_normalize_happy_path():
    out = json.loads(NormalizeLesionQueryTool()(query="open MS lesion MRI datasets"))
    assert out["lesion_key"] == "ms_lesion"
    assert out["search_terms"]


def test_normalize_rejects_out_of_scope():
    with pytest.raises(ToolInputError, match="No supported MRI lesion type"):
        NormalizeLesionQueryTool()(query="datasets of cat photos")


def test_normalize_refuses_clinical_advice():
    with pytest.raises(ToolInputError, match="Refused"):
        NormalizeLesionQueryTool()(query="here is my MRI, do I have a tumor")


def test_normalize_rejects_empty_and_overlong():
    tool = NormalizeLesionQueryTool()
    with pytest.raises(ToolInputError):
        tool(query="a")
    with pytest.raises(ToolInputError):
        tool(query="glioma " * 100)


def test_search_rejects_raw_user_phrase():
    with pytest.raises(ToolInputError, match="Unknown lesion_key"):
        SearchOpenDatasetsTool()(lesion_key="brain tumour please")


def test_search_rejects_bad_source():
    with pytest.raises(ToolInputError):
        SearchOpenDatasetsTool()(lesion_key="glioma", sources=["kaggle"])


def test_inspect_rejects_out_of_range_limit():
    with pytest.raises(ToolInputError):
        InspectDatasetFilesTool()(dataset_id="x/y", max_files_scanned=999_999)


# --------------------------------------------------------------------------- #
# output guardrail
# --------------------------------------------------------------------------- #

def test_output_schema_violation_is_caught():
    from pydantic import BaseModel

    class In(BaseModel):
        x: int

    class Out(BaseModel):
        y: int

    class Broken(ValidatedTool):
        name = "broken"
        description = "returns garbage"
        inputs = {"x": {"type": "integer", "description": "n"}}
        input_model, output_model = In, Out

        def run(self, payload):
            return {"y": "not an integer"}

    with pytest.raises(ToolOutputError, match="failed its output schema"):
        Broken()(x=1)


# --------------------------------------------------------------------------- #
# format classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path,ext,cat", [
    ("data/train/img_001.jpg", ".jpg", "photographic"),
    ("MASKS/seg.PNG", ".png", "photographic"),
    ("sub-01/anat/sub-01_T1w.nii.gz", ".nii.gz", "volumetric"),
    ("series/000123.dcm", ".dcm", "volumetric"),
    ("archive.tar.gz", ".tar.gz", "archive"),
    ("data/train-00000.parquet", ".parquet", "archive"),
    ("README.md", ".md", "other"),
])
def test_extension_and_category(path, ext, cat):
    assert file_extension(path) == ext
    assert categorize(ext) == cat


def test_summarize_photographic():
    out = summarize_files("u/d", "huggingface",
                          [f"train/{i}.jpg" for i in range(40)] + ["README.md"])
    assert out.has_photographic_images and out.photographic_image_count == 40
    assert not out.has_volumetric_images


def test_summarize_flags_archive_only():
    out = summarize_files("u/d", "huggingface", ["data/train-00000.parquet", "README.md"])
    assert out.images_possibly_in_archives
    assert not out.has_photographic_images


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #

def _inspection(dataset_id, photographic=0, volumetric=0, license_="cc-by-4.0"):
    return summarize_files(
        dataset_id, "huggingface",
        [f"a/{i}.jpg" for i in range(photographic)]
        + [f"b/{i}.nii.gz" for i in range(volumetric)],
        license_=license_,
    ).model_dump()


def test_shortlist_photographic_only_filters_nifti():
    out = json.loads(ShortlistImageDatasetsTool()(
        inspections=[_inspection("jpg/ds", photographic=500),
                     _inspection("nii/ds", volumetric=300)],
        image_policy="photographic_only",
    ))
    assert [d["dataset_id"] for d in out["accepted"]] == ["jpg/ds"]
    assert out["rejected"][0]["dataset_id"] == "nii/ds"


def test_shortlist_relaxed_policy_keeps_both():
    out = json.loads(ShortlistImageDatasetsTool()(
        inspections=[_inspection("jpg/ds", photographic=100),
                     _inspection("nii/ds", volumetric=300)],
        image_policy="photographic_or_volumetric",
    ))
    assert len(out["accepted"]) == 2


def test_shortlist_require_license():
    out = json.loads(ShortlistImageDatasetsTool()(
        inspections=[_inspection("no/license", photographic=50, license_=None)],
        require_known_license=True,
    ))
    assert not out["accepted"] and out["rejected"]


def test_shortlist_survives_junk_input():
    out = json.loads(ShortlistImageDatasetsTool()(
        inspections=[{"nonsense": True}, _inspection("good/ds", photographic=10)],
        image_policy="photographic_only",
    ))
    assert len(out["accepted"]) == 1
    assert any("unparseable" in r["dataset_id"] or "not a valid" in r["reason"]
               for r in out["rejected"])


# --------------------------------------------------------------------------- #
# search, with the network stubbed out
# --------------------------------------------------------------------------- #

def test_search_degrades_when_a_source_fails(monkeypatch):
    from lesion_finder import tools as tools_module
    from lesion_finder.sources import SourceUnavailable
    from lesion_finder.schemas import DatasetCandidate

    class OkSource:
        def search(self, terms, limit):
            return [DatasetCandidate(dataset_id="a/b", source="huggingface",
                                     title="a/b", url="https://x")]

        def inspect(self, dataset_id, max_files):
            raise NotImplementedError

    class DeadSource:
        def search(self, terms, limit):
            raise SourceUnavailable("boom")

        def inspect(self, dataset_id, max_files):
            raise NotImplementedError

    monkeypatch.setitem(tools_module.REGISTRY, "huggingface", OkSource())
    monkeypatch.setitem(tools_module.REGISTRY, "zenodo", DeadSource())

    out = json.loads(SearchOpenDatasetsTool()(lesion_key="glioma"))
    assert len(out["candidates"]) == 1
    assert "zenodo" in out["failed_sources"]


# --------------------------------------------------------------------------- #
# agent assembly — both flavours
# --------------------------------------------------------------------------- #

class DummyModel(Model):
    def generate(self, messages, **kwargs):
        return ChatMessage(role="assistant", content="ok")


TOOL_NAMES = {
    "normalize_lesion_query", "search_open_datasets",
    "inspect_dataset_files", "shortlist_image_datasets", "final_answer",
}


def test_code_agent_builds():
    agent = build_agent(model=DummyModel(), agent_type="code", verbosity_level=0)
    assert isinstance(agent, CodeAgent)
    assert set(agent.tools) == TOOL_NAMES
    assert agent.max_steps == 12


def test_tool_calling_agent_builds():
    agent = build_agent(model=DummyModel(), agent_type="tool_calling", verbosity_level=0)
    assert isinstance(agent, ToolCallingAgent)
    assert set(agent.tools) == TOOL_NAMES
    # The safety property we care about for a public Space.
    assert not hasattr(agent, "additional_authorized_imports") or not agent.tools.get("python_interpreter")


def test_tool_choice_defaults_to_auto(monkeypatch):
    """smolagents sends tool_choice='required'; many HF providers 400 on that."""
    monkeypatch.delenv("MRI_TOOL_CHOICE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "dummy")

    from smolagents import InferenceClientModel

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)
        Model.__init__(self)

    monkeypatch.setattr(InferenceClientModel, "__init__", fake_init)
    build_agent(agent_type="tool_calling", verbosity_level=0)
    assert captured["tool_choice"] == "auto"


def test_tool_choice_env_override(monkeypatch):
    from smolagents.models import REMOVE_PARAMETER

    from lesion_finder.agent import _tool_choice

    monkeypatch.setenv("MRI_TOOL_CHOICE", "required")
    assert _tool_choice() == "required"
    monkeypatch.setenv("MRI_TOOL_CHOICE", "omit")
    assert _tool_choice() is REMOVE_PARAMETER


def test_bad_agent_type_rejected():
    with pytest.raises(ValueError, match="agent_type"):
        build_agent(model=DummyModel(), agent_type="react")


def test_instructions_differ_by_agent_type():
    assert "json.loads()" in instructions_for("code")
    assert "json.loads()" not in instructions_for("tool_calling")
    assert "exact\nJSON strings" in instructions_for("tool_calling").replace("  ", " ") or \
           "JSON strings" in instructions_for("tool_calling")


def _inspection_dict(dataset_id="a/b", photographic=25):
    return summarize_files(
        dataset_id, "huggingface",
        [f"train/{i}.jpg" for i in range(photographic)],
        license_="cc-by-4.0",
    ).model_dump()


def test_shortlist_accepts_dicts():
    out = json.loads(ShortlistImageDatasetsTool()(inspections=[_inspection_dict()]))
    assert len(out["accepted"]) == 1


def test_shortlist_accepts_list_of_json_strings():
    """How a ToolCallingAgent will pass them back."""
    payload = [json.dumps(_inspection_dict("x/y"), default=str)]
    out = json.loads(ShortlistImageDatasetsTool()(inspections=payload))
    assert out["accepted"][0]["dataset_id"] == "x/y"


def test_shortlist_accepts_whole_json_array_as_one_string():
    payload = json.dumps([_inspection_dict("p/q")], default=str)
    out = json.loads(ShortlistImageDatasetsTool()(inspections=payload))
    assert out["accepted"][0]["dataset_id"] == "p/q"


def test_shortlist_rejects_unparseable_string():
    with pytest.raises(ToolInputError):
        ShortlistImageDatasetsTool()(inspections=["definitely not json"])


# --------------------------------------------------------------------------- #
# entry point — app.py serves both the UI and the CLI
# --------------------------------------------------------------------------- #

def test_app_builds_ui(monkeypatch):
    gradio = pytest.importorskip("gradio")
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-construction")

    import app

    monkeypatch.setattr(app, "build_agent", lambda **kw: build_agent(
        model=DummyModel(), agent_type=kw.get("agent_type", "tool_calling"), verbosity_level=0
    ))
    assert isinstance(app.build_ui(), gradio.Blocks)


def test_app_never_caches_examples(monkeypatch):
    """HF Spaces sets GRADIO_CACHE_EXAMPLES=true, which runs every example at
    startup — four agent runs before the app serves anyone, and a hard crash if
    one of them raises. The explicit False must survive that env var."""
    pytest.importorskip("gradio")
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-construction")
    monkeypatch.setenv("GRADIO_CACHE_EXAMPLES", "true")

    import app

    monkeypatch.setattr(app, "build_agent", lambda **kw: build_agent(
        model=DummyModel(), agent_type="tool_calling", verbosity_level=0
    ))
    assert app.build_ui().cache_examples is False


def test_app_list_lesions_needs_no_gradio(capsys):
    """The CLI path must not import gradio — that's why the import is lazy."""
    import app

    assert app.main(["--list-lesions"]) == 0
    printed = capsys.readouterr().out.split()
    assert "ms_lesion" in printed and "glioma" in printed


def test_app_cli_builds_agent_and_applies_policy(monkeypatch, capsys):
    import app

    seen = {}

    class FakeAgent:
        def run(self, task):
            seen["task"] = task
            return "report"

    def fake_build(**kwargs):
        seen["kwargs"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(app, "build_agent", fake_build)

    assert app.main(["--cli", "glioma datasets", "--policy", "any", "--quiet"]) == 0
    assert "image_policy='any'" in seen["task"]
    assert seen["kwargs"]["agent_type"] == "code"     # CLI default
    assert seen["kwargs"]["verbosity_level"] == 0     # --quiet
    assert capsys.readouterr().out.strip() == "report"
