"""The guardrail base class, and the four smolagents tools built on it.

    normalize_lesion_query -> search_open_datasets -> inspect_dataset_files
                                                  -> shortlist_image_datasets

The tools are deliberately *not* fused into one mega-tool: forcing the agent to
carry state between calls is what makes this a multi-step agent, and it lets
the agent decide which candidates are worth the cost of a file listing.

`ValidatedTool` lives here rather than in its own module because it has exactly
one set of consumers — the four classes below.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError
from smolagents import Tool

from .ontology import (
    LESION_ONTOLOGY,
    detect_clinical_intent,
    resolve_lesion_type,
    supported_lesion_keys,
)
from .schemas import (
    InspectIn,
    InspectOut,
    NormalizedQuery,
    NormalizeQueryIn,
    RankedDataset,
    SearchIn,
    SearchOut,
    ShortlistIn,
    ShortlistOut,
)
from .sources import REGISTRY, SourceUnavailable


# --------------------------------------------------------------------------- #
# The guardrail
# --------------------------------------------------------------------------- #

class ToolInputError(ValueError):
    """Raised when the agent calls a tool with arguments that fail validation."""


class ToolOutputError(RuntimeError):
    """Raised when a tool would return a payload that fails its own schema."""


def _explain(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)


class ValidatedTool(Tool):
    """Base class for tools with pydantic-enforced input and output.

    Subclasses set `input_model` / `output_model` and implement `run()`, which
    receives a validated input model and returns either the output model or a
    plain dict that must satisfy it.

    Both failure modes surface to the agent as readable error strings rather
    than tracebacks, so a multi-step agent can correct itself and retry instead
    of crashing the run.
    """

    input_model: type[BaseModel]
    output_model: type[BaseModel]
    output_type = "string"

    # `forward` takes **kwargs because the real signature lives in the pydantic
    # model; tell smolagents not to signature-check it.
    skip_forward_signature_validation = True

    def run(self, payload: BaseModel) -> BaseModel | dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def forward(self, **kwargs: Any) -> str:
        # LLMs routinely pass `null` for optional args; treat that as "omitted"
        # so the schema default applies instead of failing validation.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # ---- input guardrail -------------------------------------------------
        try:
            payload = self.input_model.model_validate(kwargs)
        except ValidationError as err:
            raise ToolInputError(
                f"Invalid arguments for `{self.name}` -> {_explain(err)}. "
                f"Expected fields: {list(self.input_model.model_fields)}."
            ) from None

        result = self.run(payload)

        # ---- output guardrail ------------------------------------------------
        try:
            validated = (
                result
                if isinstance(result, self.output_model)
                else self.output_model.model_validate(result)
            )
        except ValidationError as err:
            raise ToolOutputError(
                f"`{self.name}` produced a payload that failed its output schema "
                f"-> {_explain(err)}. No results returned; the upstream response "
                f"was probably malformed."
            ) from None

        return json.dumps(validated.model_dump(), indent=2, default=str)


# --------------------------------------------------------------------------- #
# The four pipeline steps
# --------------------------------------------------------------------------- #

class NormalizeLesionQueryTool(ValidatedTool):
    name = "normalize_lesion_query"
    description = (
        "STEP 1 — always call this first. Maps a free-text request onto a supported "
        "MRI lesion type and returns the `lesion_key` plus the search terms to use. "
        "Rejects queries that are not about an MRI lesion, and queries that ask for "
        "medical advice about a specific person. "
        f"Supported lesion keys: {', '.join(supported_lesion_keys())}."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The user's request, e.g. 'open MS white matter lesion MRI datasets'.",
        }
    }
    input_model = NormalizeQueryIn
    output_model = NormalizedQuery

    def run(self, payload: NormalizeQueryIn) -> NormalizedQuery:
        clinical = detect_clinical_intent(payload.query)
        if clinical:
            raise ToolInputError(
                f"Refused: the phrase {clinical!r} reads as a request for medical "
                "advice about a specific person. This agent only finds public "
                "research datasets and cannot interpret anyone's scan. Rephrase as "
                "a dataset request, or tell the user to consult a clinician."
            )

        lesion = resolve_lesion_type(payload.query)
        if lesion is None:
            raise ToolInputError(
                f"No supported MRI lesion type found in {payload.query!r}. "
                f"Supported keys: {', '.join(supported_lesion_keys())}. "
                "Ask the user which of these they meant instead of guessing."
            )

        return NormalizedQuery(
            lesion_key=lesion.key,
            canonical_name=lesion.canonical_name,
            body_region=lesion.body_region,
            search_terms=list(lesion.search_terms)[:8],
        )


class SearchOpenDatasetsTool(ValidatedTool):
    name = "search_open_datasets"
    description = (
        "STEP 2 — searches open dataset repositories for a validated `lesion_key` "
        "(get it from normalize_lesion_query; a raw user phrase will be rejected). "
        "Returns candidate datasets with id, url, licence and tags. It does NOT tell "
        "you what image formats are inside — use inspect_dataset_files for that. "
        "Sources: 'huggingface' (mostly .jpg/.png), 'zenodo' (mixed), "
        "'openneuro' (NIfTI brain MRI)."
    )
    inputs = {
        "lesion_key": {
            "type": "string",
            "description": "A lesion key returned by normalize_lesion_query, e.g. 'ms_lesion'.",
        },
        "sources": {
            "type": "array",
            "description": "Which repositories to query. Default ['huggingface', 'zenodo'].",
            "nullable": True,
        },
        "max_results_per_source": {
            "type": "integer",
            "description": "1-25, default 8.",
            "nullable": True,
        },
    }
    input_model = SearchIn
    output_model = SearchOut

    def run(self, payload: SearchIn) -> SearchOut:
        lesion = LESION_ONTOLOGY.get(payload.lesion_key)
        if lesion is None:
            raise ToolInputError(
                f"Unknown lesion_key {payload.lesion_key!r}. Call normalize_lesion_query "
                f"first. Valid keys: {', '.join(supported_lesion_keys())}."
            )

        candidates, failures = [], {}
        for source_name in payload.sources:
            source = REGISTRY[source_name]
            try:
                candidates.extend(
                    source.search(list(lesion.search_terms), payload.max_results_per_source)
                )
            except SourceUnavailable as exc:
                failures[source_name] = str(exc)
            except Exception as exc:  # noqa: BLE001
                failures[source_name] = f"{type(exc).__name__}: {exc}"

        candidates = candidates[:100]
        note = (
            "Candidates are unverified: names match the search terms but the actual "
            "contents are not confirmed. Inspect the promising ones before recommending."
        )
        if failures:
            note += f" Sources that failed: {', '.join(failures)}."
        if not candidates:
            note = (
                "No candidates found. Try a different source, or tell the user this "
                "lesion type has no readily discoverable open dataset."
            )

        return SearchOut(
            lesion_key=payload.lesion_key,
            queried_sources=payload.sources,
            failed_sources=failures,
            candidates=candidates,
            note=note,
        )


class InspectDatasetFilesTool(ValidatedTool):
    name = "inspect_dataset_files"
    description = (
        "STEP 3 — lists the files of ONE dataset and reports which image formats it "
        "actually contains: 'photographic' (.jpg/.jpeg/.png/.tif/.bmp — load directly "
        "with PIL), 'volumetric' (.nii/.nii.gz/.dcm — needs nibabel or pydicom), or "
        "'archive' (.zip/.parquet — images may be hidden inside). Call this once per "
        "candidate you care about, then pass the results to shortlist_image_datasets."
    )
    inputs = {
        "dataset_id": {
            "type": "string",
            "description": "Dataset id from search_open_datasets, e.g. 'user/brain-mri' or '1234567'.",
        },
        "source": {
            "type": "string",
            "description": "'huggingface', 'zenodo' or 'openneuro'. Must match where the id came from.",
            "nullable": True,
        },
        "max_files_scanned": {
            "type": "integer",
            "description": "10-5000, default 1000.",
            "nullable": True,
        },
    }
    input_model = InspectIn
    output_model = InspectOut

    def run(self, payload: InspectIn) -> InspectOut:
        source = REGISTRY[payload.source]
        try:
            return source.inspect(payload.dataset_id, payload.max_files_scanned)
        except SourceUnavailable as exc:
            raise ToolInputError(
                f"Could not list files for {payload.dataset_id!r} on {payload.source}: "
                f"{exc}. Check the id and source match, or move on to another candidate."
            ) from None


class ShortlistImageDatasetsTool(ValidatedTool):
    name = "shortlist_image_datasets"
    description = (
        "STEP 4 — final ranking. Pass the list of inspection results you collected "
        "(each one the parsed JSON from inspect_dataset_files) and an image_policy: "
        "'photographic_only' keeps datasets with .jpg/.png-style files, "
        "'photographic_or_volumetric' also allows NIfTI/DICOM, 'any' keeps everything. "
        "Returns the accepted datasets ranked, plus why each rejected one was dropped."
    )
    inputs = {
        "inspections": {
            "type": "array",
            "description": "List of dicts, each the output of inspect_dataset_files.",
        },
        "image_policy": {
            "type": "string",
            "description": "'photographic_only' (default), 'photographic_or_volumetric', or 'any'.",
            "nullable": True,
        },
        "require_known_license": {
            "type": "boolean",
            "description": "Drop datasets with no declared licence. Default false.",
            "nullable": True,
        },
        "top_k": {"type": "integer", "description": "1-20, default 5.", "nullable": True},
    }
    input_model = ShortlistIn
    output_model = ShortlistOut

    def run(self, payload: ShortlistIn) -> ShortlistOut:
        accepted_raw, rejected = [], []

        for raw in payload.inspections:
            try:
                item = InspectOut.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                rejected.append({
                    "dataset_id": str(raw.get("dataset_id", "<unparseable>"))
                    if isinstance(raw, dict) else "<unparseable>",
                    "reason": f"not a valid inspect_dataset_files result: {exc}",
                })
                continue

            reasons, ok = [], True

            if payload.image_policy == "photographic_only":
                if item.has_photographic_images:
                    reasons.append(
                        f"{item.photographic_image_count} directly-loadable 2-D image files"
                    )
                else:
                    ok = False
                    reasons.append("no .jpg/.png-style image files found")
            elif payload.image_policy == "photographic_or_volumetric":
                if item.has_photographic_images or item.has_volumetric_images:
                    reasons.append(
                        f"{item.photographic_image_count} 2-D + "
                        f"{item.volumetric_image_count} volumetric image files"
                    )
                else:
                    ok = False
                    reasons.append("no image files in any supported format")

            if payload.require_known_license and not item.license:
                ok = False
                reasons.append("no declared licence")
            elif item.license:
                reasons.append(f"licence: {item.license}")
            else:
                reasons.append("licence not declared — verify before use")

            if item.images_possibly_in_archives:
                reasons.append("images may be inside archives; listing alone is inconclusive")
            if item.truncated:
                reasons.append("file listing truncated — counts are a lower bound")

            if not ok:
                rejected.append({
                    "dataset_id": item.dataset_id,
                    "source": item.source,
                    "reason": "; ".join(r for r in reasons if r.startswith("no")),
                })
                continue

            score = (
                min(item.photographic_image_count, 5000) / 100.0
                + min(item.volumetric_image_count, 2000) / 200.0
                + (5.0 if item.license else 0.0)
                - (3.0 if item.images_possibly_in_archives else 0.0)
            )
            accepted_raw.append((round(score, 2), item, reasons))

        accepted_raw.sort(key=lambda t: t[0], reverse=True)
        accepted = [
            RankedDataset(
                rank=i,
                dataset_id=item.dataset_id,
                source=item.source,
                score=score,
                license=item.license,
                photographic_image_count=item.photographic_image_count,
                volumetric_image_count=item.volumetric_image_count,
                reasons=reasons,
            )
            for i, (score, item, reasons) in enumerate(accepted_raw[: payload.top_k], start=1)
        ]

        summary = (
            f"{len(accepted)} dataset(s) passed the '{payload.image_policy}' policy, "
            f"{len(rejected)} rejected."
        )
        if not accepted:
            summary += (
                " Nothing passed — consider relaxing image_policy to "
                "'photographic_or_volumetric', or inspecting more candidates."
            )

        return ShortlistOut(
            image_policy=payload.image_policy,
            accepted=accepted,
            rejected=rejected,
            summary=summary,
        )


def build_tools() -> list[ValidatedTool]:
    return [
        NormalizeLesionQueryTool(),
        SearchOpenDatasetsTool(),
        InspectDatasetFilesTool(),
        ShortlistImageDatasetsTool(),
    ]


__all__ = [
    "ValidatedTool", "ToolInputError", "ToolOutputError",
    "NormalizeLesionQueryTool", "SearchOpenDatasetsTool",
    "InspectDatasetFilesTool", "ShortlistImageDatasetsTool", "build_tools",
]
