"""Pydantic schemas for every tool boundary.

Guardrail design: each tool has an `*In` model (validated before the tool body
runs) and an `*Out` model (validated before anything is handed back to the
agent). The agent therefore can never see a payload that has not been through
a schema, and can never pass one that has not either.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Image format policy
# --------------------------------------------------------------------------- #

# 2-D "picture" formats — directly loadable with PIL, what most HF vision
# datasets ship and what the user asked us to check for.
PHOTOGRAPHIC_FORMATS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
)

# Native medical-imaging volume formats — still images, but need nibabel/pydicom.
VOLUMETRIC_FORMATS: frozenset[str] = frozenset(
    {".nii", ".nii.gz", ".dcm", ".dicom", ".mha", ".mhd", ".nrrd", ".img", ".hdr"}
)

# Container formats that usually *hold* images but hide them from a file listing.
ARCHIVE_FORMATS: frozenset[str] = frozenset(
    {".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".rar", ".parquet", ".arrow", ".h5", ".hdf5"}
)

ImagePolicy = Literal["photographic_only", "photographic_or_volumetric", "any"]

SourceName = Literal["huggingface", "zenodo", "openneuro"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# Tool 1 — normalise the query
# --------------------------------------------------------------------------- #

class NormalizeQueryIn(StrictModel):
    query: Annotated[str, Field(min_length=2, max_length=200)]

    @field_validator("query")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if any(ord(c) < 32 for c in v):
            raise ValueError("query must not contain control characters")
        return v


class NormalizedQuery(StrictModel):
    lesion_key: str
    canonical_name: str
    body_region: str
    search_terms: list[str] = Field(min_length=1, max_length=8)


# --------------------------------------------------------------------------- #
# Tool 2 — search
# --------------------------------------------------------------------------- #

class SearchIn(StrictModel):
    lesion_key: Annotated[str, Field(min_length=2, max_length=40)]
    sources: list[SourceName] = Field(default_factory=lambda: ["huggingface", "zenodo"])
    max_results_per_source: Annotated[int, Field(ge=1, le=25)] = 8

    @field_validator("sources")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one source is required")
        return list(dict.fromkeys(v))


class DatasetCandidate(StrictModel):
    dataset_id: str
    source: SourceName
    title: str
    url: str
    license: str | None = None
    description: str = ""
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = Field(default_factory=list)


class SearchOut(StrictModel):
    lesion_key: str
    queried_sources: list[SourceName]
    failed_sources: dict[str, str] = Field(default_factory=dict)
    candidates: list[DatasetCandidate] = Field(max_length=100)
    note: str = ""


# --------------------------------------------------------------------------- #
# Tool 3 — inspect files
# --------------------------------------------------------------------------- #

class InspectIn(StrictModel):
    dataset_id: Annotated[str, Field(min_length=1, max_length=200)]
    source: SourceName = "huggingface"
    max_files_scanned: Annotated[int, Field(ge=10, le=5000)] = 1000


class FormatBreakdown(StrictModel):
    extension: str
    count: int
    category: Literal["photographic", "volumetric", "archive", "other"]


class InspectOut(StrictModel):
    dataset_id: str
    source: SourceName
    files_scanned: int
    truncated: bool = False
    photographic_image_count: int = 0
    volumetric_image_count: int = 0
    archive_count: int = 0
    has_photographic_images: bool = False
    has_volumetric_images: bool = False
    images_possibly_in_archives: bool = False
    format_breakdown: list[FormatBreakdown] = Field(default_factory=list, max_length=40)
    license: str | None = None
    total_bytes: int | None = None
    verdict: str


# --------------------------------------------------------------------------- #
# Tool 4 — shortlist / rank
# --------------------------------------------------------------------------- #

class ShortlistIn(StrictModel):
    inspections: list[dict] = Field(min_length=1, max_length=25)
    image_policy: ImagePolicy = "photographic_only"
    require_known_license: bool = False
    top_k: Annotated[int, Field(ge=1, le=20)] = 5

    @field_validator("inspections", mode="before")
    @classmethod
    def _accept_json_strings(cls, v):
        """Accept dicts or the raw JSON strings the tools actually return.

        The model is copying `inspect_dataset_files` output between steps, and
        whether it lands here parsed or still a string depends on how the
        generated code handled it. A whole JSON array in one string is
        unwrapped too.
        """
        import json

        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("inspections was a string but not valid JSON") from None
        if not isinstance(v, list):
            raise ValueError("inspections must be a list of inspect_dataset_files results")

        parsed = []
        for item in v:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    raise ValueError(
                        "each inspection must be a dict or a JSON string; "
                        f"got unparseable text: {item[:60]!r}"
                    ) from None
            parsed.append(item)
        return parsed


class RankedDataset(StrictModel):
    rank: int
    dataset_id: str
    source: SourceName
    score: float
    license: str | None
    photographic_image_count: int
    volumetric_image_count: int
    reasons: list[str]


class ShortlistOut(StrictModel):
    image_policy: ImagePolicy
    accepted: list[RankedDataset]
    rejected: list[dict] = Field(default_factory=list)
    summary: str
