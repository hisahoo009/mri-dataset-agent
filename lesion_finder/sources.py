"""Dataset backends: HTTP plumbing, file-format classification, and one
adapter per repository.

Everything here is I/O and response-shape translation. The adapters expose the
same two calls (`search` / `inspect`) and return the validated models from
`schemas.py`, so `tools.py` never has to know which repository it is talking to.

Sections
    1. HTTP        — fixed timeouts, one retry, JSON only
    2. Formats     — extension -> photographic / volumetric / archive / other
    3. Adapters    — Hugging Face Hub, Zenodo, OpenNeuro
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any, Protocol

import requests

from .schemas import (
    ARCHIVE_FORMATS,
    PHOTOGRAPHIC_FORMATS,
    VOLUMETRIC_FORMATS,
    DatasetCandidate,
    FormatBreakdown,
    InspectOut,
)

# --------------------------------------------------------------------------- #
# 1. HTTP
# --------------------------------------------------------------------------- #

DEFAULT_TIMEOUT = 20
USER_AGENT = "mri-dataset-agent/1.0 (huggingface agents course exercise)"


class SourceUnavailable(RuntimeError):
    """A dataset backend could not be reached or answered with an error."""


def get_json(url: str, params: dict[str, Any] | None = None,
             headers: dict[str, str] | None = None,
             timeout: int = DEFAULT_TIMEOUT) -> Any:
    return _request("GET", url, params=params, headers=headers, timeout=timeout)


def post_json(url: str, json_body: dict[str, Any],
              headers: dict[str, str] | None = None,
              timeout: int = DEFAULT_TIMEOUT) -> Any:
    return _request("POST", url, json_body=json_body, headers=headers, timeout=timeout)


def _request(method: str, url: str, *, params=None, json_body=None,
             headers=None, timeout=DEFAULT_TIMEOUT) -> Any:
    merged = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    merged.update(headers or {})

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.request(
                method, url, params=params, json=json_body,
                headers=merged, timeout=timeout,
            )
            if response.status_code == 404:
                raise SourceUnavailable(f"not found: {url}")
            response.raise_for_status()
            return response.json()
        except SourceUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised below
            last_error = exc
            if attempt == 0:
                time.sleep(1.0)

    raise SourceUnavailable(f"{method} {url} failed: {type(last_error).__name__}: {last_error}")

# --------------------------------------------------------------------------- #
# 2. File-format classification
# --------------------------------------------------------------------------- #

# Multi-part extensions must be tested before the single-part fallback.
_COMPOUND_EXTENSIONS = (".nii.gz", ".tar.gz", ".img.gz")


def file_extension(path: str) -> str:
    lowered = path.lower().rsplit("/", 1)[-1]
    for compound in _COMPOUND_EXTENSIONS:
        if lowered.endswith(compound):
            return compound
    if "." not in lowered:
        return ""
    return "." + lowered.rsplit(".", 1)[-1]


def categorize(extension: str) -> str:
    if extension in PHOTOGRAPHIC_FORMATS:
        return "photographic"
    if extension in VOLUMETRIC_FORMATS:
        return "volumetric"
    if extension in ARCHIVE_FORMATS:
        return "archive"
    return "other"


def summarize_files(
    dataset_id: str,
    source: str,
    paths: list[str],
    *,
    truncated: bool = False,
    license_: str | None = None,
    total_bytes: int | None = None,
) -> InspectOut:
    """Turn a flat list of file paths into a validated InspectOut."""
    counts: Counter[str] = Counter()
    per_category: Counter[str] = Counter()

    for path in paths:
        ext = file_extension(path)
        if not ext:
            continue
        counts[ext] += 1
        per_category[categorize(ext)] += 1

    breakdown = [
        FormatBreakdown(extension=ext, count=n, category=categorize(ext))
        for ext, n in counts.most_common(40)
    ]

    photographic = per_category["photographic"]
    volumetric = per_category["volumetric"]
    archives = per_category["archive"]

    if photographic:
        verdict = (
            f"Contains {photographic} directly-loadable 2-D image files "
            f"({', '.join(sorted({file_extension(p) for p in paths if categorize(file_extension(p)) == 'photographic'}))})."
        )
    elif volumetric:
        verdict = (
            f"No .jpg/.png-style files, but {volumetric} medical volume files "
            "(NIfTI/DICOM). Usable, but needs nibabel or pydicom and a slice-export step."
        )
    elif archives:
        verdict = (
            f"No loose image files in the listing; {archives} archive/columnar files "
            "found. Images are likely inside them — check the dataset card before assuming."
        )
    else:
        verdict = "No image files of any recognised format found in the file listing."

    if truncated:
        verdict += " (File listing was truncated, counts are a lower bound.)"

    return InspectOut(
        dataset_id=dataset_id,
        source=source,  # type: ignore[arg-type]
        files_scanned=len(paths),
        truncated=truncated,
        photographic_image_count=photographic,
        volumetric_image_count=volumetric,
        archive_count=archives,
        has_photographic_images=photographic > 0,
        has_volumetric_images=volumetric > 0,
        images_possibly_in_archives=photographic == 0 and volumetric == 0 and archives > 0,
        format_breakdown=breakdown,
        license=license_,
        total_bytes=total_bytes,
        verdict=verdict,
    )


class DatasetSource(Protocol):
    """Every backend adapter implements these two calls."""

    name: str

    def search(self, terms: list[str], limit: int) -> list[DatasetCandidate]: ...

    def inspect(self, dataset_id: str, max_files: int) -> InspectOut: ...

# --------------------------------------------------------------------------- #
# 3. Adapters
# --------------------------------------------------------------------------- #

HF_API = "https://huggingface.co/api"


class HuggingFaceSource:
    name = "huggingface"

    def search(self, terms: list[str], limit: int) -> list[DatasetCandidate]:
        seen: dict[str, DatasetCandidate] = {}

        for term in terms:
            try:
                rows = get_json(
                    f"{HF_API}/datasets",
                    params={"search": term, "limit": limit, "full": "false",
                            "sort": "downloads", "direction": -1},
                )
            except SourceUnavailable:
                continue
            if not isinstance(rows, list):
                continue

            for row in rows:
                dataset_id = row.get("id")
                if not dataset_id or dataset_id in seen:
                    continue
                tags = [t for t in (row.get("tags") or []) if isinstance(t, str)]
                seen[dataset_id] = DatasetCandidate(
                    dataset_id=dataset_id,
                    source="huggingface",
                    title=dataset_id,
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                    license=_license_from_tags(tags),
                    description=f"matched search term: {term!r}",
                    downloads=_as_int(row.get("downloads")),
                    likes=_as_int(row.get("likes")),
                    tags=tags[:25],
                )

        return list(seen.values())

    def inspect(self, dataset_id: str, max_files: int) -> InspectOut:
        # /tree gives real file paths; the dataset info endpoint gives licence.
        entries = get_json(
            f"{HF_API}/datasets/{dataset_id}/tree/main",
            params={"recursive": "true", "expand": "false", "limit": max_files},
        )
        if not isinstance(entries, list):
            raise SourceUnavailable(f"unexpected tree payload for {dataset_id}")

        paths, total_bytes = [], 0
        for entry in entries:
            if entry.get("type") != "file":
                continue
            paths.append(entry.get("path", ""))
            size = entry.get("size")
            if isinstance(size, int):
                total_bytes += size

        license_ = None
        try:
            info = get_json(f"{HF_API}/datasets/{dataset_id}")
            card = info.get("cardData") or {}
            license_ = card.get("license") or _license_from_tags(info.get("tags") or [])
            if isinstance(license_, list):
                license_ = ", ".join(str(x) for x in license_)
        except SourceUnavailable:
            pass

        return summarize_files(
            dataset_id, "huggingface", paths,
            truncated=len(entries) >= max_files,
            license_=license_,
            total_bytes=total_bytes or None,
        )


def _license_from_tags(tags: list) -> str | None:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _as_int(value) -> int | None:
    return value if isinstance(value, int) else None


ZENODO_API = "https://zenodo.org/api"


class ZenodoSource:
    name = "zenodo"

    def __init__(self, token: str | None = None):
        # Optional: raises rate limits. Public search works without it.
        self.token = token or os.environ.get("ZENODO_TOKEN")

    def _params(self, extra: dict) -> dict:
        params = dict(extra)
        if self.token:
            params["access_token"] = self.token
        return params

    def search(self, terms: list[str], limit: int) -> list[DatasetCandidate]:
        seen: dict[str, DatasetCandidate] = {}

        for term in terms:
            try:
                payload = get_json(
                    f"{ZENODO_API}/records",
                    params=self._params({
                        "q": term,
                        "size": limit,
                        "type": "dataset",
                        "access_right": "open",
                        "sort": "mostrecent",
                    }),
                )
            except SourceUnavailable:
                continue

            for hit in (payload.get("hits", {}) or {}).get("hits", []) or []:
                record_id = str(hit.get("id", "")).strip()
                if not record_id or record_id in seen:
                    continue
                meta = hit.get("metadata") or {}
                seen[record_id] = DatasetCandidate(
                    dataset_id=record_id,
                    source="zenodo",
                    title=str(meta.get("title") or f"Zenodo record {record_id}")[:300],
                    url=(hit.get("links") or {}).get("self_html")
                    or f"https://zenodo.org/records/{record_id}",
                    license=_license(meta),
                    description=_strip_html(str(meta.get("description") or ""))[:500],
                    tags=[str(k.get("keyword", k)) if isinstance(k, dict) else str(k)
                          for k in (meta.get("keywords") or [])][:25],
                )

        return list(seen.values())

    def inspect(self, dataset_id: str, max_files: int) -> InspectOut:
        record = get_json(f"{ZENODO_API}/records/{dataset_id}", params=self._params({}))
        files = record.get("files") or []

        paths, total_bytes = [], 0
        for entry in files[:max_files]:
            key = entry.get("key") or (entry.get("links") or {}).get("self", "")
            if key:
                paths.append(str(key))
            size = entry.get("size")
            if isinstance(size, int):
                total_bytes += size

        return summarize_files(
            dataset_id, "zenodo", paths,
            truncated=len(files) > max_files,
            license_=_license(record.get("metadata") or {}),
            total_bytes=total_bytes or None,
        )


def _license(meta: dict) -> str | None:
    lic = meta.get("license")
    if isinstance(lic, dict):
        return lic.get("id") or lic.get("identifier") or lic.get("title")
    if isinstance(lic, str):
        return lic
    return None


def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ").strip()


OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"

_SEARCH_QUERY = """
query Search($q: String!, $first: Int!) {
  datasets(first: $first, filterBy: {}, q: $q) {
    edges {
      node {
        id
        latestSnapshot {
          tag
          description { Name Authors License }
          summary { modalities subjectMetadata { participantId } }
        }
      }
    }
  }
}
"""

_FILES_QUERY = """
query Files($id: ID!) {
  dataset(id: $id) {
    id
    latestSnapshot {
      tag
      description { Name License }
      files { filename directory size }
      size
    }
  }
}
"""


class OpenNeuroSource:
    name = "openneuro"

    def search(self, terms: list[str], limit: int) -> list[DatasetCandidate]:
        seen: dict[str, DatasetCandidate] = {}

        for term in terms:
            try:
                payload = post_json(
                    OPENNEURO_GRAPHQL,
                    {"query": _SEARCH_QUERY, "variables": {"q": term, "first": limit}},
                )
            except SourceUnavailable:
                continue
            if payload.get("errors"):
                continue

            edges = (((payload.get("data") or {}).get("datasets") or {}).get("edges")) or []
            for edge in edges:
                node = edge.get("node") or {}
                accession = node.get("id")
                if not accession or accession in seen:
                    continue
                snapshot = node.get("latestSnapshot") or {}
                desc = snapshot.get("description") or {}
                modalities = ((snapshot.get("summary") or {}).get("modalities")) or []
                # Guardrail: OpenNeuro hosts EEG/MEG/PET too. Keep MRI only.
                if modalities and not any("mr" in str(m).lower() for m in modalities):
                    continue
                seen[accession] = DatasetCandidate(
                    dataset_id=accession,
                    source="openneuro",
                    title=str(desc.get("Name") or accession)[:300],
                    url=f"https://openneuro.org/datasets/{accession}",
                    license=str(desc.get("License") or "CC0-1.0"),
                    description=f"modalities: {', '.join(map(str, modalities)) or 'unspecified'}",
                    tags=[str(m) for m in modalities][:25],
                )

        return list(seen.values())

    def inspect(self, dataset_id: str, max_files: int) -> InspectOut:
        payload = post_json(OPENNEURO_GRAPHQL, {"query": _FILES_QUERY, "variables": {"id": dataset_id}})
        if payload.get("errors"):
            raise SourceUnavailable(f"openneuro graphql error for {dataset_id}")

        dataset = (payload.get("data") or {}).get("dataset") or {}
        snapshot = dataset.get("latestSnapshot") or {}
        files = [f for f in (snapshot.get("files") or []) if not f.get("directory")]

        paths = [str(f.get("filename", "")) for f in files[:max_files]]
        total = snapshot.get("size") if isinstance(snapshot.get("size"), int) else None

        return summarize_files(
            dataset_id, "openneuro", paths,
            truncated=len(files) > max_files,
            license_=str((snapshot.get("description") or {}).get("License") or "CC0-1.0"),
            total_bytes=total,
        )

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, DatasetSource] = {
    "huggingface": HuggingFaceSource(),
    "zenodo": ZenodoSource(),
    "openneuro": OpenNeuroSource(),
}

__all__ = [
    "SourceUnavailable", "get_json", "post_json",
    "file_extension", "categorize", "summarize_files",
    "DatasetSource", "HuggingFaceSource", "ZenodoSource", "OpenNeuroSource",
    "REGISTRY",
]
