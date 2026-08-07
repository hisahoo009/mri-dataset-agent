"""The three tools the agent can call.

Each is a plain Python function with an @tool decorator. smolagents reads the
type hints and docstring to describe the tool to the model, so the docstring is
part of the program, not a comment.

Guardrails, in order of appearance:
  1. normalize_lesion_query  validates the user's request before anything runs
  2. search_datasets         only accepts a lesion_key produced by step 1
  3. inspect_dataset         validates its own output against a schema
"""

from collections import Counter

import requests
from pydantic import BaseModel, ValidationError
from smolagents import tool

from lesions import LESIONS, asks_for_medical_advice, find_lesion

HF_API = "https://huggingface.co/api"
TIMEOUT = 20

# What counts as an image, and what a reader would have to do to open it.
PHOTO_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VOLUME_FORMATS = {".nii", ".nii.gz", ".dcm", ".mha", ".nrrd"}
ARCHIVE_FORMATS = {".zip", ".tar", ".tar.gz", ".parquet", ".arrow", ".h5"}


class DatasetReport(BaseModel):
    """The shape inspect_dataset promises to return."""

    dataset_id: str
    files_checked: int
    photo_images: int
    volume_images: int
    archives: int
    formats: dict
    license: str
    verdict: str


@tool
def normalize_lesion_query(query: str) -> dict:
    """Turn a free-text request into a lesion_key the other tools accept.

    Call this first. It rejects anything that is not a supported MRI lesion
    type, and refuses requests for medical advice about a specific person.

    Args:
        query: What the user asked for, e.g. "open MS lesion datasets".
    """
    if asks_for_medical_advice(query):
        raise ValueError(
            "Refused: this reads as a request for medical advice about a person. "
            "This agent only finds public research datasets and cannot interpret "
            "anyone's scan."
        )

    lesion_key = find_lesion(query)
    if lesion_key is None:
        raise ValueError(
            f"No supported lesion type found in {query!r}. "
            f"Supported: {', '.join(LESIONS)}. Ask the user which they meant."
        )

    return {"lesion_key": lesion_key, "search_terms": LESIONS[lesion_key][1]}


@tool
def search_datasets(lesion_key: str) -> list:
    """Search the Hugging Face Hub for datasets matching a lesion type.

    Returns candidates whose *names* match. It does not prove they contain
    images -- use inspect_dataset for that.

    Args:
        lesion_key: A key from normalize_lesion_query, e.g. "ms_lesion".
    """
    if lesion_key not in LESIONS:
        raise ValueError(
            f"Unknown lesion_key {lesion_key!r}. Call normalize_lesion_query first. "
            f"Valid keys: {', '.join(LESIONS)}."
        )

    found = {}
    for term in LESIONS[lesion_key][1]:
        response = requests.get(
            f"{HF_API}/datasets",
            params={"search": term, "limit": 10, "sort": "downloads", "direction": -1},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        for row in response.json():
            dataset_id = row.get("id")
            if dataset_id and dataset_id not in found:
                found[dataset_id] = {
                    "dataset_id": dataset_id,
                    "url": f"https://huggingface.co/datasets/{dataset_id}",
                    "downloads": row.get("downloads", 0),
                }

    return list(found.values())[:20]


@tool
def inspect_dataset(dataset_id: str) -> dict:
    """List a dataset's files and report which image formats it really contains.

    Args:
        dataset_id: An id from search_datasets, e.g. "user/brain-mri".
    """
    if "/" not in dataset_id:
        raise ValueError(f"dataset_id should look like 'owner/name', got {dataset_id!r}.")

    response = requests.get(
        f"{HF_API}/datasets/{dataset_id}/tree/main",
        params={"recursive": "true", "limit": 1000},
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    counts = Counter()
    for entry in response.json():
        if entry.get("type") == "file":
            counts[_extension(entry.get("path", ""))] += 1
    counts.pop("", None)

    photo = sum(n for ext, n in counts.items() if ext in PHOTO_FORMATS)
    volume = sum(n for ext, n in counts.items() if ext in VOLUME_FORMATS)
    archive = sum(n for ext, n in counts.items() if ext in ARCHIVE_FORMATS)

    if photo:
        verdict = f"{photo} ready-to-use image files. Open them with PIL."
    elif volume:
        verdict = f"{volume} NIfTI/DICOM volumes. Needs nibabel or pydicom."
    elif archive:
        verdict = f"No loose images; {archive} archives. Images may be inside them."
    else:
        verdict = "No image files found."

    report = {
        "dataset_id": dataset_id,
        "files_checked": sum(counts.values()),
        "photo_images": photo,
        "volume_images": volume,
        "archives": archive,
        "formats": dict(counts.most_common(10)),
        "license": _license(dataset_id),
        "verdict": verdict,
    }

    # Output guardrail: never hand the agent a payload of the wrong shape.
    try:
        return DatasetReport(**report).model_dump()
    except ValidationError as error:
        raise ValueError(f"inspect_dataset built a malformed report: {error}") from None


def _extension(path):
    name = path.lower().rsplit("/", 1)[-1]
    for compound in (".nii.gz", ".tar.gz"):
        if name.endswith(compound):
            return compound
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def _license(dataset_id):
    try:
        response = requests.get(f"{HF_API}/datasets/{dataset_id}", timeout=TIMEOUT)
        tags = response.json().get("tags", [])
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.split(":", 1)[1]
    except Exception:
        pass
    return "unknown"


TOOLS = [normalize_lesion_query, search_datasets, inspect_dataset]
