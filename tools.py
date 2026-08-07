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

# The only formats we accept: plain images you can open with PIL.
IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}


class DatasetReport(BaseModel):
    """The shape inspect_dataset promises to return."""

    dataset_id: str
    total_files: int
    image_count: int
    image_formats: dict
    license: str
    has_images: bool
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
    """Count the .jpg, .jpeg and .png files in a dataset.

    This is the only thing that proves a dataset is usable. A dataset with zero
    image files should not be recommended, whatever its name says.

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
    files = [e.get("path", "") for e in response.json() if e.get("type") == "file"]

    images = Counter(ext for ext in map(_extension, files) if ext in IMAGE_FORMATS)
    image_count = sum(images.values())

    if image_count:
        verdict = f"{image_count} image files ({', '.join(sorted(images))}). Open with PIL."
    else:
        verdict = "No .jpg/.jpeg/.png files found. Skip this one."

    report = {
        "dataset_id": dataset_id,
        "total_files": len(files),
        "image_count": image_count,
        "image_formats": dict(images),
        "license": _license(dataset_id),
        "has_images": image_count > 0,
        "verdict": verdict,
    }

    # Output guardrail: never hand the agent a payload of the wrong shape.
    try:
        return DatasetReport(**report).model_dump()
    except ValidationError as error:
        raise ValueError(f"inspect_dataset built a malformed report: {error}") from None


def _extension(path):
    name = path.lower().rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def _license(dataset_id):
    try:
        response = requests.get(f"{HF_API}/datasets/{dataset_id}", timeout=TIMEOUT)
        for tag in response.json().get("tags", []):
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.split(":", 1)[1]
    except Exception:
        pass
    return "unknown"


TOOLS = [normalize_lesion_query, search_datasets, inspect_dataset]
