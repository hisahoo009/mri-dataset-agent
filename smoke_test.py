#!/usr/bin/env python3
"""Live check of the three source adapters — no LLM, no token needed.

    python smoke_test.py

Run this first. It calls each backend directly and prints what came back, so you
can tell an API-contract problem apart from an agent-reasoning problem.
"""

from __future__ import annotations

import json
import sys

from lesion_finder.tools import (
    InspectDatasetFilesTool,
    NormalizeLesionQueryTool,
    SearchOpenDatasetsTool,
    ShortlistImageDatasetsTool,
)

LESION = "brain tumor glioma"


def main() -> int:
    failures = []

    print("1) normalize_lesion_query")
    normalized = json.loads(NormalizeLesionQueryTool()(query=LESION))
    print("   ->", normalized["lesion_key"], "|", normalized["search_terms"])

    print("\n2) search_open_datasets")
    search = json.loads(SearchOpenDatasetsTool()(
        lesion_key=normalized["lesion_key"],
        sources=["huggingface", "zenodo", "openneuro"],
        max_results_per_source=5,
    ))
    for name, err in search["failed_sources"].items():
        print(f"   !! {name} failed: {err}")
        failures.append(name)
    by_source: dict[str, list[str]] = {}
    for c in search["candidates"]:
        by_source.setdefault(c["source"], []).append(c["dataset_id"])
    for source, ids in by_source.items():
        print(f"   {source}: {len(ids)} candidates -> {ids[:3]}")
    if not search["candidates"]:
        print("   !! no candidates from any source")
        return 1

    print("\n3) inspect_dataset_files (first 3 candidates)")
    inspections = []
    for candidate in search["candidates"][:3]:
        try:
            result = json.loads(InspectDatasetFilesTool()(
                dataset_id=candidate["dataset_id"], source=candidate["source"],
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"   !! {candidate['dataset_id']}: {exc}")
            continue
        inspections.append(result)
        print(f"   {result['dataset_id']}: {result['files_scanned']} files, "
              f"{result['photographic_image_count']} 2-D / "
              f"{result['volumetric_image_count']} volumetric, "
              f"licence={result['license']}")
        print(f"      {result['verdict']}")

    if not inspections:
        print("   !! nothing could be inspected")
        return 1

    print("\n4) shortlist_image_datasets")
    shortlist = json.loads(ShortlistImageDatasetsTool()(
        inspections=inspections, image_policy="photographic_or_volumetric", top_k=5,
    ))
    print("  ", shortlist["summary"])
    for d in shortlist["accepted"]:
        print(f"   #{d['rank']} {d['dataset_id']} (score {d['score']}) — {'; '.join(d['reasons'])}")

    print("\nDone." + (f" Sources that failed: {failures}" if failures else " All sources OK."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
