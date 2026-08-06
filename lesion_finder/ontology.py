"""Controlled vocabulary for MRI lesion types.

This is the *input* side of the guardrail: free-text user queries are mapped
onto a small, closed set of lesion families. Anything that does not map is
rejected before a single network call is made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LesionType:
    """One lesion family in the ontology."""

    key: str
    canonical_name: str
    body_region: str
    # Terms the user might type. Matched case-insensitively as whole words.
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    # Terms fed to dataset search backends (broader than synonyms).
    search_terms: tuple[str, ...] = field(default_factory=tuple)


LESION_ONTOLOGY: dict[str, LesionType] = {
    "glioma": LesionType(
        key="glioma",
        canonical_name="Glioma / glioblastoma",
        body_region="brain",
        synonyms=("glioma", "gliomas", "glioblastoma", "gbm", "hgg", "lgg",
                  "astrocytoma", "oligodendroglioma"),
        search_terms=("brain tumor MRI", "glioma MRI segmentation", "BraTS",
                      "glioblastoma MRI"),
    ),
    "meningioma": LesionType(
        key="meningioma",
        canonical_name="Meningioma",
        body_region="brain",
        synonyms=("meningioma", "meningiomas"),
        search_terms=("meningioma MRI", "brain tumor classification MRI"),
    ),
    "pituitary_tumor": LesionType(
        key="pituitary_tumor",
        canonical_name="Pituitary tumour / adenoma",
        body_region="brain",
        synonyms=("pituitary", "pituitary tumor", "pituitary tumour",
                  "pituitary adenoma", "adenoma"),
        search_terms=("pituitary tumor MRI", "brain tumor classification MRI"),
    ),
    "brain_metastasis": LesionType(
        key="brain_metastasis",
        canonical_name="Brain metastasis",
        body_region="brain",
        synonyms=("metastasis", "metastases", "brain mets", "secondary brain tumor"),
        search_terms=("brain metastases MRI", "BrainMetShare", "metastasis MRI segmentation"),
    ),
    "ms_lesion": LesionType(
        key="ms_lesion",
        canonical_name="Multiple sclerosis white-matter lesion",
        body_region="brain",
        synonyms=("ms", "multiple sclerosis", "demyelination", "demyelinating",
                  "white matter lesion", "wml", "plaque"),
        search_terms=("multiple sclerosis MRI lesion", "MSSEG", "white matter"
                      " hyperintensity MRI"),
    ),
    "stroke_lesion": LesionType(
        key="stroke_lesion",
        canonical_name="Ischaemic stroke lesion / infarct",
        body_region="brain",
        synonyms=("stroke", "ischemic", "ischaemic", "infarct", "infarction",
                  "cerebral infarct"),
        search_terms=("ischemic stroke MRI lesion", "ISLES", "stroke lesion segmentation"),
    ),
    "hemorrhage": LesionType(
        key="hemorrhage",
        canonical_name="Intracranial haemorrhage",
        body_region="brain",
        synonyms=("hemorrhage", "haemorrhage", "bleed", "hematoma", "haematoma",
                  "microbleed"),
        search_terms=("intracranial hemorrhage MRI", "cerebral microbleed MRI"),
    ),
    "wmh": LesionType(
        key="wmh",
        canonical_name="White-matter hyperintensity (small-vessel disease)",
        body_region="brain",
        synonyms=("wmh", "hyperintensity", "hyperintensities", "leukoaraiosis",
                  "small vessel disease"),
        search_terms=("white matter hyperintensity segmentation MRI", "WMH challenge"),
    ),
    "breast_lesion": LesionType(
        key="breast_lesion",
        canonical_name="Breast lesion / breast cancer",
        body_region="breast",
        synonyms=("breast", "breast cancer", "breast lesion", "mammary", "dce breast"),
        search_terms=("breast MRI lesion", "DCE breast MRI cancer"),
    ),
    "prostate_lesion": LesionType(
        key="prostate_lesion",
        canonical_name="Prostate lesion",
        body_region="prostate",
        synonyms=("prostate", "prostate cancer", "prostate lesion", "pi-rads", "picai"),
        search_terms=("prostate MRI lesion", "PI-CAI prostate MRI"),
    ),
    "liver_lesion": LesionType(
        key="liver_lesion",
        canonical_name="Liver lesion",
        body_region="liver",
        synonyms=("liver", "hepatic", "hepatocellular", "hcc", "liver lesion"),
        search_terms=("liver MRI lesion segmentation", "hepatic MRI tumor"),
    ),
    "spinal_cord_lesion": LesionType(
        key="spinal_cord_lesion",
        canonical_name="Spinal cord lesion",
        body_region="spine",
        synonyms=("spinal cord", "spine lesion", "myelitis", "cord lesion"),
        search_terms=("spinal cord MRI lesion segmentation",),
    ),
    "knee_lesion": LesionType(
        key="knee_lesion",
        canonical_name="Knee cartilage / meniscal lesion",
        body_region="knee",
        synonyms=("knee", "meniscus", "meniscal", "cartilage", "acl"),
        search_terms=("knee MRI lesion", "MRNet knee MRI", "cartilage segmentation MRI"),
    ),
}

# Phrases that mean the person wants a clinical opinion rather than a dataset.
# Kept deliberately small and explicit — this is an input validator, not a
# content moderation system.
_CLINICAL_INTENT_PATTERNS = (
    r"\bmy (scan|mri|results?|report|tumou?r|lesion)\b",
    r"\b(do|does) i have\b",
    r"\bdiagnos(e|is|ing) (me|my|this patient)\b",
    r"\bshould i (take|start|stop)\b",
    r"\bis (this|it) (cancer|malignant|benign|serious)\b",
    r"\b(prognosis|treatment plan|how long do i have)\b",
)


def detect_clinical_intent(text: str) -> str | None:
    """Return the matched phrase if the query reads as a request for medical advice."""
    lowered = text.lower()
    for pattern in _CLINICAL_INTENT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return match.group(0)
    return None


def resolve_lesion_type(text: str) -> LesionType | None:
    """Map free text onto a LesionType, or None if nothing matches."""
    lowered = " " + re.sub(r"[^a-z0-9\- ]+", " ", text.lower()) + " "

    best: tuple[int, LesionType] | None = None
    for lesion in LESION_ONTOLOGY.values():
        if lesion.key.replace("_", " ") in lowered:
            return lesion
        for synonym in lesion.synonyms:
            if re.search(rf"(?<![a-z]){re.escape(synonym)}(?![a-z])", lowered):
                score = len(synonym)
                if best is None or score > best[0]:
                    best = (score, lesion)
    return best[1] if best else None


def supported_lesion_keys() -> list[str]:
    return sorted(LESION_ONTOLOGY)
