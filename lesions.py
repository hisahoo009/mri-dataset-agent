"""The lesion types this agent knows about.

This is the guardrail's vocabulary. A query that doesn't map to one of these is
rejected before any network call happens.
"""

# lesion_key -> (words a user might type, terms to search the Hub with)
LESIONS = {
    "glioma": (
        ["glioma", "glioblastoma", "gbm", "brain tumor", "brain tumour"],
        ["brain tumor MRI", "glioma MRI"],
    ),
    "meningioma": (
        ["meningioma"],
        ["meningioma MRI", "brain tumor classification MRI"],
    ),
    "pituitary_tumor": (
        ["pituitary", "adenoma"],
        ["pituitary tumor MRI", "brain tumor classification MRI"],
    ),
    "ms_lesion": (
        ["ms", "multiple sclerosis", "demyelination", "white matter lesion"],
        ["multiple sclerosis MRI", "white matter lesion MRI"],
    ),
    "stroke_lesion": (
        ["stroke", "ischemic", "ischaemic", "infarct"],
        ["ischemic stroke MRI", "stroke lesion segmentation"],
    ),
    "breast_lesion": (
        ["breast", "breast cancer", "mammary"],
        ["breast MRI lesion", "breast cancer MRI"],
    ),
    "prostate_lesion": (
        ["prostate", "pi-rads"],
        ["prostate MRI lesion", "prostate cancer MRI"],
    ),
    "knee_lesion": (
        ["knee", "meniscus", "meniscal", "cartilage", "acl"],
        ["knee MRI", "knee MRI lesion"],
    ),
}

# Phrases that mean "interpret my scan" rather than "find me data".
MEDICAL_ADVICE_PHRASES = [
    "my scan", "my mri", "my results", "my tumor", "my tumour",
    "do i have", "diagnose me", "is this cancer", "should i",
]


def find_lesion(text):
    """Return the lesion_key mentioned in `text`, or None."""
    lowered = text.lower()
    for key, (synonyms, _) in LESIONS.items():
        if any(word in lowered for word in synonyms):
            return key
    return None


def asks_for_medical_advice(text):
    lowered = text.lower()
    return any(phrase in lowered for phrase in MEDICAL_ADVICE_PHRASES)
