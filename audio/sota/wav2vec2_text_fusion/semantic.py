from transformers import pipeline
import yaml
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MAPPING_PATH  = PROJECT_ROOT / "common" / "configs" / "label_mapping.yaml"
ROBERTA_MODEL = "PushkarOM/roberta-head-goemotion"

_classifier = None
_go_to_canonical = None  # dict: goemotions label -> canonical label


def load_resources(device=0):
    """
    Lazy-load the RoBERTa classifier pipeline and GoEmotions->canonical mapping.
    device=0 means GPU 0, device=-1 means CPU (HF pipeline convention).
    """
    global _classifier, _go_to_canonical
    if _classifier is None:
        # load HF pipeline for text classification
        # top_k=None returns all 28 class scores, not just the top 1
        _classifier = pipeline(
            task="text-classification",
            model=ROBERTA_MODEL,
            top_k=None,              # return all class probabilities
            device=device,
        )
    
    if _go_to_canonical is None:
        # load label_mapping.yaml, extract the "goemotions" section
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f)

        _go_to_canonical = mapping["goemotions"]


def classify_transcript(text, device=0):
    """
    Run GoEmotions classification on a transcript and map to canonical label.

    Args:
        text (str | None): Transcript from Whisper. Returns None if text is None.
        device (int): 0 for GPU, -1 for CPU.

    Returns:
        dict | None: {
            "emotion": canonical label string,
            "confidence": float (mapped canonical score)
            "raw": list of all 28 GoEmotions scores
        }
        or None if text is None/empty.
    """

    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    load_resources(device)

    raw_scores = _classifier(text)[0]

    canonical_scores = {}

    for pred in raw_scores:
        go_label = pred["label"]
        score = pred["score"]

        canonical = _go_to_canonical.get(go_label)

        if canonical is None:
            continue

        # Multiple GoEmotions labels may map to the same canonical label.
        # Keep the highest confidence.
        canonical_scores[canonical] = max(
            canonical_scores.get(canonical, 0.0),
            score,
        )

    if not canonical_scores:
        return None

    emotion = max(canonical_scores, key=canonical_scores.get)

    return {
        "emotion": emotion,
        "confidence": float(canonical_scores[emotion]),
        "model": ROBERTA_MODEL,
        "raw": raw_scores,
    }
