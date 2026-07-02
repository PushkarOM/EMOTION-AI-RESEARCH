def fuse(acoustic_result, semantic_result, acoustic_weight=0.6, semantic_weight=0.4):
    """
    Combine acoustic and semantic branch outputs via confidence-weighted fusion.

    Args:
        acoustic_result (dict | None): Output from acoustic branch:
            {"emotion": str, "confidence": float, "model": str}
        semantic_result (dict | None): Output from semantic branch:
            {"emotion": str, "confidence": float, "model": str}
        acoustic_weight (float): Base weight for acoustic branch (default 0.6).
        semantic_weight (float): Base weight for semantic branch (default 0.4).

    Returns:
        dict: Standardized fusion output schema.
    """

    if acoustic_result is None and semantic_result is None:
        return None

    if acoustic_result is None:
        return {
            "source": "audio_multimodal",
            "emotion": semantic_result["emotion"],
            "confidence": semantic_result["confidence"],
            "components": {
                "acoustic": None,
                "semantic": semantic_result,
            },
        }
    

    if semantic_result is None:
        return {
            "source": "audio_multimodal",
            "emotion": acoustic_result["emotion"],
            "confidence": acoustic_result["confidence"],
            "components": {
                "acoustic": acoustic_result,
                "semantic": None,
            },
        }


    acoustic_score = (
        acoustic_result["confidence"] * acoustic_weight
    )

    semantic_score = (
        semantic_result["confidence"] * semantic_weight
    )

    if acoustic_score >= semantic_score:
        final_emotion     = acoustic_result["emotion"]
        final_confidence  = acoustic_result["confidence"]  # raw, not weighted
    else:
        final_emotion     = semantic_result["emotion"]
        final_confidence  = semantic_result["confidence"]  # raw, not weighted

    return {
        "source": "audio_multimodal",
        "emotion": final_emotion,
        "confidence": final_confidence,
        "components": {
            "acoustic": acoustic_result,
            "semantic": semantic_result,
        },
    }
