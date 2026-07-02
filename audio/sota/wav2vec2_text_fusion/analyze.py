from transcribe import transcribe
from semantic import classify_transcript
from fusion import fuse

import torch
from pathlib import Path
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)
import soundfile as sf
import librosa
import numpy as np

ACOUSTIC_MODEL_ID = "PushkarOM/wav2vec2-ser-v1"
SEMANTIC_MODEL_ID = "PushkarOM/roberta-head-goemotion"

TARGET_SR = 16000
MAX_SAMPLES = 6 * TARGET_SR

LABEL_TO_IDX = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6,
    "frustration": 7,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

_acoustic_model = None
_feature_extractor = None


def load_acoustic_model(device):
    """
    Lazy-load the acoustic SER model.
    """

    global _acoustic_model, _feature_extractor

    if _feature_extractor is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            ACOUSTIC_MODEL_ID
        )

    if _acoustic_model is None:
        _acoustic_model = (
            Wav2Vec2ForSequenceClassification.from_pretrained(
                ACOUSTIC_MODEL_ID
            )
            .to(device)
            .eval()
        )


def run_acoustic(audio_path, device):
    """
    Run acoustic emotion recognition.

    Returns
    -------
    {
        "emotion": str,
        "confidence": float,
        "model": ACOUSTIC_MODEL_ID
    }
    """

    load_acoustic_model(device)


    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR,
        )

    # Pad / truncate

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio)),
            mode="constant",
        )

    # Feature extraction

    inputs = _feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference

    with torch.no_grad():
        outputs = _acoustic_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)

        confidence, prediction = torch.max(probs, dim=-1)

    emotion = IDX_TO_LABEL[prediction.item()]

    return {
        "emotion": emotion,
        "confidence": float(confidence.item()),
        "model": ACOUSTIC_MODEL_ID,
    }


def analyze(audio_path, device=None):
    """
    Full multimodal emotion analysis pipeline.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_device = 0 if device == "cuda" else -1

    acoustic_result = run_acoustic(audio_path, device)

    transcript = transcribe(
        str(audio_path),
        device=hf_device,
    )

    semantic_result = classify_transcript(
        transcript,
        device=hf_device,
    )

    return fuse(
        acoustic_result,
        semantic_result,
    )


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path_to_wav>")
        sys.exit(1)

    result = analyze(sys.argv[1])

    print(json.dumps(result, indent=2))
