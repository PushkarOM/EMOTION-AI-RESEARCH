import whisper
import torch
from pathlib import Path

# Load once at module level
_models = {}


def get_whisper_model(model_size="small", device=None):
    """
    Lazy-load and cache the Whisper model at module level.
    Called once on first transcription, reused after that.

    Args:
        model_size (str): Whisper model size.
        device (str | None): 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        whisper.Whisper: Loaded Whisper model.
    """
    global _models

    if model_size not in _models:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(device, int):
            device = "cuda" if device >= 0 else "cpu"

        _models[model_size] = whisper.load_model(model_size, device=device)
        
    return _models[model_size]


def transcribe(audio_path, model_size="small", device=None):
    """
    Transcribe a .wav file to text using Whisper.

    Args:
        audio_path (str | Path): Path to the audio file.
        model_size (str): Whisper model size.
        device (str | None): 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        str | None: Transcript string, or None if empty/failed.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        model = get_whisper_model(model_size, device)

        result = model.transcribe(str(audio_path))
        text = result["text"].strip()

        return text if text else None

    except Exception as e:
        print(f"Transcription failed for {audio_path}: {e}")
        return None
