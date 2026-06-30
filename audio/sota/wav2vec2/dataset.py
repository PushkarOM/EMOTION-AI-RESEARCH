import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa

MAX_LENGTH_SECONDS = 6
TARGET_SAMPLE_RATE = 16000

LABEL_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "frustration": 7,
}

class SERDataset(Dataset):
    """
    PyTorch Dataset for Speech Emotion Recognition (SER).

    Loads audio files from a manifest DataFrame, converts them to a
    standardized format (mono, 16 kHz, fixed length), extracts model-ready
    features using a Hugging Face feature extractor, and returns the
    corresponding emotion label as a tensor.
    """

    def __init__(self, manifest_df, feature_extractor, max_length_seconds=MAX_LENGTH_SECONDS):
        """
        Initialize the dataset.

        Args:
            manifest_df (pd.DataFrame):
                DataFrame containing audio file paths and labels.
                Expected columns:
                    - filepath
                    - canonical_label

            feature_extractor:
                Hugging Face feature extractor used to convert raw audio
                waveforms into model inputs.

            max_length_seconds (int):
                Maximum duration of audio clips in seconds. Audio longer
                than this value is truncated, while shorter audio is
                zero-padded.
        """
        self.df = manifest_df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.max_samples = int(max_length_seconds * TARGET_SAMPLE_RATE)

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of rows in the manifest DataFrame.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and preprocess a single audio sample.

        Processing steps:
            1. Load audio from disk.
            2. Convert stereo audio to mono.
            3. Resample audio to 16 kHz if necessary.
            4. Truncate or zero-pad audio to a fixed length.
            5. Extract model-ready features.
            6. Convert the emotion label to its integer index.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                {
                    "input_values": Tensor containing processed audio features,
                    "label": Tensor containing the encoded emotion label
                }
        """
        row = self.df.iloc[idx]

        audio, sr = sf.read(row["filepath"])

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        inputs = self.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs.input_values.squeeze(0)

        label_idx = LABEL_TO_IDX[row["canonical_label"]]

        return {
            "input_values": input_values,
            "label": torch.tensor(label_idx, dtype=torch.long)
        }