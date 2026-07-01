import torch
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from huggingface_hub import HfApi

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
SAVE_DIR        = Path("hf_export/wav2vec2-ser-v1")
HF_REPO_ID      = "PushkarOM/wav2vec2-ser-v1"  # adjust if needed
MODEL_NAME      = "facebook/wav2vec2-base"
NUM_CLASSES     = 8

LABEL_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "frustration": 7,
}
ID2LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
LABEL2ID = LABEL_TO_IDX

# load checkpoint into model
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# save locally in HF format
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
feature_extractor.save_pretrained(SAVE_DIR)
print(f"saved to {SAVE_DIR}")

# push to hub
api = HfApi()
api.create_repo(HF_REPO_ID, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path=str(SAVE_DIR),
    repo_id=HF_REPO_ID,
    repo_type="model",
)
print(f"pushed to https://huggingface.co/{HF_REPO_ID}")
