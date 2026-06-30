import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from dataset import SERDataset


PROJECT_ROOT    = Path(__file__).resolve().parents[3]
MANIFESTS_DIR   = PROJECT_ROOT / "common" / "manifests"
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_NAME      = "facebook/wav2vec2-base"
NUM_CLASSES     = 8
BATCH_SIZE      = 16
ACCUM_STEPS     = 2        # effective batch = 16 * 2 = 32
NUM_EPOCHS      = 10
LR              = 2e-5
WARMUP_RATIO    = 0.1      # 10% of total steps used for warmup
MAX_LENGTH_SEC  = 6
SEED            = 42

LABEL_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "frustration": 7,
}


def get_class_weights(train_df, label_to_idx, device):
    labels = train_df["canonical_label"].values
    classes = list(label_to_idx.keys())
    
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=labels
    )
    
    # compute_class_weight returns weights in the order of `classes`
    # need to reorder to match LABEL_TO_IDX integer ordering
    ordered_weights = np.array([
        weights[classes.index(c)]
        for c in sorted(
            label_to_idx,
            key=label_to_idx.get
        )
    ])
    
    return torch.tensor(ordered_weights, dtype=torch.float32).to(device)

def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device, accum_steps):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_values = batch["input_values"].to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_values=input_values)
            loss = loss_fn(outputs.logits, labels)

        # Track the original (unscaled) loss
        total_loss += loss.item()

        # Scale loss for gradient accumulation
        loss = loss / accum_steps
        scaler.scale(loss).backward()

        # Update weights after accum_steps or at the final batch
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        # Store predictions for F1
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, weighted_f1


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values=input_values)
                loss = loss_fn(outputs.logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        

    avg_loss = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, weighted_f1

 
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. load manifests
    train_df = pd.read_csv(MANIFESTS_DIR / "train.csv")
    val_df   = pd.read_csv(MANIFESTS_DIR / "val.csv")

    # 2. feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # 3. datasets + dataloaders
    train_ds = SERDataset(train_df, feature_extractor, max_length_seconds=MAX_LENGTH_SEC)
    val_ds   = SERDataset(val_df,   feature_extractor, max_length_seconds=MAX_LENGTH_SEC)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    model.freeze_feature_encoder()
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
        
    # 5. class weights + loss
    class_weights = get_class_weights(train_df, LABEL_TO_IDX, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 6. optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 7. scheduler
    total_steps  = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 8. mixed precision scaler
    scaler = torch.amp.GradScaler(device="cuda")

    # 9. epoch loop 
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        # call train_one_epoch -> get train_loss, train_f1
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, device, ACCUM_STEPS)
        # call evaluate -> get val_loss, val_f1
        val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)
        # print epoch summary
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, CHECKPOINT_DIR / "best_model.pt")
        
    print(f"Training complete. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
