import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from dataset import SERDataset


PROJECT_ROOT   = Path(__file__).resolve().parents[3]
MANIFESTS_DIR  = PROJECT_ROOT / "common" / "manifests"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_NAME     = "facebook/wav2vec2-base"
NUM_CLASSES    = 8
BATCH_SIZE     = 16
ACCUM_STEPS    = 2        # effective batch = 16 × 2 = 32
NUM_EPOCHS     = 10
LR             = 2e-5
WARMUP_RATIO   = 0.1     # 10% of total steps used for warmup
MAX_LENGTH_SEC = 6
SEED           = 42
NUM_WORKERS    = 0        # set to 2-4 on Linux / Kaggle for faster loading

KAGGLE = False            # flip to True when running on Kaggle

LABEL_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "frustration": 7,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
LABEL_NAMES  = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]


def remap_path(filepath, kaggle=KAGGLE):
    """
    Remap a local absolute filepath to its Kaggle dataset mount equivalent.

    No-op when kaggle=False (local runs). When kaggle=True, resolves the
    dataset name from the path and rewrites the prefix to match Kaggle's
    /kaggle/input/<dataset>/ mount structure.

    Args:
        filepath (str): Absolute filepath from the manifest CSV.
        kaggle (bool): Whether to apply the Kaggle path remapping.

    Returns:
        str: Remapped filepath (or original if kaggle=False).
    """
    if not kaggle:
        return filepath
    p = Path(filepath)
    parts = p.parts
    if "ravdess" in parts:
        idx = parts.index("ravdess")
        return str(Path("/kaggle/input/ravdess") / Path(*parts[idx + 1:]))
    elif "crema_d" in parts:
        idx = parts.index("crema_d")
        return str(Path("/kaggle/input/cremad") / Path(*parts[idx + 1:]))
    elif "iemocap" in parts:
        idx = parts.index("iemocap")
        return str(Path("/kaggle/input/iemocap") / Path(*parts[idx + 1:]))
    return filepath


def get_class_weights(train_df, label_to_idx, device):
    """
    Compute balanced class weights from the training set.

    Uses sklearn's 'balanced' strategy: n_samples / (n_classes × class_count).
    Weights are reordered to match the integer label mapping.

    Args:
        train_df (pd.DataFrame): Training manifest with a 'canonical_label' column.
        label_to_idx (dict): Mapping from class name to integer index.
        device (torch.device): Target device for the returned tensor.

    Returns:
        torch.Tensor: Float tensor of shape (num_classes,) on `device`.
    """
    labels  = train_df["canonical_label"].values
    classes = list(label_to_idx.keys())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=labels,
    )

    ordered_weights = np.array([
        weights[classes.index(c)]
        for c in sorted(label_to_idx, key=label_to_idx.get)
    ])

    return torch.tensor(ordered_weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device, accum_steps):
    """
    Train the model for one full epoch.

    Uses mixed-precision (fp16), gradient accumulation, and gradient clipping.
    Collects predictions across the epoch to compute the weighted F1-score.

    Args:
        model (torch.nn.Module): SER classification model.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Parameter optimizer.
        scheduler: Learning rate scheduler.
        scaler (torch.amp.GradScaler): Mixed-precision gradient scaler.
        loss_fn (torch.nn.Module): Weighted cross-entropy loss.
        device (torch.device): Compute device.
        accum_steps (int): Gradient accumulation steps per optimizer update.

    Returns:
        tuple[float, float]: (avg_loss, weighted_f1) for this epoch.
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_values = batch["input_values"].to(device)
        labels       = batch["label"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_values=input_values)
            loss    = loss_fn(outputs.logits, labels)

        total_loss += loss.item()

        # scale loss for gradient accumulation before backward
        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss    = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, weighted_f1


def evaluate(model, loader, loss_fn, device):
    """
    Evaluate the model on a validation set.

    Runs inference without gradient computation, computes average loss and
    weighted F1-score.

    Args:
        model (torch.nn.Module): SER classification model.
        loader (DataLoader): Validation data loader.
        loss_fn (torch.nn.Module): Weighted cross-entropy loss.
        device (torch.device): Compute device.

    Returns:
        tuple[float, float]: (avg_loss, weighted_f1).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels       = batch["label"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values=input_values)
                loss    = loss_fn(outputs.logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss    = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, weighted_f1


def predict(model, loader, device):
    """
    Run inference over a loader and return ground-truth + predicted labels.

    Args:
        model (torch.nn.Module): Trained classification model.
        loader (DataLoader): Data loader for inference.
        device (torch.device): Compute device.

    Returns:
        tuple[list[int], list[int]]: (y_true, y_pred).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            labels       = batch["label"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values=input_values)

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def print_confusion_matrix(cm, labels):
    """
    Print a compact confusion matrix with abbreviated class labels.

    Args:
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes).
        labels (list[str]): Ordered list of class name strings.
    """
    header = "         " + "".join(f"{l[:4]:>7}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = f"{l[:6]:>8}" + "".join(f"{cm[i, j]:>7}" for j in range(len(labels)))
        print(row)


def test(model, loader, device, label_names):
    """
    Evaluate the trained model on the test set and print a full report.

    Prints per-class precision / recall / F1 via classification_report and
    a compact confusion matrix.

    Args:
        model (torch.nn.Module): Trained classification model.
        loader (DataLoader): Test data loader.
        device (torch.device): Compute device.
        label_names (list[str]): Ordered class name strings.

    Returns:
        tuple[list[int], list[int]]: (y_true, y_pred).
    """
    y_true, y_pred = predict(model, loader, device)
    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=predicted):")
    print_confusion_matrix(cm, label_names)
    return y_true, y_pred


def evaluate_only():
    """
    Load the best checkpoint and run test-set evaluation without retraining.

    Entry point when the script is called with: python train.py evaluate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_df = pd.read_csv(MANIFESTS_DIR / "test.csv")
    test_df["filepath"] = test_df["filepath"].apply(remap_path)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    test_ds     = SERDataset(test_df, feature_extractor, max_length_seconds=MAX_LENGTH_SEC)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    print(f"loaded checkpoint — epoch {ckpt['epoch'] + 1}, val F1: {ckpt['val_f1']:.4f}")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    test(model, test_loader, device, LABEL_NAMES)


def main():
    """
    Execute the full training pipeline.

    Steps:
        1.  Set random seed and select compute device.
        2.  Load and (optionally) remap manifest filepaths for Kaggle.
        3.  Initialise Wav2Vec2FeatureExtractor.
        4.  Build SERDataset instances and DataLoaders for train / val / test.
        5.  Load pretrained Wav2Vec2ForSequenceClassification and freeze the
            CNN feature encoder.
        6.  Compute balanced class weights; build weighted CrossEntropyLoss.
        7.  Configure AdamW optimizer and linear warmup + decay scheduler.
        8.  Initialise mixed-precision GradScaler.
        9.  Train for NUM_EPOCHS, evaluate on val after each epoch, and save
            the checkpoint that achieves the best val weighted F1.
        10. Load the best checkpoint and run full test-set evaluation.
    """
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. load + remap manifests
    train_df = pd.read_csv(MANIFESTS_DIR / "train.csv")
    val_df   = pd.read_csv(MANIFESTS_DIR / "val.csv")
    test_df  = pd.read_csv(MANIFESTS_DIR / "test.csv")

    for df in (train_df, val_df, test_df):
        df["filepath"] = df["filepath"].apply(remap_path)

    # 2. feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # 3. datasets + dataloaders
    train_ds = SERDataset(train_df, feature_extractor, max_length_seconds=MAX_LENGTH_SEC)
    val_ds   = SERDataset(val_df,   feature_extractor, max_length_seconds=MAX_LENGTH_SEC)
    test_ds  = SERDataset(test_df,  feature_extractor, max_length_seconds=MAX_LENGTH_SEC)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # 4. model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    )
    model.freeze_feature_encoder()
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # 5. class weights + loss
    class_weights = get_class_weights(train_df, LABEL_TO_IDX, device)
    loss_fn       = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 6. optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # 7. scheduler
    total_steps  = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 8. mixed precision scaler
    scaler = torch.amp.GradScaler(device="cuda")

    # 9. epoch loop
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, ACCUM_STEPS
        )
        val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.module if hasattr(model, "module") else model
                    ).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                },
                CHECKPOINT_DIR / "best_model.pt",
            )
            print(f"checkpoint saved (val F1: {val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    # 10. test evaluation using best checkpoint
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    print(f"\nLoading best checkpoint — epoch {ckpt['epoch'] + 1}, val F1: {ckpt['val_f1']:.4f}")

    eval_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    )
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model = eval_model.to(device)

    test(eval_model, test_loader, device, LABEL_NAMES)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_only()
    else:
        main()
