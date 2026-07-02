import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from analyze import run_acoustic, load_acoustic_model
from semantic import classify_transcript
from fusion import fuse

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = PROJECT_ROOT / "common" / "manifests"

LABEL_TO_IDX = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
    "fear": 4, "disgust": 5, "surprise": 6, "frustration": 7,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
LABEL_NAMES  = [IDX_TO_LABEL[i] for i in range(8)]


def run_evaluation(test_df, device, acoustic_weight=0.6, semantic_weight=0.4):
    """
    Run acoustic-only and fusion evaluation over the test set.

    Performs acoustic inference and semantic classification for every clip,
    caches both sets of results, then computes fusion predictions at the
    specified weights. Returns metrics alongside the raw caches so that
    subsequent weight sweeps can reuse the inference results without
    redundant forward passes.

    Args:
        test_df (pd.DataFrame): Test manifest with a 'transcript' column.
        device (str): 'cuda' or 'cpu'.
        acoustic_weight (float): Weight for the acoustic branch in fusion.
        semantic_weight (float): Weight for the semantic branch in fusion.

    Returns:
        dict: {
            "acoustic": {"weighted_f1", "macro_f1", "y_true", "y_pred"},
            "fusion":   {"weighted_f1", "macro_f1", "y_true", "y_pred"},
            "acoustic_cache": list[dict],
            "semantic_cache": list[dict | None],
        }
    """
    hf_device = 0 if device == "cuda" else -1
    load_acoustic_model(device)

    y_true          = []
    y_pred_acoustic = []
    y_pred_fusion   = []
    acoustic_cache  = []
    semantic_cache  = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="evaluating"):
        true_label = row["canonical_label"]
        y_true.append(LABEL_TO_IDX[true_label])

        # ── acoustic branch ──────────────────────────────────────
        acoustic_result = run_acoustic(row["filepath"], device)
        acoustic_cache.append(acoustic_result)
        y_pred_acoustic.append(LABEL_TO_IDX[acoustic_result["emotion"]])

        # ── semantic branch (precomputed transcript) ─────────────
        transcript = row["transcript"] if pd.notna(row["transcript"]) else None
        semantic_result = classify_transcript(transcript, device=hf_device)
        if semantic_result is not None:
            semantic_result["model"] = "PushkarOM/roberta-head-goemotion"
        semantic_cache.append(semantic_result)

        # ── fusion ───────────────────────────────────────────────
        fused = fuse(
            acoustic_result, semantic_result,
            acoustic_weight=acoustic_weight,
            semantic_weight=semantic_weight,
        )
        y_pred_fusion.append(LABEL_TO_IDX[fused["emotion"]])

    return {
        "acoustic": {
            "weighted_f1": f1_score(y_true, y_pred_acoustic, average="weighted"),
            "macro_f1":    f1_score(y_true, y_pred_acoustic, average="macro"),
            "y_true":      y_true,
            "y_pred":      y_pred_acoustic,
        },
        "fusion": {
            "weighted_f1": f1_score(y_true, y_pred_fusion, average="weighted"),
            "macro_f1":    f1_score(y_true, y_pred_fusion, average="macro"),
            "y_true":      y_true,
            "y_pred":      y_pred_fusion,
        },
        "acoustic_cache": acoustic_cache,
        "semantic_cache": semantic_cache,
    }


def run_evaluation_from_cache(test_df, acoustic_cache, semantic_cache,
                               acoustic_weight=0.6, semantic_weight=0.4):
    """
    Recompute fusion predictions from cached branch results.

    Avoids redundant acoustic / semantic inference when sweeping over
    different fusion weights. Only the fusion decision is recomputed.

    Args:
        test_df (pd.DataFrame): Test manifest (used for ground-truth labels).
        acoustic_cache (list[dict]): Cached acoustic branch outputs.
        semantic_cache (list[dict | None]): Cached semantic branch outputs.
        acoustic_weight (float): Weight for the acoustic branch.
        semantic_weight (float): Weight for the semantic branch.

    Returns:
        dict: {
            "acoustic": {"weighted_f1", "macro_f1", "y_true", "y_pred"},
            "fusion":   {"weighted_f1", "macro_f1", "y_true", "y_pred"},
        }
    """
    y_true          = []
    y_pred_acoustic = []
    y_pred_fusion   = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        true_label = row["canonical_label"]
        y_true.append(LABEL_TO_IDX[true_label])

        acoustic_result = acoustic_cache[i]
        semantic_result = semantic_cache[i]

        y_pred_acoustic.append(LABEL_TO_IDX[acoustic_result["emotion"]])

        fused = fuse(
            acoustic_result, semantic_result,
            acoustic_weight=acoustic_weight,
            semantic_weight=semantic_weight,
        )
        y_pred_fusion.append(LABEL_TO_IDX[fused["emotion"]])

    return {
        "acoustic": {
            "weighted_f1": f1_score(y_true, y_pred_acoustic, average="weighted"),
            "macro_f1":    f1_score(y_true, y_pred_acoustic, average="macro"),
            "y_true":      y_true,
            "y_pred":      y_pred_acoustic,
        },
        "fusion": {
            "weighted_f1": f1_score(y_true, y_pred_fusion, average="weighted"),
            "macro_f1":    f1_score(y_true, y_pred_fusion, average="macro"),
            "y_true":      y_true,
            "y_pred":      y_pred_fusion,
        },
    }


def print_comparison(results, label="default weights (0.6 / 0.4)"):
    """
    Print a side-by-side acoustic vs fusion summary table and per-class reports.

    Args:
        results (dict): Output from run_evaluation or run_evaluation_from_cache.
        label (str): Descriptive label printed in the header.
    """
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"{'':20} {'Acoustic':>15} {'Fusion':>15}")
    print(f"{'-' * 60}")
    print(f"{'Weighted F1':20} "
          f"{results['acoustic']['weighted_f1']:>15.4f} "
          f"{results['fusion']['weighted_f1']:>15.4f}")
    print(f"{'Macro F1':20} "
          f"{results['acoustic']['macro_f1']:>15.4f} "
          f"{results['fusion']['macro_f1']:>15.4f}")
    print(f"{'=' * 60}")

    print("\n--- Acoustic per-class ---")
    print(classification_report(
        results["acoustic"]["y_true"],
        results["acoustic"]["y_pred"],
        target_names=LABEL_NAMES, digits=3,
    ))

    print("\n--- Fusion per-class ---")
    print(classification_report(
        results["fusion"]["y_true"],
        results["fusion"]["y_pred"],
        target_names=LABEL_NAMES, digits=3,
    ))


def weight_sweep(test_df, acoustic_cache, semantic_cache):
    """
    Sweep over acoustic/semantic weight splits using cached inference results.

    Tests splits from 0.5/0.5 to 1.0/0.0 in 0.1 increments. Since branch
    inference is reused from cache, each split is nearly instant to evaluate.

    Args:
        test_df (pd.DataFrame): Test manifest for ground-truth labels.
        acoustic_cache (list[dict]): Cached acoustic branch outputs.
        semantic_cache (list[dict | None]): Cached semantic branch outputs.

    Returns:
        tuple[float, float]: Best (acoustic_weight, semantic_weight) by weighted F1.
    """
    print("\n--- Weight sweep ---")
    print(f"{'acoustic_w':>12} {'semantic_w':>12} {'weighted_f1':>14} {'macro_f1':>12}")
    print("-" * 54)

    best_f1, best_weights = 0.0, (0.6, 0.4)

    for aw in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sw = round(1.0 - aw, 1)
        results = run_evaluation_from_cache(
            test_df, acoustic_cache, semantic_cache,
            acoustic_weight=aw, semantic_weight=sw,
        )
        wf1 = results["fusion"]["weighted_f1"]
        mf1 = results["fusion"]["macro_f1"]
        marker = " ← best" if wf1 > best_f1 else ""
        print(f"{aw:>12.1f} {sw:>12.1f} {wf1:>14.4f} {mf1:>12.4f}{marker}")
        if wf1 > best_f1:
            best_f1, best_weights = wf1, (aw, sw)

    print(f"\nbest: acoustic={best_weights[0]}, semantic={best_weights[1]}, "
          f"weighted_f1={best_f1:.4f}")
    return best_weights


def save_results(results, best_weights, best_results, out_dir):
    """
    Persist evaluation metrics and per-class reports to disk.

    Args:
        results (dict): Default-weight evaluation results.
        best_weights (tuple[float, float]): Best (acoustic_w, semantic_w).
        best_results (dict | None): Evaluation at best weights (None if same as default).
        out_dir (Path): Directory to write output files.
    """
    # JSON metrics 
    to_save = {
        "default_weights": {
            "acoustic_weight": 0.6,
            "semantic_weight": 0.4,
            "acoustic": {
                "weighted_f1": results["acoustic"]["weighted_f1"],
                "macro_f1":    results["acoustic"]["macro_f1"],
            },
            "fusion": {
                "weighted_f1": results["fusion"]["weighted_f1"],
                "macro_f1":    results["fusion"]["macro_f1"],
            },
        }
    }

    if best_results is not None:
        to_save["best_weights"] = {
            "acoustic_weight": best_weights[0],
            "semantic_weight": best_weights[1],
            "fusion": {
                "weighted_f1": best_results["fusion"]["weighted_f1"],
                "macro_f1":    best_results["fusion"]["macro_f1"],
            },
        }

    json_path = out_dir / "fusion_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"\nmetrics  saved → {json_path}")

    # text report 
    report_path = out_dir / "fusion_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=== ACOUSTIC ONLY ===\n")
        f.write(classification_report(
            results["acoustic"]["y_true"],
            results["acoustic"]["y_pred"],
            target_names=LABEL_NAMES, digits=3,
        ))
        f.write("\n=== FUSION (default 0.6 / 0.4) ===\n")
        f.write(classification_report(
            results["fusion"]["y_true"],
            results["fusion"]["y_pred"],
            target_names=LABEL_NAMES, digits=3,
        ))
        if best_results is not None:
            f.write(f"\n=== FUSION (best {best_weights[0]} / {best_weights[1]}) ===\n")
            f.write(classification_report(
                best_results["fusion"]["y_true"],
                best_results["fusion"]["y_pred"],
                target_names=LABEL_NAMES, digits=3,
            ))
    print(f"report   saved → {report_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    transcripts_path = MANIFESTS_DIR / "test_transcripts.csv"
    if not transcripts_path.exists():
        raise FileNotFoundError(
            "test_transcripts.csv not found — run precompute_transcripts.py first"
        )

    test_df = pd.read_csv(transcripts_path)
    print(f"test set: {len(test_df)} clips")
    print(f"null transcripts: {test_df['transcript'].isna().sum()}")

    # 1. full inference pass at default weights — caches acoustic + semantic
    print("\nrunning evaluation at default weights (0.6 / 0.4) ...")
    results = run_evaluation(test_df, device)
    print_comparison(results, label="default weights (0.6 / 0.4)")

    acoustic_cache = results["acoustic_cache"]
    semantic_cache = results["semantic_cache"]

    # 2. weight sweep — instant, uses cache
    best_weights = weight_sweep(test_df, acoustic_cache, semantic_cache)

    # 3. full report at best weights (skip if same as default)
    best_results = None
    if best_weights != (0.6, 0.4):
        print(f"\nfull report at best weights {best_weights} ...")
        best_results = run_evaluation_from_cache(
            test_df, acoustic_cache, semantic_cache,
            acoustic_weight=best_weights[0],
            semantic_weight=best_weights[1],
        )
        print_comparison(
            best_results,
            label=f"best weights ({best_weights[0]} / {best_weights[1]})",
        )

    # 4. save to disk
    save_results(results, best_weights, best_results, Path(__file__).parent)


if __name__ == "__main__":
    main()
