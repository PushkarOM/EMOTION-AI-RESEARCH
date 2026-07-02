# EmotionSense — Research Plan

---

## 1. Research Motivation and Potential Contribution

The main research question behind EmotionSense is:

> **Can explicit multimodal emotional state modeling and persistent affective memory improve the quality, empathy, and personalization of LLM-based conversational agents?**

Modern LLMs are capable of inferring emotions directly from text. However, they have several limitations:

- They rely primarily on linguistic information and often lack access to non-verbal cues such as facial expressions and vocal tone.
- Their emotional understanding is implicit and not explicitly modeled as a persistent state.
- They do not naturally maintain a structured history of a user's emotional evolution across interactions.

EmotionSense addresses these limitations by introducing an external emotional intelligence layer that exists alongside the LLM.

The proposed contribution consists of three major ideas:

### 1.1 Multimodal Emotion Perception

Rather than relying only on text, the system combines:

- What the user **says** (semantic / transcript branch).
- **How** the user says it (acoustic / prosodic branch).
- What the user **expresses visually** (facial expression branch).

This more closely approximates human emotional communication, where meaning, tone, and expression operate in parallel and sometimes in contradiction.

### 1.2 Persistent Affective Memory

Instead of treating every conversation as an independent event, the system maintains an evolving emotional history. This enables the agent to reason about emotional trends, changes, and patterns over time — moving beyond per-turn emotion classification toward longitudinal affective modeling.

### 1.3 Controlled Emotional Conditioning of LLMs

Rather than expecting the LLM to infer emotions implicitly, the system provides explicit emotional context as structured input. This allows responses to be more consistent, more controllable, and potentially more empathetic — without requiring changes to the underlying LLM.

---

## 2. Research Questions

- Does multimodal emotion information produce better conversational responses than text-only systems?
- Does emotional memory improve perceived personalization in long-term conversations?
- Which modality contributes most to accurate emotional understanding?
- What is the optimal strategy for combining emotional signals from different modalities — and does that optimum change across emotion classes and speaking styles?
- When do acoustic and semantic signals *disagree*, and what does that disagreement reveal about the user's emotional state?

---

## 3. Possible Paper Titles

- **"EmotionSense: A Multimodal Affective Memory Framework for Emotionally Adaptive LLM-Based Conversational Agents"**
- **"Beyond Text: Integrating Multimodal Emotion Recognition and Affective Memory for Personalized AI Conversations"**
- **"Persistent Emotional State Modeling for Adaptive Human–AI Interaction"**

---

## 4. Overall Vision

EmotionSense is not trying to replace the reasoning ability of modern LLMs. It aims to provide them with a **human-like emotional perception and memory system**, enabling AI agents to understand not only the meaning of a conversation, but the emotional journey behind it.

---

## 5. Architecture Overview

```
Audio Input
    ├── Acoustic Branch   (Wav2Vec2 → SER)
    └── Semantic Branch   (Whisper → RoBERTa-GoEmotions → canonical mapping)
            ↓
    Internal Audio Fusion  (confidence-weighted / dynamic / intermediate)
            ↓
Face Input → Face Branch  (FER, deferred)
            ↓
    Multimodal Fusion Layer
            ↓
    Emotion Orchestrator
            ↓
    Affective Memory Module
            ↓
    LLM (Llama 3.1 8B Instruct) — with explicit emotional context injected
            ↓
    Response
```

Model versioning convention:

```
Audio Models
  v1 → Wav2Vec2-base, acoustic-only             ✅ done
  v2 → Late fusion, dynamic confidence weights  ← current
  v3 → Late fusion, full evaluation + ablations
  v4 → Intermediate fusion (cross-attention)
  v5 → Proposed improvement / final model

Face Models (deferred)
  v1 → CNN baseline
  v2 → ResNet50
  v3 → EfficientNet
  v4 → ViT
  v5 → Proposed improvement
```

---

## 6. Canonical Emotion Taxonomy

8-class canonical label set (locked):

| ID | Label | Notes |
|----|-------|-------|
| 0 | neutral | Covers RAVDESS "calm" (folded in) |
| 1 | happy | Covers excited (IEMOCAP) |
| 2 | sad | |
| 3 | angry | |
| 4 | fear | |
| 5 | disgust | |
| 6 | surprise | RAVDESS-only in current datasets |
| 7 | frustration | IEMOCAP-only; kept distinct from angry |

`frustration` is kept as a distinct class because it is IEMOCAP's largest label and is acoustically distinguishable from hot anger (lower arousal, more restrained prosody). Merging it into `angry` would both lose this distinction and severely imbalance IEMOCAP's class distribution.

Native-to-canonical label mappings for all four sources (RAVDESS, CREMA-D, IEMOCAP, GoEmotions-28) are versioned in `common/configs/label_mapping.yaml`. Original native labels are also preserved in the manifest for per-taxonomy reporting.

---

## 7. Datasets

| Dataset | Clips | Speakers | Style | Notes |
|---------|-------|----------|-------|-------|
| RAVDESS | 1,440 | 24 | Acted, scripted | 8 emotions, clean studio audio |
| CREMA-D | 7,442 | 91 | Acted | 6 emotions, more speaker diversity |
| IEMOCAP | 7,529* | 10 | Spontaneous + scripted | 5 sessions, manual transcripts |
| **Total** | **16,411** | **125** | | |

*After dropping `oth`/`xxx` (no majority-vote consensus) — 2,510 utterances excluded (~25% of IEMOCAP raw, within expected published range).

**Known class imbalance:**

- `disgust` and `fear` near-absent in IEMOCAP (carried by RAVDESS/CREMA-D).
- `surprise` is RAVDESS-only.
- `frustration` is IEMOCAP-only and is IEMOCAP's largest class.
- Mitigation: class-weighted `CrossEntropyLoss` at training time (`sklearn.compute_class_weight`, `balanced` strategy).

---

## 8. Splits

Speaker-disjoint train/val/test splits (fixed seed = 42):

| Dataset | Train speakers | Val speakers | Test speakers |
|---------|---------------|-------------|--------------|
| RAVDESS | 17 | 4 | 3 |
| CREMA-D | 64 | 14 | 13 |
| IEMOCAP | 6 | 2 | 2 |

| Split | Clips |
|-------|-------|
| Train | 10,647 |
| Val | 3,010 |
| Test | 2,754 |

Speaker-disjoint splitting is non-negotiable for honest SER evaluation — random utterance-level splits allow the model to partially learn speaker identity rather than emotion, inflating reported accuracy.

---

## 9. Results

### 9.1 Acoustic Branch — Wav2Vec2-base (v1)

**Model**: `facebook/wav2vec2-base`, CNN feature encoder frozen, fine-tuned via `Wav2Vec2ForSequenceClassification` (8-way head).

**Training**: Kaggle dual-GPU (`DataParallel`), 10 epochs, batch size 16 + gradient accumulation ×2 (effective batch 32), AdamW (lr=2e-5, weight_decay=0.01), linear warmup/decay (10% warmup), mixed precision fp16, class-weighted CrossEntropyLoss, fixed seed=42.

**Checkpoint**: best val weighted F1, epoch 6 (val F1=0.5921). Train F1 continued climbing to 0.90 by epoch 10 while val plateaued — mild overfitting in later epochs, mitigated by checkpoint-on-val.

| Metric | Val (n=3,010) | Test (n=2,754) |
|--------|--------------|----------------|
| Weighted F1 | 0.592 | 0.635 |
| Macro F1 | 0.595 | 0.643 |
| Accuracy | 0.592 | 0.635 |

**Per-class test results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| neutral | 0.609 | 0.621 | 0.615 | 625 |
| happy | 0.734 | 0.567 | 0.639 | 510 |
| sad | 0.533 | 0.775 | 0.632 | 453 |
| angry | 0.829 | 0.638 | 0.721 | 365 |
| fear | 0.644 | 0.773 | 0.702 | 220 |
| disgust | 0.716 | 0.709 | 0.712 | 206 |
| surprise | 0.840 | 0.512 | 0.636 | 41 |
| frustration | 0.526 | 0.458 | 0.490 | 334 |

**Key findings:**

- `frustration` is consistently the hardest class (F1≈0.49 on both val and test). Confused primarily with `neutral` (28% of true frustration predicted as neutral vs. 3.6% as angry) — likely reflects IEMOCAP's frustration utterances skewing toward restrained, conversational prosody, acoustically closer to flat speech than to hot anger.
- `neutral→sad` is the largest single off-diagonal confusion (138/625 neutral clips predicted as sad) — both are low-arousal, low-energy classes, and the acted sad speech in RAVDESS/CREMA-D may be more exaggerated than real neutral IEMOCAP utterances, creating a systematic bias toward sad for flat prosody.
- Test slightly outperforms val (0.635 vs 0.592 weighted F1) — attributed to speaker-level variance at small-N speaker-disjoint splits rather than leakage; test speakers happen to have more prototypical emotional expressions than val speakers.
- SOTA comparison is **not directly applicable**: published SER benchmarks use single-corpus protocols (IEMOCAP 5/10-fold, RAVDESS LOSO) on different label sets. Literature suggests specialized architectures buy ~8% average improvement over standard fine-tuning baselines, making this v1 a credible starting point.

**HuggingFace**: `PushkarOM/wav2vec2-ser-v1`

---

### 9.2 Semantic Branch (v1)

**Pipeline**: Whisper-small (transcription) → `PushkarOM/roberta-head-goemotion` (GoEmotions-28 classification) → canonical 8-class mapping via `label_mapping.yaml`.

**Aggregation**: when multiple GoEmotions labels map to the same canonical class, the maximum score across those labels is used (e.g., `joy`, `amusement`, `excitement`, `love` → `happy`; the highest individual score is taken as the canonical confidence).

**Null transcript rate on test set**: 18/2,754 (0.65%) — very short or silent clips where Whisper returned empty string; these fall back to acoustic-only in fusion.

**Known limitation**: `frustration` has no direct GoEmotions equivalent. It maps through `annoyance`/`anger` → canonical `angry`, causing systematic mislabeling of IEMOCAP frustration utterances as `angry` at the semantic branch level.

---

### 9.3 Late Fusion Evaluation (v1)

**Method**: confidence-weighted combination of acoustic and semantic top-1 predictions. Final emotion is the branch whose `confidence × weight` is higher; final confidence is the winning branch's raw confidence (not the weighted score).

**Weight sweep results (test set, n=2,754):**

| Acoustic weight | Semantic weight | Weighted F1 | Macro F1 |
|----------------|----------------|-------------|----------|
| 0.5 | 0.5 | 0.4971 | 0.4961 |
| 0.6 | 0.4 | 0.6193 | 0.6271 |
| 0.7 | 0.3 | 0.6341 | 0.6410 |
| 0.8 | 0.2 | **0.6357** | **0.6435** |
| 0.9 | 0.1 | 0.6357 | 0.6435 |
| 1.0 | 0.0 | 0.6357 | 0.6435 |

**Finding: late fusion with fixed confidence weighting does not improve over acoustic-only on this test set.**

Root causes identified:

1. **Scripted-speech corpora (RAVDESS/CREMA-D) use emotionally neutral text** spoken with emotional prosody. The semantic branch consistently predicts `neutral` regardless of true emotion (e.g., "Kids are talking by the door" spoken angrily → semantic: `neutral` at 96% confidence). This systematically pulls fusion predictions toward `neutral` for the majority of the test set.

2. **GoEmotions taxonomy gap for `frustration`**: the semantic branch has no direct mapping for frustration, systematically mislabeling IEMOCAP frustration utterances as `angry`. This causes a measurable degradation in the `frustration` class F1 under fusion (0.490 acoustic-only → 0.457 at 0.6/0.4 weights).

3. **Domain mismatch**: `PushkarOM/roberta-head-goemotion` was trained on written internet text (Reddit/social media). ASR-transcribed spontaneous speech has disfluencies, incomplete sentences, and lacks punctuation — a different distribution the model was not trained on.

**Paper-worthy observation**: *"Late fusion with confidence weighting degrades performance on scripted-speech corpora where lexical content is emotionally neutral, and on the frustration class where the semantic model's taxonomy lacks a direct equivalent. These findings motivate intermediate fusion architectures that can learn cross-modal alignment rather than naive confidence weighting, and highlight the importance of taxonomy alignment between modalities."*

**Fusion is most useful when modalities carry complementary rather than redundant information** — e.g., a person speaking in a calm voice while using aggressive language (suppressed/masked anger). This case is underrepresented in RAVDESS/CREMA-D (scripted speech where prosody IS the emotion signal) but present in real conversational use cases (EmotionSense's target domain). The evaluation corpus does not adequately test the cases fusion is designed for.

---

## 10. Next Steps (Audio / SER)

### Immediate — v2 (Dynamic Weighting)

**Motivation**: fixed weights assume both branches are equally reliable across all clips. In reality, reliability varies per clip — when the acoustic model is uncertain (low max softmax confidence), the semantic branch should receive more weight, and vice versa.

**Proposed dynamic weight formula:**

```python
acoustic_uncertainty = 1 - acoustic_confidence
semantic_uncertainty = 1 - semantic_confidence

acoustic_w = semantic_uncertainty / (acoustic_uncertainty + semantic_uncertainty)
semantic_w = acoustic_uncertainty / (acoustic_uncertainty + semantic_uncertainty)
```

**Expected benefit**: on clips where the acoustic model is genuinely uncertain (ambiguous prosody, background noise), semantic gets more weight → better handling of masked/suppressed emotion. On scripted-speech clips where acoustic is highly confident, acoustic dominates naturally without hard-coded weights.

**Evaluation plan**: same test set + targeted evaluation on modality-disagreement clips (clips where acoustic and semantic predict different canonical classes) — this subset is where fusion either helps or hurts, not on easy agreement cases. The disagreement-clip breakdown is a planned paper figure.

### Short-term — v3 (Ablations + Analysis)

- LOSO (Leave-One-Speaker-Out) evaluation on RAVDESS — produces comparable numbers to published RAVDESS-only baselines (which almost universally use LOSO).
- Embedding-space domain-confound analysis: t-SNE over pooled Wav2Vec2 hidden states, colored by `dataset × canonical_label` — tests whether the model learned emotion-discriminative features or partially learned corpus-style features (e.g., "IEMOCAP-style speech" vs "RAVDESS-style speech").
- Per-corpus breakdown of val/test performance (RAVDESS-only, CREMA-D-only, IEMOCAP-only within the test split) — root-causes the val/test gap and identifies which corpus drives which class errors.

### Medium-term — v4 (Intermediate Fusion)

**Motivation**: late fusion can only combine top-1 predictions. Intermediate fusion learns the interaction between modalities at the representation level — the suppressed-anger case (calm voice, aggressive words) becomes learnable rather than hand-engineered.

**Proposed architecture:**

```
Wav2Vec2 hidden states   (acoustic features, shape: [T_a, 768])
         +
RoBERTa hidden states    (transcript features, shape: [T_t, 768])
         ↓
Cross-attention layer    (acoustic attends to semantic and vice versa)
         ↓
Mean pool both streams
         ↓
Concatenate → [1536]
         ↓
Joint classification head → 8 classes
```

**Paper contribution**: compare late fusion vs intermediate fusion on the same canonical-label test sets, with analysis of *which emotion classes and which corpora* benefit most from cross-modal interaction. The finding that intermediate fusion helps more on IEMOCAP (spontaneous speech, lexical content carries real emotion) than RAVDESS (scripted speech, lexical content is neutral) would be a clean, interpretable contribution.

### Longer-term — v5 (Proposed Improvement)

Options under consideration:

- Domain-adapted semantic model: fine-tune RoBERTa directly on IEMOCAP transcripts + emotion labels, producing a model calibrated for spoken/transcribed language rather than written text.
- Extend GoEmotions mapping to include `frustration` as a distinct class via targeted data augmentation or additional fine-tuning.
- Multimodal contrastive pretraining: learn aligned acoustic/semantic representations before fine-tuning for classification.

---

## 11. Face Branch (Deferred)

Face emotion recognition (FER) branch is planned but deliberately deferred until the audio pipeline (acoustic + semantic + fusion) reaches a stable v3 state. The face branch will follow the same versioned model structure:

```
face/baselines/simple_cnn/
face/baselines/resnet/
face/sota/efficientnet/
face/sota/vision_transformer/
```

The 3-modality fusion (audio + face + text) design — including how late vs intermediate fusion generalizes from 2 to 3 modalities — will be informed by the findings from the audio-only fusion experiments above.

---

## 12. Integration with EmotionSense V2

The `emotion-ai-research` repo produces models; `EmotionSense-V2` consumes them via HuggingFace Hub API calls. The interface is the standardized output schema:

```json
{
  "source": "audio_multimodal",
  "emotion": "angry",
  "confidence": 0.94,
  "components": {
    "acoustic": {
      "model": "PushkarOM/wav2vec2-ser-v1",
      "emotion": "angry",
      "confidence": 0.94
    },
    "semantic": {
      "model": "PushkarOM/roberta-head-goemotion",
      "emotion": "neutral",
      "confidence": 0.96
    }
  }
}
```

This decoupling means model upgrades (v1 → v2 → v5) require only a HuggingFace endpoint URL change in EmotionSense — no frontend, fusion layer, memory module, or LLM changes needed.

**Current EmotionSense integration status**: acoustic branch (`wav2vec2-ser-v1`) ready for integration into `backend/models/audio_emotion/`. Semantic branch and fusion layer pending dynamic-weight evaluation before integration.

---

## 13. Repository Structure

```
emotion-ai-research/
├── audio/
│   ├── datasets/          # raw data (gitignored, download instructions in each subfolder)
│   │   ├── ravdess/
│   │   ├── iemocap/
│   │   ├── crema_d/
│   │   └── msp_podcast/   # future
│   ├── baselines/         # MFCC+SVM, CNN-spectrogram, LSTM (planned)
│   ├── sota/
│   │   ├── wav2vec2/      # acoustic branch ✅
│   │   └── wav2vec2_text_fusion/  # semantic branch + fusion ✅
│   ├── evaluation/
│   └── papers/
├── face/                  # deferred
├── common/
│   ├── configs/
│   │   └── label_mapping.yaml     # canonical taxonomy + all dataset mappings
│   ├── manifests/
│   │   ├── manifest.csv           # unified 16,411-row manifest
│   │   ├── train.csv              # 10,647 clips
│   │   ├── val.csv                # 3,010 clips
│   │   ├── test.csv               # 2,754 clips
│   │   └── test_transcripts.csv   # precomputed Whisper transcripts for test set
│   └── scripts/
│       ├── build_manifest.py
│       ├── make_splits.py
│       └── fix_roberta_config.py
└── docs/
```

---

## 14. Git Tags (Milestones)

| Tag | Description |
|-----|-------------|
| `v1-acoustic-baseline` | Wav2Vec2-base fine-tuned, test weighted F1=0.635 |
| `v2-semantic-fusion` | Whisper + RoBERTa semantic branch + late fusion pipeline |
| `v3-fusion-evaluation` | Late fusion evaluation — finding: no improvement over acoustic-only |
| `v4-dynamic-weights` | Dynamic confidence weighting (in progress) |