### Research Plan Section
---
# 12. Research Motivation and Potential Contribution

The main research question behind EmotionSense is:

> **Can explicit multimodal emotional state modeling and persistent affective memory improve the quality, empathy, and personalization of LLM-based conversational agents?**

Modern LLMs are capable of inferring emotions directly from text. However, they have several limitations:

* They rely primarily on linguistic information and often lack access to non-verbal cues such as facial expressions and vocal tone.
* Their emotional understanding is implicit and not explicitly modeled as a persistent state.
* They do not naturally maintain a structured history of a user's emotional evolution across interactions.

EmotionSense addresses these limitations by introducing an external emotional intelligence layer that exists alongside the LLM.

The proposed contribution consists of three major ideas:

### 1. Multimodal Emotion Perception

Rather than relying only on text, the system combines:

* What the user says.
* How the user says it.
* What the user expresses visually.

This more closely approximates human emotional communication.

---

### 2. Persistent Affective Memory

Instead of treating every conversation as an independent event, the system maintains an evolving emotional history.

This enables the agent to reason about emotional trends, changes, and patterns over time.

---

### 3. Controlled Emotional Conditioning of LLMs

Rather than expecting the LLM to infer emotions implicitly, the system provides explicit emotional context.

This allows responses to be:

* More consistent.
* More controllable.
* Potentially more empathetic.

---

## Potential Research Questions

Some possible directions:

* Does multimodal emotion information produce better conversational responses than text-only systems?
* Does emotional memory improve perceived personalization in long-term conversations?
* Which modality contributes most to accurate emotional understanding?
* What is the optimal strategy for combining emotional signals from different modalities?

---

## Possible Paper Title

**"EmotionSense: A Multimodal Affective Memory Framework for Emotionally Adaptive LLM-Based Conversational Agents"**

Alternative titles:

* **"Beyond Text: Integrating Multimodal Emotion Recognition and Affective Memory for Personalized AI Conversations"**
* **"Persistent Emotional State Modeling for Adaptive Human–AI Interaction"**

---

## Overall Vision

EmotionSense is not trying to replace the reasoning ability of modern LLMs.

It aims to provide them with a **human-like emotional perception and memory system**, enabling AI agents to understand not only the meaning of a conversation, but the emotional journey behind it.

---


## Status

### (Audio / SER)

Here's the Audio/SER status section updated with the acoustic branch work and both eval rounds:

---

### (Audio / SER)

#### Completed

**Taxonomy**
- Canonical 8-class emotion label set locked: `neutral, happy, sad, angry, fear, disgust, surprise, frustration`
- `frustration` kept distinct from `angry` (IEMOCAP's largest class, acoustically/semantically distinct)
- Full native-to-canonical label mapping for RAVDESS, CREMA-D, IEMOCAP, and GoEmotions-28 documented in `common/configs/label_mapping.yaml`
- Intensity mapping also standardized (`low/medium/high`) across RAVDESS and CREMA-D; IEMOCAP has no native intensity annotation

**Datasets acquired & processed**
- RAVDESS (speech-only, 1,440 clips, 24 actors)
- CREMA-D (7,442 clips, 91 actors)
- IEMOCAP (7,529 labeled utterances after dropping `oth`/`xxx` consensus-disagreement labels, 5 sessions / 10 speakers)
- Unified manifest (`common/manifests/manifest.csv`, 16,411 rows) built via `common/scripts/build_manifest.py` — joins all three datasets into one schema: `filepath, dataset, speaker_id, actor_gender, native_label, canonical_label, native_intensity, canonical_intensity, duration_sec`

**Splits**
- Speaker-disjoint train/val/test splits built via `common/scripts/make_splits.py` (fixed seed = 42), to prevent speaker leakage across splits
- Split ratios (by speaker count, not clip count):
  - RAVDESS: 17 / 4 / 3 (train/val/test)
  - CREMA-D: 64 / 14 / 13
  - IEMOCAP: 6 / 2 / 2 (10 total speakers: 5 sessions × 2)
- Final split sizes: train = 10,647, val = 3,010, test = 2,754
- Known class imbalance documented:
  - `disgust` and `fear` near-absent in IEMOCAP (carried by RAVDESS/CREMA-D in the combined set)
  - `surprise` is RAVDESS-only (not present in CREMA-D or IEMOCAP)
  - `frustration` is IEMOCAP-only, and is IEMOCAP's largest class
  - Mitigation: class-weighted `CrossEntropyLoss` at training time (`sklearn.compute_class_weight`, `balanced`), not split-level rebalancing, to preserve speaker-disjointness

**Acoustic branch v1 (Wav2Vec2)** — *new*
- `facebook/wav2vec2-base`, feature encoder frozen, fine-tuned end-to-end via `Wav2Vec2ForSequenceClassification` (8-way classification head)
- Trained on Kaggle (dual-GPU via `DataParallel`), 10 epochs, batch size 16 with gradient accumulation 2 (effective batch 32), AdamW (lr 2e-5, weight decay 0.01), linear warmup/decay schedule (10% warmup), mixed precision (fp16 autocast + grad scaler), fixed seed = 42
- Checkpoint selection: best val weighted F1, hit at epoch 6 (val F1 0.5921); train F1 continued climbing to 0.90 by epoch 10 while val plateaued/declined slightly — flags some overfitting in later epochs, mitigated by checkpointing on val rather than final epoch
- **Val set** (n=3,010): weighted F1 0.592, macro F1 0.595, accuracy 0.592
- **Test set** (n=2,754): weighted F1 0.635, macro F1 0.643, accuracy 0.635 — outperforms val, consistent with val being the checkpoint-selection set (mildly optimistic by construction) rather than evidence of unusually strong generalization
- Per-class pattern stable across both splits: `frustration` is consistently the hardest class (F1 ≈ 0.49 on both val and test), confused primarily with `neutral` (not `angry` as might be expected) — 28% of true frustration test samples predicted as neutral vs. 3.6% as angry. Likely reflects IEMOCAP's frustration annotations skewing toward restrained/conversational rather than dramatic, acoustically closer to flat prosody than to anger
- Other classes shifted differently between val→test (angry: precision up, recall down; sad: recall up, precision down) — read as different error profiles rather than a clean improvement, and not yet root-caused; pending the speaker-disjointness/per-corpus sanity checks noted below
- SOTA comparison attempted and found **not directly comparable**: published SER benchmarks use single-corpus protocols (e.g., IEMOCAP 5/10-fold, RAVDESS LOSO) on different label sets, while this result is an 8-class taxonomy across 3 merged corpora on a single fixed speaker-disjoint split. Literature suggests specialized architectures (multitask, cross-corpus regularization) only buy ~8% average improvement over standard fine-tuning, so this baseline is considered a credible v1, not an obviously suboptimal one
- Domain-confound check on frustration misclassifications run and found **inconclusive/uninformative** by construction (frustration only exists in IEMOCAP, so its confusions are trivially IEMOCAP-only); a real test would need embedding-space analysis (t-SNE over pooled Wav2Vec2 representations, colored by dataset+label) comparing IEMOCAP neutral/happy to RAVDESS/CREMA-D neutral/happy — deferred to the paper's analysis section, not done yet
- **Decision: locked as v1.** Checkpoint saved (`checkpoints/best_model.pt`); further acoustic-only tuning deferred until the full multimodal pipeline is working end-to-end

#### Next
- **Sanity checks on the val/test gap** (cheap, should precede further analysis): confirm true speaker-disjointness across splits (no accidental leakage), and check whether per-corpus class composition (RAVDESS/CREMA-D/IEMOCAP mix) differs between val and test for the classes that moved most (`sad`, `angry`)
- **Semantic branch** (in progress): Whisper transcription pipeline → existing `PushkarOM/roberta-head-goemotion` model → map GoEmotions-28 output to the canonical 8-class taxonomy via the existing label mapping
- **Internal audio fusion**: confidence-weighted combination of acoustic + semantic branch outputs (`/emotion/audio/analyze` endpoint, schema with `source/emotion/confidence/components` already drafted)
- Embedding-space domain-confound analysis (t-SNE, deferred from above) — planned for the paper's analysis section, not blocking v1
- LOSO (Leave-One-Speaker-Out) evaluation on RAVDESS planned as a separate ablation, alongside the primary fixed-split methodology, to allow direct comparison with published RAVDESS-only baselines
- Face branch (deferred, separate from audio work)
  