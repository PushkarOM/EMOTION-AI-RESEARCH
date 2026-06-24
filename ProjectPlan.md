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
  - Mitigation planned: class-weighted loss at training time (not split-level rebalancing, to preserve speaker-disjointness)

### Next
- Wav2Vec2 dataloader + feature extraction pipeline (acoustic branch)
- LOSO (Leave-One-Speaker-Out) evaluation on RAVDESS planned as a separate ablation, alongside the primary fixed-split methodology, to allow direct comparison with published RAVDESS-only baselines
  