# LLaVA-Style Visual Instruction Tuning in JAX/TPU

This repository contains a lightweight reproduction of the LLaVA-style multimodal instruction tuning pipeline using:
- **CLIP ViT** as the frozen vision encoder
- **Llama 3.2 1B Instruct** as the language model
- **JAX + Flax + Tunix** for TPU-compatible training
- **COCO** as the initial training dataset

The project focuses on reproducing the two-stage visual instruction tuning workflow described in the LLaVA paper while maintaining a TPU-friendly and modular research pipeline.

---

## Overview

The training process is divided into two stages:

### Stage 1 — Vision-Language Alignment
The goal of Stage 1 is to align CLIP visual embeddings with the Llama embedding space.

#### Training Setup
- **CLIP Vision Encoder:** Frozen
- **Llama 3.2 1B:** Frozen
- **Trainable Parameters:** Projection Matrix only

#### Workflow
\[\text{Image} \rightarrow \text{CLIP Vision Encoder} \rightarrow \text{Visual Features} \rightarrow \text{Projection Matrix} \rightarrow \text{Llama Embedding Space} \rightarrow \text{Caption Prediction}\]

The model is trained on image-caption style supervision using COCO-derived prompts such as:
```text
USER: Describe this image briefly.
ASSISTANT: A bicycle replica with a clock as the front wheel.
```
The output of this stage is a trained projection matrix capable of mapping visual features into the language embedding manifold.

### Stage 2 — Visual Instruction Tuning
Stage 2 extends the aligned model into a multimodal instruction-following assistant.

#### Training Setup
- **CLIP Vision Encoder:** Frozen
- **Projection Matrix:** Trainable
- **Llama 3.2 1B:** Trainable

#### Workflow
\[\text{Image} \rightarrow \text{CLIP Vision Encoder} \rightarrow \text{Projection Matrix} \rightarrow \text{Llama 3.2 1B} \rightarrow \text{Instruction-Following Response}\]

Training data consists of image-instruction-response triples derived from COCO annotations. Example:
```text
USER: What objects are visible in the image?
ASSISTANT: The image shows a bicycle replica with a clock as the front wheel.
```

---

## Repository Structure

### `scripts/`
Contains the core preprocessing, feature extraction, and model helper utilities.

- **`prepare_stage1_dataset.py`**
  Parses the raw COCO caption dataset and converts it into a simplified alignment dataset format.
  - *Output format:* `{ "image": "...jpg", "caption": "..." }`
- **`convert_alignment_format.py`**
  Converts alignment captions into instruction-following format for Stage 1 training.
  - *Example output:* `{ "image": "...jpg", "instruction": "Describe this image briefly.", "response": "A bicycle replica with a clock as the front wheel." }`
- **`clip_helpers.py`**
  Provides helper utilities for downloading CLIP checkpoints, loading Flax CLIP Vision models locally, and building TPU-compatible CLIP feature extraction pipelines. Uses `FlaxCLIPVisionModel` and JAX/Flax HuggingFace Transformers for JIT-compiled vision feature extraction.
- **`precompute_clip_features.py`**
  Extracts and saves CLIP visual features to disk for faster training. Features are computed once, stored as `.npy` files, and loaded during training instead of recomputing CLIP every step. Following the LLaVA design choice, the pipeline uses the penultimate CLIP hidden layer.
- **`build_stage1_manifest.py`**
  Builds the final Stage 1 training manifest. Each training sample contains:
  ```json
  { "vision_path": "...npy", "input_ids": [...], "labels": [...] }
  ```
  This manifest is consumed directly by the training loop.

### Root Files
- **`model.py`**
  Contains the TPU-compatible Tunix/Flax implementation of the Llama 3 architecture, attention layers, RoPE embeddings, RMSNorm decoder blocks, embedding layers, and sharding configuration. The model is modified to support `forward_from_embeddings(...)` which enables direct multimodal embedding injection.
- **`params.py`**
  Handles loading HuggingFace safetensor checkpoints, converting weights into Tunix-compatible parameter structures, and constructing JAX-native Llama models. This file bridges HuggingFace checkpoints and Tunix Flax/NNX models.
- **`LLaVA_Public.ipynb`**
  Main research notebook containing preprocessing orchestration, CLIP feature extraction, Stage 1 training, Stage 2 training, debugging utilities, and TPU experiments. This notebook serves as the primary experimental driver for the repository.

---

## Model Architecture

- **Vision Encoder:** `openai/clip-vit-base-patch32`
  - Frozen during all stages.
  - Visual features are extracted from the penultimate hidden layer.
- **Language Model:** `meta-llama/Llama-3.2-1B-Instruct`
  - Loaded into Tunix/Flax format.
  - TPU-compatible JAX execution.
- **Projection Layer:** A trainable linear projection matrix:
  $$H_v = WZ_v$$
  Where:
  - $Z_v$ = CLIP visual features
  - $W$ = Trainable projection matrix
  - $H_v$ = Projected language-space embeddings

---

## Dataset

Initial experiments use a subset of the COCO 2017 dataset. For faster experimentation and TPU debugging, only the first/random 10,000 samples are used initially.

---

## Training Notes


| Parameter Group | Stage 1 | Stage 2 |
| :--- | :--- | :--- |
| **CLIP Vision Encoder** | Frozen | Frozen |
| **Projection Matrix** | Trainable | Trainable |
| **Llama 3.2 1B** | Frozen | Trainable |

---

## TPU Compatibility

The entire pipeline is designed for Google TPU VMs, JAX/Flax execution, and Tunix distributed training. Features include:
- JIT-compiled CLIP extraction
- BF16 support
- Tunix-compatible parameter loading
- Sharding-ready architecture

---

## Future Work

Planned extensions include:
- Larger Stage 1 datasets (CC3M / LAION)
- Multi-turn instruction tuning
- Projector MLP upgrades
- Q-Former integration
- Multimodal RLHF
- Video-language extensions

---

## References

- [LLaVA: Visual Instruction Tuning](https://arxiv.org)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org)
- [Llama 3 Model Architecture](https://meta.com)
- [Tunix Framework](https://github.com)
- [JAX Ecosystem (Flax / Optax)](https://github.com)
