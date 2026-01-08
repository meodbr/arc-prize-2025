# ARC Prize 2025 — Neighborhood-Causal Grid Models

This repository contains my ongoing experiments for the **ARC Prize 2025**, a data science competition inspired by François Chollet’s work on measuring intelligence and abstract reasoning.

- ARC Prize website: https://arcprize.org/
- Reference paper: *On the Measure of Intelligence* — François Chollet  
  https://arxiv.org/abs/1911.01547

This is a **personal research project**, focused on architectural ideas rather than leaderboard optimization.

---

## Motivation

Most strong ARC approaches today rely on **large language models fine-tuned with Test-Time Training (TTT)**. These models treat ARC grids as sequences of tokens and generate outputs in a **fixed writing order** (left → right, top → bottom).

From my perspective, this is a major mismatch with the structure of ARC tasks:

- ARC grids are **2D objects**, not 1D sequences
- Reasoning often propagates **spatially**, not linearly
- Writing order introduces arbitrary causal constraints

### Core hypothesis

> A significant source of error in LLM-based ARC solvers comes from forcing a **1D autoregressive order** onto a **2D problem**.

This project explores **2D-aware causal generation** as an alternative.

---

## Model idea (high-level)

Instead of predicting pixels strictly in raster order, I experiment with a **neighborhood-causal autoregressive model**:

- The grid is treated as a graph
- A cell can be generated once **any of its neighbors** (top, bottom, left, right) has been generated
- At each step, the model predicts **any valid frontier cell** -> We must decide which
- The dependency graph is a **dynamic oriented graph**, induced by generation order

This preserves:
- Causality (no future leakage)
- Autoregressive generation
- Flexibility in spatial reasoning

While this is not a full departure from language-model style decoding, it relaxes the strongest constraint: **a single fixed writing direction**.

---

## Implementation directions

I currently explore three main variants:

1. **Base model** : Reproducing the 2024 winning arc solution ([ARChitects](https://github.com/da-fr/arc-prize-2024))
   - Standard token embeddings
   - Completion masking
   - Batched inference

2. **2D Positional Encoding (2DPE)**
   - Tokenizer wrapper with 2D positional information
   - Tweaked RoPe positionnal encoding

3. **Convolutional Embeddings (ConvEmbedding)**
   - This gives the spatial context needed for heighboorhood-Causal generation
   - Works by changing the transformer's embedding level (Neighboors embedding instead of previous/left token embedding)
   - Designed to better match grid locality

---

## Current TODO / Roadmap

### New implementation

- [x] Completion mask (base)
- [x] LoRA merge
- [x] Base batch inference
- [x] TTT backbone
- [x] Tokenizer wrapper for 2DPE
- [ ] 2DPE batch inference
- [x] Inference from HF datasets
- [x] Test-after-train pipeline

- [x] ConvEmbedding embeddings
- [x] ConvEmbedding embedding layer tweaks
- [x] ConvEmbedding tokenizer wrapper
- [x] ConvEmbedding inference

---

### Training

- [x] Smol model (reduced embeddings)
- [x] Big base model
- [x] Base TTT (Kaggle)
- [x] Larger context experiments
- [x] Hyperparameter tuning
- [x] Debug 2DPE
- [ ] Train smol 2DPE
- [ ] Train big 2DPE

- [x] Q-8 evaluation
- [x] Q-4 evaluation
- [ ] Train smol ConvEmbedding
- [x] Train big ConvEmbedding
- [ ] ConvEmbedding TTT (Kaggle)

- [ ] xFormers flash-attention training
- [ ] xFormers flash-attention TTT (Kaggle)

---

## Status & disclaimer

This project is:
- Experimental
- Iterative
- Very much a work in progress

Some ideas may fail, some implementations may be scrapped, and results are not guaranteed to be competitive. The goal is to **learn what kinds of causal structures actually help on ARC**, not to claim a finished solution.

If you’re interested in:
- 2D-aware autoregressive models
- Alternatives to raster-scan generation
- ARC-specific architectural biases

…then this repo might be useful.
