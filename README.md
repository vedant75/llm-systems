# LLM Systems — Transformer Training Stack from Scratch

A from-scratch implementation of a modern decoder-only language model stack in PyTorch, covering the full path from **byte-level BPE tokenization to Transformer training and autoregressive text generation**.

The goal of this project is not to wrap high-level PyTorch Transformer APIs, but to implement and understand the systems underneath an LLM: attention, normalization, positional encoding, tokenization, optimization, numerical stability, training infrastructure, checkpointing, and decoding.

The project also includes a performance-engineering pass on BPE training that reduced tokenizer training time by up to **4.37×** through profiling, multiprocessing, and incremental pair-index updates.

---

## What This Project Implements

### Transformer Architecture

Implemented core neural-network components directly in PyTorch:

- Bias-free linear layers
- Token embeddings
- RMSNorm
- Stable softmax
- Scaled dot-product attention
- Causal masking
- Rotary positional embeddings (RoPE)
- Multi-head self-attention
- SwiGLU feed-forward networks
- Pre-normalization Transformer blocks
- Final RMSNorm
- Vocabulary projection / language-model head

The model follows the standard decoder-only flow:

```text
Token IDs [B, T]
      │
      ▼
Embedding
      │
      ▼
┌─────────────────────────┐
│ Transformer Block × N   │
│                         │
│ RMSNorm                 │
│   ↓                     │
│ Multi-Head Attention    │
│   ↓                     │
│ Residual                │
│   ↓                     │
│ RMSNorm                 │
│   ↓                     │
│ SwiGLU FFN              │
│   ↓                     │
│ Residual                │
└─────────────────────────┘
      │
      ▼
Final RMSNorm
      │
      ▼
Vocabulary Projection
      │
      ▼
Logits [B, T, V]
```

---

## Byte-Level BPE Tokenizer

Implemented a byte-level BPE tokenizer from scratch, including:

- UTF-8 byte vocabulary
- Regex-based pre-tokenization
- Frequency-weighted pair counting
- Deterministic merge selection
- Lexicographic tie-breaking
- Special-token boundaries
- Left-to-right non-overlapping merges
- Runtime encoding and decoding
- Streaming `encode_iterable`
- Merge-rank-based inference

The implementation preserves byte-level correctness and avoids merging across pre-token or special-token boundaries.

---

## BPE Performance Engineering

The initial BPE implementation was intentionally kept as a simple reference implementation.

I then profiled the tokenizer and optimized the actual bottlenecks rather than prematurely rewriting the system.

### Optimization 1 — Parallel Pre-tokenization

Pre-tokenization was parallelized using multiple processes while preserving safe chunk boundaries.

Key considerations included:

- Python GIL behavior
- Process-based parallelism
- Serialization overhead
- Safe byte offsets
- UTF-8 boundaries
- Special-token boundaries
- Map/reduce-style aggregation

### Optimization 2 — Incremental Pair Updates

The original implementation rescanned the complete vocabulary representation after every BPE merge.

The optimized implementation maintains:

```text
word_id → current token sequence
word_id → corpus frequency
pair → global weighted count
pair → affected word IDs
```

When a pair is merged, only words containing that pair are updated.

This avoids repeatedly rescanning unaffected words.

### Benchmark

TinyStories validation corpus:

```text
~22.5 MB
~157K lines
```

| Vocabulary Size | Reference | Optimized | Speedup |
|---|---:|---:|---:|
| 300 | 17.216 s | 5.695 s | **3.02×** |
| 500 | 24.545 s | 5.622 s | **4.37×** |

The important result was not only the final speedup, but the bottleneck shift:

```text
Initial implementation
        │
        ▼
Pre-tokenization dominates
        │
        ▼
Parallelize pre-tokenization
        │
        ▼
Merge rescanning becomes dominant
        │
        ▼
Incremental pair indexing
        │
        ▼
~4.4× end-to-end improvement
```

The reference implementation is preserved as a correctness oracle for the optimized implementation.

---

## Training Stack

The repository implements the core training components rather than delegating them to high-level training frameworks.

### Numerically Stable Cross Entropy

Cross entropy is implemented directly from logits using a numerically stable log-sum-exp formulation.

For logits \(x\) and target token \(y\):

\[
L =
\log \left(\sum_j e^{x_j-m}\right)
-
(x_y-m)
\]

where:

\[
m = \max_j x_j
\]

This avoids overflow and reduces catastrophic cancellation for large logits.

---

### AdamW from Scratch

Implemented a stateful AdamW optimizer including:

- First moment estimate
- Second moment estimate
- Bias correction
- Per-parameter optimizer state
- Decoupled weight decay
- Parameter groups

Conceptually:

\[
m_t =
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
\]

\[
v_t =
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
\]

followed by the Adam update and decoupled weight decay.

---

### Learning-Rate Scheduling

Implemented linear warmup followed by cosine decay:

```text
Learning Rate

αmax       /\
          /  \
         /    \
        /      \
       /        \____ αmin

       ↑        ↑
     warmup   cosine decay
```

The learning rate is updated directly through optimizer parameter groups at every training step.

---

### Global Gradient Clipping

Gradient clipping computes the global L2 norm across all model gradients:

\[
\|g\|_2 =
\sqrt{\sum_i g_i^2}
\]

If the norm exceeds the configured threshold, every gradient is scaled by the same factor.

This preserves the direction of the global gradient while limiting its magnitude.

---

## Memory-Efficient Data Pipeline

Training operates on one long stream of token IDs.

For a sampled starting position \(i\):

```text
input  = tokens[i     : i + T]
target = tokens[i + 1 : i + T + 1]
```

Across a batch:

```text
inputs  → [B, T]
targets → [B, T]
```

Tokenized datasets are loaded with NumPy memory mapping:

```python
np.load(path, mmap_mode="r")
```

This allows datasets larger than available RAM to remain on disk while only the required training windows are accessed.

The storage and compute representations are intentionally separated:

```text
Disk
np.uint16
     │
     ▼
Sample batch
     │
     ▼
PyTorch token IDs
torch.long
     │
     ▼
Transformer
float32 / accelerator-compatible compute
```

---

## Training Loop

The complete training path is implemented end-to-end:

```text
Memory-mapped token dataset
          │
          ▼
Random batch sampling
          │
          ▼
Transformer forward pass
          │
          ▼
Cross entropy
          │
          ▼
Backpropagation
          │
          ▼
Global gradient clipping
          │
          ▼
AdamW + scheduled learning rate
          │
          ▼
Updated parameters
          │
          ├──────────────► Validation
          │
          └──────────────► Checkpoint
```

Training and validation modes are handled explicitly using:

```python
model.train()
model.eval()
torch.no_grad()
```

Validation loss is estimated periodically across multiple validation batches.

---

## Checkpointing and Resume

Training state can be persisted and restored using:

```text
Model state
+
Optimizer state
+
Global training iteration
```

This allows interrupted runs to resume without restarting the optimizer or learning-rate schedule.

The checkpoint iteration represents the **next training step to execute**, avoiding duplicated optimizer updates after resume.

Example:

```text
Run 1
0 ───────────────────► 20
                       │
                       ▼
                  checkpoint

Run 2
                       20 ─────────► 30
```

Checkpoint/resume behavior was tested end-to-end.

---

## Autoregressive Text Generation

The repository also implements inference without relying on high-level generation APIs.

Generation proceeds autoregressively:

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
Transformer
  │
  ▼
Last-position logits
  │
  ▼
Temperature scaling
  │
  ▼
Top-p sampling
  │
  ▼
Sample next token
  │
  ▼
Append token
  │
  └──────────── repeat
```

Implemented decoding features include:

- Temperature-controlled sampling
- Nucleus / top-p sampling
- Context-window truncation
- EOS-based stopping
- Maximum generated-token limits
- Restoration of model train/eval state

### Top-p Sampling

Tokens are sorted by probability and the smallest set whose cumulative probability exceeds \(p\) is retained.

For example:

```text
Probability     Cumulative

0.50            0.50
0.25            0.75
0.15            0.90   ← crosses p = 0.80
0.10            1.00
```

The first three tokens remain candidates, are renormalized, and the next token is sampled from that reduced distribution.

---

## Repository Structure

```text
llm-systems/
│
├── src/
│   └── llm_systems/
│       │
│       ├── nn/
│       │   ├── linear
│       │   ├── embedding
│       │   ├── rmsnorm
│       │   ├── attention
│       │   ├── rope
│       │   ├── feedforward
│       │   ├── transformer_block
│       │   └── transformer_lm
│       │
│       ├── tokenization/
│       │   ├── tokenizer
│       │   ├── reference BPE implementation
│       │   └── optimized BPE implementation
│       │
│       ├── training/
│       │   ├── loss
│       │   ├── optimizer
│       │   ├── data loading
│       │   ├── checkpointing
│       │   ├── training utilities
│       │   └── training loop
│       │
│       └── generation/
│           └── decoding
│
├── scripts/
│   ├── BPE benchmarking
│   ├── dataset preprocessing
│   ├── language-model training
│   └── text generation
│
└── tests/
    └── unit and integration tests
```

---

## Testing

The project includes unit and integration tests covering:

- Tensor shapes
- Transformer components
- Attention behavior
- Numerical stability
- Gradient propagation
- Optimizer state
- Gradient clipping
- Learning-rate schedules
- BPE merge correctness
- Special-token boundaries
- BPE frequency weighting
- Deterministic tie-breaking
- Reference vs optimized BPE parity
- Random training-window construction
- Next-token target alignment
- Checkpoint save/load
- Training resume semantics
- Top-p sampling
- Vocabulary-index restoration after sorting
- EOS generation behavior
- Train/eval mode restoration

Run the test suite with:

```bash
uv run pytest -v
```

---

## Running Training

The training launcher exposes experiment configuration through command-line arguments rather than hardcoding individual runs.

Inspect available options:

```bash
uv run python scripts/train_lm.py --help
```

Example:

```bash
uv run python scripts/train_lm.py \
    --train-data data/tokenized/train.npy \
    --val-data data/tokenized/valid.npy \
    --num-steps 10000 \
    --batch-size 16 \
    --context-length 256 \
    --d-model 512 \
    --num-heads 8 \
    --num-layers 8 \
    --alpha-max 3e-4 \
    --device auto
```

External datasets and generated token arrays are intentionally not committed to the repository.

---

## Generating Text

A trained checkpoint can be used for autoregressive generation with configurable decoding parameters.

```bash
uv run python scripts/generate.py \
    --checkpoint checkpoints/model.pt \
    --prompt "Once upon a time" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9
```

---

## Engineering Principles

A few principles guided the implementation:

**Correctness before optimization.**  
Reference implementations and unit tests were established before performance changes.

**Measure before optimizing.**  
Profiling was used to identify actual tokenizer bottlenecks before changing the implementation.

**Preserve a correctness oracle.**  
The simple BPE implementation remains available for differential testing against optimized versions.

**Optimize algorithms before adding complexity.**  
Incremental pair updates delivered substantial gains without immediately introducing heaps, native extensions, or custom kernels.

**Separate implementation from orchestration.**  
Core model and training logic lives under `src/`, while executable workflows live under `scripts/`.

**Keep external data external.**  
Large datasets and generated artifacts are not stored in Git.

---

## Current Scope

This repository currently focuses on a **single-device, from-scratch language-model training stack**.

The implementation intentionally prioritizes understanding and correctness of the core systems before introducing higher-level distributed or inference optimizations.

Potential future extensions include:

- Mixed-precision training
- Training throughput profiling
- FlashAttention / fused kernels
- KV-cache-based decoding
- Distributed data parallelism
- FSDP / model sharding
- Larger-scale training experiments

---

## Why This Project

Modern LLM frameworks make it possible to train models with only a few lines of high-level code.

That abstraction is useful, but it hides many of the systems decisions that determine whether a model is **correct, numerically stable, memory efficient, and fast**.

This project implements those layers directly to build a deeper understanding of:

```text
tokenization
→ tensor representations
→ Transformer computation
→ autograd
→ optimization
→ memory-efficient data access
→ checkpointing
→ inference
→ performance bottlenecks
```

The emphasis throughout the project is not just on making the model run, but on understanding **why each subsystem works, where it becomes expensive, and how to improve it based on evidence**.
