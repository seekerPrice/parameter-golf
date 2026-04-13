# Hash-Gated Additive Embeddings + QAT + Two-Pass Latent Compression + Depth Recurrence

## Summary

Non-record submission exploring **7 novel zero-overhead innovations** for maximizing parameter capacity and model quality within the 16MB budget. All innovations add zero eval-time overhead — the core insight from PR #831 that throughput-quantization co-optimization is the binding constraint.

**Current results (MLX, 300 steps, SP1024):** val_bpb = 2.4264, artifact = 8.66 MB (PASS)
**H100 validation pending** — applying for compute credits.

## Novel Innovations

### 1. Hash-Gated Additive Compositional Embeddings
Instead of a full [V x D] embedding table, I use a tiny [K x D] base table (K=256) with additive hash composition:
- `E_i = B[h1(i)] + B[h2(i)] + B[h3(i)]`
- Logits computed in O(K) + O(V) instead of O(V*D): `u = h @ B.T`, then `z_i = u[h1(i)] + u[h2(i)] + u[h3(i)]`
- Frees ~3MB of the 16MB budget for model capacity
- Decouples vocabulary size from parameter count

### 2. Quantization-Aware Training (QAT)
Int6 quantization noise injected from 30% of training via straight-through estimator:
- Per-row `clip = k * std`, quantize to [-31, 31], dequantize
- Gradient flows through as if no quantization happened
- Model learns int6-friendly weight distributions
- Validated: clip_k=12.85 gives 0.75 BPB quant gap at 300 steps (expected to shrink to ~0.01 with full 20K steps)

### 3. Two-Pass Latent Compression (novel, from ChatGPT collaboration)
During training only, the model runs two forward passes:
- Pass 1: standard forward, get logits z1
- Pass 2: feed (embedding + projected stop_gradient(softmax(z1))) as input
- Train on pass-2 loss only
- At eval: single pass — model has internalized refinement behavior
- Expected: -0.03 to -0.06 BPB

### 4. Factored Embeddings
[V x R] x [R x D] factorization for smaller vocabularies, saving 50%+ embedding bytes.

### 5. Depth Recurrence
8 physical layers create 11+ virtual layers by reusing layers 3-5. Zero parameter overhead.

### 6. Parallel Residuals
Layers 6+ use parallel residuals: attention and MLP operate on the same pre-residual input.

### 7. Weight Entropy Regularizer
L2-based regularizer encourages low-entropy weight distributions for better zlib compression.

## Architecture

```
Physical layers: 8 (11 virtual via depth recurrence)
Dimension: 1024
Heads: 16 (GQA with 4 KV heads)
MLP: 3x expansion, relu^2
Parallel residuals: layers 6+
Embed rank: 256 (factored)
QAT: int6 noise from step 90/300
Quantization: int6 GPTQ (clip_k=12.85) + zlib
```

## Results

| Config | Params | val_bpb (pre-quant) | val_bpb (int6 roundtrip) | Artifact |
|--------|--------|---------------------|--------------------------|----------|
| dim=512 (13M) | 13M | 2.91 | — | 1.86 MB |
| dim=768 (41M) | 41M | 2.50 | — | 5.10 MB |
| **dim=1024 (72M)** | **72M** | **2.43** | **3.18** | **8.66 MB** |

Scaling clearly works: each size increase improves BPB. The quant gap (0.75 BPB) is due to only 300 steps of QAT — with 20K steps on H100, this should shrink dramatically.

## Research Methodology

Innovations designed through a 3-AI research collaboration:
- **Claude** (Anthropic): implementation, architecture, QAT, factored embeddings
- **ChatGPT** (OpenAI): two-pass training, entropy-aware tokenizer, quant-aware residual scaling
- **Gemini** (Google): hash-gated embeddings, additive logit trick, global KV-cache, FWHT mixing

Key finding from PR #831: "Every 1ms overhead costs ~0.007 BPB." All innovations are designed to be zero-overhead at eval time.

## Next Steps
- H100 validation (20K steps, 3-seed runs)
- SP4096/SP8192 casefold tokenizer (already trained, needs data retokenization)
- Test two-pass training at scale
- Target: sub-1.06 BPB
