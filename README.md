# **StarLM — Deformer Architecture**

StarLM introduces the **Deformer**, a **linear-time (O(T))**, fully **parallel** alternative to quadratic self-attention for sequence modeling.

Instead of computing dense attention over all pairs of tokens, the Deformer **deforms** the sequence: each head and embedding dimension learns a continuous shift over time and **re-samples features from other token positions**. This lets different dimensions **read from different timesteps** while keeping the computation simple and highly vectorized.

Because everything is implemented with standard tensor ops (projections + interpolation), the Deformer offers:

- **Linear-time training (O(T))** with respect to sequence length, without the O(T²) cost of self-attention.
- **O(1) per-token compute with respect to context length** during autoregressive decoding: generation cost does not grow as the context gets longer.
- **Fully parallel encoding** over all tokens in a sequence (no recurrence or scan needed).
- **Rich cross-token communication**, since each head/dimension can independently select which past positions to read from.
- **Lightweight, hardware-friendly implementation** built from dense layers, normalization, and gathers—no custom kernels or attention softmax required.
- **Inherently position-aware**, using absolute time indices plus learned shifts to read from different timesteps, so it can model order without relying solely on external positional encodings.

In practice, this makes the Deformer a fast, scalable building block for high-throughput sequence models, preserving much of the flexibility of attention while avoiding its quadratic time complexity.


## License

- Original upstream code: MIT License (see LICENSE-MIT)
- Modifications in this fork: AGPL-3.0 (see LICENSE-AGPL)

Additional licensing information is provided in the NOTICE file.
