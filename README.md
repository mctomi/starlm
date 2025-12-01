# **StarLM — Deformer Architecture**

StarLM is introducing the **Deformer**, a **linear-time**, fully **parallel** architecture for efficient sequence processing.  
Instead of relying on quadratic self-attention, the Deformer shifts and re-samples each token’s representation across the sequence. Every head and embedding dimension can **read from a different token position**, enabling flexible cross-token communication with minimal computational cost.

Through continuous learned shifts and vectorized operations, the Deformer provides fast temporal alignment and feature mixing, making it a lightweight and scalable component for high-throughput sequence models.

## License

- Original upstream code: MIT License (see LICENSE-MIT)
- Modifications in this fork: AGPL-3.0 (see LICENSE-AGPL)

Additional licensing information is provided in the NOTICE file.
