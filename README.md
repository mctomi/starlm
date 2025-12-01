# **StarLM — Deformer Architecture**

StarLM is introducing the **Deformer**, a **linear-time (O(T))**, fully **parallel** architecture for efficient sequence processing.  
Instead of relying on quadratic self-attention, the Deformer shifts and re-samples each token’s representation across the sequence. Every head and embedding dimension can **read from a different token position**, enabling flexible cross-token communication with minimal computational cost.

Because all operations are vectorized, the Deformer supports **O(T) training and inference**, providing fast temporal alignment and feature mixing at scale. This makes it a lightweight and efficient component for high-throughput sequence models.


## License

- Original upstream code: MIT License (see LICENSE-MIT)
- Modifications in this fork: AGPL-3.0 (see LICENSE-AGPL)

Additional licensing information is provided in the NOTICE file.
