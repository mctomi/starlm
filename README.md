# **StarLM — Deformer Architecture**

I developed the *deformer*: a form of linear attention where each dimension selects and learns a temporal shift, combining tokens through linear interpolation. It outperforms transformers as the context window grows larger, both in computational speed and learning efficiency. Because the shift is token-dependent, the model becomes highly adaptive, allowing each dimension to choose a different shift to extract information from different tokens. It inherently encodes positional information without requiring RoPE or positional embeddings, and without increasing the parameter count.

It supports O(1) inference and O(n) parallel training while remaining mathematically simple.


## File reference  
You can view the code here:  
[gpt.py — line 37](https://github.com/mctomi/starlm/blob/65043b5b0c319ad25029da3b16f8f5914d084fb9/nanochat/gpt.py#L37)  

## License

- Original upstream code: MIT License (see LICENSE-MIT)
- Modifications in this fork: AGPL-3.0 (see LICENSE-AGPL)

Additional licensing information is provided in the NOTICE file.
