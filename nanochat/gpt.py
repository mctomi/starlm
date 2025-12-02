# This file is part of a modified version of a project originally licensed under MIT.
# 
# Copyright (c) 2025 mctomi
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
# You should have received a copy of the AGPL License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

class Deformer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        dim = config.n_embd
        h = config.n_head
        self.layer_idx = layer_idx
        self.h = h
        self.dh = dim // h

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.shift_q = nn.Linear(dim, dim, bias=False)
        self.shift_k = nn.Linear(dim, dim, bias=False)

    def _project(self, x):
        B, T, _ = x.shape
        H, Dh = self.h, self.dh

        q = self.q_proj(x).view(B, T, H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)

        q = norm(q)
        k = norm(k)

        sq = F.softplus(self.shift_q(x)).view(B, T, H, Dh)
        sk = F.softplus(self.shift_k(x)).view(B, T, H, Dh)

        return q, k, sq, sk

    def forward(self, x, kv_cache=None):
        if kv_cache is not None:
            return self._forward_incremental(x, kv_cache)

        if torch.is_grad_enabled():
            return cp.checkpoint(self._forward_full, x, use_reentrant=False)
        else:
            return self._forward_full(x)

    def _forward_full(self, x):
        B, T, D = x.shape
        q, k, sq, sk = self._project(x)

        t_idx = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1, 1)
        zero = torch.zeros_like(sq)

        posq = torch.maximum(t_idx - sq, zero)
        posq = torch.minimum(posq, t_idx)

        posk = torch.maximum(t_idx - sk, zero)
        posk = torch.minimum(posk, t_idx)

        q_def = self._interp(q, posq)
        k_def = self._interp(k, posk)

        return (q_def * k_def).reshape(B, T, D)

    def _forward_incremental(self, x, kv_cache):
        B, T, D = x.shape
        q, k, sq, sk = self._project(x)

        Q_all, K_all, T_prev = kv_cache.insert_deformer(self.layer_idx, q, k)

        t_idx = torch.arange(T_prev, T_prev + T, device=x.device, dtype=x.dtype)
        t_idx = t_idx.view(1, T, 1, 1)

        zero = torch.zeros_like(sq)

        posq = torch.maximum(t_idx - sq, zero)
        posq = torch.minimum(posq, t_idx)

        posk = torch.maximum(t_idx - sk, zero)
        posk = torch.minimum(posk, t_idx)

        q_def = self._interp(Q_all, posq)
        k_def = self._interp(K_all, posk)

        return (q_def * k_def).reshape(B, T, D)

    def _interp(self, x, pos):
        B, T_x, H, Dh = x.shape

        pos_floor = pos.floor()
        pos0 = pos_floor.clamp(0, T_x - 1).long()
        pos1 = (pos0 + 1).clamp(0, T_x - 1)

        frac = pos - pos_floor

        x0 = x.gather(1, pos0)
        x1 = x.gather(1, pos1)

        return x0 + (x1 - x0) * frac



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Deformer(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache):
        x = x + self.attn(norm(x), kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)


    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()


        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
