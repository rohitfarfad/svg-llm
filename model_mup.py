from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import mup


@dataclass
class GPTMuPConfig:
    vocab_size: int = 4096
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttentionMuP(nn.Module):
    def __init__(self, config: GPTMuPConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # µP transformer detail:
        # use 1/d attention scaling rather than 1/sqrt(d)
        att = (q @ k.transpose(-2, -1)) * (head_dim ** -1.0)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(y)
        y = self.resid_dropout(y)

        return y


class MLPMuP(nn.Module):
    def __init__(self, config: GPTMuPConfig):
        super().__init__()

        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class BlockMuP(nn.Module):
    def __init__(self, config: GPTMuPConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttentionMuP(config)

        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLPMuP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTMuP(nn.Module):
    def __init__(self, config: GPTMuPConfig):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([BlockMuP(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # For µP, use MuReadout and do not tie weights for this MVP.
        self.lm_head = mup.MuReadout(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} is larger than block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)

        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build_mup_models(
    target_config: GPTMuPConfig,
    base_width: int = 128,
    delta_width: int = 256,
):
    """
    Builds target, base, and delta models for mup.set_base_shapes.

    For this MVP, depth is held fixed and width is varied.
    """
    base_config = GPTMuPConfig(
        vocab_size=target_config.vocab_size,
        block_size=target_config.block_size,
        n_layer=target_config.n_layer,
        n_head=4,
        n_embd=base_width,
        dropout=target_config.dropout,
        bias=target_config.bias,
    )

    delta_config = GPTMuPConfig(
        vocab_size=target_config.vocab_size,
        block_size=target_config.block_size,
        n_layer=target_config.n_layer,
        n_head=4,
        n_embd=delta_width,
        dropout=target_config.dropout,
        bias=target_config.bias,
    )

    target_model = GPTMuP(target_config)
    base_model = GPTMuP(base_config)
    delta_model = GPTMuP(delta_config)

    mup.set_base_shapes(target_model, base_model, delta=delta_model)

    return target_model, base_model, delta_model