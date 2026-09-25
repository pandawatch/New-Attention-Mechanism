"""A small decoder-only transformer used by the benchmark."""

import torch
from torch import Tensor, nn

from attention import ScaledDotProductMultiHeadSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = ScaledDotProductMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x), is_causal=True)
        return x + self.feed_forward(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, 0.0) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        _, length = tokens.shape
        if length > self.block_size:
            raise ValueError(f"sequence length {length} exceeds block size {self.block_size}")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def replace_attention(model: TinyGPT, attention_type: type[nn.Module], attention_kwargs: dict | None = None) -> None:
    attention_kwargs = attention_kwargs or {}
    for block in model.blocks:
        old = block.attention
        replacement = attention_type(old.embed_dim, old.num_heads, dropout=old.dropout, **attention_kwargs)
        replacement.load_state_dict(old.state_dict(), strict=False)
        block.attention = replacement