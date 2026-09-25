"""Drop-in self-attention layers for the toy decoder model."""

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_input(x: Tensor, embed_dim: int, num_heads: int) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"expected [batch, sequence, embedding], got {tuple(x.shape)}")
    if x.size(-1) != embed_dim:
        raise ValueError(f"expected embedding size {embed_dim}, got {x.size(-1)}")
    if embed_dim % num_heads:
        raise ValueError("embed_dim must be divisible by num_heads")
    return x.size(0), x.size(1), embed_dim // num_heads


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    batch, length, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch, length, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    batch, _, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, length, -1)


def _apply_mask(scores: Tensor, allowed: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            allowed = allowed & ~attn_mask
        else:
            scores = scores + attn_mask
    return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)


class ScaledDotProductMultiHeadSelfAttention(nn.Module):
    """Batch-first scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch, length, head_dim = _validate_input(x, self.embed_dim, self.num_heads)
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)
        if not need_weights:
            fused_mask = None
            fused_is_causal = is_causal and attn_mask is None and key_padding_mask is None
            if attn_mask is not None or key_padding_mask is not None or (is_causal and not fused_is_causal):
                mask_shape = (batch, self.num_heads, length, length)
                allowed = torch.ones(mask_shape, dtype=torch.bool, device=x.device)
                additive_mask = None
                if is_causal:
                    allowed = allowed & torch.ones(length, length, dtype=torch.bool, device=x.device).tril()[None, None]
                if attn_mask is not None:
                    if attn_mask.ndim == 2:
                        expanded_mask = attn_mask[None, None, :, :]
                    elif attn_mask.ndim == 3 and attn_mask.size(0) == batch:
                        expanded_mask = attn_mask[:, None, :, :]
                    elif attn_mask.ndim == 3 and attn_mask.size(0) == batch * self.num_heads:
                        expanded_mask = attn_mask.view(batch, self.num_heads, length, length)
                    elif attn_mask.ndim == 4:
                        expanded_mask = attn_mask
                    else:
                        raise ValueError("attn_mask must have shape [L, L], [B, L, L], [B*H, L, L], or [B, H, L, L]")
                    if expanded_mask.dtype == torch.bool:
                        allowed = allowed & ~expanded_mask
                    else:
                        additive_mask = expanded_mask
                if key_padding_mask is not None:
                    allowed = allowed & ~key_padding_mask[:, None, None, :].to(torch.bool)
                if additive_mask is None:
                    fused_mask = allowed
                else:
                    fused_mask = additive_mask.masked_fill(~allowed, float("-inf"))
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=fused_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=fused_is_causal,
            )
            return self.out_proj(_merge_heads(attended))
        scores = torch.matmul(query, key.transpose(-2, -1)) / head_dim**0.5
        allowed = torch.ones(batch, length, length, dtype=torch.bool, device=x.device)
        if is_causal:
            allowed = allowed & torch.ones_like(allowed).tril()
        if key_padding_mask is not None:
            allowed = allowed & ~key_padding_mask[:, None, :].to(torch.bool)
        scores = _apply_mask(scores, allowed[:, None], attn_mask)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = self.out_proj(_merge_heads(torch.matmul(weights, value)))
        return (output, weights) if need_weights else output


class ImportanceRoutedSelfAttention(ScaledDotProductMultiHeadSelfAttention):
    """Attend locally plus a small learned set of globally important keys.

    Every query sees its local window. The importance predictor scores keys,
    and each query additionally sees the top ``global_token_budget`` keys. In
    causal mode the top-k set is selected from the current prefix only.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        local_window: int = 5,
        global_token_budget: int = 8,
        router_dim: int = 32,
        routing_chunk_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if local_window < 0 or global_token_budget < 1:
            raise ValueError("local_window must be non-negative and global_token_budget must be positive")
        if router_dim < 1:
            raise ValueError("router_dim must be positive")
        if routing_chunk_size < 1:
            raise ValueError("routing_chunk_size must be positive")
        self.local_window = local_window
        self.global_token_budget = global_token_budget
        self.router_dim = router_dim
        self.routing_chunk_size = routing_chunk_size
        self.router_query = nn.Linear(embed_dim, router_dim, bias=False)
        self.router_key = nn.Linear(embed_dim, router_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch, length, head_dim = _validate_input(x, self.embed_dim, self.num_heads)
        local_candidate_count = 2 * self.local_window + 1
        if local_candidate_count + min(self.global_token_budget, length) >= length:
            return super().forward(x, attn_mask, key_padding_mask, is_causal, need_weights)
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        value = _split_heads(value, self.num_heads)
        router_query = self.router_query(x)
        router_key = self.router_key(x)
        offsets = torch.arange(-self.local_window, self.local_window + 1, device=x.device)
        query_positions = torch.arange(length, device=x.device)
        raw_local_indices = query_positions[:, None] + offsets[None, :]
        local_allowed = (raw_local_indices >= 0) & (raw_local_indices < length)
        if is_causal:
            local_allowed = local_allowed & (raw_local_indices <= query_positions[:, None])
        local_indices = raw_local_indices.clamp(0, length - 1)
        global_count = min(self.global_token_budget, length)
        global_index_chunks = []
        global_score_chunks = []
        global_valid_chunks = []
        for chunk_start in range(0, length, self.routing_chunk_size):
            chunk_end = min(chunk_start + self.routing_chunk_size, length)
            chunk_query = router_query[:, chunk_start:chunk_end]
            route_scores = torch.matmul(chunk_query, router_key.transpose(-2, -1)) / self.router_dim**0.5
            if is_causal:
                chunk_positions = query_positions[chunk_start:chunk_end]
                causal_allowed = query_positions[None, :] <= chunk_positions[:, None]
                route_scores = route_scores.masked_fill(~causal_allowed[None, :, :], float("-inf"))
            selected_scores, selected_indices = route_scores.topk(global_count, dim=-1)
            global_index_chunks.append(selected_indices)
            global_score_chunks.append(selected_scores)
            global_valid_chunks.append(torch.isfinite(selected_scores))
        global_indices = torch.cat(global_index_chunks, dim=1)
        global_scores = torch.cat(global_score_chunks, dim=1)
        global_valid = torch.cat(global_valid_chunks, dim=1)
        global_allowed = global_valid
        global_allowed = global_allowed & ~(
            global_indices[:, :, :, None] == local_indices[None, :, None, :]
        ).any(dim=-1)
        key_indices = torch.cat(
            (local_indices[None, :, :].expand(batch, -1, -1), global_indices),
            dim=-1,
        )
        candidate_allowed = torch.cat(
            (local_allowed[None, :, :].expand(batch, -1, -1), global_allowed),
            dim=-1,
        )
        candidate_count = key_indices.size(-1)
        gather_index = key_indices[:, None, :, :, None].expand(-1, self.num_heads, -1, -1, head_dim)
        expanded_key = key[:, :, None, :, :].expand(-1, -1, length, -1, -1)
        expanded_value = value[:, :, None, :, :].expand(-1, -1, length, -1, -1)
        gathered_key = expanded_key.gather(3, gather_index)
        gathered_value = expanded_value.gather(3, gather_index)
        score = torch.matmul(
            query[:, :, :, None, :],
            gathered_key.transpose(-2, -1),
        ).squeeze(-2) / head_dim**0.5
        score = score + torch.cat(
            (
                x.new_zeros(batch, 1, length, local_indices.size(-1)),
                global_scores[:, None, :, :],
            ),
            dim=-1,
        )
        allowed = candidate_allowed[:, None, :, :]
        if key_padding_mask is not None:
            padding = key_padding_mask[:, None, :].expand(-1, length, -1).gather(2, key_indices)
            allowed = allowed & ~padding[:, None, :, :].to(torch.bool)
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expanded_mask = attn_mask[None, None, :, :].expand(batch, 1, -1, -1)
            elif attn_mask.ndim == 3 and attn_mask.size(0) == batch:
                expanded_mask = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 3 and attn_mask.size(0) == batch * self.num_heads:
                expanded_mask = attn_mask.view(batch, self.num_heads, length, length)
            elif attn_mask.ndim == 4 and attn_mask.size(0) == batch and attn_mask.shape[-2:] == (length, length):
                expanded_mask = attn_mask
            else:
                raise ValueError("attn_mask must have shape [L, L], [B, L, L], [B*H, L, L], or [B, H, L, L]")
            mask_index = key_indices[:, None, :, :].expand(-1, expanded_mask.size(1), -1, -1)
            mask = expanded_mask.gather(3, mask_index)
            if mask.dtype == torch.bool:
                allowed = allowed & ~mask
            else:
                score = score + mask
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        weights = F.softmax(score, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = self.out_proj(_merge_heads((weights.unsqueeze(-2) @ gathered_value).squeeze(-2)))
        if need_weights:
            all_weights = x.new_zeros(batch, self.num_heads, length, length)
            all_weights.scatter_add_(
                3,
                key_indices[:, None, :, :].expand(-1, self.num_heads, -1, -1),
                weights,
            )
            return output, all_weights
        return output