import unittest

import torch

from attention import ImportanceRoutedSelfAttention, ScaledDotProductMultiHeadSelfAttention


class TestAttention(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)

    def test_fused_dense_matches_weight_return_path_with_masks(self) -> None:
        layer = ScaledDotProductMultiHeadSelfAttention(16, 4).eval()
        x = torch.randn(2, 12, 16)
        attn_mask = torch.zeros(2, 4, 12, 12, dtype=torch.bool)
        attn_mask[:, :, :, 8:] = True
        padding_mask = torch.zeros(2, 12, dtype=torch.bool)
        padding_mask[:, 11] = True
        fused = layer(x, attn_mask=attn_mask, key_padding_mask=padding_mask, is_causal=True)
        reference, _ = layer(
            x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            is_causal=True,
            need_weights=True,
        )
        self.assertTrue(torch.allclose(fused, reference, atol=1e-6, rtol=1e-5))

    def test_routed_outputs_are_invariant_to_future_tokens(self) -> None:
        layer = ImportanceRoutedSelfAttention(
            16, 4, local_window=2, global_token_budget=3, routing_chunk_size=4
        ).eval()
        x = torch.randn(2, 32, 16)
        changed = x.clone()
        changed[:, 20:] += torch.randn_like(changed[:, 20:]) * 100
        baseline = layer(x, is_causal=True)
        perturbed = layer(changed, is_causal=True)
        self.assertTrue(torch.allclose(baseline[:, :20], perturbed[:, :20], atol=1e-6))

    def test_routed_supports_head_specific_masks_and_gradients(self) -> None:
        layer = ImportanceRoutedSelfAttention(16, 4, local_window=2, global_token_budget=3)
        x = torch.randn(2, 32, 16, requires_grad=True)
        mask = torch.zeros(2, 4, 32, 32, dtype=torch.bool)
        mask[:, :, :, 31] = True
        output, weights = layer(x, attn_mask=mask, is_causal=True, need_weights=True)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(weights.shape, (2, 4, 32, 32))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 32), atol=1e-6))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)
        output.square().mean().backward()
        self.assertIsNotNone(layer.router_query.weight.grad)
        self.assertIsNotNone(layer.router_key.weight.grad)
        self.assertTrue(torch.isfinite(layer.router_query.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.router_key.weight.grad).all())


if __name__ == "__main__":
    unittest.main()