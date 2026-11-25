import unittest

import torch

from fmgad.detector import ResFlowGAD
from fmgad.losses import conditional_flow_matching_loss, flow_matching_loss
from fmgad.residuals import compute_dual_residuals_with_degree


class IdentityEncoder:
    def encode(self, x, edge_index):
        return x


class ResidualTrainingTests(unittest.TestCase):
    def test_constant_embeddings_and_empty_neighborhood(self):
        h = torch.tensor([[10.1, -3.3]]).repeat(65, 1)
        src = torch.arange(64).repeat(9)
        edges = torch.stack([src, src.roll(1)])
        global_residual, local_residual, degree = compute_dual_residuals_with_degree(h, edges, stable=True)
        torch.testing.assert_close(global_residual, torch.zeros_like(h), atol=0, rtol=0)
        torch.testing.assert_close(local_residual[:-1], torch.zeros_like(h[:-1]), atol=0, rtol=0)
        torch.testing.assert_close(local_residual[-1], h[-1], atol=0, rtol=0)
        self.assertEqual(degree[-1].item(), 0)
        model = ResFlowGAD(residual_mode="stable")
        for inputs in (h, h[:1]):
            torch.testing.assert_close(model._normalize_clip(inputs), torch.zeros_like(inputs), atol=0, rtol=0)

    def test_residual_values_and_gradients_match_double_reference(self):
        generator = torch.Generator().manual_seed(4)
        h = torch.randn(19, 5, generator=generator, dtype=torch.float64, requires_grad=True)
        edges = torch.randint(0, 18, (2, 260), generator=generator)
        weights = [torch.randn(h.shape, generator=generator, dtype=h.dtype) for _ in range(2)]
        legacy = compute_dual_residuals_with_degree(h, edges)
        stable = compute_dual_residuals_with_degree(h, edges, stable=True)
        for old, new in zip(legacy, stable):
            torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
        gradients = [
            torch.autograd.grad(sum((a * w).sum() for a, w in zip(values[:2], weights)), h)[0]
            for values in (legacy, stable)
        ]
        torch.testing.assert_close(*gradients, rtol=1e-12, atol=1e-12)

    def test_ablation_removes_only_residual_target(self):
        h = torch.tensor([[1., 4.], [3., 6.], [7., 1.]])
        edges = torch.tensor([[0, 2, 0], [1, 1, 2]])
        for target in ("full", "zero"):
            model = ResFlowGAD(residual_mode="stable", residual_target=target,
                               residual_scale=15., use_virtual_neighbors=False)
            model.ae = IdentityEncoder()
            z, _, residual = model._build_z(h, edges)
            self.assertEqual(model.residual_scale, 1.)
            self.assertEqual(z.shape, (3, 4))
            self.assertTrue(torch.equal(z[:, :2], h))
            self.assertEqual(bool(residual.any()), target == "full")
            self.assertEqual(bool(model._normalize_clip(z)[:, 2:].any()), target == "full")

    def test_both_velocity_losses_use_block_weight_and_keep_sample_weights(self):
        target = torch.tensor([[1., 2., 4., 8.], [2., -4., 1., -1.]])
        context = torch.zeros(1, 4)
        sample_weights = torch.tensor([0.25, 1.5])
        def zero_velocity(state, time, context):
            return torch.zeros_like(state)

        for loss in (flow_matching_loss, conditional_flow_matching_loss):
            torch.manual_seed(42)
            torch.rand(2, 1)
            raw_error = (target - torch.randn_like(target)).square()
            for beta in (1., 2.):
                torch.manual_seed(42)
                actual = loss(zero_velocity, target, context, reduction="none", weight=sample_weights,
                              residual_loss_weight=beta)
                expected = (raw_error[:, :2].mean(1) + beta * raw_error[:, 2:].mean(1)) / (1 + beta)
                torch.testing.assert_close(actual, expected * sample_weights)


if __name__ == "__main__":
    unittest.main()
