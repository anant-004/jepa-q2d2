"""SIGReg: Sliced Isotropic Gaussian Regularizer for JEPA encoders.

Replaces the EMA target encoder with a regularization loss that directly
pushes the latent space toward N(0, I). This is the optimal source
distribution for lattice quantizers (minimum MSE by Zador's formula).

The loss has three components:
  1. Variance: push per-dim variance toward 1
  2. Covariance: push off-diagonal covariance toward 0
  3. Sliced Wasserstein: match 1D projections to N(0,1)

By the Cramér-Wold theorem, matching all 1D marginals to Gaussian
guarantees the joint distribution is Gaussian. The sliced approach
approximates this with K random projections.

Reference: Bardes et al. "VICReg" (ICLR 2022) for the var/cov structure;
Kolouri et al. "Sliced Wasserstein" (CVPR 2019) for the sliced component.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """Sliced Isotropic Gaussian Regularizer.

    Args:
        lambda_var: Weight for variance regularization (push var -> 1)
        lambda_cov: Weight for covariance regularization (push cov -> 0)
        lambda_sw: Weight for sliced Wasserstein to N(0,1)
        n_slices: Number of random projections for sliced Wasserstein
        eps: Stability epsilon for variance computation
    """

    def __init__(
        self,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_sw: float = 1.0,
        n_slices: int = 64,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_sw = lambda_sw
        self.n_slices = n_slices
        self.eps = eps

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Push per-dimension variance toward 1.

        Uses hinge loss: max(0, 1 - sqrt(var + eps)) so there's no penalty
        when variance exceeds 1 (only penalizes collapsed dimensions).
        """
        var = z.var(dim=0)
        std = (var + self.eps).sqrt()
        return F.relu(1.0 - std).mean()

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Push off-diagonal covariance toward 0 (decorrelation)."""
        N, D = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(N - 1, 1)
        # Zero out diagonal, penalize off-diagonal
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0)
        return off_diag.sum() / D

    def _sliced_wasserstein_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Sliced Wasserstein distance to N(0, I).

        Projects z onto random directions and compares sorted projections
        to sorted samples from N(0, 1). The 1D Wasserstein distance between
        sorted samples is just the L2 distance of the sorted vectors.
        """
        N, D = z.shape
        # Random projection directions (unit vectors)
        directions = torch.randn(D, self.n_slices, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)

        # Project data onto random directions: [N, n_slices]
        projections = z @ directions

        # Sort projections
        sorted_proj, _ = projections.sort(dim=0)

        # Reference: sorted samples from N(0, 1)
        # Use quantile function (inverse CDF) for deterministic reference
        quantiles = torch.linspace(0.5 / N, 1.0 - 0.5 / N, N,
                                   device=z.device, dtype=z.dtype)
        ref = torch.erfinv(2 * quantiles - 1) * math.sqrt(2)
        ref = ref.unsqueeze(1).expand_as(sorted_proj)

        # 1D Wasserstein = MSE of sorted samples
        return F.mse_loss(sorted_proj, ref)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute SIGReg loss.

        Args:
            z: [N, D] encoder embeddings (flattened batch × time)

        Returns:
            loss: scalar loss
            metrics: dict of component losses for logging
        """
        l_var = self._variance_loss(z)
        l_cov = self._covariance_loss(z)
        l_sw = self._sliced_wasserstein_loss(z)

        loss = (self.lambda_var * l_var
                + self.lambda_cov * l_cov
                + self.lambda_sw * l_sw)

        metrics = {
            "sigreg/var": l_var.item(),
            "sigreg/cov": l_cov.item(),
            "sigreg/sw": l_sw.item(),
            "sigreg/total": loss.item(),
        }
        return loss, metrics


# ═══════════════════════════════════════════════════════════
# Gaussianity Metrics
# ═══════════════════════════════════════════════════════════


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Effective rank of the embedding matrix via eigenvalue entropy.

    erank = exp(H(p)) where p_i = sigma_i / sum(sigma_i) and H is Shannon
    entropy. erank = D means all dimensions carry equal variance (isotropic).
    erank = 1 means all variance is in one dimension (collapsed).

    Args:
        z: [N, D] embedding matrix

    Returns:
        Effective rank (float in [1, D])
    """
    z_centered = z - z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


@torch.no_grad()
def gaussianity_metrics(z: torch.Tensor) -> Dict[str, float]:
    """Comprehensive Gaussianity assessment of embeddings.

    Args:
        z: [N, D] embedding matrix

    Returns:
        Dict with: erank, mean_var, var_of_var, max_abs_corr, mean_kurtosis,
                   sw_distance (sliced Wasserstein to N(0,I))
    """
    N, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Per-dimension statistics
    per_dim_var = z.var(dim=0)
    per_dim_mean = z.mean(dim=0)

    # Kurtosis: for Gaussian, excess kurtosis = 0
    z_std = z_centered / (per_dim_var.sqrt().clamp(min=1e-8))
    kurtosis = (z_std.pow(4).mean(dim=0) - 3.0)  # excess kurtosis

    # Correlation matrix
    cov = (z_centered.T @ z_centered) / max(N - 1, 1)
    std_outer = per_dim_var.sqrt().unsqueeze(1) * per_dim_var.sqrt().unsqueeze(0)
    corr = cov / std_outer.clamp(min=1e-8)
    corr.fill_diagonal_(0)

    # Sliced Wasserstein
    n_slices = min(128, D)
    directions = torch.randn(D, n_slices, device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=0)
    projections = z @ directions
    sorted_proj, _ = projections.sort(dim=0)
    quantiles = torch.linspace(0.5 / N, 1 - 0.5 / N, N,
                               device=z.device, dtype=z.dtype)
    ref = torch.erfinv(2 * quantiles - 1) * math.sqrt(2)
    sw = F.mse_loss(sorted_proj, ref.unsqueeze(1).expand_as(sorted_proj)).item()

    return {
        "erank": effective_rank(z),
        "erank_ratio": effective_rank(z) / D,
        "mean_var": per_dim_var.mean().item(),
        "var_of_var": per_dim_var.var().item(),
        "mean_abs_mean": per_dim_mean.abs().mean().item(),
        "max_abs_corr": corr.abs().max().item(),
        "mean_abs_corr": corr.abs().mean().item(),
        "mean_kurtosis": kurtosis.mean().item(),
        "kurtosis_std": kurtosis.std().item(),
        "sw_distance": sw,
    }


@torch.no_grad()
def epps_pulley_test(z: torch.Tensor, n_projections: int = 100) -> float:
    """Epps-Pulley-style multivariate normality test via sliced 1D tests.

    Projects onto random directions and runs Shapiro-Wilk-like tests on
    each projection. Returns the fraction of projections that reject
    Gaussianity at the 5% level.

    A fully Gaussian embedding should have rejection_rate ≈ 0.05.

    Uses the D'Agostino-Pearson omnibus test (skewness + kurtosis) as a
    fast surrogate for Shapiro-Wilk.
    """
    N, D = z.shape
    directions = torch.randn(D, n_projections, device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=0)
    projections = z @ directions  # [N, n_projections]

    rejections = 0
    for i in range(n_projections):
        p = projections[:, i]
        p_std = (p - p.mean()) / p.std().clamp(min=1e-8)

        # Skewness test
        skew = p_std.pow(3).mean().item()
        se_skew = math.sqrt(6.0 / N)
        z_skew = skew / se_skew

        # Kurtosis test
        kurt = p_std.pow(4).mean().item() - 3.0
        se_kurt = math.sqrt(24.0 / N)
        z_kurt = kurt / se_kurt

        # Omnibus: chi2(2) under H0
        chi2 = z_skew ** 2 + z_kurt ** 2
        if chi2 > 5.991:  # p < 0.05 threshold for chi2(2)
            rejections += 1

    return rejections / n_projections


if __name__ == "__main__":
    torch.manual_seed(42)
    D = 128

    print("=" * 60)
    print("SIGReg Loss + Gaussianity Metrics Test")
    print("=" * 60)

    # Test 1: Perfectly Gaussian data
    print("\n--- Test 1: N(0, I) data ---")
    z_gauss = torch.randn(1000, D)
    metrics = gaussianity_metrics(z_gauss)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    rejection = epps_pulley_test(z_gauss)
    print(f"  epps_pulley_rejection: {rejection:.3f} (expect ~0.05)")

    # Test 2: Collapsed data (low rank)
    print("\n--- Test 2: Collapsed (rank 4) data ---")
    z_collapsed = torch.randn(1000, 4) @ torch.randn(4, D) * 0.1
    metrics = gaussianity_metrics(z_collapsed)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test 3: Non-isotropic (high variance in some dims)
    print("\n--- Test 3: Non-isotropic data ---")
    scale = torch.logspace(-1, 1, D)
    z_noniso = torch.randn(1000, D) * scale
    metrics = gaussianity_metrics(z_noniso)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test 4: SIGReg loss values
    print("\n--- Test 4: SIGReg loss ---")
    sigreg = SIGRegLoss()

    loss_gauss, m_gauss = sigreg(z_gauss)
    print(f"  N(0,I):      loss={loss_gauss.item():.4f}  {m_gauss}")

    loss_coll, m_coll = sigreg(z_collapsed)
    print(f"  Collapsed:   loss={loss_coll.item():.4f}  {m_coll}")

    loss_noniso, m_noniso = sigreg(z_noniso)
    print(f"  Non-iso:     loss={loss_noniso.item():.4f}  {m_noniso}")

    # Test 5: Gradient flow
    print("\n--- Test 5: Gradient flow ---")
    z_param = torch.randn(500, D, requires_grad=True)
    loss, _ = sigreg(z_param)
    loss.backward()
    print(f"  grad norm: {z_param.grad.norm().item():.4f}")
    print(f"  grad shape: {z_param.grad.shape}")
