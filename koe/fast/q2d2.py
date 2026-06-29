"""Q2D2: Quantization on 2D Grids (arxiv 2512.01537).

Drop-in replacement for FiniteScalarQuantizer. Groups dimensions into pairs
and quantizes each pair on a 2D grid (square or rhombic/hexagonal), capturing
inter-dimensional correlations that scalar quantization misses.

Interface matches FiniteScalarQuantizer exactly:
    forward(z_e)        -> (z_q, indices, aux_loss)
    dequantize(indices)  -> z_q
    entropy_metric(indices) -> float

Tensor conventions:
    z_e, z_q: [B, D, T]  (channel-first, matches encoder output)
    indices:  [B, T, P]   where P = D // 2 = num_pairs, values in [0, K^2 - 1]
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_square_grid(K: int) -> torch.Tensor:
    """Build a K x K square grid of points uniformly spaced in [-1, 1]^2.

    Returns: [K^2, 2] tensor of grid coordinates.
    """
    # Centers of K uniform bins in [-1, 1], e.g. K=4 -> [-0.75, -0.25, 0.25, 0.75]
    ticks = torch.linspace(-1 + 1 / K, 1 - 1 / K, K)
    gy, gx = torch.meshgrid(ticks, ticks, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [K^2, 2]
    return grid


def _build_rhombic_grid(K: int) -> torch.Tensor:
    """Build a rhombic (hexagonal-packed) grid of ~K^2 points in [-1, 1]^2.

    Hexagonal packing: rows are offset by half the horizontal spacing on
    alternating rows. This gives ~15% better packing density than square grids.

    We generate K rows of K points. Even rows are aligned; odd rows are offset
    by half the column spacing. The result is scaled to fit within [-1, 1]^2.

    Returns: [N, 2] tensor where N = K^2.
    """
    # Hex packing: vertical spacing is sqrt(3)/2 times horizontal spacing
    # for optimal circle packing. We use K rows and K columns.
    col_spacing = 2.0 / K
    row_spacing = col_spacing * math.sqrt(3) / 2

    points = []
    for row in range(K):
        y = -1 + row_spacing / 2 + row * row_spacing
        offset = col_spacing / 2 if row % 2 == 1 else 0.0
        for col in range(K):
            x = -1 + col_spacing / 2 + col * col_spacing + offset
            points.append((x, y))

    grid = torch.tensor(points, dtype=torch.float32)  # [K^2, 2]

    # Rescale to [-1, 1]^2 (the hex offset may push points slightly outside)
    for d in range(2):
        lo, hi = grid[:, d].min(), grid[:, d].max()
        if hi - lo > 1e-6:
            grid[:, d] = 2.0 * (grid[:, d] - lo) / (hi - lo) - 1.0

    return grid


class Q2D2Quantizer(nn.Module):
    """Quantization on 2D Grids — geometry-aware drop-in replacement for FSQ.

    Groups D dimensions into D//2 pairs and quantizes each pair jointly on a
    2D grid (square or rhombic). A learnable per-pair affine transform (2x2
    matrix) rotates and scales the pair before snapping to the grid, then
    inverts the transform after quantization.

    Args:
        dim:        Feature dimension (must be even). Default 128.
        num_levels: Grid resolution K per axis. Grid has K^2 points per pair.
                    K=4 -> 16 codes/pair (comparable to FSQ levels=[4,4,4,4]).
        grid_type:  "square" or "rhombic" (hexagonal packing).
    """

    def __init__(
        self,
        dim: int = 128,
        num_levels: int = 4,
        grid_type: str = "rhombic",
        commitment_weight: float = 0.0,
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        assert grid_type in ("square", "rhombic"), f"Unknown grid_type: {grid_type}"

        self.dim = dim
        self.num_levels = num_levels
        self.num_pairs = dim // 2
        self.codebook_size = num_levels ** 2  # K^2 codes per pair
        self.grid_type = grid_type
        self.commitment_weight = commitment_weight

        # Build the 2D grid codebook (shared across all pairs)
        if grid_type == "square":
            grid = _build_square_grid(num_levels)
        else:
            grid = _build_rhombic_grid(num_levels)
        self.register_buffer("grid", grid)  # [K^2, 2]

        # Normalization: LayerNorm + learnable scale + tanh bounding
        self.pre_norm = nn.LayerNorm(dim)
        self.pre_scale = nn.Parameter(torch.ones(1) * 0.5)

        # Per-pair learnable affine: 2x2 matrix per pair
        # Initialized to identity so the grid starts axis-aligned.
        affine = torch.eye(2).unsqueeze(0).expand(self.num_pairs, -1, -1).clone()
        self.affine = nn.Parameter(affine)  # [P, 2, 2]

    def _normalize(self, z_e: torch.Tensor) -> torch.Tensor:
        """Normalize encoder outputs into [-1, 1] via LayerNorm + scale + tanh.

        Args:
            z_e: [B, D, T] channel-first raw encoder output.

        Returns:
            z_norm: [B, D, T] normalized and bounded in roughly [-1, 1].
        """
        B, D, T = z_e.shape
        z = z_e.permute(0, 2, 1).contiguous()  # [B, T, D]
        z_flat = z.reshape(-1, D)               # [B*T, D]
        z_flat = self.pre_norm(z_flat) * self.pre_scale
        z_flat = torch.tanh(z_flat)
        return z_flat.reshape(B, T, D).permute(0, 2, 1).contiguous()  # [B, D, T]

    def _apply_affine(self, pairs: torch.Tensor) -> torch.Tensor:
        """Apply per-pair learned rotation/scale.

        Args:
            pairs: [N, P, 2] where N = B*T, P = num_pairs.

        Returns:
            rotated: [N, P, 2]
        """
        # pairs: [N, P, 2] -> [P, N, 2] for batched matmul
        # affine: [P, 2, 2]
        # We want output[p] = pairs[:, p, :] @ affine[p].T  for each pair p
        # Using einsum: out[n, p, j] = sum_i pairs[n, p, i] * affine[p, j, i]
        return torch.einsum("npi,pji->npj", pairs, self.affine)

    def _apply_inverse_affine(self, pairs: torch.Tensor) -> torch.Tensor:
        """Apply inverse of per-pair affine transform.

        Args:
            pairs: [N, P, 2]

        Returns:
            unrotated: [N, P, 2]
        """
        # Inverse of 2x2 matrix via closed-form
        # For each [P, 2, 2] matrix [[a,b],[c,d]], inv = (1/det) * [[d,-b],[-c,a]]
        a = self.affine[:, 0, 0]  # [P]
        b = self.affine[:, 0, 1]  # [P]
        c = self.affine[:, 1, 0]  # [P]
        d = self.affine[:, 1, 1]  # [P]
        det = a * d - b * c       # [P]
        inv_det = 1.0 / (det + 1e-8)  # [P]

        # Build inverse: [P, 2, 2]
        inv = torch.stack([
            torch.stack([d * inv_det, -b * inv_det], dim=-1),
            torch.stack([-c * inv_det, a * inv_det], dim=-1),
        ], dim=1)  # [P, 2, 2]

        return torch.einsum("npi,pji->npj", pairs, inv)

    def _quantize_pairs(
        self, pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap each 2D pair to the nearest grid point.

        Args:
            pairs: [N, P, 2] in the affine-transformed space.

        Returns:
            quantized: [N, P, 2] snapped to grid points.
            indices:   [N, P] index into the grid (0 to K^2-1).
        """
        # pairs: [N, P, 2], grid: [K^2, 2]
        # Compute L2 distance from each pair to each grid point
        # Expand for broadcasting: [N, P, 1, 2] - [1, 1, K^2, 2] -> [N, P, K^2]
        diff = pairs.unsqueeze(2) - self.grid.unsqueeze(0).unsqueeze(0)  # [N, P, K^2, 2]
        dist_sq = (diff * diff).sum(dim=-1)  # [N, P, K^2]
        indices = dist_sq.argmin(dim=-1)      # [N, P]
        quantized = self.grid[indices]         # [N, P, 2]
        return quantized, indices

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with straight-through estimator.

        Args:
            z_e: [B, D, T] raw encoder output.

        Returns:
            z_q:      [B, D, T] quantized features (STE gradients flow through).
            indices:  [B, T, P] where P = D//2, values in [0, K^2-1].
            aux_loss: Scalar tensor (always 0 — no commitment loss needed).
        """
        B, D, T = z_e.shape
        assert D == self.dim

        # 1. Normalize into [-1, 1]
        z_norm = self._normalize(z_e)  # [B, D, T]

        # 2. Reshape to pairs: [B, D, T] -> [B, T, D] -> [B*T, P, 2]
        z_bt = z_norm.permute(0, 2, 1).contiguous()  # [B, T, D]
        N = B * T
        pairs = z_bt.reshape(N, self.num_pairs, 2)    # [N, P, 2]

        # 3. Apply per-pair affine rotation/scale
        pairs_rot = self._apply_affine(pairs)  # [N, P, 2]

        # 4. Snap to nearest grid point (non-differentiable)
        pairs_q_grid, indices = self._quantize_pairs(pairs_rot)  # [N, P, 2], [N, P]

        # 5. STE around the quantization only: gradient flows through pairs_rot
        #    as if the snap-to-grid didn't happen. This lets the affine learn.
        pairs_q = pairs_rot + (pairs_q_grid - pairs_rot).detach()  # [N, P, 2]

        # 6. Inverse affine to get back to original space (differentiable)
        pairs_q_inv = self._apply_inverse_affine(pairs_q)  # [N, P, 2]

        # 7. Reshape back: [N, P, 2] -> [B, T, D] -> [B, D, T]
        z_q = pairs_q_inv.reshape(B, T, D).permute(0, 2, 1).contiguous()  # [B, D, T]

        # 8. Reshape indices: [N, P] -> [B, T, P]
        indices = indices.reshape(B, T, self.num_pairs)

        # Commitment loss: push encoder outputs toward quantized values
        if self.commitment_weight > 0:
            aux_loss = self.commitment_weight * F.mse_loss(z_norm, z_q.detach())
        else:
            aux_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        return z_q, indices, aux_loss

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct quantized features from indices.

        Looks up grid coordinates for each pair index, then applies the
        inverse affine to map back to the original feature space.

        Args:
            indices: [B, T, P] with values in [0, K^2-1].

        Returns:
            z_q: [B, D, T] channel-first quantized features.
        """
        B, T, P = indices.shape
        assert P == self.num_pairs
        N = B * T

        # Lookup grid coordinates: [N, P, 2]
        flat_idx = indices.reshape(N, P)
        pairs_q = self.grid[flat_idx]  # [N, P, 2]

        # Inverse affine
        pairs_out = self._apply_inverse_affine(pairs_q)  # [N, P, 2]

        # Reshape: [N, P, 2] -> [B, T, D] -> [B, D, T]
        z_q = pairs_out.reshape(B, T, self.dim).permute(0, 2, 1).contiguous()
        return z_q

    @torch.no_grad()
    def entropy_metric(self, indices: torch.Tensor) -> float:
        """Compute average per-pair codebook utilization as normalized entropy.

        Measures how uniformly the K^2 grid points are used across all pairs.

        Args:
            indices: [B, T, P] with values in [0, K^2-1].

        Returns:
            Float in [0, 1] where 1.0 = perfectly uniform usage across all
            grid points for every pair.
        """
        B, T, P = indices.shape
        K2 = self.codebook_size
        max_entropy = math.log(K2)
        if max_entropy < 1e-12:
            return 1.0

        total = 0.0
        for p in range(P):
            idx_p = indices[:, :, p].reshape(-1)  # [B*T]
            counts = torch.zeros(K2, device=indices.device, dtype=torch.float32)
            counts.scatter_add_(
                0, idx_p.long(),
                torch.ones_like(idx_p, dtype=torch.float32),
            )
            probs = counts / counts.sum().clamp(min=1)
            entropy = -(probs * (probs + 1e-8).log()).sum().item()
            total += entropy / max_entropy

        return total / P


# ======================================================================
# Index packing / unpacking (for token-based AR models)
# ======================================================================


@torch.no_grad()
def q2d2_pack_indices(
    indices: torch.Tensor,
    codebook_size: int,
    group_size: int = 7,
) -> torch.Tensor:
    """Pack Q2D2 pair indices into mixed-radix tokens via Horner's method.

    Each pair index is in [0, K^2-1], so the radix is K^2 for every position.
    Groups of `group_size` pair-indices are packed into a single integer.

    Args:
        indices:       [B, T, P] where P = num_pairs, values in [0, K^2-1].
        codebook_size: K^2 (the radix for each pair).
        group_size:    Number of pair indices to pack into one token.

    Returns:
        packed: [B, T, G] where G = ceil(P / group_size).
                Max token value = K^(2*group_size) - 1.
    """
    B, T, P = indices.shape
    G = (P + group_size - 1) // group_size

    # Pad to multiple of group_size
    pad = G * group_size - P
    if pad > 0:
        indices = torch.cat(
            [indices, torch.zeros(B, T, pad, dtype=indices.dtype, device=indices.device)],
            dim=2,
        )

    tokens = []
    for g in range(G):
        s = g * group_size
        e = s + group_size
        chunk = indices[:, :, s:e].long()  # [B, T, group_size]

        # Horner's method: fold right-to-left
        tok = torch.zeros(B, T, dtype=torch.long, device=indices.device)
        for k in range(group_size - 1, -1, -1):
            tok = chunk[:, :, k] + tok * codebook_size
        tokens.append(tok.unsqueeze(-1))

    return torch.cat(tokens, dim=-1)  # [B, T, G]


@torch.no_grad()
def q2d2_unpack_indices(
    packed: torch.Tensor,
    codebook_size: int,
    num_pairs: int,
    group_size: int = 7,
) -> torch.Tensor:
    """Unpack mixed-radix tokens back to per-pair Q2D2 indices.

    Args:
        packed:        [B, T, G] packed tokens.
        codebook_size: K^2 (the radix).
        num_pairs:     Total number of pairs (= D // 2).
        group_size:    Same group_size used during packing.

    Returns:
        indices: [B, T, P] with values in [0, K^2-1].
    """
    B, T, G = packed.shape
    P_padded = G * group_size

    indices = torch.zeros(B, T, P_padded, dtype=torch.long, device=packed.device)
    for g in range(G):
        s = g * group_size
        tok = packed[:, :, g].clone()
        for k in range(group_size):
            indices[:, :, s + k] = tok % codebook_size
            tok = tok // codebook_size

    return indices[:, :, :num_pairs]
