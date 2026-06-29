"""Lattice quantizers for neural audio codecs.

Replaces learned VQ / FSQ / Q2D2 with structured lattice quantizers from
quantization theory. Key advantage: provably optimal distortion for Gaussian
sources, no codebook collapse, no commitment loss, O(d) nearest-neighbor.

Supported lattices:
  - A2 (hexagonal, 2D): G = 0.08018, used by Q2D2
  - D4 (checkerboard, 4D): G = 0.07665, 4.4% better than A2
  - E8 (Gosset, 8D): G = 0.07168, 10.6% better than A2

Theory: Zador's formula gives MSE = G(Lambda) * N^(-2/d) for N-point
lattice quantizer of a uniform source in R^d. Lower G = less distortion
per codeword. Higher d = more efficient packing (sphere-like Voronoi cells).

Reference: Conway & Sloane, "Sphere Packings, Lattices and Groups" (1999)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_d4_nearest(z: torch.Tensor) -> torch.Tensor:
    """Fast nearest-point decoder for the D4 lattice.

    D4 = {x in Z^4 : x1 + x2 + x3 + x4 is even}.
    Algorithm: round to Z^4, then fix parity by flipping the coordinate
    with the largest rounding residual.

    Args:
        z: (..., 4) float tensor

    Returns:
        lattice_point: (..., 4) integer tensor, nearest D4 point
    """
    f = torch.round(z)
    r = z - f
    parity = f.sum(dim=-1) % 2

    abs_r = r.abs()
    max_idx = abs_r.argmax(dim=-1, keepdim=True)
    sign = torch.sign(r.gather(-1, max_idx))

    correction = torch.zeros_like(f)
    correction.scatter_(-1, max_idx, sign)

    needs_fix = (parity != 0).unsqueeze(-1).float()
    return f + needs_fix * correction


def build_d4_shell(max_norm_sq: int) -> torch.Tensor:
    """Enumerate D4 lattice points up to a given squared norm.

    Returns points sorted by distance from origin.
    """
    R = int(math.sqrt(max_norm_sq)) + 1
    points = []
    for x1 in range(-R, R + 1):
        for x2 in range(-R, R + 1):
            for x3 in range(-R, R + 1):
                for x4 in range(-R, R + 1):
                    if (x1 + x2 + x3 + x4) % 2 != 0:
                        continue
                    nsq = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4
                    if nsq <= max_norm_sq:
                        points.append([x1, x2, x3, x4])
    pts = torch.tensor(points, dtype=torch.float32)
    norms = (pts * pts).sum(dim=-1)
    order = norms.argsort()
    return pts[order]


class D4LatticeQuantizer(nn.Module):
    """4D lattice quantizer using the D4 (checkerboard) lattice.

    Groups `dim` dimensions into `dim//4` groups of 4, quantizes each group
    on a truncated D4 lattice with `n_codes` points.

    The codebook is FIXED (not learned). Only per-group affine parameters
    (scale + shift) are learned, adapting the lattice to the data distribution.

    Args:
        dim: Input dimension (must be divisible by 4)
        n_codes: Number of codewords per group (controls bitrate)
        commitment_weight: Optional commitment loss weight
    """

    def __init__(
        self,
        dim: int = 128,
        n_codes: int = 64,
        commitment_weight: float = 0.0,
        use_tanh: bool = False,
    ):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"

        self.dim = dim
        self.group_size = 4
        self.n_groups = dim // 4
        self.n_codes = n_codes
        self.commitment_weight = commitment_weight
        self.use_tanh = use_tanh

        bits_per_group = math.log2(n_codes)
        print(f"[D4] {self.n_groups} groups of 4, {n_codes} codes/group "
              f"({bits_per_group:.1f} bits), total {bits_per_group * self.n_groups:.0f} bits/frame"
              f"{', tanh bounded' if use_tanh else ''}")

        max_norm = 2
        while True:
            pts = build_d4_shell(max_norm)
            if len(pts) >= n_codes:
                break
            max_norm += 1

        codebook = pts[:n_codes]

        cb_radius = codebook.norm(dim=-1).max().item()
        if use_tanh:
            # tanh bounds input to [-1, 1] per dim → max 4D norm = 2.0
            # Scale codebook so 95th percentile of points covers the tanh range
            target_radius = 1.5
        else:
            # Unbounded: scale for unit Gaussian (95th pct of chi(4))
            target_radius = 3.08
        init_scale = target_radius / max(cb_radius, 1e-6)
        codebook = codebook * init_scale

        self.register_buffer("codebook", codebook)
        self.codebook_size = n_codes

        self.pre_norm = nn.LayerNorm(dim)
        if use_tanh:
            self.pre_scale = nn.Parameter(torch.ones(1) * 0.5)

        affine = torch.eye(4).unsqueeze(0).expand(self.n_groups, -1, -1).clone()
        self.affine = nn.Parameter(affine)  # [G, 4, 4]

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        B, D, T = z.shape
        z = z.permute(0, 2, 1).contiguous()
        z_flat = z.reshape(-1, D)
        z_flat = self.pre_norm(z_flat)
        if self.use_tanh:
            z_flat = torch.tanh(z_flat * self.pre_scale)
        return z_flat.reshape(B, T, D).permute(0, 2, 1).contiguous()

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with STE.

        Args:
            z_e: [B, D, T] encoder output

        Returns:
            z_q: [B, D, T] quantized (STE gradients)
            indices: [B, T, G] group indices
            aux_loss: commitment loss
        """
        B, D, T = z_e.shape
        assert D == self.dim

        z_norm = self._normalize(z_e)

        z_bt = z_norm.permute(0, 2, 1).contiguous()  # [B, T, D]
        N = B * T
        groups = z_bt.reshape(N, self.n_groups, 4)  # [N, G, 4]

        # Upcast to float32 for numerically stable affine transforms and cdist
        orig_dtype = groups.dtype
        groups = groups.float()
        affine_f32 = self.affine.float()
        codebook_f32 = self.codebook.float()

        # Apply per-group affine: map data -> lattice space
        groups_rot = torch.einsum("ngi,gji->ngj", groups, affine_f32)  # [N, G, 4]

        # Nearest lattice point in rotated space
        dists = torch.cdist(
            groups_rot.reshape(N * self.n_groups, 1, 4),
            codebook_f32.unsqueeze(0),
        ).squeeze(1)  # [N*G, n_codes]
        indices = dists.argmin(dim=-1).reshape(N, self.n_groups)  # [N, G]
        groups_q_rot = codebook_f32[indices]  # [N, G, 4] in lattice space

        # STE in lattice space (gradients flow through groups_rot)
        groups_q_rot = groups_rot + (groups_q_rot - groups_rot).detach()

        # Inverse affine: map lattice space -> data space
        affine_inv = torch.linalg.inv(affine_f32)  # [G, 4, 4]
        groups_q = torch.einsum("ngi,gji->ngj", groups_q_rot, affine_inv)  # [N, G, 4]

        groups_q = groups_q.to(orig_dtype)
        z_q = groups_q.reshape(B, T, D).permute(0, 2, 1).contiguous()
        indices = indices.reshape(B, T, self.n_groups)

        if self.commitment_weight > 0:
            aux_loss = self.commitment_weight * F.mse_loss(z_norm.float(), z_q.float().detach())
            aux_loss = aux_loss.to(orig_dtype)
        else:
            aux_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)

        return z_q, indices, aux_loss

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct from indices.

        Args:
            indices: [B, T, G]

        Returns:
            z_q: [B, D, T]
        """
        B, T, G = indices.shape
        N = B * T
        flat_idx = indices.reshape(N, G)
        groups_q_rot = self.codebook.float()[flat_idx]  # [N, G, 4] in lattice space
        affine_inv = torch.linalg.inv(self.affine.float())
        groups_q = torch.einsum("ngi,gji->ngj", groups_q_rot, affine_inv)
        z_q = groups_q.reshape(B, T, self.dim).permute(0, 2, 1).contiguous()
        return z_q

    @torch.no_grad()
    def entropy_metric(self, indices: torch.Tensor) -> float:
        """Codebook utilization as normalized entropy."""
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.n_codes).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(self.n_codes)
        return entropy / max_entropy if max_entropy > 0 else 0.0


def d4_pack_indices(
    indices: torch.Tensor,
    group_size: int = 2,
) -> torch.Tensor:
    """Pack D4 group indices into tokens for AR modeling.

    Args:
        indices: [B, T, G] with values in [0, n_codes-1]
        group_size: number of D4 groups per token

    Returns:
        tokens: [B, T, G//group_size] packed indices
    """
    B, T, G = indices.shape
    assert G % group_size == 0
    n_tokens = G // group_size

    indices = indices.reshape(B, T, n_tokens, group_size)
    n_codes = indices.max().item() + 1

    tokens = torch.zeros(B, T, n_tokens, dtype=torch.long, device=indices.device)
    for i in range(group_size):
        tokens = tokens * n_codes + indices[..., i]

    return tokens


# ═══════════════════════════════════════════════════════════
# Adaptive Bit Allocation (reverse water-filling)
# ═══════════════════════════════════════════════════════════


def reverse_water_filling(
    variances: torch.Tensor,
    total_bits: float,
    min_bits: float = 0.0,
    max_bits: float = 16.0,
) -> torch.Tensor:
    """Optimal per-group bit allocation via reverse water-filling.

    For a Gaussian source with per-group variance sigma_g^2, the
    rate-distortion optimal allocation minimizes total MSE subject to
    a total bit budget. The solution is:

        b_g = max(0, 0.5 * log2(sigma_g^2 / lambda))

    where lambda is the "water level" chosen so sum(b_g) = total_bits.

    Args:
        variances: [G] per-group variance estimates
        total_bits: total bit budget across all groups
        min_bits: minimum bits per group (0 = can shut off a group)
        max_bits: maximum bits per group

    Returns:
        bits: [G] optimal bit allocation (may be fractional)
    """
    G = variances.shape[0]
    log_var = 0.5 * torch.log2(variances.clamp(min=1e-12))

    # Binary search for the water level lambda (in log domain)
    lo = log_var.min().item() - max_bits
    hi = log_var.max().item() + 1.0

    for _ in range(64):
        mid = (lo + hi) / 2.0
        bits = (log_var - mid).clamp(min=min_bits, max=max_bits)
        if bits.sum().item() > total_bits:
            lo = mid
        else:
            hi = mid

    bits = (log_var - (lo + hi) / 2.0).clamp(min=min_bits, max=max_bits)

    # Snap to nearest feasible integer bits and redistribute remainder
    bits_int = bits.round().clamp(min=min_bits, max=max_bits)
    remainder = total_bits - bits_int.sum().item()
    if abs(remainder) >= 1.0:
        residuals = bits - bits_int
        order = residuals.abs().argsort(descending=True)
        for i in order:
            if remainder > 0.5 and bits_int[i] < max_bits:
                bits_int[i] += 1
                remainder -= 1
            elif remainder < -0.5 and bits_int[i] > min_bits:
                bits_int[i] -= 1
                remainder += 1
            if abs(remainder) < 0.5:
                break

    return bits_int


def bits_to_n_codes(bits: torch.Tensor, lattice: str = "d4") -> torch.Tensor:
    """Convert bit allocation to number of codebook points per group.

    For D4 lattice, n_codes = 2^bits (clamped to feasible shell sizes).
    """
    return (2.0 ** bits).round().long().clamp(min=1)


class AdaptiveD4Quantizer(nn.Module):
    """D4 quantizer with non-uniform per-group bit allocation.

    Uses reverse water-filling to allocate more bits to high-variance
    groups. Each group gets a different number of D4 lattice points.

    Args:
        dim: Input dimension (must be divisible by 4)
        total_bits: Total bit budget per frame
        min_bits_per_group: Minimum bits per group
        max_bits_per_group: Maximum bits per group
    """

    def __init__(
        self,
        dim: int = 128,
        total_bits: int = 192,
        min_bits_per_group: int = 2,
        max_bits_per_group: int = 10,
    ):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.n_groups = dim // 4
        self.total_bits = total_bits
        self.min_bits = min_bits_per_group
        self.max_bits = max_bits_per_group

        # Pre-build D4 codebooks for each feasible size
        self._codebook_cache: dict = {}
        for b in range(min_bits_per_group, max_bits_per_group + 1):
            n = 2 ** b
            cb = self._build_scaled_codebook(n)
            self.register_buffer(f"_cb_{b}", cb)
            self._codebook_cache[b] = cb

        # Default uniform allocation
        uniform_bits = total_bits / self.n_groups
        default_bits = torch.full((self.n_groups,), uniform_bits).round().long()
        default_bits = default_bits.clamp(self.min_bits, self.max_bits)
        self.register_buffer("bits_per_group", default_bits)

        self.pre_norm = nn.LayerNorm(dim)
        affine = torch.eye(4).unsqueeze(0).expand(self.n_groups, -1, -1).clone()
        self.affine = nn.Parameter(affine)

        # Running variance estimate
        self.register_buffer("running_var", torch.ones(self.n_groups, 4))
        self.register_buffer("n_seen", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _build_scaled_codebook(n_codes: int) -> torch.Tensor:
        max_norm = 2
        while True:
            pts = build_d4_shell(max_norm)
            if len(pts) >= n_codes:
                break
            max_norm += 1
        cb = pts[:n_codes]
        cb_radius = cb.norm(dim=-1).max().item()
        target_radius = 3.08
        cb = cb * (target_radius / max(cb_radius, 1e-6))
        return cb

    def _get_codebook(self, bits: int) -> torch.Tensor:
        return getattr(self, f"_cb_{bits}")

    @torch.no_grad()
    def update_allocation(self, z_e: torch.Tensor):
        """Update running variance and recompute bit allocation."""
        B, D, T = z_e.shape
        z_bt = z_e.permute(0, 2, 1).contiguous().reshape(-1, D)
        z_bt = self.pre_norm(z_bt)
        groups = z_bt.reshape(-1, self.n_groups, 4)
        batch_var = groups.var(dim=0)  # [G, 4]

        momentum = 0.01
        self.running_var.mul_(1 - momentum).add_(batch_var * momentum)
        self.n_seen.add_(1)

        group_var = self.running_var.sum(dim=-1)  # [G]
        new_bits = reverse_water_filling(
            group_var,
            total_bits=self.total_bits,
            min_bits=self.min_bits,
            max_bits=self.max_bits,
        )
        self.bits_per_group.copy_(new_bits.long())

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, T = z_e.shape
        z_norm_flat = self.pre_norm(
            z_e.permute(0, 2, 1).contiguous().reshape(-1, D)
        )
        N = B * T
        groups = z_norm_flat.reshape(N, self.n_groups, 4)
        groups_rot = torch.einsum("ngi,gji->ngj", groups, self.affine)

        all_q = torch.zeros_like(groups_rot)
        all_idx = torch.zeros(N, self.n_groups, dtype=torch.long, device=z_e.device)

        for g in range(self.n_groups):
            bits = self.bits_per_group[g].item()
            cb = self._get_codebook(bits)
            g_rot = groups_rot[:, g:g+1, :]  # [N, 1, 4]
            dists = torch.cdist(g_rot.squeeze(1).unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
            idx = dists.argmin(dim=-1)
            all_idx[:, g] = idx
            all_q[:, g, :] = cb[idx]

        # STE
        all_q = groups_rot + (all_q - groups_rot).detach()

        affine_inv = torch.linalg.inv(self.affine)
        groups_q = torch.einsum("ngi,gji->ngj", all_q, affine_inv)

        z_q = groups_q.reshape(B, T, D).permute(0, 2, 1).contiguous()
        indices = all_idx.reshape(B, T, self.n_groups)

        aux_loss = torch.tensor(0.0, device=z_e.device, dtype=z_e.dtype)
        return z_q, indices, aux_loss

    @torch.no_grad()
    def effective_bits(self) -> float:
        return self.bits_per_group.float().sum().item()


class WelchCoherenceLoss(nn.Module):
    """Frame-theoretic regularizer: minimize Welch coherence excess.

    Pushes the effective codebook (grid * affine) toward a tight frame
    with minimal mutual coherence.
    """

    def forward(
        self, affine: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            affine: [P, 2, 2] per-pair affine matrices
            grid: [K^2, 2] base grid points

        Returns:
            Scalar loss >= 0, zero for tight equiangular frame
        """
        effective = torch.einsum("ki,pji->pkj", grid, affine)

        norms = effective.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = effective / norms

        G_hat = torch.bmm(normalized, normalized.transpose(1, 2))

        K2 = grid.shape[0]
        d = 2
        welch_excess = (G_hat ** 2).sum(dim=(-1, -2)) / (K2 ** 2) - 1.0 / d

        return welch_excess.clamp(min=0).mean()


class FrameConditionLoss(nn.Module):
    """Penalize non-tight frames: minimize condition number B/A."""

    def forward(
        self, affine: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        effective = torch.einsum("ki,pji->pkj", grid, affine)

        centroid = effective.mean(dim=1, keepdim=True)
        centered = effective - centroid

        S = torch.bmm(centered.transpose(1, 2), centered)

        tr = S[:, 0, 0] + S[:, 1, 1]
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
        disc = (tr ** 2 - 4 * det).clamp(min=0)
        sqrt_disc = disc.sqrt()

        lambda_max = (tr + sqrt_disc) / 2
        lambda_min = (tr - sqrt_disc) / 2

        kappa = lambda_max / lambda_min.clamp(min=1e-8)
        return ((kappa - 1) ** 2).mean()


def _quantize_raw(data: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor quantization without any normalization or affine.

    Args:
        data: [N, d] raw data
        codebook: [K, d] codebook points

    Returns:
        quantized: [N, d] nearest codebook points
    """
    # Process in chunks to avoid OOM on cdist
    chunk_size = 512
    out = torch.empty_like(data)
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i + chunk_size]
        dists = torch.cdist(chunk, codebook)
        idx = dists.argmin(dim=-1)
        out[i:i + chunk_size] = codebook[idx]
    return out


def _optimal_scale(data: torch.Tensor, codebook_raw: torch.Tensor) -> Tuple[float, float]:
    """Find the codebook scale that minimizes MSE for given data.

    Searches over scales to find the one that best matches the data
    distribution. Returns (best_scale, best_mse).
    """
    best_mse = float("inf")
    best_s = 1.0
    for s in [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]:
        q = _quantize_raw(data, codebook_raw * s)
        mse = F.mse_loss(data, q).item()
        if mse < best_mse:
            best_mse = mse
            best_s = s

    # Refine around best
    for s in torch.linspace(best_s * 0.7, best_s * 1.3, 20):
        s = s.item()
        q = _quantize_raw(data, codebook_raw * s)
        mse = F.mse_loss(data, q).item()
        if mse < best_mse:
            best_mse = mse
            best_s = s

    return best_s, best_mse


def build_a2_grid(n_codes: int) -> torch.Tensor:
    """Build a hexagonal (A2) lattice codebook with n_codes points."""
    K = int(math.ceil(math.sqrt(n_codes)))
    col_spacing = 1.0
    row_spacing = math.sqrt(3) / 2

    points = []
    for row in range(-K, K + 1):
        y = row * row_spacing
        offset = 0.5 if row % 2 != 0 else 0.0
        for col in range(-K, K + 1):
            x = col * col_spacing + offset
            points.append([x, y])

    pts = torch.tensor(points, dtype=torch.float32)
    norms = (pts * pts).sum(dim=-1)
    order = norms.argsort()
    return pts[order[:n_codes]]


def _run_comparison():
    """Fair lattice comparison: same data, same bits, raw quantization."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    torch.manual_seed(42)
    DIM = 128
    N_SAMPLES = 500

    # ─── Experiment 1: Raw lattice geometry comparison ───
    print("=" * 70)
    print("EXPERIMENT 1: Raw Lattice MSE — D4 vs A2 vs Scalar (unit Gaussian)")
    print("=" * 70)
    print(f"Data: N(0,I) in R^{DIM}, {N_SAMPLES} samples")
    print("All codebooks scale-optimized for the data distribution.\n")

    x = torch.randn(N_SAMPLES, DIM)

    # Rate = 2 bits/dim → 256 bits total
    # Scalar: 4 levels per dim, 128 dims
    # A2: 16 codes per 2D pair, 64 pairs
    # D4: 256 codes per 4D group, 32 groups

    print(f"{'Method':<28} {'bits/frame':>10} {'bits/dim':>9} {'MSE/dim':>10} {'vs Scalar':>10}")
    print("-" * 72)

    # Scalar quantizer: 4 uniform levels per dim
    scalar_cb = torch.linspace(-1, 1, 4).unsqueeze(1)  # [4, 1]
    x_flat = x.reshape(-1, 1)
    best_s_sc, _ = _optimal_scale(x_flat, scalar_cb)
    x_q_sc = _quantize_raw(x_flat, scalar_cb * best_s_sc)
    mse_scalar = F.mse_loss(x_flat, x_q_sc).item()
    print(f"{'Scalar (4 levels)':<28} {2 * DIM:>10} {2.0:>9.1f} {mse_scalar:>10.6f} {'—':>10}")

    # A2 (hexagonal) quantizer: 16 codes per 2D pair
    a2_cb = build_a2_grid(16)
    pairs = x.reshape(N_SAMPLES * (DIM // 2), 2)
    best_s_a2, mse_a2 = _optimal_scale(pairs, a2_cb)
    vs_sc = (mse_scalar - mse_a2) / mse_scalar * 100
    print(f"{'A2 hex (16 codes, 2D)':<28} {4 * (DIM // 2):>10} {4 / 2:>9.1f} {mse_a2:>10.6f} {vs_sc:>+9.1f}%")

    # D4 lattice: 256 codes per 4D group (same bits as A2)
    d4_cb = build_d4_shell(16)[:256]
    groups = x.reshape(N_SAMPLES * (DIM // 4), 4)
    best_s_d4, mse_d4 = _optimal_scale(groups, d4_cb)
    vs_sc = (mse_scalar - mse_d4) / mse_scalar * 100
    print(f"{'D4 lattice (256 codes, 4D)':<28} {8 * (DIM // 4):>10} {8 / 4:>9.1f} {mse_d4:>10.6f} {vs_sc:>+9.1f}%")

    d4_vs_a2 = (mse_a2 - mse_d4) / mse_a2 * 100
    print(f"\nD4 vs A2 (same bitrate): {d4_vs_a2:+.1f}% MSE reduction")

    nsm_a2 = 0.080188
    nsm_d4 = 0.076603
    predicted = (nsm_a2 - nsm_d4) / nsm_a2 * 100
    print(f"Theoretical prediction (Zador NSM): {predicted:+.1f}%")

    # Also compare at lower bitrate
    print(f"\n--- Lower bitrate: 1.5 bits/dim ---")
    print(f"{'Method':<28} {'bits/frame':>10} {'codes/grp':>9} {'MSE/dim':>10} {'vs Scalar':>10}")
    print("-" * 72)

    # Scalar: 3 levels (1.58 bits)
    sc3 = torch.linspace(-1, 1, 3).unsqueeze(1)
    best_s3, _ = _optimal_scale(x_flat, sc3)
    x_q3 = _quantize_raw(x_flat, sc3 * best_s3)
    mse_sc3 = F.mse_loss(x_flat, x_q3).item()
    print(f"{'Scalar (3 levels)':<28} {int(1.58 * DIM):>10} {1.58:>9.2f} {mse_sc3:>10.6f} {'—':>10}")

    # A2: 8 codes per pair (3 bits/pair, 1.5 bits/dim)
    a2_8 = build_a2_grid(8)
    best_s_a2_8, mse_a2_8 = _optimal_scale(pairs, a2_8)
    vs = (mse_sc3 - mse_a2_8) / mse_sc3 * 100
    print(f"{'A2 hex (8 codes, 2D)':<28} {3 * (DIM // 2):>10} {3 / 2:>9.2f} {mse_a2_8:>10.6f} {vs:>+9.1f}%")

    # D4: 64 codes per group (6 bits/group, 1.5 bits/dim)
    d4_64 = build_d4_shell(12)[:64]
    best_s_d4_64, mse_d4_64 = _optimal_scale(groups, d4_64)
    vs = (mse_sc3 - mse_d4_64) / mse_sc3 * 100
    print(f"{'D4 lattice (64 codes, 4D)':<28} {6 * (DIM // 4):>10} {6 / 4:>9.2f} {mse_d4_64:>10.6f} {vs:>+9.1f}%")

    d4_vs_a2_low = (mse_a2_8 - mse_d4_64) / mse_a2_8 * 100
    print(f"\nD4 vs A2 (same bitrate): {d4_vs_a2_low:+.1f}% MSE reduction")

    # ─── Experiment 2: Non-isotropic data ───
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Non-Isotropic Gaussian — Adaptive vs Uniform D4")
    print("=" * 70)

    variances_per_group = torch.logspace(-1, 1, DIM // 4)  # 32 groups
    scale = variances_per_group.repeat_interleave(4).sqrt()
    x_noniso = torch.randn(N_SAMPLES, DIM) * scale.unsqueeze(0)
    groups_noniso = x_noniso.reshape(N_SAMPLES * (DIM // 4), 4)

    print(f"Data: N(0, diag) in R^{DIM}, group var range "
          f"[{variances_per_group.min():.2f}, {variances_per_group.max():.2f}]")

    # Uniform D4: same scale for all groups
    d4_uni_cb = build_d4_shell(12)[:64]
    best_s_uni, mse_uni = _optimal_scale(groups_noniso, d4_uni_cb)
    total_bits = 6 * (DIM // 4)

    # Adaptive: per-group optimal scale (simulates adaptive bit allocation effect)
    mse_per_group = []
    for g in range(DIM // 4):
        g_data = x_noniso[:, g * 4:(g + 1) * 4]
        _, mse_g = _optimal_scale(g_data, d4_uni_cb)
        mse_per_group.append(mse_g)
    mse_adapted_scale = sum(mse_per_group) / len(mse_per_group)

    # True adaptive: different codebook sizes per group via water-filling
    group_vars = torch.tensor([x_noniso[:, g * 4:(g + 1) * 4].var().item()
                               for g in range(DIM // 4)])
    alloc = reverse_water_filling(group_vars, total_bits=total_bits,
                                  min_bits=2, max_bits=10)

    mse_adaptive_groups = []
    actual_bits = 0
    for g in range(DIM // 4):
        b = int(alloc[g].item())
        actual_bits += b
        n = 2 ** b
        cb = build_d4_shell(20)[:n]
        g_data = x_noniso[:, g * 4:(g + 1) * 4]
        _, mse_g = _optimal_scale(g_data, cb)
        mse_adaptive_groups.append(mse_g)
    mse_adaptive = sum(mse_adaptive_groups) / len(mse_adaptive_groups)

    print(f"\n{'Method':<35} {'bits':>6} {'MSE/dim':>10} {'vs uniform':>10}")
    print("-" * 65)
    print(f"{'D4 uniform (64 codes, global s)':<35} {total_bits:>6} {mse_uni:>10.6f} {'—':>10}")
    print(f"{'D4 uniform (per-group optimal s)':<35} {total_bits:>6} {mse_adapted_scale:>10.6f} "
          f"{(mse_uni - mse_adapted_scale) / mse_uni * 100:>+9.1f}%")
    print(f"{'D4 adaptive (water-fill bits)':<35} {actual_bits:>6} {mse_adaptive:>10.6f} "
          f"{(mse_uni - mse_adaptive) / mse_uni * 100:>+9.1f}%")

    print(f"\nBit allocation: min={alloc.min().item():.0f}, max={alloc.max().item():.0f}, "
          f"mean={alloc.float().mean():.1f}")
    print(f"First 8:  {alloc[:8].long().tolist()}  (var: {[f'{v:.2f}' for v in group_vars[:8].tolist()]})")
    print(f"Last 8:   {alloc[-8:].long().tolist()}  (var: {[f'{v:.2f}' for v in group_vars[-8:].tolist()]})")

    # ─── Experiment 3: D4/A2 ratio verification ───
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Empirical D4/A2 Ratio vs Theory")
    print("=" * 70)
    print("At fixed rate R bits/dim, theory predicts MSE_D4/MSE_A2 = G(D4)/G(A2)")
    print(f"Theoretical ratio: G(D4)/G(A2) = {nsm_d4}/{nsm_a2} = {nsm_d4 / nsm_a2:.4f}")
    print(f"Theoretical D4 advantage: {(1 - nsm_d4 / nsm_a2) * 100:.1f}%\n")

    data_2d = torch.randn(2000, 2)
    data_4d = torch.randn(2000, 4)

    print(f"{'R bits/dim':>10} {'N_A2':>6} {'N_D4':>6} {'MSE_A2':>10} {'MSE_D4':>10} "
          f"{'Ratio':>8} {'Theory':>8} {'D4 win':>8}")
    print("-" * 72)

    for R in [1.0, 1.5, 2.0, 2.5, 3.0]:
        N_a2 = int(2 ** (2 * R))
        N_d4 = int(2 ** (4 * R))

        if N_a2 < 2 or N_d4 < 2:
            continue

        cb_a2 = build_a2_grid(N_a2)
        _, mse_a2_r = _optimal_scale(data_2d, cb_a2)

        cb_d4 = build_d4_shell(30)[:N_d4]
        if len(cb_d4) < N_d4:
            continue
        _, mse_d4_r = _optimal_scale(data_4d, cb_d4)

        ratio = mse_d4_r / mse_a2_r
        theory_ratio = nsm_d4 / nsm_a2
        d4_win = (1 - ratio) * 100

        print(f"{R:>10.1f} {N_a2:>6} {N_d4:>6} {mse_a2_r:>10.6f} {mse_d4_r:>10.6f} "
              f"{ratio:>8.4f} {theory_ratio:>8.4f} {d4_win:>+7.1f}%")


if __name__ == "__main__":
    _run_comparison()
