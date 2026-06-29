"""Architecture diagram for Paper A.

Simple horizontal flow: waveform -> JEPA encoder -> Q2D2 quantizer ->
HiFi-GAN decoder -> waveform. SIGReg annotated on the Stage-1 encoder;
Stage-2 losses (MR-STFT + L1 + frozen-WavLM perceptual, no GAN) annotated
on the decoder.

Run: python scripts/plot_architecture.py
Output: paper/figures/architecture.{pdf,png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 240,
    "savefig.dpi": 240,
})

fig, ax = plt.subplots(figsize=(7.1, 2.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 38)
ax.axis("off")

# ---- palette ----
C_ENC = "#3d6fb4"
C_ENC_F = "#dce6f4"
C_QNT = "#2e7d32"
C_QNT_F = "#d9ead0"
C_DEC = "#b5651d"
C_DEC_F = "#f3e3d0"
C_TXT = "#1a1a1a"
C_SIG = "#2e7d32"
C_LOSS = "#7a3b8f"

BOX_Y = 16.0
BOX_H = 9.0


def box(x, w, label, sub, edge, fill):
    p = FancyBboxPatch((x, BOX_Y), w, BOX_H,
                       boxstyle="round,pad=0.3,rounding_size=0.8",
                       linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, BOX_Y + BOX_H * 0.62, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_TXT, zorder=4)
    ax.text(x + w / 2, BOX_Y + BOX_H * 0.24, sub, ha="center", va="center",
            fontsize=6.7, color="#444444", zorder=4)
    return x + w


def arrow(x0, x1, y=BOX_Y + BOX_H / 2, label=None, color="#333333"):
    a = FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12,
                        linewidth=1.6, color=color, zorder=2)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, y + 2.0, label, ha="center", va="bottom",
                fontsize=6.3, color=color, style="italic")


def waveform(cx, cy, w=7.0, h=4.5, color="#555555"):
    t = np.linspace(0, 4 * np.pi, 220)
    env = np.exp(-0.12 * np.abs(t - 2 * np.pi)) * 0.6 + 0.4
    y = np.sin(t * 2.3) * env
    ax.plot(cx + (t / t.max() - 0.5) * w, cy + y * h / 2,
            color=color, linewidth=0.9, zorder=4)


# ---- input waveform ----
waveform(7, BOX_Y + BOX_H / 2)
ax.text(7, BOX_Y - 3.2, "input\nwaveform", ha="center", va="top",
        fontsize=6.8, color="#444444")

# ---- main blocks ----
x = 14
arrow(12, x)
x = box(x, 23, "JEPA Encoder",
        "conv + Conformer\n128-d latent", C_ENC, C_ENC_F)
arrow(x, x + 8, label="z")
x += 8
x = box(x, 21, "Q2D2 Quantizer",
        "rhombic $A_2$ lattice\nper-pair affine, K=4", C_QNT, C_QNT_F)
arrow(x, x + 8, label="tokens")
x += 8
x = box(x, 21, "HiFi-GAN Decoder",
        "multi-receptive-field\nupsampling", C_DEC, C_DEC_F)
arrow(x, x + 5)
waveform(x + 9, BOX_Y + BOX_H / 2)
ax.text(x + 9, BOX_Y - 3.2, "reconstructed\nwaveform", ha="center", va="top",
        fontsize=6.8, color="#444444")

# ---- token-stream tag (between quantizer and decoder) ----
ax.text(64, BOX_Y + BOX_H + 4.2, "8 tokens / frame  $\\cdot$  12.5 Hz  $\\cdot$  1.6 kbps",
        ha="center", va="center", fontsize=6.6, color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                  edgecolor="#bbbbbb", linewidth=0.7))

# ---- SIGReg tag on the encoder (Stage 1) ----
ax.annotate("Stage 1: masked-latent prediction\n+ SIGReg (Gaussianizes $z$)",
            xy=(25.5, BOX_Y + BOX_H), xytext=(25.5, BOX_Y + BOX_H + 7.2),
            ha="center", va="center", fontsize=6.7, color=C_SIG,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_SIG, linewidth=1.0))

# ---- Stage-2 loss tag on the decoder ----
ax.annotate("Stage 2 losses: MR-STFT + $\\ell_1$ + frozen-WavLM\nperceptual  "
            "(no GAN discriminator)",
            xy=(x - 10.5, BOX_Y), xytext=(x - 10.5, BOX_Y - 8.8),
            ha="center", va="center", fontsize=6.7, color=C_LOSS,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_LOSS, linewidth=1.0))

fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"architecture.{ext}"), bbox_inches="tight")
print("saved", os.path.join(OUT, "architecture.pdf"))
