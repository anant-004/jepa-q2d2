"""Hero figure for Paper A: SIGReg distribution-quantizer co-design.

Panel (a): training curves, Q2D2 codec WITH vs WITHOUT SIGReg on the encoder.
  Matched pair, verified from source 2026-05-14:
    SIGReg ON  = q2d2_sigreg          (W&B r0mj5oy5, sigreg_200k_stage1 lambda=0.05)
    SIGReg OFF = q2d2_ema200k_control (train.log,    ema_200k_stage1   lambda=0.0)
  Both: 25 Hz, code_dim 32, Q2D2 K=4, single-phase, 100k steps. Only lambda_sigreg differs.

Panel (b): encoder feature pairs (post-normalize, post-affine, i.e. exactly what the
  Q2D2 lattice snaps) with the rhombic lattice overlaid. Data from
  extract_sigreg_feature_dist.py (real features from the two trained codecs).

Run: python scripts/plot_sigreg_hero.py
Outputs: paper/figures/sigreg_codesign.{pdf,png}  and  sigreg_panel_a.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "eval_results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType (Type 42), not Type 3 — required by EDAS / IEEE
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 220,
    "savefig.dpi": 220,
})

C_ON = "#2e7d32"    # green  = works
C_OFF = "#c62828"   # red    = collapses
C_ON_FILL = "#a5d6a7"
C_OFF_FILL = "#ef9a9a"

on = pd.read_csv(os.path.join(EVAL, "sigreg_on_curve.csv"))
off = pd.read_csv(os.path.join(EVAL, "sigreg_off_curve.csv"))


def style_panel_a(ax):
    """Polished training-curve panel."""
    ax.set_axisbelow(True)
    ax.grid(True, color="0.88", linewidth=0.6, zorder=0)

    # soft fill under the working curve only (down to the zero baseline):
    # the area between the green curve and zero = quality the codec gained.
    ax.fill_between(on["step"], on["stoi"], 0.0, color=C_ON_FILL,
                    alpha=0.35, linewidth=0, zorder=1)

    # zero reference
    ax.axhline(0, color="0.45", linewidth=0.9, linestyle=(0, (5, 3)), zorder=2)

    # curves: the "without SIGReg" curve stays a clean line (its flatness
    # at zero is the point, no fill needed).
    ax.plot(on["step"], on["stoi"], color=C_ON, linewidth=2.2,
            solid_capstyle="round", zorder=4, label="with SIGReg")
    ax.plot(off["step"], off["stoi"], color=C_OFF, linewidth=2.0,
            solid_capstyle="round", zorder=4, label="without SIGReg")

    # endpoint markers + value tags
    xe = on["step"].iloc[-1]
    ye_on, ye_off = on["stoi"].iloc[-1], off["stoi"].iloc[-1]
    ax.scatter([xe], [ye_on], s=26, color=C_ON, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.scatter([xe], [ye_off], s=26, color=C_OFF, zorder=5,
               edgecolor="white", linewidth=0.8)
    ax.annotate(f"{ye_on:.3f}", xy=(xe, ye_on), xytext=(-6, 7),
                textcoords="offset points", color=C_ON, fontsize=8.5,
                fontweight="bold", ha="right")
    ax.annotate(f"{ye_off:.3f}  (collapsed)", xy=(xe, ye_off),
                xytext=(-6, -13), textcoords="offset points", color=C_OFF,
                fontsize=8.5, fontweight="bold", ha="right")

    ax.set_xlabel("training steps")
    ax.set_ylabel("ESTOI")
    ax.set_xlim(0, 100000)
    ax.set_ylim(-0.12, 0.9)
    ax.set_xticks([0, 20000, 40000, 60000, 80000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    leg = ax.legend(loc="center right", frameon=True, handlelength=1.6,
                    borderpad=0.6, edgecolor="0.8")
    leg.get_frame().set_linewidth(0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(a) Q2D2 codec training (25 Hz, cd32)", loc="left",
                 fontweight="bold")


def style_panel_b(ax):
    """Encoder feature pairs vs. the rhombic Q2D2 lattice."""
    dist_path = os.path.join(EVAL, "sigreg_feature_dist.npz")
    if not os.path.exists(dist_path):
        ax.text(0.5, 0.5,
                "panel (b): run\nscripts/extract_sigreg_feature_dist.py\non the VM first",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="0.45")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(b) Encoder features vs. Q2D2 lattice", loc="left",
                     fontweight="bold")
        return

    d = np.load(dist_path)
    off_xy, on_xy = d["off_xy"], d["on_xy"]
    grid = d["grid"]            # (K^2, 2) the actual rhombic lattice points
    lim = float(d["lim"])

    ax.set_axisbelow(True)
    ax.scatter(off_xy[:, 0], off_xy[:, 1], s=3, c=C_OFF, alpha=0.30,
               linewidths=0, zorder=2, label="without SIGReg")
    ax.scatter(on_xy[:, 0], on_xy[:, 1], s=3, c=C_ON, alpha=0.30,
               linewidths=0, zorder=3, label="with SIGReg")
    ax.scatter(grid[:, 0], grid[:, 1], s=42, marker="P", c="#1a1a1a",
               edgecolor="white", linewidth=0.6, zorder=5,
               label="Q2D2 lattice")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("pair coordinate 1")
    ax.set_ylabel("pair coordinate 2")
    leg = ax.legend(loc="upper right", frameon=True, handlelength=1.2,
                    borderpad=0.5, markerscale=1.8, edgecolor="0.8")
    leg.get_frame().set_linewidth(0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(b) Encoder features vs. Q2D2 lattice", loc="left",
                 fontweight="bold")


# ---- standalone polished panel (a) ----
figA, axA = plt.subplots(figsize=(4.0, 3.0))
style_panel_a(axA)
figA.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    figA.savefig(os.path.join(OUT, f"sigreg_panel_a.{ext}"), bbox_inches="tight")

# ---- combined 2-panel hero figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
style_panel_a(ax1)
style_panel_b(ax2)
fig.tight_layout(pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"sigreg_codesign.{ext}"), bbox_inches="tight")

print("saved:")
print("  ", os.path.join(OUT, "sigreg_panel_a.pdf"), "(standalone panel a)")
print("  ", os.path.join(OUT, "sigreg_codesign.pdf"), "(combined a+b)")
print(f"  panel (a): SIGReg-on final ESTOI {on['stoi'].iloc[-1]:.4f}, "
      f"off {off['stoi'].iloc[-1]:.4f}")
b = os.path.join(EVAL, "sigreg_feature_dist.npz")
print(f"  panel (b): {'built from ' + b if os.path.exists(b) else 'PLACEHOLDER'}")
