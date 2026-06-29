"""Paper A figures: regenerated FSQ-vs-Q2D2 curves + consolidated cross-lingual.

(1) fsq_vs_q2d2_clean.{pdf,png}: PESQ and ESTOI vs training step, clean x-axis.
    Data: eval_results/fsq_cd64_curve.csv, q2d2_cd64_curve.csv (pulled from the
    two cd64 train.logs, 400 eval points each, verified 2026-05-15).

(2) crosslingual_summary.{pdf,png}: two clean panels replacing the cluttered
    4-panel figure -- (a) 5-NN language separability across 9 models,
    (b) cluster NMI (k=64). All values from verified analysis JSONs.

Run: python scripts/plot_paper_a_figures.py
"""
import os
import pandas as pd
import numpy as np
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

C_Q2D2 = "#2e7d32"
C_FSQ = "#c62828"
C_OURS = "#2e7d32"
C_OTHER = "#7f8c9b"

# ============================================================
# (1) FSQ vs Q2D2 -- clean regeneration
# ============================================================
fsq = pd.read_csv(os.path.join(EVAL, "fsq_cd64_curve.csv"))
q2d2 = pd.read_csv(os.path.join(EVAL, "q2d2_cd64_curve.csv"))

fig, (axp, axs) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, col, ylab, title in [
    (axp, "pesq", "PESQ", "(a) PESQ"),
    (axs, "stoi", "ESTOI", "(b) ESTOI"),
]:
    ax.set_axisbelow(True)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.plot(q2d2["step"], q2d2[col], color=C_Q2D2, linewidth=2.0,
            solid_capstyle="round", label="Q2D2", zorder=4)
    ax.plot(fsq["step"], fsq[col], color=C_FSQ, linewidth=2.0,
            solid_capstyle="round", label="FSQ", zorder=4)
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 400000)
    ax.set_xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    ax.set_title(title, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # final value tags
    xe = q2d2["step"].iloc[-1]
    ax.scatter([xe], [q2d2[col].iloc[-1]], s=22, color=C_Q2D2, zorder=5,
               edgecolor="white", linewidth=0.7)
    ax.scatter([xe], [fsq[col].iloc[-1]], s=22, color=C_FSQ, zorder=5,
               edgecolor="white", linewidth=0.7)
    ax.annotate(f"{q2d2[col].iloc[-1]:.2f}", xy=(xe, q2d2[col].iloc[-1]),
                xytext=(-4, 6), textcoords="offset points", color=C_Q2D2,
                fontsize=8, fontweight="bold", ha="right")
    ax.annotate(f"{fsq[col].iloc[-1]:.2f}", xy=(xe, fsq[col].iloc[-1]),
                xytext=(-4, -11), textcoords="offset points", color=C_FSQ,
                fontsize=8, fontweight="bold", ha="right")

axp.legend(loc="lower right", frameon=False, handlelength=1.5)
fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fsq_vs_q2d2_clean.{ext}"), bbox_inches="tight")
print("saved fsq_vs_q2d2_clean.pdf  "
      f"(Q2D2 PESQ {q2d2['pesq'].iloc[-1]:.2f} / FSQ {fsq['pesq'].iloc[-1]:.2f})")

# ============================================================
# (2) Cross-lingual summary -- 2 clean panels
# ============================================================
# Verified from analysis JSONs (with_mimi_embeddings/5_codec_classifier_results,
# ssl_baselines/ssl_classifier_results, comprehensive_embeddings/cluster_purity).
# 5-NN language separability (FLEURS 6 languages, chance = 0.167):
sep = [
    ("wav2vec2-base", 0.925, "SSL"),
    ("JEPA-EMA (ours)", 0.854, "OURS"),
    ("JEPA-SIGReg (ours)", 0.793, "OURS"),
    ("XLS-R-300m", 0.752, "SSL"),
    ("Mimi", 0.634, "CODEC"),
    ("mHuBERT-147", 0.612, "SSL"),
    ("EnCodec", 0.605, "CODEC"),
    ("HuBERT-base", 0.564, "SSL"),
    ("DAC", 0.450, "CODEC"),
]
# NMI at k=64 (codecs only -- the 4-model comprehensive run):
nmi = [
    ("JEPA-EMA (ours)", 0.261, "OURS"),
    ("JEPA-SIGReg (ours)", 0.199, "OURS"),
    ("EnCodec", 0.103, "CODEC"),
    ("DAC", 0.069, "CODEC"),
]

def barcolor(tag):
    return {"OURS": C_OURS, "SSL": "#9b8cc4", "CODEC": C_OTHER}[tag]

fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.5, 5.95),
                               gridspec_kw={"height_ratios": [1.55, 1], "hspace": 0.45})

# panel (a): language separability, horizontal bars
names = [n for n, _, _ in sep][::-1]
vals = [v for _, v, _ in sep][::-1]
cols = [barcolor(t) for _, _, t in sep][::-1]
ypos = np.arange(len(names))
axa.barh(ypos, vals, color=cols, height=0.66, zorder=3)
axa.axvline(0.167, color="0.4", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
axa.text(0.167, len(names) - 0.3, " chance", fontsize=6.5, color="0.4",
         va="center", ha="left")
for i, v in enumerate(vals):
    axa.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=6.8)
axa.set_yticks(ypos)
axa.set_yticklabels(names, fontsize=7.3)
axa.set_xlim(0, 1.04)
axa.set_xlabel("5-NN language-ID accuracy")
axa.set_title("(a) Cross-lingual separability (FLEURS, 6 languages)",
              loc="left", fontweight="bold", fontsize=8.6)
axa.set_axisbelow(True)
axa.grid(True, axis="x", color="0.9", linewidth=0.6)
for s in ("top", "right"):
    axa.spines[s].set_visible(False)

# panel (b): NMI, vertical bars
nm = [n for n, _, _ in nmi]
nv = [v for _, v, _ in nmi]
nc = [barcolor(t) for _, _, t in nmi]
xpos = np.arange(len(nm))
axb.bar(xpos, nv, color=nc, width=0.62, zorder=3)
for i, v in enumerate(nv):
    axb.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.8)
axb.set_xticks(xpos)
axb.set_xticklabels(["JEPA-\nEMA", "JEPA-\nSIGReg", "EnCodec", "DAC"],
                    fontsize=8.5)
axb.set_ylim(0, 0.30)
axb.set_ylabel("cluster NMI vs. language ($k{=}64$)")
axb.set_title("(b) Cluster structure", loc="left", fontweight="bold", fontsize=8.6)
axb.set_axisbelow(True)
axb.grid(True, axis="y", color="0.9", linewidth=0.6)
for s in ("top", "right"):
    axb.spines[s].set_visible(False)

# shared legend pushed well below the x-axis labels of both panels
from matplotlib.patches import Patch
handles = [Patch(facecolor=C_OURS, label="ours"),
           Patch(facecolor="#9b8cc4", label="SSL model"),
           Patch(facecolor=C_OTHER, label="other codec")]

# Explicit margins for single-column vertical layout, extra bottom space so the
# legend sits cleanly below panel (b)'s x-tick labels (avoids overlap)
fig.subplots_adjust(left=0.34, right=0.97, top=0.95, bottom=0.14, hspace=0.55)
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.005),
           columnspacing=3.5, handletextpad=0.6, handlelength=1.8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"crosslingual_summary.{ext}"), bbox_inches="tight")
print("saved crosslingual_summary.pdf")
