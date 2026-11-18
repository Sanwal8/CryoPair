"""Draw the overview figure of the README: inputs, the two transfer routes and the software commands."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in Path(sys.argv[1]).glob("*.ttf") if len(sys.argv) > 1 else []:
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "svg.hashsalt": "cryopair", "svg.fonttype": "none", "pdf.fonttype": 42})

FROZEN, FFPE, ADAPT, MAP, INK, MUTED, LINE = "#2b6ca3", "#b0601f", "#3d7f5f", "#7a5aa6", "#1f2933", "#52606d", "#9aa5b1"
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 110); ax.set_ylim(0, 54); ax.axis("off")


def box(x, y, w, h, title, lines=(), edge=INK, fill="white", size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.3, edgecolor=edge, facecolor=fill))
    top = y + h - 2.6 if lines else y + h / 2
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=size, fontweight="bold", color=edge)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, top - 3.0 - 2.7 * k, line, ha="center", va="center", fontsize=9, color=INK)


def arrow(a, b, color=LINE, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, linewidth=1.4, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


def heading(x, text):
    ax.text(x, 51.5, text, ha="center", va="center", fontsize=10, color=MUTED)


heading(10, "Same patient, two preparations")
heading(46.5, "Two ways to transfer")
heading(96, "Output")

box(1, 34, 18, 10, "Frozen section", ("H&E slide",), edge=FROZEN)
box(1, 14, 18, 10, "FFPE slide", ("H&E slide",), edge=FFPE)
ax.plot([10, 10], [24.6, 33.4], color=LINE, linewidth=1.2, linestyle=(0, (3, 2)))
ax.text(11, 29, "paired", ha="left", va="center", fontsize=9, color=MUTED, style="italic")

ax.add_patch(FancyBboxPatch((25, 31), 43, 16.5, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.0, edgecolor=ADAPT, facecolor="#f1f7f4"))
ax.text(26.5, 45.3, "CryoPair", ha="left", va="center", fontsize=11, fontweight="bold", color=ADAPT)
ax.text(66.5, 45.3, "adapts the encoder", ha="right", va="center", fontsize=9, color=MUTED, style="italic")
box(27, 33, 17, 9.5, "Encoder", ("with low-rank", "adapters"), edge=ADAPT)
for y, line in ((41.2, "Pairing term: frozen features"), (38.8, "approach the patient's FFPE ones."),
                (35.8, "Retention term penalizes"), (33.4, "changes in FFPE features.")):
    ax.text(46.2, y, line, ha="left", va="center", fontsize=8.7, color=INK)

ax.add_patch(FancyBboxPatch((25, 10.5), 43, 16.5, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.0, edgecolor=MAP, facecolor="#f5f2f9"))
ax.text(26.5, 24.8, "PAIRMAP", ha="left", va="center", fontsize=11, fontweight="bold", color=MAP)
ax.text(66.5, 24.8, "leaves the encoder unchanged", ha="right", va="center", fontsize=9, color=MUTED, style="italic")
box(27, 12.5, 17, 9.5, "Encoder", ("original",), edge=MAP)
box(48, 12.5, 18, 9.5, "Linear map", ("frozen features to", "FFPE features"), edge=MAP)
arrow((44, 17.25), (48, 17.25), MAP)

box(74, 22.5, 14, 13, "RNA reader", ("fitted on", "FFPE slides,", "not refitted"), edge=FFPE)
box(92.5, 22.5, 16, 13, "Predicted RNA", ("per patient", "from frozen slides"), edge=INK)

arrow((19, 39), (25, 39), FROZEN)
ax.plot([19, 22, 22], [39, 39, 17.25], color=FROZEN, linewidth=1.4)
arrow((22, 17.25), (25, 17.25), FROZEN)
ax.plot([68, 71, 71], [39, 39, 31.5], color=ADAPT, linewidth=1.4); arrow((71, 31.5), (74, 31.5), ADAPT)
ax.plot([68, 71, 71], [17.25, 17.25, 26.5], color=MAP, linewidth=1.4); arrow((71, 26.5), (74, 26.5), MAP)
arrow((88, 29), (92.5, 29), INK)
ax.text(10, 11.6, "pairs are used to train the", ha="center", va="center", fontsize=8.5, color=MUTED)
ax.text(10, 9.2, "adapters and to fit the map", ha="center", va="center", fontsize=8.5, color=MUTED)

ax.plot([1, 109], [6.2, 6.2], color="#d9dee3", linewidth=1.0)
for x, command, what in ((25, "cryopair-encode", "patient features"), (46.5, "cryopair-train", "adapters"), (64, "cryopair-fit", "map and reader"), (91, "cryopair-predict", "RNA from frozen slides")):
    ax.text(x, 3.9, command, ha="center", va="center", fontsize=9.5, family="DejaVu Sans Mono", color=INK)
    ax.text(x, 1.3, what, ha="center", va="center", fontsize=8.5, color=MUTED)
ax.text(1, 3.9, "Commands", ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)

out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
fig.savefig(out / "overview.png", dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
fig.savefig(out / "overview.svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.15, facecolor="white")
