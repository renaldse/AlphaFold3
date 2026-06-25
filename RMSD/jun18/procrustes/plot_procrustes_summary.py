"""
plot_procrustes_summary.py
---------------------------
Bar chart summarizing Procrustes template-bias results across sequences.

X-axis : sequence
Y-axis : % below null mean (effect size)
Color  : bias interpretation (no / moderate / strong template bias)
         based on % below null mean AND p-value significance

Edit the DATA list below with your own results, or adapt this script
to read directly from your *_results.txt files.

Run:
    python plot_procrustes_summary.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data: (seq, mode, mean_self_disparity, null_mean, p_value, z_score)
# ---------------------------------------------------------------------------
DATA = [
    ("SEQ1",  "ALLATOM", 0.013080, 0.014720, 0.000100, -4.601476),
    ("SEQ1",  "CA",      0.007482, 0.009020, 0.000100, -4.626735),
    ("SEQ2",  "ALLATOM", 0.018730, 0.019018, 0.051900, -1.688397),
    ("SEQ2",  "CA",      0.008371, 0.008696, 0.000100, -3.970342),
    ("SEQ3",  "ALLATOM", 0.062706, 0.103989, 0.000100, -5.000153),
    ("SEQ3",  "CA",      0.048692, 0.088029, 0.000100, -4.722773),
    ("SEQ5",  "ALLATOM", 0.015582, 0.024683, 0.000100, -4.946321),
    ("SEQ5",  "CA",      0.008302, 0.016580, 0.000100, -4.493695),
    ("SEQ6",  "ALLATOM", 0.939592, 0.945410, 0.103100, -1.280953),
    ("SEQ6",  "CA",      0.941076, 0.945757, 0.150600, -1.046653),
    ("SEQ7",  "ALLATOM", 0.476086, 0.532959, 0.000400, -3.092210),
    ("SEQ7",  "CA",      0.463509, 0.521988, 0.000400, -3.078580),
    ("SEQ9",  "ALLATOM", 0.419874, 0.468859, 0.001000, -3.056595),
    ("SEQ9",  "CA",      0.442517, 0.491872, 0.001300, -3.036869),
    ("SEQ10", "ALLATOM", 0.811146, 0.814508, 0.374600, -0.317207),
    ("SEQ10", "CA",      0.810192, 0.812865, 0.399700, -0.251289),
    ("SEQ11", "ALLATOM", 0.466885, 0.527890, 0.001900, -2.705162),
    ("SEQ11", "CA",      0.466914, 0.528704, 0.002100, -2.710910),
    ("SEQ12", "ALLATOM", 0.489711, 0.547159, 0.000700, -2.968354),
    ("SEQ12", "CA",      0.490531, 0.546859, 0.001200, -2.886206),
    ("SEQ13", "ALLATOM", 0.663899, 0.681005, 0.123100, -1.204110),
    ("SEQ13", "CA",      0.667381, 0.683515, 0.122300, -1.202876),
    ("SEQ14", "ALLATOM", 0.015116, 0.017730, 0.000100, -5.287920),
    ("SEQ14", "CA",      0.010119, 0.012452, 0.000100, -4.974458),
    ("SEQ15", "ALLATOM", 0.090180, 0.135647, 0.000900, -3.057291),
    ("SEQ15", "CA",      0.071481, 0.111667, 0.001600, -2.839335),
    ("SEQ16", "ALLATOM", 0.045532, 0.053777, 0.006500, -2.386143),
    ("SEQ16", "CA",      0.038744, 0.046622, 0.012400, -2.144137),
    ("SEQ17", "ALLATOM", 0.021897, 0.027222, 0.000100, -6.738085),
    ("SEQ17", "CA",      0.015479, 0.019935, 0.000100, -6.393943),
]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
STRONG_THRESH   = 30.0   # % below null mean
MODERATE_THRESH = 5.0
SIG_THRESH      = 0.05

COLORS = {
    "Strong template bias":        "#2E75B6",  # blue
    "Moderate template bias":      "#70AD47",  # green
    "No meaningful template bias": "#BFBFBF",  # gray
}

def pct_below_null(self_d, null_m):
    return (null_m - self_d) / null_m * 100

def classify(self_d, null_m, p):
    pct = pct_below_null(self_d, null_m)
    if p >= SIG_THRESH or pct < MODERATE_THRESH:
        return "No meaningful template bias", pct
    elif pct < STRONG_THRESH:
        return "Moderate template bias", pct
    else:
        return "Strong template bias", pct

# ---------------------------------------------------------------------------
# Build plotting data
# ---------------------------------------------------------------------------
seqs  = sorted(set(d[0] for d in DATA), key=lambda s: int(s.replace("SEQ", "")))
modes = ["ALLATOM", "CA"]

results = {}  # (seq, mode) -> (pct, category)
for seq, mode, self_d, null_m, p, z in DATA:
    cat, pct = classify(self_d, null_m, p)
    results[(seq, mode)] = (pct, cat)

# ---------------------------------------------------------------------------
# Plot: grouped bars (ALLATOM vs CA) per sequence
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 7))

x = np.arange(len(seqs))
width = 0.38

for i, mode in enumerate(modes):
    heights = []
    colors  = []
    for seq in seqs:
        pct, cat = results.get((seq, mode), (0, "No meaningful template bias"))
        heights.append(pct)
        colors.append(COLORS[cat])

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, heights, width=width, color=colors,
                  edgecolor="black", linewidth=0.6,
                  label=mode, zorder=3,
                  hatch="" if mode == "ALLATOM" else "//")

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

# threshold lines
ax.axhline(STRONG_THRESH, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.axhline(MODERATE_THRESH, color="black", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(seqs) - 0.5, STRONG_THRESH + 1, "Strong threshold (30%)",
        fontsize=7, ha="right", color="dimgray")
ax.text(len(seqs) - 0.5, MODERATE_THRESH + 1, "Moderate threshold (5%)",
        fontsize=7, ha="right", color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels(seqs, fontsize=9)
ax.set_xlabel("Sequence", fontsize=11)
ax.set_ylabel("% Below Null Mean (effect size)", fontsize=11)
ax.set_title(
    "Template Bias Summary Across Sequences (Procrustes Analysis)\n"
    "Bar height = % below null mean   |   Color = bias category   |   "
    "Solid = ALLATOM, Hatched = CA",
    fontsize=11, fontweight="bold"
)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

legend_handles = [
    Patch(facecolor=COLORS["Strong template bias"], edgecolor="black",
          label="Strong template bias (≥30% below null, p<0.05)"),
    Patch(facecolor=COLORS["Moderate template bias"], edgecolor="black",
          label="Moderate template bias (5–30% below null, p<0.05)"),
    Patch(facecolor=COLORS["No meaningful template bias"], edgecolor="black",
          label="No meaningful template bias (<5% below null or p≥0.05)"),
    Patch(facecolor="white", edgecolor="black", label="Solid = ALLATOM"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="Hatched = CA"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
          bbox_to_anchor=(1.0, 1.0))

plt.tight_layout()
out_path = "procrustes_bias_summary.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
