import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo"

APO_BASE   = f"{OUTPUTS_BASE}/seq2_apo_May26_2026_10.52AM/seq2"
TEMP1_BASE = f"{OUTPUTS_BASE}/seq2_templated_apo_May27_2026_9.24AM/seq2"  # from Seed1 model 0
TEMP2_BASE = f"{OUTPUTS_BASE}/seq2_templated_apo_May27_2026_9.26AM/seq2"  # from Seed2 model 0

SEED1 = "seed-1747042453"
SEED2 = "seed-1747042454"
SAMPLES = [0, 1, 2, 3, 4]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_structure(path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("model", path)

def get_ca_atoms(path):
    structure = load_structure(path)
    chain = next(iter(structure[0].get_chains()))
    return [r["CA"] for r in chain.get_residues() if "CA" in r]

def ca_rmsd(ref, mob):
    n = min(len(ref), len(mob))
    sup = Superimposer()
    sup.set_atoms(ref[:n], mob[:n])
    return round(sup.rms, 3)

def compute_rmsds(base, seed):
    ref = get_ca_atoms(f"{base}/{seed}_sample-0/model.cif")
    vals = [0.0]
    for i in [1, 2, 3, 4]:
        vals.append(ca_rmsd(ref, get_ca_atoms(f"{base}/{seed}_sample-{i}/model.cif")))
    return vals

def plot_bargraph(rmsds, colors, labels, title, out_path):
    x = np.arange(len(SAMPLES))
    width = 0.2
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    fig, ax = plt.subplots(figsize=(13, 6))

    for rmsd, offset, color, label in zip(rmsds, offsets, colors, labels):
        bars = ax.bar(x + offset, rmsd, width,
                      color=color, edgecolor='black', linewidth=0.5,
                      label=label, zorder=3)
        for bar in bars:
            if bar.get_height() > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Cα RMSD vs own Sample-0 (Å)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'sample-{i}' for i in SAMPLES], fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.annotate('Reference\n(0 Å)', xy=(0, 0.01), ha='center',
                fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")

# ── Compute all RMSDs ─────────────────────────────────────────────────────────
print("Computing RMSDs...")
r_apo1     = compute_rmsds(APO_BASE,   SEED1)
r_apo2     = compute_rmsds(APO_BASE,   SEED2)
r_temp1_s1 = compute_rmsds(TEMP1_BASE, SEED1)
r_temp1_s2 = compute_rmsds(TEMP1_BASE, SEED2)
r_temp2_s1 = compute_rmsds(TEMP2_BASE, SEED1)
r_temp2_s2 = compute_rmsds(TEMP2_BASE, SEED2)

# ── Graph 1: Templated from Seed 1 model 0 ───────────────────────────────────
print("\nPlotting Graph 1...")
plot_bargraph(
    rmsds  = [r_apo1, r_apo2, r_temp1_s1, r_temp1_s2],
    colors = ['lightgreen', 'darkgreen', 'lightskyblue', 'dodgerblue'],
    labels = [
        f'Apo Seed 1 ({SEED1[-10:]})',
        f'Apo Seed 2 ({SEED2[-10:]})',
        f'Templated Apo (from Seed1 m0) — Seed 1',
        f'Templated Apo (from Seed1 m0) — Seed 2',
    ],
    title="SEQ2 — Cα RMSD vs Own Sample-0\nApo (green) + Templated Apo from Seed 1 Model 0 (blue)",
    out_path=f"{OUT_DIR}/seq2_apo_vs_templated_seed1_bargraph.png"
)

# ── Graph 2: Templated from Seed 2 model 0 ───────────────────────────────────
print("Plotting Graph 2...")
plot_bargraph(
    rmsds  = [r_apo1, r_apo2, r_temp2_s1, r_temp2_s2],
    colors = ['lightgreen', 'darkgreen', 'lightyellow', 'gold'],
    labels = [
        f'Apo Seed 1 ({SEED1[-10:]})',
        f'Apo Seed 2 ({SEED2[-10:]})',
        f'Templated Apo (from Seed2 m0) — Seed 1',
        f'Templated Apo (from Seed2 m0) — Seed 2',
    ],
    title="SEQ2 — Cα RMSD vs Own Sample-0\nApo (green) + Templated Apo from Seed 2 Model 0 (yellow/silver)",
    out_path=f"{OUT_DIR}/seq2_apo_vs_templated_seed2_bargraph.png"
)

print("\nDone!")
