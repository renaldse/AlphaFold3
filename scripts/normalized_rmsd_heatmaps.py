"""
normalized_rmsd_heatmaps.py
---------------------------
For each sequence, builds a normalized all-atom RMSD heatmap.
Rows: 10 cluster representative CIFs
Cols: 2 samples per cluster (seed-*_sample-0), 10 clusters = 20 columns
Cell(x, y) = RMSD(sample_y -> template_x) / RMSD(sample_y -> home_template_x)

Output: /mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/june15_heatmaps/
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs/june11"
CIF_BASE     = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins"
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/june15_heatmaps"

SEQS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17]
N_CLUSTERS = 10

HEAVY_ATOMS = {
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
    "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
    "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PARSER = MMCIFParser(QUIET=True)

def load_heavy_atoms(path):
    structure = PARSER.get_structure("s", path)
    chain = next(iter(structure[0].get_chains()))
    res_atoms = {}
    for residue in chain.get_residues():
        resnum = residue.get_id()[1]
        atoms = [a for a in residue.get_atoms() if a.get_name() in HEAVY_ATOMS]
        if atoms:
            res_atoms[resnum] = atoms
    return res_atoms

def allatom_rmsd(ref_res, mob_res):
    common = sorted(set(ref_res.keys()) & set(mob_res.keys()))
    ref_atoms, mob_atoms = [], []
    for resnum in common:
        ref_names = {a.get_name(): a for a in ref_res[resnum]}
        mob_names = {a.get_name(): a for a in mob_res[resnum]}
        for name in sorted(set(ref_names) & set(mob_names)):
            ref_atoms.append(ref_names[name])
            mob_atoms.append(mob_names[name])
    if len(ref_atoms) < 4:
        return np.nan
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    return round(sup.rms, 3)

def find_cluster_output_dir(seq_num, cluster_idx):
    pattern = os.path.join(OUTPUTS_BASE, f"seq{seq_num}_cluster{cluster_idx}_templated*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No output dir for seq{seq_num} cluster{cluster_idx}")
    return matches[0]

def collect_sample_cifs(seq_num, cluster_idx):
    out_dir = find_cluster_output_dir(seq_num, cluster_idx)
    inner   = os.path.join(out_dir, f"seq{seq_num}_cluster{cluster_idx}")
    cifs    = sorted(glob.glob(os.path.join(inner, "seed-*_sample-0", "model.cif")))
    return cifs

def plot_heatmap(matrix, row_labels, col_labels, title, out_path):
    fig_w = max(16, len(col_labels) * 0.65)
    fig_h = max(5,  len(row_labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list("bwr_norm", ["blue", "white", "red"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=2.0, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Normalised All-Atom RMSD\n(sample→row_template) / (sample→home_template)", fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0\n(= home)", "1.5", "2.0"])

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(1, N_CLUSTERS):
        ax.axvline(x=i * 2 - 0.5, color="gray", linewidth=0.8, linestyle="--")

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            if np.isnan(val):
                ax.text(c, r, "nan", ha="center", va="center", fontsize=5, color="gray")
                continue
            norm_val = val / 2.0
            dist_from_mid = abs(norm_val - 0.5)
            text_color = "white" if dist_from_mid > 0.35 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for seq_num in SEQS:
    seq = f"seq{seq_num}"
    print(f"\n{'='*60}")
    print(f"  Processing {seq.upper()}")
    print(f"{'='*60}")

    try:
        print("  Loading cluster templates...")
        tmpl_atoms = {}
        for i in range(N_CLUSTERS):
            cif_path = os.path.join(CIF_BASE, f"{seq}_cluster_repr_{i}.cif")
            tmpl_atoms[i] = load_heavy_atoms(cif_path)
            print(f"    Cluster {i}: loaded")

        print("  Collecting sample CIFs...")
        col_labels = []
        col_paths  = []
        col_home   = []

        for cid in range(N_CLUSTERS):
            cifs = collect_sample_cifs(seq_num, cid)
            if len(cifs) != 2:
                print(f"  [warn] expected 2 samples for cluster {cid}, found {len(cifs)}")
            for s_idx, cif in enumerate(cifs):
                col_labels.append(f"C{cid}\ns{s_idx+1}")
                col_paths.append(cif)
                col_home.append(cid)

        n_cols = len(col_paths)
        n_rows = N_CLUSTERS
        matrix = np.full((n_rows, n_cols), np.nan)

        print(f"  Computing RMSD matrix ({n_rows}x{n_cols})...")
        for c, (col_path, home_cid) in enumerate(zip(col_paths, col_home)):
            print(f"    Col {c+1}/{n_cols} (home=C{home_cid})...")
            try:
                mob = load_heavy_atoms(col_path)
                denom = allatom_rmsd(tmpl_atoms[home_cid], mob)
                if denom == 0 or np.isnan(denom):
                    print(f"      [warn] home RMSD is {denom}, skipping")
                    continue
                for r in range(N_CLUSTERS):
                    numerator = allatom_rmsd(tmpl_atoms[r], mob)
                    matrix[r, c] = round(numerator / denom, 4)
            except Exception as e:
                print(f"      [warn] {e}")

        row_labels = [f"Cluster {i}" for i in range(N_CLUSTERS)]
        plot_heatmap(
            matrix, row_labels, col_labels,
            title=(
                f"{seq.upper()} — Normalised All-Atom RMSD Heatmap\n"
                f"Cell = RMSD(sample→row_template) / RMSD(sample→home_template)  "
                f"[1.0 = home; <1 = closer; >1 = further]"
            ),
            out_path=os.path.join(OUT_DIR, f"{seq}_allatom_normalised_heatmap.png")
        )

    except Exception as e:
        print(f"\n[ERROR] {seq} failed: {e}")
        continue

print("\nAll done!")
