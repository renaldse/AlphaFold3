import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
import os
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
CIF_BASE     = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins"
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/seq15"

HEAVY_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
               "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
               "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
               "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"}

# ── Cluster config: (cluster_id, folder, seed1, seed2) ────────────────────────
CLUSTERS = [
    (0, "seq15_cluster0_templated_apo_Jun1_2026_2.14.29PM_1",  "seed-1194167029", "seed-1194167030"),
    (1, "seq15_cluster1_templated_apo_Jun1_2026_2.14.29PM_2",  "seed-808106595",  "seed-808106596"),
    (2, "seq15_cluster2_templated_apo_Jun1_2026_2.15.47PM_3",  "seed-690219995",  "seed-690219996"),
    (3, "seq15_cluster3_templated_apo_Jun1_2026_2.15.46PM_4",  "seed-1579473745", "seed-1579473746"),
    (4, "seq15_cluster4_templated_apo_Jun1_2026_2.17.04PM_5",  "seed-1077956334", "seed-1077956335"),
    (5, "seq15_cluster5_templated_apo_Jun1_2026_2.17.03PM_6",  "seed-927432316",  "seed-927432317"),
    (6, "seq15_cluster6_templated_apo_Jun1_2026_2.18.23PM_7",  "seed-776875972",  "seed-776875973"),
    (7, "seq15_cluster7_templated_apo_Jun1_2026_2.18.22PM_8",  "seed-1497964378", "seed-1497964379"),
    (8, "seq15_cluster8_templated_apo_Jun1_2026_2.19.41PM_9",  "seed-1240859335", "seed-1240859336"),
    (9, "seq15_cluster9_templated_apo_Jun1_2026_2.19.40PM_10", "seed-1055994449", "seed-1055994450"),
]

SAMPLES = [0, 1, 2, 3, 4]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_structure(path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("model", path)

def get_ca_atoms(path):
    structure = load_structure(path)
    chain = next(iter(structure[0].get_chains()))
    return [r["CA"] for r in chain.get_residues() if "CA" in r]

def get_heavy_atoms(path):
    structure = load_structure(path)
    chain = next(iter(structure[0].get_chains()))
    res_atoms = {}
    for residue in chain.get_residues():
        resnum = residue.get_id()[1]
        atoms = [a for a in residue.get_atoms() if a.get_name() in HEAVY_ATOMS]
        if atoms:
            res_atoms[resnum] = atoms
    return res_atoms

def ca_rmsd(ref, mob):
    n = min(len(ref), len(mob))
    sup = Superimposer()
    sup.set_atoms(ref[:n], mob[:n])
    return round(sup.rms, 3)

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
        return float("nan")
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    return round(sup.rms, 3)

def plot_heatmap(matrix, row_labels, col_labels, title, cbar_label, out_path):
    fig_w = max(22, len(col_labels) * 0.6)
    fig_h = max(5,  len(row_labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["white", "blue"])
    if mode == 'calpha':
        vmin, vmax = 0, 9
    else:
        vmin, vmax = 4.5, 9  # change these for all-atom
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Vertical dividers between clusters
    for i in range(1, 10):
        ax.axvline(x=i*10 - 0.5, color='gray', linewidth=0.8, linestyle='--')

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            norm_val = (val - vmin) / (vmax - vmin) if not np.isnan(val) else 0
            text_color = "white" if norm_val > 0.5 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for mode in ['calpha', 'allatom']:
    print(f"\n=== {mode.upper()} ===")
    cbar_label = "Cα RMSD (Å)" if mode == 'calpha' else "All-Atom RMSD (Å)"

    def get_atoms(path):
        return get_ca_atoms(path) if mode == 'calpha' else get_heavy_atoms(path)

    def rmsd_fn(ref, mob):
        return ca_rmsd(ref, mob) if mode == 'calpha' else allatom_rmsd(ref, mob)

    # ── Row references: cluster CIF files ─────────────────────────────────────
    row_labels = [f"Cluster {i} CIF" for i in range(10)]
    row_refs = []
    for i in range(10):
        cif_path = f"{CIF_BASE}/seq15_cluster_repr_{i}.cif"
        print(f"  Loading row ref: cluster {i} CIF...")
        row_refs.append(get_atoms(cif_path))

    # ── Column paths: all samples for all clusters ─────────────────────────────
    col_labels = []
    col_paths  = []
    for cid, folder, seed1, seed2 in CLUSTERS:
        base = f"{OUTPUTS_BASE}/{folder}/seq15_cluster{cid}"
        for sample in SAMPLES:
            col_labels.append(f"C{cid} S1\ns{sample}")
            col_paths.append(f"{base}/{seed1}_sample-{sample}/model.cif")
        for sample in SAMPLES:
            col_labels.append(f"C{cid} S2\ns{sample}")
            col_paths.append(f"{base}/{seed2}_sample-{sample}/model.cif")

    n_rows = len(row_refs)   # 10
    n_cols = len(col_paths)  # 100

    matrix = np.zeros((n_rows, n_cols))

    for c, col_path in enumerate(col_paths):
        print(f"  Computing column {c+1}/{n_cols}: {col_path.split('/')[-2]}...")
        try:
            mob = get_atoms(col_path)
            for r, ref in enumerate(row_refs):
                matrix[r, c] = rmsd_fn(ref, mob)
        except Exception as e:
            print(f"  WARNING: {e}")
            matrix[:, c] = float("nan")

    plot_heatmap(
        matrix, row_labels, col_labels,
        title=f"Seq15 Cluster Templates — {cbar_label.split(' (')[0]} Heatmap\n"
              f"Y: Cluster CIF Templates | X: Templated Apo Samples (Seed1+Seed2, samples 0–4)",
        cbar_label=cbar_label,
        out_path=f"{OUT_DIR}/seq15_cluster_{mode}_heatmap.png"
    )

print("\nAll done!")
