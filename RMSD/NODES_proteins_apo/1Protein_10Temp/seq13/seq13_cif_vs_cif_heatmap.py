import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
CIF_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins"
OUT_DIR  = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/1Protein_10Temp"

PDB_REF  = f"{CIF_BASE}/seq13_PDB_cleaned_8EF5.cif"

HEAVY_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
               "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
               "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
               "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"}

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

def plot_heatmap(matrix, labels, title, cbar_label, out_path, n_clusters=10):
    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    vmin, vmax = 11.00, 25.00
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("Cluster CIF (X)", fontsize=12)
    ax.set_ylabel("Cluster CIF (Y)", fontsize=12)

    # Dividing lines after the PDB reference row/column (index 0)
    ax.axhline(y=0.5, color='red', linewidth=1.5, linestyle='--')
    ax.axvline(x=0.5, color='red', linewidth=1.5, linestyle='--')

    for r in range(len(labels)):
        for c in range(len(labels)):
            val = matrix[r, c]
            norm_val = (val - vmin) / (vmax - vmin) if not np.isnan(val) else 0
            text_color = "white" if norm_val > 0.5 else "black"
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────
cluster_labels = [f"Cluster {i}" for i in range(10)]
cif_paths      = [f"{CIF_BASE}/seq13_cluster_repr_{i}.cif" for i in range(10)]

# PDB reference first, then clusters
all_labels = ["8EF5 (PDB)"] + cluster_labels
all_paths  = [PDB_REF] + cif_paths
n          = len(all_paths)   # 11

for mode in ['calpha', 'allatom']:
    print(f"\n=== {mode.upper()} ===")
    cbar_label = "Cα RMSD (Å)" if mode == 'calpha' else "All-Atom RMSD (Å)"

    def get_atoms(path):
        return get_ca_atoms(path) if mode == 'calpha' else get_heavy_atoms(path)

    def rmsd_fn(ref, mob):
        return ca_rmsd(ref, mob) if mode == 'calpha' else allatom_rmsd(ref, mob)

    print("  Loading all CIF files...")
    all_atoms = [get_atoms(p) for p in all_paths]

    matrix = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            matrix[r, c] = rmsd_fn(all_atoms[r], all_atoms[c])
            print(f"  {all_labels[r]} vs {all_labels[c]}: {matrix[r, c]:.3f} Å")

    print(f"\n  Min: {np.nanmin(matrix):.3f} Å")
    print(f"  Max: {np.nanmax(matrix):.3f} Å")

    plot_heatmap(
        matrix, all_labels,
        title=f"Seq13 Cluster CIF vs CIF — {cbar_label.split(' (')[0]}\n(10 cluster representatives + 8EF5 PDB reference)",
        cbar_label=cbar_label,
        out_path=f"{OUT_DIR}/seq13_cif_vs_cif_{mode}_heatmap.png"
    )

print("\nAll done!")