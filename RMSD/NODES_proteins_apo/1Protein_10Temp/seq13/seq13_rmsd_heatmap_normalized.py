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
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/1Protein_10Temp"

PDB_REF      = f"{CIF_BASE}/seq13_PDB_cleaned_8EF5.cif"

HEAVY_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
               "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
               "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
               "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"}

# ── Cluster config: (cluster_id, folder, seed1, seed2) ────────────────────────
CLUSTERS = [
    (0, "seq13_cluster0_templated_apo_Jun4_2026_2.58.42PM_1",  "seed-1244920500", "seed-1244920501"),
    (1, "seq13_cluster1_templated_apo_Jun4_2026_2.59.46PM_2",  "seed-1026869798", "seed-1026869799"),
    (2, "seq13_cluster2_templated_apo_Jun4_2026_3.00.32PM_3",  "seed-608017314",  "seed-608017315"),
    (3, "seq13_cluster3_templated_apo_Jun4_2026_3.01.37PM_4",  "seed-1396353804", "seed-1396353805"),
    (4, "seq13_cluster4_templated_apo_Jun4_2026_3.02.22PM_5",  "seed-1298022059", "seed-1298022060"),
    (5, "seq13_cluster5_templated_apo_Jun4_2026_3.03.24PM_6",  "seed-978914877",  "seed-978914878"),
    (6, "seq13_cluster6_templated_apo_Jun4_2026_3.04.09PM_7",  "seed-592469897",  "seed-592469898"),
    (7, "seq13_cluster7_templated_apo_Jun4_2026_3.05.13PM_8",  "seed-1415016211", "seed-1415016212"),
    (8, "seq13_cluster8_templated_apo_Jun4_2026_3.05.59PM_9",  "seed-1156084354", "seed-1156084355"),
    (9, "seq13_cluster9_templated_apo_Jun4_2026_3.08.13PM_10", "seed-871318040",  "seed-871318041"),
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

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bwr_norm", ["blue", "white", "red"]
    )
    vmin, vmax = (0.7, 1.3) if "All-Atom" in cbar_label else (0.7, 1.3)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=11)
    mid = (vmin + vmax) / 2
    cbar.set_ticks([vmin, (vmin + mid) / 2, mid, (mid + vmax) / 2, vmax])
    cbar.set_ticklabels([f"{vmin:.1f}", f"{(vmin+mid)/2:.2f}", f"{mid:.1f}\n(= home)",
                         f"{(mid+vmax)/2:.2f}", f"{vmax:.1f}"])

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Vertical dividers between clusters
    n_clusters = len(CLUSTERS)
    for i in range(1, n_clusters):
        ax.axvline(x=i*10 - 0.5, color='gray', linewidth=0.8, linestyle='--')

    # Horizontal divider separating PDB reference row (row 0) from cluster rows
    ax.axhline(y=0.5, color='red', linewidth=1.5, linestyle='--')

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            if np.isnan(val):
                ax.text(c, r, "nan", ha="center", va="center",
                        fontsize=5, color="gray")
                continue
            norm_val = (val - vmin) / (vmax - vmin)
            dist_from_mid = abs(norm_val - 0.5)
            text_color = "white" if dist_from_mid > 0.35 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

n_clusters = len(CLUSTERS)

for mode in ['calpha', 'allatom']:
    print(f"\n=== {mode.upper()} ===")
    cbar_label = "Normalised Cα RMSD  (sample→y_template) / (sample→home_template)" \
                 if mode == 'calpha' else \
                 "Normalised All-Atom RMSD  (sample→y_template) / (sample→home_template)"

    def get_atoms(path):
        return get_ca_atoms(path) if mode == 'calpha' else get_heavy_atoms(path)

    def rmsd_fn(ref, mob):
        return ca_rmsd(ref, mob) if mode == 'calpha' else allatom_rmsd(ref, mob)

    # ── Load cluster CIF references + PDB reference ────────────────────────────
    print("  Loading cluster CIF references...")
    cif_refs = {}
    for i in range(n_clusters):
        cif_path = f"{CIF_BASE}/seq13_cluster_repr_{i}.cif"
        print(f"    Cluster {i}: {cif_path}")
        cif_refs[i] = get_atoms(cif_path)

    print(f"  Loading PDB reference: {PDB_REF}")
    pdb_ref_atoms = get_atoms(PDB_REF)

    # ── Build column metadata ─────────────────────────────────────────────────
    col_labels = []
    col_paths  = []
    col_home   = []

    for cid, folder, seed1, seed2 in CLUSTERS:
        base = f"{OUTPUTS_BASE}/{folder}/seq13_cluster{cid}"
        for sample in SAMPLES:
            col_labels.append(f"C{cid} S1\ns{sample}")
            col_paths.append(f"{base}/{seed1}_sample-{sample}/model.cif")
            col_home.append(cid)
        for sample in SAMPLES:
            col_labels.append(f"C{cid} S2\ns{sample}")
            col_paths.append(f"{base}/{seed2}_sample-{sample}/model.cif")
            col_home.append(cid)

    # 1 PDB reference row (top) + 10 cluster rows
    row_labels = ["8EF5 (PDB)"] + [f"Cluster {i} CIF" for i in range(n_clusters)]
    n_rows = len(row_labels)   # 11
    n_cols = len(col_paths)    # 100

    matrix = np.full((n_rows, n_cols), np.nan)

    for c, (col_path, home_cid) in enumerate(zip(col_paths, col_home)):
        print(f"  Column {c+1}/{n_cols}: home=C{home_cid}  {col_path.split('/')[-2]}")
        try:
            mob = get_atoms(col_path)

            denom = rmsd_fn(cif_refs[home_cid], mob)

            if denom == 0 or np.isnan(denom):
                print(f"    WARNING: home RMSD is {denom} — skipping column")
                continue

            # Row 0: PDB reference
            pdb_numerator = rmsd_fn(pdb_ref_atoms, mob)
            matrix[0, c] = round(pdb_numerator / denom, 4)

            # Rows 1–10: cluster CIF references
            for r in range(n_clusters):
                numerator = rmsd_fn(cif_refs[r], mob)
                matrix[r + 1, c] = round(numerator / denom, 4)

        except Exception as e:
            print(f"    WARNING: {e}")

    plot_heatmap(
        matrix, row_labels, col_labels,
        title=(
            f"Seq13 — Normalised {'Cα' if mode == 'calpha' else 'All-Atom'} RMSD Heatmap\n"
            f"Cell = RMSD(sample → y_template) / RMSD(sample → home_template)  "
            f"[1.0 = same distance as home; <1 = closer; >1 = further]"
        ),
        cbar_label=cbar_label,
        out_path=f"{OUT_DIR}/seq13_cluster_{mode}_normalised_heatmap.png"
    )

print("\nAll done!")