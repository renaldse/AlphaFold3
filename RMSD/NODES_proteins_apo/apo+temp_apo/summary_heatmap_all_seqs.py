import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Sequence config ────────────────────────────────────────────────────────────
SEQUENCES = [
    ("seq1",  "seq1_apo_May26_2026_10.52AM",  "seed-1898484009", "seed-1898484010",
     "seq1_templated_apo_May27_2026_10.19.06AM",  "seq1_templated_apo_May27_2026_10.21.33AM"),
    ("seq2",  "seq2_apo_May26_2026_10.52AM",  "seed-1747042453",  "seed-1747042454",
     "seq2_templated_apo_May27_2026_10.23.04AM",  "seq2_templated_apo_May27_2026_10.23.23AM"),
    ("seq3",  "seq3_apo_May26_2026_10.52AM",  "seed-522645511",   "seed-522645512",
     "seq3_templated_apo_May27_2026_10.24.26AM",  "seq3_templated_apo_May27_2026_10.24.35AM"),
    ("seq5",  "seq5_apo_May26_2026_10.53AM",  "seed-1984255282",  "seed-1984255283",
     "seq5_templated_apo_May27_2026_10.25.47AM",  "seq5_templated_apo_May27_2026_10.25.49AM"),
    ("seq6",  "seq6_apo_May26_2026_10.53AM",  "seed-1867252878",  "seed-1867252879",
     "seq6_templated_apo_May27_2026_11.15.47AM_1", "seq6_templated_apo_May27_2026_11.15.47AM_2"),
    ("seq7",  "seq7_apo_May26_2026_10.53AM",  "seed-407581728",   "seed-407581729",
     "seq7_templated_apo_May27_2026_10.31.53AM",  "seq7_templated_apo_May27_2026_10.32.03AM"),
    ("seq8",  "seq8_apo_May26_2026_6.20PM",   "seed-150219149",   "seed-150219150",
     "seq8_templated_apo_May27_2026_10.56.55AM",  "seq8_templated_apo_May27_2026_11.02.16AM"),
    ("seq9",  "seq9_apo_May26_2026_10.55AM",  "seed-2146638107",  "seed-2146638108",
     "seq9_templated_apo_May27_2026_10.33.18AM",  "seq9_templated_apo_May27_2026_10.33.28AM"),
    ("seq10", "seq10_apo_May26_2026_10.55AM", "seed-1092575053",  "seed-1092575054",
     "seq10_templated_apo_May27_2026_10.35.04AM", "seq10_templated_apo_May27_2026_10.35.13AM"),
    ("seq11", "seq11_apo_May26_2026_10.55AM", "seed-907571167",   "seed-907571168",
     "seq11_templated_apo_May27_2026_10.36.49AM", "seq11_templated_apo_May27_2026_10.36.59AM"),
    ("seq12", "seq12_apo_May26_2026_10.55AM", "seed-789659239",   "seed-789659240",
     "seq12_templated_apo_May27_2026_10.38.33AM", "seq12_templated_apo_May27_2026_10.38.43AM"),
    ("seq13", "seq13_apo_May26_2026_10.55AM", "seed-1477840625",  "seed-1477840626",
     "seq13_templated_apo_May27_2026_10.39.17AM", "seq13_templated_apo_May27_2026_10.40.13AM"),
    ("seq14", "seq14_apo_May26_2026_10.55AM", "seed-1181915990",  "seed-1181915991",
     "seq14_templated_apo_May27_2026_10.40.13AM", "seq14_templated_apo_May27_2026_10.40.18AM"),
    ("seq15", "seq15_apo_May26_2026_11.05AM", "seed-829795272",   "seed-829795273",
     "seq15_templated_apo_May27_2026_10.40.27AM", "seq15_templated_apo_May27_2026_10.41.03AM"),
    ("seq16", "seq16_apo_May26_2026_11.05AM", "seed-679279232",   "seed-679279233",
     "seq16_templated_apo_May27_2026_10.41.37AM", "seq16_templated_apo_May27_2026_10.42.03AM"),
    ("seq17", "seq17_apo_May26_2026_11.15AM", "seed-1601948394",  "seed-1601948395",
     "seq17_templated_apo_May27_2026_10.42.03AM", "seq17_templated_apo_May27_2026_10.42.09AM"),
]

OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/apo+temp_apo"
SAMPLES      = [1, 2, 3, 4]

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

# ── Build heatmap ──────────────────────────────────────────────────────────────
for mode in ['calpha', 'allatom']:
    print(f"\n=== {mode.upper()} ===")
    cbar_label = "Cα RMSD (Å)" if mode == 'calpha' else "All-Atom RMSD (Å)"

    def get_atoms(path):
        return get_ca_atoms(path) if mode == 'calpha' else get_heavy_atoms(path)

    def rmsd_fn(ref, mob):
        return ca_rmsd(ref, mob) if mode == 'calpha' else allatom_rmsd(ref, mob)

    # ── Preload all row references ─────────────────────────────────────────────
    print("  Loading row references...")
    seq1_apo_base   = f"{OUTPUTS_BASE}/seq1_apo_May26_2026_10.52AM/seq1"
    seq1_seed1      = "seed-1898484009"
    seq1_seed2      = "seed-1898484010"
    seq1_temp1_base = f"{OUTPUTS_BASE}/seq1_templated_apo_May27_2026_10.19.06AM/seq1"
    seq1_temp2_base = f"{OUTPUTS_BASE}/seq1_templated_apo_May27_2026_10.21.33AM/seq1"

    # Rows 1-2: fixed (seq1 apo model 0s)
    fixed_ref1 = get_atoms(f"{seq1_apo_base}/{seq1_seed1}_sample-0/model.cif")
    fixed_ref2 = get_atoms(f"{seq1_apo_base}/{seq1_seed2}_sample-0/model.cif")

    # Rows 3+: all sequences' templated apo model 0s (2 per sequence = 32 rows + 2 fixed = 34 total)
    row_refs   = [fixed_ref1, fixed_ref2]
    row_labels = [
        "Apo Seq1 S1 m0\n(fixed ref 1)",
        "Apo Seq1 S2 m0\n(fixed ref 2)",
    ]

    for seqname, apo_folder, seed1, seed2, temp1_folder, temp2_folder in SEQUENCES:
        temp1_base = f"{OUTPUTS_BASE}/{temp1_folder}/{seqname}"
        temp2_base = f"{OUTPUTS_BASE}/{temp2_folder}/{seqname}"
        row_refs.append(get_atoms(f"{temp1_base}/{seed1}_sample-0/model.cif"))
        row_refs.append(get_atoms(f"{temp2_base}/{seed2}_sample-0/model.cif"))
        row_labels.append(f"{seqname} Temp S1 m0")
        row_labels.append(f"{seqname} Temp S2 m0")

    n_rows = len(row_refs)  # 34
    n_cols = len(SEQUENCES) * 2  # 32

    # ── X-axis labels ──────────────────────────────────────────────────────────
    col_labels = []
    for seqname, _, _, _, _, _ in SEQUENCES:
        col_labels.append(f"{seqname}\nTemp S1")
        col_labels.append(f"{seqname}\nTemp S2")

    # ── Fill matrix ────────────────────────────────────────────────────────────
    matrix = np.zeros((n_rows, n_cols))

    for col_idx, (seqname, apo_folder, seed1, seed2, temp1_folder, temp2_folder) in enumerate(SEQUENCES):
        print(f"  Computing columns for {seqname}...")
        temp1_base = f"{OUTPUTS_BASE}/{temp1_folder}/{seqname}"
        temp2_base = f"{OUTPUTS_BASE}/{temp2_folder}/{seqname}"

        # Col 1: avg templated seed1 samples 1-4
        for r, ref in enumerate(row_refs):
            vals = [rmsd_fn(ref, get_atoms(f"{temp1_base}/{seed1}_sample-{i}/model.cif")) for i in SAMPLES]
            matrix[r, col_idx*2] = round(np.nanmean(vals), 3)

        # Col 2: avg templated seed2 samples 1-4
        for r, ref in enumerate(row_refs):
            vals = [rmsd_fn(ref, get_atoms(f"{temp2_base}/{seed2}_sample-{i}/model.cif")) for i in SAMPLES]
            matrix[r, col_idx*2 + 1] = round(np.nanmean(vals), 3)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig_w = max(22, n_cols * 0.75)
    fig_h = max(8,  n_rows * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    vmin, vmax = 0, max(np.nanmax(matrix), 4.0)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Vertical dividers between sequences
    for i in range(1, len(SEQUENCES)):
        ax.axvline(x=i*2 - 0.5, color='gray', linewidth=0.8, linestyle='--')

    # Horizontal divider after fixed rows
    ax.axhline(y=1.5, color='blue', linewidth=1.5, linestyle='--')

    # Horizontal dividers between sequences on y-axis
    for i in range(1, len(SEQUENCES)):
        ax.axhline(y=1.5 + i*2 + 0.5, color='gray', linewidth=0.5, linestyle='--')

    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            norm_val = (val - vmin) / (vmax - vmin)
            text_color = "white" if norm_val > 0.5 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color=text_color)

    ax.set_title(
        f"Summary {cbar_label.split(' (')[0]} Heatmap — All Sequences\n"
        f"X: Avg Templated Apo Samples 1–4 | Y: Reference Model 0s",
        fontsize=13, fontweight='bold', pad=12
    )

    plt.tight_layout()
    out_path = f"{OUT_DIR}/summary_{mode}_heatmap_all_seqs.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

print("\nAll done!")
