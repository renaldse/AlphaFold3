import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
OUT_DIR      = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/apo+temp_apo"
SAMPLES      = [1, 2, 3, 4]  # exclude sample-0 (reference)

HEAVY_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
               "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
               "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
               "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"}

# ── Sequence config ────────────────────────────────────────────────────────────
# (seqname, apo_folder, seed1, seed2, temp1_folder, temp2_folder)
# subfolder is always == seqname
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_structure(path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("model", path)

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

def compute_aa_rmsds(base, seed, samples):
    """RMSDs for each sample vs sample-0 of that same run."""
    ref = get_heavy_atoms(f"{base}/{seed}_sample-0/model.cif")
    return np.array([
        allatom_rmsd(ref, get_heavy_atoms(f"{base}/{seed}_sample-{i}/model.cif"))
        for i in samples
    ])

# ── Main loop ──────────────────────────────────────────────────────────────────
seq_labels = []
b1_means, b1_stds = [], []   # Apo Seed1 - Templated(Seed1 m0) Seed1  [lightgreen - lightskyblue]
b2_means, b2_stds = [], []   # Apo Seed1 - Templated(Seed1 m0) Seed2  [lightgreen - dodgerblue]
b3_means, b3_stds = [], []   # Apo Seed2 - Templated(Seed2 m0) Seed1  [darkgreen  - lightyellow]
b4_means, b4_stds = [], []   # Apo Seed2 - Templated(Seed2 m0) Seed2  [darkgreen  - gold]

for seqname, apo_folder, seed1, seed2, temp1_folder, temp2_folder in SEQUENCES:
    print(f"\n{'='*50}")
    print(f"Processing {seqname}...")

    apo_base   = f"{OUTPUTS_BASE}/{apo_folder}/{seqname}"
    temp1_base = f"{OUTPUTS_BASE}/{temp1_folder}/{seqname}"
    temp2_base = f"{OUTPUTS_BASE}/{temp2_folder}/{seqname}"

    try:
        print("  Computing all-atom RMSDs...")
        aa_apo1     = compute_aa_rmsds(apo_base,   seed1, SAMPLES)
        aa_apo2     = compute_aa_rmsds(apo_base,   seed2, SAMPLES)
        aa_temp1_s1 = compute_aa_rmsds(temp1_base, seed1, SAMPLES)
        aa_temp1_s2 = compute_aa_rmsds(temp1_base, seed2, SAMPLES)
        aa_temp2_s1 = compute_aa_rmsds(temp2_base, seed1, SAMPLES)
        aa_temp2_s2 = compute_aa_rmsds(temp2_base, seed2, SAMPLES)

        d1 = aa_apo1 - aa_temp1_s1
        d2 = aa_apo1 - aa_temp1_s2
        d3 = aa_apo2 - aa_temp2_s1
        d4 = aa_apo2 - aa_temp2_s2

        seq_labels.append(seqname)
        b1_means.append(np.nanmean(d1)); b1_stds.append(np.nanstd(d1))
        b2_means.append(np.nanmean(d2)); b2_stds.append(np.nanstd(d2))
        b3_means.append(np.nanmean(d3)); b3_stds.append(np.nanstd(d3))
        b4_means.append(np.nanmean(d4)); b4_stds.append(np.nanstd(d4))

        print(f"  {seqname} done.")
    except Exception as e:
        print(f"  {seqname} SKIPPED — {e}")

# ── Plot ───────────────────────────────────────────────────────────────────────
n_seq   = len(seq_labels)
width   = 0.18
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
x       = np.arange(n_seq)

fig, ax = plt.subplots(figsize=(max(16, n_seq * 1.3), 7))

bar_specs = [
    (b1_means, b1_stds, 'lightgreen',  'Apo Seed1 − Templated (Seed1 m0) Seed1'),
    (b2_means, b2_stds, 'dodgerblue',  'Apo Seed1 − Templated (Seed1 m0) Seed2'),
    (b3_means, b3_stds, 'darkgreen',   'Apo Seed2 − Templated (Seed2 m0) Seed1'),
    (b4_means, b4_stds, 'gold',        'Apo Seed2 − Templated (Seed2 m0) Seed2'),
]

for (means, stds, color, label), offset in zip(bar_specs, offsets):
    ax.bar(
        x + offset, means, width,
        color=color, edgecolor='black', linewidth=0.5,
        label=label, zorder=3,
        yerr=stds, capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='black', capthick=1.2)
    )

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)
ax.set_xlabel('Sequence', fontsize=12)
ax.set_ylabel('Mean Δ All-Atom RMSD (Å)\n(Apo − Templated)', fontsize=12)
ax.set_title(
    'All-Atom RMSD Difference: Apo vs Templated\n'
    'Mean across samples 1–4 ± std dev  |  Positive = Apo more variable than Templated',
    fontsize=13, fontweight='bold'
)
ax.set_xticks(x)
ax.set_xticklabels(seq_labels, fontsize=10, rotation=45, ha='right')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3, zorder=1)

plt.tight_layout()
out_path = f"{OUT_DIR}/summary_allatom_rmsd_diff_bargraph.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSummary graph saved → {out_path}")
print("\nAll done!")
