import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Sequence config — (seqname, folder_date, subfolder, seed1, seed2) ─────────
SEQUENCES = [
    ("seq1",  "seq1_apo_May26_2026_10.52AM",  "seq1",  "seed-1898484009", "seed-1898484010"),
    ("seq2",  "seq2_apo_May26_2026_10.52AM",  "seq2",  "seed-1747042453", "seed-1747042454"),
    ("seq3",  "seq3_apo_May26_2026_10.52AM",  "seq3",  "seed-522645511",  "seed-522645512"),
    ("seq5",  "seq5_apo_May26_2026_10.53AM",  "seq5",  "seed-1984255282", "seed-1984255283"),
    ("seq6",  "seq6_apo_May26_2026_10.53AM",  "seq6",  "seed-1867252878", "seed-1867252879"),
    ("seq7",  "seq7_apo_May26_2026_10.53AM",  "seq7",  "seed-407581728",  "seed-407581729"),
    ("seq9",  "seq9_apo_May26_2026_10.55AM",  "seq9",  "seed-2146638107", "seed-2146638108"),
    ("seq10", "seq10_apo_May26_2026_10.55AM", "seq10", "seed-1092575053", "seed-1092575054"),
    ("seq11", "seq11_apo_May26_2026_10.55AM", "seq11", "seed-907571167",  "seed-907571168"),
    ("seq12", "seq12_apo_May26_2026_10.55AM", "seq12", "seed-789659239",  "seed-789659240"),
    ("seq13", "seq13_apo_May26_2026_10.55AM", "seq13", "seed-1477840625", "seed-1477840626"),
    ("seq14", "seq14_apo_May26_2026_10.55AM", "seq14", "seed-1181915990", "seed-1181915991"),
    ("seq15", "seq15_apo_May26_2026_11.05AM", "seq15", "seed-829795272",  "seed-829795273"),
    ("seq16", "seq16_apo_May26_2026_11.05AM", "seq16", "seed-679279232",  "seed-679279233"),
    ("seq17", "seq17_apo_May26_2026_11.15AM", "seq17", "seed-1601948394", "seed-1601948395"),
]

OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
OUT_DIR = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo"
SAMPLES = [0, 1, 2, 3, 4]

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

def plot_bargraph(rmsd1, rmsd2, seed1, seed2, ylabel, title, out_path):
    x = np.arange(len(SAMPLES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width/2, rmsd1, width, color='steelblue',
                   label=f'Seed 1 ({seed1[-10:]}) vs own sample-0', zorder=3)
    bars2 = ax.bar(x + width/2, rmsd2, width, color='coral',
                   label=f'Seed 2 ({seed2[-10:]}) vs own sample-0', zorder=3)

    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='steelblue')
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='coral')

    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'sample-{i}' for i in SAMPLES], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.annotate('Reference\n(0 Å)', xy=(0, 0.01), ha='center',
                fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

# ── Main loop ─────────────────────────────────────────────────────────────────
for seqname, folder, subfolder, seed1, seed2 in SEQUENCES:
    print(f"\n{'='*50}")
    print(f"Processing {seqname}...")
    base = f"{OUTPUTS_BASE}/{folder}/{subfolder}"

    # ── Cα RMSD ───────────────────────────────────────────────────────────────
    ref1_ca = get_ca_atoms(f"{base}/{seed1}_sample-0/model.cif")
    ref2_ca = get_ca_atoms(f"{base}/{seed2}_sample-0/model.cif")

    rmsd1_ca = [0.0]
    rmsd2_ca = [0.0]
    for i in [1, 2, 3, 4]:
        rmsd1_ca.append(ca_rmsd(ref1_ca, get_ca_atoms(f"{base}/{seed1}_sample-{i}/model.cif")))
        rmsd2_ca.append(ca_rmsd(ref2_ca, get_ca_atoms(f"{base}/{seed2}_sample-{i}/model.cif")))

    plot_bargraph(
        rmsd1_ca, rmsd2_ca, seed1, seed2,
        ylabel="Cα RMSD vs own Sample-0 (Å)",
        title=f"{seqname.upper()} Apo — Cα RMSD vs Own Sample-0\n(Sample-0 = reference, RMSD = 0 Å for both seeds)",
        out_path=f"{OUT_DIR}/{seqname}_apo_calpha_bargraph.png"
    )

    # ── All-Atom RMSD ─────────────────────────────────────────────────────────
    ref1_aa = get_heavy_atoms(f"{base}/{seed1}_sample-0/model.cif")
    ref2_aa = get_heavy_atoms(f"{base}/{seed2}_sample-0/model.cif")

    rmsd1_aa = [0.0]
    rmsd2_aa = [0.0]
    for i in [1, 2, 3, 4]:
        rmsd1_aa.append(allatom_rmsd(ref1_aa, get_heavy_atoms(f"{base}/{seed1}_sample-{i}/model.cif")))
        rmsd2_aa.append(allatom_rmsd(ref2_aa, get_heavy_atoms(f"{base}/{seed2}_sample-{i}/model.cif")))

    plot_bargraph(
        rmsd1_aa, rmsd2_aa, seed1, seed2,
        ylabel="All-Atom RMSD vs own Sample-0 (Å)",
        title=f"{seqname.upper()} Apo — All-Atom RMSD vs Own Sample-0\n(Sample-0 = reference, RMSD = 0 Å for both seeds)",
        out_path=f"{OUT_DIR}/{seqname}_apo_allatom_bargraph.png"
    )

print("\nAll done!")
