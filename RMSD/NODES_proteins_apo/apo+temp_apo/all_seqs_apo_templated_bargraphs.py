import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Sequence config ────────────────────────────────────────────────────────────
SEQUENCES = [
     ("seq8",  "seq8_apo_May26_2026_6.20PM",  "seq8",  "seed-150219149", "seed-150219150",
      "seq8_templated_apo_May27_2026_10.56.55AM", "seq8_templated_apo_May27_2026_11.02.16AM"),
]

OUTPUTS_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs"
OUT_DIR = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/NODES_proteins_apo/apo+temp_apo"
SAMPLES      = [0, 1, 2, 3, 4]

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

def compute_ca_rmsds(base, seed):
    ref = get_ca_atoms(f"{base}/{seed}_sample-0/model.cif")
    return [0.0] + [ca_rmsd(ref, get_ca_atoms(f"{base}/{seed}_sample-{i}/model.cif")) for i in [1,2,3,4]]

def compute_aa_rmsds(base, seed):
    ref = get_heavy_atoms(f"{base}/{seed}_sample-0/model.cif")
    return [0.0] + [allatom_rmsd(ref, get_heavy_atoms(f"{base}/{seed}_sample-{i}/model.cif")) for i in [1,2,3,4]]

def plot_bargraph(rmsds, colors, labels, ylabel, title, out_path):
    x = np.arange(len(SAMPLES))
    width = 0.25
    offsets = [-width, 0, width]
    fig, ax = plt.subplots(figsize=(12, 6))

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
                        fontsize=8, fontweight='bold')

    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
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
    print(f"  Saved → {out_path}")

# ── Main loop ─────────────────────────────────────────────────────────────────
for seqname, apo_folder, subfolder, seed1, seed2, temp1_folder, temp2_folder in SEQUENCES:
    print(f"\n{'='*50}")
    print(f"Processing {seqname}...")

    apo_base   = f"{OUTPUTS_BASE}/{apo_folder}/{subfolder}"
    temp1_base = f"{OUTPUTS_BASE}/{temp1_folder}/{subfolder}"
    temp2_base = f"{OUTPUTS_BASE}/{temp2_folder}/{subfolder}"

    # ── Cα RMSDs ──────────────────────────────────────────────────────────────
    print("  Computing Cα RMSDs...")
    ca_apo1     = compute_ca_rmsds(apo_base,   seed1)
    ca_apo2     = compute_ca_rmsds(apo_base,   seed2)
    ca_temp1_s1 = compute_ca_rmsds(temp1_base, seed1)
    ca_temp1_s2 = compute_ca_rmsds(temp1_base, seed2)
    ca_temp2_s1 = compute_ca_rmsds(temp2_base, seed1)
    ca_temp2_s2 = compute_ca_rmsds(temp2_base, seed2)

    # ── All-Atom RMSDs ────────────────────────────────────────────────────────
    print("  Computing All-Atom RMSDs...")
    aa_apo1     = compute_aa_rmsds(apo_base,   seed1)
    aa_apo2     = compute_aa_rmsds(apo_base,   seed2)
    aa_temp1_s1 = compute_aa_rmsds(temp1_base, seed1)
    aa_temp1_s2 = compute_aa_rmsds(temp1_base, seed2)
    aa_temp2_s1 = compute_aa_rmsds(temp2_base, seed1)
    aa_temp2_s2 = compute_aa_rmsds(temp2_base, seed2)

    # Graph 1 — Apo Seed1 + Templated Seed1 + Templated Seed2 (from Seed1 m0)
    for rmsds, ylabel, suffix in [
        ([ca_apo1, ca_temp1_s1, ca_temp1_s2], "Cα RMSD vs own Sample-0 (Å)", "calpha"),
        ([aa_apo1, aa_temp1_s1, aa_temp1_s2], "All-Atom RMSD vs own Sample-0 (Å)", "allatom"),
    ]:
        plot_bargraph(
            rmsds,
            colors=['lightgreen', 'lightskyblue', 'dodgerblue'],
            labels=[
                f'Apo {seed1[-10:]} (Seed 1)',
                f'Templated (Seed1 m0) — Seed 1',
                f'Templated (Seed1 m0) — Seed 2',
            ],
            ylabel=ylabel,
            title=f"{seqname.upper()} — {ylabel.split(' vs')[0]} vs Own Sample-0\nApo Seed 1 (green) + Templated from Seed 1 Model 0 (blue)",
            out_path=f"{OUT_DIR}/{seqname}_{suffix}_seed1_bargraph.png"
        )

    # Graph 2 — Apo Seed2 + Templated Seed1 + Templated Seed2 (from Seed2 m0)
    for rmsds, ylabel, suffix in [
        ([ca_apo2, ca_temp2_s1, ca_temp2_s2], "Cα RMSD vs own Sample-0 (Å)", "calpha"),
        ([aa_apo2, aa_temp2_s1, aa_temp2_s2], "All-Atom RMSD vs own Sample-0 (Å)", "allatom"),
    ]:
        plot_bargraph(
            rmsds,
            colors=['darkgreen', 'lightyellow', 'gold'],
            labels=[
                f'Apo {seed2[-10:]} (Seed 2)',
                f'Templated (Seed2 m0) — Seed 1',
                f'Templated (Seed2 m0) — Seed 2',
            ],
            ylabel=ylabel,
            title=f"{seqname.upper()} — {ylabel.split(' vs')[0]} vs Own Sample-0\nApo Seed 2 (green) + Templated from Seed 2 Model 0 (yellow)",
            out_path=f"{OUT_DIR}/{seqname}_{suffix}_seed2_bargraph.png"
        )

print("\nAll done!")
