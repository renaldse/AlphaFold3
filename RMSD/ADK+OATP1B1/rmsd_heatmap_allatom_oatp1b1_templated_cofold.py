import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs/OATP1B1_E3S_templated_cofold_May21_2026_11.14AM/oatp1b1"
SEED = "seed-1655790885"
STRUCT_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures"

COL_LABELS = ["Templated\nSample-1", "Templated\nSample-2",
               "Templated\nSample-3", "Templated\nSample-4"]
COL_PATHS  = [f"{BASE}/{SEED}_sample-{i}/model.cif" for i in range(1, 5)]

ROW_LABELS = ["Sample-0", "OATP1B1\nOutward", "OATP1B1\nInward"]
ROW_PATHS  = [
    f"{BASE}/{SEED}_sample-0/model.cif",
    f"{STRUCT_BASE}/OATP1B1_outward.cif",
    f"{STRUCT_BASE}/OATP1B1_inward.cif",
]

# Backbone + side chain heavy atoms to include
HEAVY_ATOMS = {"N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1",
               "CD2", "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
               "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
               "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
               "SD", "SG"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_structure(path):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("model", path)

def get_chain(structure, chain_id="A"):
    model = structure[0]
    if chain_id in model:
        return model[chain_id]
    return next(iter(model.get_chains()))

def get_heavy_atoms_by_resnum(chain):
    """Return dict of {res_num: [atoms]} for all backbone+sidechain heavy atoms."""
    res_atoms = {}
    for residue in chain.get_residues():
        resnum = residue.get_id()[1]
        atoms = [a for a in residue.get_atoms() if a.get_name() in HEAVY_ATOMS]
        if atoms:
            res_atoms[resnum] = atoms
    return res_atoms

def align_and_rmsd_allatom(ref_path, mob_path):
    """Align by residue sequence number, then compute all-atom RMSD on shared residues."""
    ref_struct = load_structure(ref_path)
    mob_struct = load_structure(mob_path)

    ref_chain = get_chain(ref_struct)
    mob_chain = get_chain(mob_struct)

    ref_res = get_heavy_atoms_by_resnum(ref_chain)
    mob_res = get_heavy_atoms_by_resnum(mob_chain)

    common_resnums = sorted(set(ref_res.keys()) & set(mob_res.keys()))

    if len(common_resnums) < 10:
        print(f"  WARNING: only {len(common_resnums)} common residues")
        return float("nan"), 0

    # Build matched atom lists — only atoms present in both residues
    ref_atoms = []
    mob_atoms = []
    for resnum in common_resnums:
        ref_atom_names = {a.get_name(): a for a in ref_res[resnum]}
        mob_atom_names = {a.get_name(): a for a in mob_res[resnum]}
        shared_names = sorted(set(ref_atom_names.keys()) & set(mob_atom_names.keys()))
        for name in shared_names:
            ref_atoms.append(ref_atom_names[name])
            mob_atoms.append(mob_atom_names[name])

    if len(ref_atoms) < 4:
        return float("nan"), 0

    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    return round(sup.rms, 3), len(ref_atoms)

# ── Compute RMSD matrix ────────────────────────────────────────────────────────
print("Computing all-atom RMSD (backbone + side chains) …\n")

rmsd_matrix = np.zeros((len(ROW_LABELS), len(COL_LABELS)))

for r, (rlabel, rpath) in enumerate(zip(ROW_LABELS, ROW_PATHS)):
    for c, (clabel, cpath) in enumerate(zip(COL_LABELS, COL_PATHS)):
        val, n_atoms = align_and_rmsd_allatom(rpath, cpath)
        rmsd_matrix[r, c] = val
        print(f"  {rlabel.replace(chr(10),' ')} vs {clabel.replace(chr(10),' ')}: {val} Å  ({n_atoms} atoms)")

print("\nRMSD matrix computed successfully.")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
vmin, vmax = 0, max(rmsd_matrix.max(), 4.0)
im = ax.imshow(rmsd_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("All-Atom RMSD (Å)", fontsize=11)

ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, fontsize=10)
ax.set_yticks(range(len(ROW_LABELS)))
ax.set_yticklabels(ROW_LABELS, fontsize=11)

ax.set_xlabel("Templated Cofolded Models", fontsize=12, labelpad=8)
ax.set_ylabel("Reference Structures", fontsize=12, labelpad=8)
ax.set_title("OATP1B1 All-Atom RMSD Heatmap\n(Backbone + Side Chains, Templated Cofolded Samples vs References)",
             fontsize=13, fontweight="bold", pad=12)

for r in range(len(ROW_LABELS)):
    for c in range(len(COL_LABELS)):
        val = rmsd_matrix[r, c]
        norm_val = (val - vmin) / (vmax - vmin)
        text_color = "white" if norm_val > 0.5 else "black"
        ax.text(c, r, f"{val:.3f} Å", ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

plt.tight_layout()
out_path = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/rmsd_heatmap_allatom_oatp1b1_templated_cofold.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved heatmap → {out_path}")
