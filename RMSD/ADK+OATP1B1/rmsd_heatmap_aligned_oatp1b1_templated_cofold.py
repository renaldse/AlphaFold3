import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer, CEAligner
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_structure(path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", path)
    return structure

def get_chain(structure, chain_id="A"):
    model = structure[0]
    if chain_id in model:
        return model[chain_id]
    return next(iter(model.get_chains()))

def ce_align_rmsd(ref_struct, mob_struct):
    """Use CEAligner for structure-based alignment — handles different lengths."""
    aligner = CEAligner()
    aligner.set_reference(ref_struct)
    aligner.align(mob_struct)
    return round(aligner.rms, 3)

# ── Load structures ────────────────────────────────────────────────────────────
print("Loading structures …\n")

for label, path in zip(ROW_LABELS, ROW_PATHS):
    s = load_structure(path)
    chain = get_chain(s)
    ca = [r for r in chain.get_residues() if "CA" in r]
    print(f"  {label.replace(chr(10),' ')}: {len(ca)} Cα atoms  ({path})")

for label, path in zip(COL_LABELS, COL_PATHS):
    s = load_structure(path)
    chain = get_chain(s)
    ca = [r for r in chain.get_residues() if "CA" in r]
    print(f"  {label.replace(chr(10),' ')}: {len(ca)} Cα atoms  ({path})")

print()
rmsd_matrix = np.zeros((len(ROW_LABELS), len(COL_LABELS)))

for r, rlabel in enumerate(ROW_LABELS):
    for c, clabel in enumerate(COL_LABELS):
        # Reload fresh each time — CEAligner modifies coordinates in place
        rs = load_structure(ROW_PATHS[r])
        cs = load_structure(COL_PATHS[c])
        val = ce_align_rmsd(rs, cs)
        rmsd_matrix[r, c] = val
        print(f"  {rlabel.replace(chr(10),' ')} vs {clabel.replace(chr(10),' ')}: {val} Å  (CE-aligned)")

print("\nRMSD matrix computed successfully.")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
vmin, vmax = 0, max(rmsd_matrix.max(), 4.0)
im = ax.imshow(rmsd_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Cα RMSD (Å)", fontsize=11)

ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, fontsize=10)
ax.set_yticks(range(len(ROW_LABELS)))
ax.set_yticklabels(ROW_LABELS, fontsize=11)

ax.set_xlabel("Templated Cofolded Models", fontsize=12, labelpad=8)
ax.set_ylabel("Reference Structures", fontsize=12, labelpad=8)
ax.set_title("OATP1B1 Cα RMSD Heatmap — CE-Aligned\n(Templated Cofolded Samples vs References)",
             fontsize=13, fontweight="bold", pad=12)

for r in range(len(ROW_LABELS)):
    for c in range(len(COL_LABELS)):
        val = rmsd_matrix[r, c]
        norm_val = (val - vmin) / (vmax - vmin)
        text_color = "white" if norm_val > 0.5 else "black"
        ax.text(c, r, f"{val:.3f} Å", ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

plt.tight_layout()
out_path = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/rmsd_heatmap_aligned_oatp1b1_templated_cofold.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved heatmap → {out_path}")
