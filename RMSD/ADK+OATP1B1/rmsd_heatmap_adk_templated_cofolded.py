import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
TEMPLATED_BASE = (
    "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs/ADK_ATP/"
    "adk_20260519_152943/seed-1346664333_sample-{i}/model.cif"
)
PDB_BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs/PDB files"

# Columns  – templated cofolded samples 1–4
COL_LABELS = ["Templated\nSample-1", "Templated\nSample-2",
               "Templated\nSample-3", "Templated\nSample-4"]
COL_PATHS  = [TEMPLATED_BASE.format(i=i) for i in range(1, 5)]

# Rows – sample-0, 1AKE, 4AKE
ROW_LABELS = ["Sample-0", "1AKE", "4AKE"]
ROW_PATHS  = [
    TEMPLATED_BASE.format(i=0),
    f"{PDB_BASE}/1AKE.cif",
    f"{PDB_BASE}/4AKE.cif",
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_ca_atoms(path, chain_id="A"):
    """Return (structure, [CA atoms]) for chain chain_id."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", path)
    ca_atoms = []
    # Try requested chain first, fall back to first available chain
    model = structure[0]
    if chain_id in model:
        chain = model[chain_id]
    else:
        chain = next(iter(model.get_chains()))
        print(f"  Chain '{chain_id}' not found in {path}; using '{chain.id}'")
    for residue in chain.get_residues():
        if "CA" in residue:
            ca_atoms.append(residue["CA"])
    return structure, ca_atoms


def compute_rmsd(ref_ca, mob_ca):
    """Superimpose mob onto ref and return RMSD. Trims to shared length."""
    n = min(len(ref_ca), len(mob_ca))
    if n == 0:
        return float("nan")
    sup = Superimposer()
    sup.set_atoms(ref_ca[:n], mob_ca[:n])
    return round(sup.rms, 3)


# ── Compute RMSD matrix  (rows × cols) ────────────────────────────────────────
print("Loading structures and computing RMSDs …\n")

# Pre-load all structures
row_cas = []
for label, path in zip(ROW_LABELS, ROW_PATHS):
    _, ca = load_ca_atoms(path)
    row_cas.append(ca)
    print(f"  {label}: {len(ca)} Cα atoms  ({path})")

col_cas = []
for label, path in zip(COL_LABELS, COL_PATHS):
    _, ca = load_ca_atoms(path)
    col_cas.append(ca)
    print(f"  {label.replace(chr(10),' ')}: {len(ca)} Cα atoms  ({path})")

print()
rmsd_matrix = np.zeros((len(ROW_LABELS), len(COL_LABELS)))

for r, (rlabel, rca) in enumerate(zip(ROW_LABELS, row_cas)):
    for c, (clabel, cca) in enumerate(zip(COL_LABELS, col_cas)):
        val = compute_rmsd(rca, cca)
        rmsd_matrix[r, c] = val
        print(f"  {rlabel} vs {clabel.replace(chr(10),' ')}: {val} Å")

print("\nRMSD matrix computed successfully.")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])  # low RMSD = white, high = red
vmin, vmax = 0, max(rmsd_matrix.max(), 4.0)   # floor ceiling at 4 Å minimum
im = ax.imshow(rmsd_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

# Colour bar
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Cα RMSD (Å)", fontsize=11)

# Axis labels
ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, fontsize=10)
ax.set_yticks(range(len(ROW_LABELS)))
ax.set_yticklabels(ROW_LABELS, fontsize=11)

ax.set_xlabel("Templated Cofolded Models", fontsize=12, labelpad=8)
ax.set_ylabel("Reference Structures", fontsize=12, labelpad=8)
ax.set_title("ADK Cα RMSD Heatmap\n(Templated Cofolded Samples vs References)",
             fontsize=13, fontweight="bold", pad=12)

# Annotate each cell with the RMSD value
for r in range(len(ROW_LABELS)):
    for c in range(len(COL_LABELS)):
        val = rmsd_matrix[r, c]
        # Choose white text on dark cells, black on light
        norm_val = (val - vmin) / (vmax - vmin)
        text_color = "white" if norm_val > 0.5 else "black"
        ax.text(c, r, f"{val:.3f} Å", ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

plt.tight_layout()
out_path = "/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/rmsd_heatmap_adk.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved heatmap → {out_path}")
