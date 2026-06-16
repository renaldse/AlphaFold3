import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser, Superimposer
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Path to CIF files
BASE = "/mnt/gs21/scratch/renaldse/AlphaFold3/outputs/ADK_lig-4c021a9d_cofold_May20_2026_9.56AM/adk"
SEED = "seed-2063740159"
SAMPLES = [0, 1, 2, 3, 4]

def load_ca_atoms(path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", path)
    ca_atoms = []
    for residue in structure[0]["A"].get_residues():
        if "CA" in residue:
            ca_atoms.append(residue["CA"])
    return structure, ca_atoms

# load reference (sample-0)
ref_path = f"{BASE}/{SEED}_sample-0/model.cif"
ref_struct, ref_ca = load_ca_atoms(ref_path)
print(f"Reference has {len(ref_ca)} Ca atoms")

# Compare each sample
samples = []
ca_rmsd = []

for i in SAMPLES:
    path = f"{BASE}/{SEED}_sample-{i}/model.cif"
    mob_struct, mob_ca = load_ca_atoms(path)
    sup = Superimposer()
    sup.set_atoms(ref_ca, mob_ca)
    rmsd = round(sup.rms, 3)
    print(f"sample-{i}: Ca RMSD = {rmsd} Angstroms")
    samples.append(f"sample-{i}")
    ca_rmsd.append(rmsd)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(samples, ca_rmsd, color='steelblue', label='Ca RMSD')
ax.set_xlabel('ADK SAMPLE')
ax.set_ylabel('RMSD (Angstroms)')
ax.set_title('ADK - Ca RMSD vs sample-0 (Cofolded)')
ax.legend()
plt.tight_layout()
plt.savefig('/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/adk_cofolded_rmsd.png', dpi=150)
plt.close()
print("Saved adk_cofolded_rmsd.png!")
