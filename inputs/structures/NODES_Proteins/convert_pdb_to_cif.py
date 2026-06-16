"""
Convert PDB cluster representative files to properly formatted mmCIF for AF3.
Run this on HPCC after uploading the zip file.
"""
import os
import zipfile
from Bio.PDB import PDBParser, MMCIFIO

ZIP_PATH   = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins/seq17.zip"
OUT_DIR    = "/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins"
NUM_REPRS  = 10  # repr_0 through repr_9

os.makedirs(OUT_DIR, exist_ok=True)

parser = PDBParser(QUIET=True)
io     = MMCIFIO()

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for i in range(NUM_REPRS):
        pdb_name = f"seq17_cluster_repr_{i}.pdb"
        cif_name = f"seq17_cluster_repr_{i}.cif"
        out_path = os.path.join(OUT_DIR, cif_name)

        print(f"Converting {pdb_name} → {cif_name}...")

        tmp_path = f"/tmp/{pdb_name}"
        with zf.open(pdb_name) as f_in, open(tmp_path, 'wb') as f_out:
            f_out.write(f_in.read())

        structure = parser.get_structure(f"cluster_repr_{i}", tmp_path)
        io.set_structure(structure)
        io.save(out_path)
        os.remove(tmp_path)
        print(f"  Saved → {out_path}")

print("\nAll done! Verify with:")
print(f"  grep '^ATOM' {OUT_DIR}/seq17_cluster_repr_0.cif | head -3")
