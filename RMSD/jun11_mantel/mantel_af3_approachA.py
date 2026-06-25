"""
mantel_af3_approachA.py
-----------------------
Approach A: 10x10 Mantel test.

Template matrix : 10x10 pairwise RMSD between cluster representative CIFs.
Sample matrix   : 10x10 where cell (i,j) = mean RMSD across all pairwise
                  comparisons of the 10 samples from cluster i vs cluster j.

Run:
    python mantel_af3_approachA.py
    python mantel_af3_approachA.py --seqs seq15 seq13 seq5
    python mantel_af3_approachA.py --mode ca
    python mantel_af3_approachA.py --seqs seq5 --mode allatom
"""

import argparse
import glob
import itertools
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

from Bio.PDB import MMCIFParser, Superimposer
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent.resolve()
OUTPUTS_DIR = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/outputs")
NODES_DIR   = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins")
N_CLUSTERS          = 10
SAMPLES_PER_CLUSTER = 2    # sample-0 from each seed only
N_PERMUTATIONS      = 9999

ALL_SEQS = [
    "seq1",  "seq2",  "seq3",  "seq5",  "seq6",  "seq7",
    "seq9",  "seq10", "seq11", "seq12", "seq13", "seq14",
    "seq15", "seq16", "seq17"
]

# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------
PARSER = MMCIFParser(QUIET=True)

def load_structure(cif_path: Path, structure_id: str = "s"):
    return PARSER.get_structure(structure_id, str(cif_path))

def get_atoms(structure, mode: str):
    """Return ordered atom list from first model, first chain."""
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if mode == "ca":
                    if "CA" in residue:
                        atoms.append(residue["CA"])
                else:
                    for atom in residue.get_atoms():
                        atoms.append(atom)
            break
        break
    return atoms

def compute_rmsd(atoms_fixed: list, atoms_moving: list, mode: str) -> float:
    """Superimpose on Cα, measure RMSD on requested atom set."""
    if mode == "ca":
        fit_fixed, fit_moving = atoms_fixed, atoms_moving
        meas_fixed, meas_moving = atoms_fixed, atoms_moving
    else:
        ca_f = [a for a in atoms_fixed  if a.name == "CA"]
        ca_m = [a for a in atoms_moving if a.name == "CA"]
        n = min(len(ca_f), len(ca_m))
        fit_fixed, fit_moving = ca_f[:n], ca_m[:n]
        meas_fixed, meas_moving = atoms_fixed, atoms_moving

    n_fit = min(len(fit_fixed), len(fit_moving))
    if n_fit < 3:
        return np.nan

    sup = Superimposer()
    sup.set_atoms(fit_fixed[:n_fit], fit_moving[:n_fit])
    sup.apply(meas_moving)

    n_meas = min(len(meas_fixed), len(meas_moving))
    diff = np.array([meas_fixed[k].coord - meas_moving[k].coord
                     for k in range(n_meas)])
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------
def find_cluster_output_dir(seq: str, cluster_idx: int) -> Path:
    pattern = str(OUTPUTS_DIR / f"june11/{seq}_cluster{cluster_idx}_templated*")
    matches = sorted(glob.glob(pattern))
    if seq.lower() == "seq5":
        matches = [m for m in matches if "run2" not in m]
    if not matches:
        raise FileNotFoundError(
            f"No output folder for {seq} cluster {cluster_idx}\n  pattern: {pattern}")
    if len(matches) > 1:
        print(f"  [warn] multiple matches for {seq} cluster {cluster_idx}, "
              f"using: {matches[0]}")
    return Path(matches[0])

def collect_sample_cifs(seq: str, cluster_idx: int) -> list[Path]:
    out_dir = find_cluster_output_dir(seq, cluster_idx)
    inner   = out_dir / f"{seq}_cluster{cluster_idx}"
    cifs    = sorted(glob.glob(str(inner / "seed-*_sample-0" / "model.cif")))
    if len(cifs) != SAMPLES_PER_CLUSTER:
        print(f"  [warn] expected {SAMPLES_PER_CLUSTER} samples for "
              f"{seq} cluster {cluster_idx}, found {len(cifs)}")
    return [Path(c) for c in cifs]

def get_template_cif(seq: str, cluster_idx: int) -> Path:
    p = NODES_DIR / f"{seq}_cluster_repr_{cluster_idx}.cif"
    if not p.exists():
        raise FileNotFoundError(f"Template CIF not found: {p}")
    return p

# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------
def build_template_matrix(seq: str, mode: str, tmpl_atoms: list) -> np.ndarray:
    print(f"  Building template matrix ({mode}) ...")
    mat = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i, j in itertools.combinations(range(N_CLUSTERS), 2):
        n    = min(len(tmpl_atoms[i]), len(tmpl_atoms[j]))
        rmsd = compute_rmsd(tmpl_atoms[i][:n], tmpl_atoms[j][:n], mode)
        mat[i, j] = mat[j, i] = rmsd
        print(f"    tmpl {i} vs {j}: {rmsd:.3f} Å")
    return mat

def build_sample_matrix_A(mode: str, sample_atoms: list[list]) -> np.ndarray:
    """
    10x10: cell (i,j) = mean RMSD of all sample pairs from cluster i vs j.
    Diagonal = mean pairwise RMSD within cluster i.
    """
    print(f"  Building sample matrix A ({mode}) ...")
    mat = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i, j in itertools.combinations_with_replacement(range(N_CLUSTERS), 2):
        pairs = (itertools.combinations(range(SAMPLES_PER_CLUSTER), 2)
                 if i == j
                 else itertools.product(range(SAMPLES_PER_CLUSTER),
                                        range(SAMPLES_PER_CLUSTER)))
        rmsds = []
        for si, sj in pairs:
            ai, aj = sample_atoms[i][si], sample_atoms[j][sj]
            n = min(len(ai), len(aj))
            if n < 3:
                continue
            rmsds.append(compute_rmsd(ai[:n], aj[:n], mode))
        val = float(np.nanmean(rmsds)) if rmsds else np.nan
        mat[i, j] = mat[j, i] = val
        print(f"    cluster {i} vs {j}: {val:.3f} Å  ({len(rmsds)} pairs)")
    return mat

# ---------------------------------------------------------------------------
# Mantel test
# ---------------------------------------------------------------------------
def mantel_test(mat_x: np.ndarray, mat_y: np.ndarray,
                n_perm: int = N_PERMUTATIONS) -> tuple[float, float]:
    n   = mat_x.shape[0]
    idx = np.triu_indices(n, k=1)
    xv  = mat_x[idx]
    yv  = mat_y[idx]
    mask = ~(np.isnan(xv) | np.isnan(yv))
    xv, yv = xv[mask], yv[mask]

    obs_r, _ = pearsonr(xv, yv)

    rng      = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perm):
        perm   = rng.permutation(n)
        yp     = mat_y[np.ix_(perm, perm)][idx][mask]
        r_p, _ = pearsonr(xv, yp)
        if r_p >= obs_r:
            count_ge += 1

    return float(obs_r), float((count_ge + 1) / (n_perm + 1))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_matrices(tmpl_mat, samp_mat, seq, mode, r, p, out_path: Path):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"{seq.upper()} — Mantel Test (Approach A: 10x10)\n"
        f"{mode.upper()} RMSD  |  r = {r:.4f}  p = {p:.4f}  "
        f"(n_perm={N_PERMUTATIONS})",
        fontsize=12, fontweight="bold"
    )
    gs   = gridspec.GridSpec(1, 2, wspace=0.4)
    vmax = np.nanmax([np.nanmax(tmpl_mat), np.nanmax(samp_mat)])
    labels = [f"C{i}" for i in range(N_CLUSTERS)]

    for ax_idx, (mat, title) in enumerate([
        (tmpl_mat, "Template pairwise RMSD (Å)"),
        (samp_mat, "Sample mean pairwise RMSD (Å)")
    ]):
        ax  = fig.add_subplot(gs[ax_idx])
        im  = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("RMSD (Å)", fontsize=8)

        for i in range(N_CLUSTERS):
            for j in range(N_CLUSTERS):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val > vmax * 0.5 else "black")

        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(N_CLUSTERS))
        ax.set_yticks(range(N_CLUSTERS))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_xticks(np.arange(-0.5, N_CLUSTERS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_CLUSTERS, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seq: str, mode: str):
    seq = seq.lower()
    print(f"\n{'='*60}")
    print(f"  Approach A | Sequence: {seq.upper()} | Mode: {mode.upper()}")
    print(f"{'='*60}")

    print("\n[1/4] Loading template structures ...")
    tmpl_atoms = []
    for i in range(N_CLUSTERS):
        s = load_structure(get_template_cif(seq, i), f"tmpl_{i}")
        tmpl_atoms.append(get_atoms(s, mode))
    print(f"  Loaded {N_CLUSTERS} templates")

    print("\n[2/4] Loading sample structures ...")
    sample_atoms = []
    for ci in range(N_CLUSTERS):
        cifs  = collect_sample_cifs(seq, ci)
        atoms = []
        for cif in cifs:
            s = load_structure(cif, f"{seq}_c{ci}_{cif.parent.name}")
            atoms.append(get_atoms(s, mode))
        sample_atoms.append(atoms)
        print(f"  Cluster {ci}: {len(atoms)} samples")

    print("\n[3/4] Computing RMSD matrices ...")
    tmpl_mat = build_template_matrix(seq, mode, tmpl_atoms)
    samp_mat = build_sample_matrix_A(mode, sample_atoms)

    print("\n[4/4] Running Mantel test ...")
    r, p = mantel_test(tmpl_mat, samp_mat)
    print(f"\n  r = {r:.4f}   p = {p:.4f}")

    results_path = SCRIPT_DIR / f"mantel_{seq}_{mode}_approachA_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Mantel Test Results — Approach A (10x10)\n")
        f.write(f"Sequence     : {seq.upper()}\n")
        f.write(f"Mode         : {mode.upper()}\n")
        f.write(f"Permutations : {N_PERMUTATIONS}\n\n")
        f.write(f"r = {r:.6f}\n")
        f.write(f"p = {p:.6f}\n\n")
        f.write("Template RMSD matrix (Å):\n")
        f.write(np.array2string(tmpl_mat, precision=3) + "\n\n")
        f.write("Sample mean RMSD matrix (Å):\n")
        f.write(np.array2string(samp_mat, precision=3) + "\n")

    plot_matrices(tmpl_mat, samp_mat, seq, mode, r, p,
                  SCRIPT_DIR / f"mantel_{seq}_{mode}_approachA.png")

    print(f"  Results saved to: {results_path}")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mantel test (Approach A: 10x10) for AF3 template bias")
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQS,
                        help="Sequence names to process (default: all June 11 seqs)")
    parser.add_argument("--mode", default="allatom",
                        choices=["ca", "allatom"],
                        help="RMSD mode (default: allatom)")
    args = parser.parse_args()

    for seq in args.seqs:
        try:
            run(seq, args.mode)
        except Exception as e:
            print(f"\n[ERROR] {seq} failed: {e}")
            continue