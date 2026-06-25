"""
mantel_af3_top_confidence.py
----------------------------
Mantel test for AF3 template bias using the jwcarr/mantel package.

For each cluster, reads ranking_score from summary_confidences.json in
each seed-*_model_0 directory, selects the highest-scoring seed, and
uses that model.cif for a 10x10 pairwise RMSD matrix against the
template matrix.

Tail: upper (testing for positive correlation — template similarity
predicts output similarity). Note: two-tail would test for any
correlation regardless of direction; upper is appropriate here since
the hypothesis is that templates bias AF3 outputs toward similar
structures.

Run:
    python mantel_af3_top_confidence.py
    python mantel_af3_top_confidence.py --seqs seq15 seq13 seq5
    python mantel_af3_top_confidence.py --mode ca
    python mantel_af3_top_confidence.py --seqs seq5 --mode allatom
"""

import argparse
import glob
import itertools
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import mantel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Bio.PDB import MMCIFParser, Superimposer
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

# ---------------------------------------------------------------------------
# Datetime tag (generated once at script start)
# ---------------------------------------------------------------------------
def make_datetime_tag() -> str:
    now = datetime.now()
    return now.strftime("%b%-d_%y_%-I-%M%p").lower()  # e.g. jun18_26_9-33am

DATETIME_TAG = make_datetime_tag()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
OUTPUTS_DIR = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/outputs")
NODES_DIR   = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins")
RESULTS_DIR = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/jun18")

N_CLUSTERS     = 10
N_PERMUTATIONS = 9999

OUTPUT_SUBDIR = "june18"   # <-- update for each new batch

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
        ca_f = [a for a in atoms_fixed if a.name == "CA"]
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
    pattern = str(OUTPUTS_DIR / f"{OUTPUT_SUBDIR}/{seq}_cluster{cluster_idx}_templated*")
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

def find_seed_dirs(seq: str, cluster_idx: int) -> list[Path]:
    """
    Find all seed-*_model_0 dirs. New batch always has an inner subfolder
    seq#_cluster#/ — check there first, then fall back to direct.
    """
    out_dir = find_cluster_output_dir(seq, cluster_idx)
    inner   = out_dir / f"{seq}_cluster{cluster_idx}"

    # prefer inner subfolder (consistent in new batch)
    cif_glob = sorted(glob.glob(str(inner / "seed-*_model_0" / "model.cif")))
    if not cif_glob:
        # fallback: seeds directly in outer folder (old batch structure)
        cif_glob = sorted(glob.glob(str(out_dir / "seed-*_model_0" / "model.cif")))
    if not cif_glob:
        raise FileNotFoundError(
            f"No seed-*_model_0/model.cif found under {out_dir}")

    return [Path(p).parent for p in cif_glob]

def get_ranking_score(seed_dir: Path) -> float:
    """Read ranking_score from summary_confidences.json inside a seed-*_model_0 dir."""
    json_path = seed_dir / "summary_confidences.json"
    if not json_path.exists():
        return -np.inf
    with open(json_path) as f:
        data = json.load(f)
    return float(data.get("ranking_score", -np.inf))

def select_best_seed(seq: str, cluster_idx: int) -> tuple[Path, int, float]:
    """
    Find all seed-*_model_0 dirs sorted alphabetically (= seed rank order).
    Pick the one with the highest ranking_score.
    Returns (model.cif path, seed_rank 1-indexed, score).
    """
    seed_dirs = find_seed_dirs(seq, cluster_idx)
    ranked    = [(sd, rank + 1, get_ranking_score(sd))
                 for rank, sd in enumerate(seed_dirs)]
    best      = max(ranked, key=lambda x: x[2])
    best_dir, best_rank, best_score = best

    print(f"    cluster {cluster_idx}: seed {best_rank} ({best_dir.name})  "
          f"ranking_score = {best_score:.4f}")
    return best_dir / "model.cif", best_rank, best_score

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

def build_sample_matrix(mode: str, sample_atoms: list) -> np.ndarray:
    """10x10: cell (i,j) = RMSD between best-seed model_0 of cluster i vs j."""
    print(f"  Building sample matrix ({mode}) ...")
    mat = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i, j in itertools.combinations(range(N_CLUSTERS), 2):
        ai, aj = sample_atoms[i], sample_atoms[j]
        n = min(len(ai), len(aj))
        if n < 3:
            mat[i, j] = mat[j, i] = np.nan
            continue
        rmsd = compute_rmsd(ai[:n], aj[:n], mode)
        mat[i, j] = mat[j, i] = rmsd
        print(f"    cluster {i} vs {j}: {rmsd:.3f} Å")
    return mat

# ---------------------------------------------------------------------------
# Mantel test (jwcarr/mantel package, upper tail)
# ---------------------------------------------------------------------------
def run_mantel(tmpl_mat: np.ndarray, samp_mat: np.ndarray) -> tuple[float, float, float]:
    result = mantel.test(
        tmpl_mat, samp_mat,
        perms=N_PERMUTATIONS,
        method="pearson",
        tail="upper",
        ignore_nans=True
    )
    return float(result.r), float(result.p), float(result.z)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_matrices(tmpl_mat, samp_mat, seq, mode, r, p, z,
                  best_ranks: list[int], best_scores: list[float], out_path: Path):

    tick_labels = [f"C{i} (S{best_ranks[i]})" for i in range(N_CLUSTERS)]

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"{seq.upper()} — Mantel Test (top-confidence model_0, 10x10)\n"
        f"{mode.upper()} RMSD  |  r = {r:.4f}   p = {p:.4f}   z = {z:.4f}"
        f"   (n_perm={N_PERMUTATIONS}, upper tail)",
        fontsize=10, fontweight="bold"
    )

    gs = gridspec.GridSpec(1, 2, wspace=0.4, left=0.06, right=0.78)
    vmax = np.nanmax([np.nanmax(tmpl_mat), np.nanmax(samp_mat)])

    for ax_idx, (mat, title) in enumerate([
        (tmpl_mat, "Template pairwise RMSD (Å)"),
        (samp_mat, "Sample RMSD — top-confidence model_0 (Å)")
    ]):
        ax   = fig.add_subplot(gs[ax_idx])
        im   = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
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
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Cluster (seed used)", fontsize=8)
        ax.set_ylabel("Cluster (seed used)", fontsize=8)

        ax.set_xticks(np.arange(-0.5, N_CLUSTERS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_CLUSTERS, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # key panel
    key_ax = fig.add_axes([0.80, 0.08, 0.19, 0.84])
    key_ax.axis("off")
    key_lines = ["Key", ""]
    key_lines += [f"C{i} = Cluster {i}" for i in range(N_CLUSTERS)]
    key_lines += ["", "S1 = seed rank 1", "  (highest ranking_score)",
                  "S2 = seed rank 2"]
    key_lines += ["", "Selected seed per cluster:"]
    key_lines += [f"  C{i}: S{best_ranks[i]}  ({best_scores[i]:.3f})"
                  for i in range(N_CLUSTERS)]
    key_lines += ["", "Tail: upper", "  (positive correlation only)"]

    key_ax.text(0.0, 1.0, "\n".join(key_lines),
                transform=key_ax.transAxes,
                fontsize=7, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke",
                          edgecolor="gray", linewidth=0.8))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(seq: str, mode: str):
    seq = seq.lower()
    print(f"\n{'='*60}")
    print(f"  Sequence: {seq.upper()} | Mode: {mode.upper()}")
    print(f"  Datetime tag: {DATETIME_TAG}")
    print(f"{'='*60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading template structures ...")
    tmpl_atoms = []
    for i in range(N_CLUSTERS):
        s = load_structure(get_template_cif(seq, i), f"tmpl_{i}")
        tmpl_atoms.append(get_atoms(s, mode))
    print(f"  Loaded {N_CLUSTERS} templates")

    print("\n[2/4] Selecting top-confidence model_0 per cluster ...")
    sample_atoms = []
    best_ranks   = []
    best_scores  = []
    for ci in range(N_CLUSTERS):
        cif, seed_rank, score = select_best_seed(seq, ci)
        s = load_structure(cif, f"{seq}_c{ci}_s{seed_rank}")
        sample_atoms.append(get_atoms(s, mode))
        best_ranks.append(seed_rank)
        best_scores.append(score)

    print("\n[3/4] Computing RMSD matrices ...")
    tmpl_mat = build_template_matrix(seq, mode, tmpl_atoms)
    samp_mat = build_sample_matrix(mode, sample_atoms)

    print("\n[4/4] Running Mantel test (upper tail, Pearson) ...")
    r, p, z = run_mantel(tmpl_mat, samp_mat)
    print(f"\n  r = {r:.4f}   p = {p:.4f}   z = {z:.4f}")

    stem = f"mantel_{seq}_{mode}_{DATETIME_TAG}"

    results_path = RESULTS_DIR / f"{stem}_results.txt"
    with open(results_path, "w") as f:
        f.write("Mantel Test Results — top-confidence model_0 (10x10)\n")
        f.write(f"Sequence     : {seq.upper()}\n")
        f.write(f"Mode         : {mode.upper()}\n")
        f.write(f"Permutations : {N_PERMUTATIONS}\n")
        f.write(f"Tail         : upper\n")
        f.write(f"Method       : pearson\n\n")
        f.write(f"r = {r:.6f}\n")
        f.write(f"p = {p:.6f}\n")
        f.write(f"z = {z:.6f}\n\n")
        f.write("Best seed per cluster (S = seed rank in sorted order):\n")
        for ci, (rank, score) in enumerate(zip(best_ranks, best_scores)):
            f.write(f"  cluster {ci}: S{rank}  ranking_score = {score:.4f}\n")
        f.write("\nTemplate RMSD matrix (Å):\n")
        f.write(np.array2string(tmpl_mat, precision=3) + "\n\n")
        f.write("Sample RMSD matrix (Å):\n")
        f.write(np.array2string(samp_mat, precision=3) + "\n")

    plot_matrices(tmpl_mat, samp_mat, seq, mode, r, p, z,
                  best_ranks, best_scores,
                  RESULTS_DIR / f"{stem}.png")

    print(f"  Results saved to: {results_path}")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mantel test using top-confidence model_0 per cluster")
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQS,
                        help="Sequence names to process (default: all seqs)")
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
