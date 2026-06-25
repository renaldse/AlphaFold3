"""
procrustes_af3_top_confidence.py
---------------------------------
Procrustes analysis for AF3 template bias.

For each cluster, selects the highest-ranking seed's model_0 CIF
by ranking_score in summary_confidences.json and measures the
Procrustes disparity between that output and its own template.

Statistical test: permutation test on mean self-disparity.
  - Observed stat: mean disparity of each output to its OWN template
  - Null: shuffle output-to-template assignments 9999 times
  - p-value: lower tail (bias = outputs closer to own template than chance)

"""

import argparse
import glob
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

# Datetime
def make_datetime_tag() -> str:
    now = datetime.now()
    return now.strftime("%b%-d_%y_%-I-%M%p").lower()

DATETIME_TAG = make_datetime_tag()

# Paths & constants

OUTPUTS_DIR = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/outputs")
NODES_DIR   = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/inputs/structures/NODES_Proteins")
RESULTS_DIR = Path("/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/jun18/procrustes")

N_CLUSTERS     = 10    #CHANGE ME
N_PERMUTATIONS = 9999
OUTPUT_SUBDIR  = "june18"   #CHANGE ME

ALL_SEQS = [
    "seq1",  "seq2",  "seq3",  "seq5",  "seq6",  "seq7",
    "seq9",  "seq10", "seq11", "seq12", "seq13", "seq14",
    "seq15", "seq16", "seq17"
]

# Structure helpers

PARSER = MMCIFParser(QUIET=True)

def load_structure(cif_path: Path, structure_id: str = "s"):
    return PARSER.get_structure(structure_id, str(cif_path))

def get_ca_coords(structure) -> np.ndarray:
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    coords.append(residue["CA"].coord)
            break
        break
    return np.array(coords)

def get_allatom_coords(structure) -> np.ndarray:
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                for atom in residue.get_atoms():
                    coords.append(atom.coord)
            break
        break
    return np.array(coords)

def get_coords(structure, mode: str) -> np.ndarray:
    if mode == "ca":
        return get_ca_coords(structure)
    else:
        return get_allatom_coords(structure)


# Paths

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
    out_dir  = find_cluster_output_dir(seq, cluster_idx)
    inner    = out_dir / f"{seq}_cluster{cluster_idx}"
    cif_glob = sorted(glob.glob(str(inner / "seed-*_model_0" / "model.cif")))
    if not cif_glob:
        cif_glob = sorted(glob.glob(str(out_dir / "seed-*_model_0" / "model.cif")))
    if not cif_glob:
        raise FileNotFoundError(f"No seed-*_model_0/model.cif found under {out_dir}")
    return [Path(p).parent for p in cif_glob]

def get_ranking_score(seed_dir: Path) -> float:
    json_path = seed_dir / "summary_confidences.json"
    if not json_path.exists():
        return -np.inf
    with open(json_path) as f:
        data = json.load(f)
    return float(data.get("ranking_score", -np.inf))

def select_best_seed(seq: str, cluster_idx: int) -> tuple[Path, int, float]:
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

# Procrustes disparity

def procrustes_disparity(ref_coords: np.ndarray,
                         mob_coords: np.ndarray) -> float:
    n = min(len(ref_coords), len(mob_coords))
    _, _, disparity = procrustes(ref_coords[:n], mob_coords[:n])
    return float(disparity)


# Permutation test

def permutation_test(tmpl_coords, samp_coords, n_perm=N_PERMUTATIONS):
    n = len(tmpl_coords)

    obs_disparities = []
    for i in range(n):
        d = procrustes_disparity(tmpl_coords[i], samp_coords[i])
        obs_disparities.append(d)
        print(f"    cluster {i}: self-disparity = {d:.5f}")
    obs_stat = float(np.mean(obs_disparities))

    rng        = np.random.default_rng(42)
    null_stats = []
    for _ in range(n_perm):
        perm   = rng.permutation(n)
        null_d = [procrustes_disparity(tmpl_coords[i], samp_coords[perm[i]])
                  for i in range(n)]
        null_stats.append(float(np.mean(null_d)))

    null_arr = np.array(null_stats)
    p_value  = float((np.sum(null_arr <= obs_stat) + 1) / (n_perm + 1))
    z_score  = float((obs_stat - null_arr.mean()) / null_arr.std())

    return obs_stat, p_value, z_score, obs_disparities, null_arr


def plot_self_disparity_bar(obs_disparities, null_arr, obs_stat, p, z,
                             best_ranks, seq, mode, out_path):
    """
    Bar chart of per-cluster self-disparity vs. null distribution,
    with observed stat, p-value, and z-score in the title.
    Saves figure to out_path.
    """
    n = N_CLUSTERS
    labels = [f"C{i}\n(S{best_ranks[i]})" for i in range(n)]

    null_mean = null_arr.mean()
    null_std = null_arr.std()

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        f"{seq.upper()} — Procrustes Analysis ({mode.upper()} Cα)\n"
        f"mean self-disparity = {obs_stat:.5f}   p = {p:.4f}   z = {z:.4f}"
        f"   (n_perm={N_PERMUTATIONS}, lower tail)",
        fontsize=10, fontweight="bold"
    )

    colors = ["steelblue" if d < null_mean else "tomato" for d in obs_disparities]
    bars = ax.bar(range(n), obs_disparities, color=colors, edgecolor="white",
                  linewidth=0.5, zorder=3)

    ax.axhline(null_mean, color="black", linewidth=1.2, linestyle="--",
               label=f"Null mean ({null_mean:.4f})")
    ax.axhspan(null_mean - null_std, null_mean + null_std,
               alpha=0.15, color="gray", label="Null ±1 SD")

    for bar, val in zip(bars, obs_disparities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Cluster (seed used)", fontsize=9)
    ax.set_ylabel("Procrustes disparity", fontsize=9)
    ax.set_title("Per-cluster self-disparity vs null mean", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # color legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                   label=f"Null mean ({null_mean:.4f})"),
        Patch(facecolor="gray", alpha=0.3, label="Null ±1 SD"),
        Patch(facecolor="steelblue", label="Below null mean (more biased)"),
        Patch(facecolor="tomato",    label="Above null mean (less biased)"),
    ], fontsize=7, loc="upper right")


    plt.tight_layout()
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

    print("\n[1/3] Loading template structures ...")
    tmpl_coords = []
    for i in range(N_CLUSTERS):
        s = load_structure(get_template_cif(seq, i), f"tmpl_{i}")
        tmpl_coords.append(get_coords(s, mode))
    print(f"  Loaded {N_CLUSTERS} templates")

    print("\n[2/3] Selecting top-confidence model_0 per cluster ...")
    samp_coords = []
    best_ranks  = []
    best_scores = []
    for ci in range(N_CLUSTERS):
        cif, seed_rank, score = select_best_seed(seq, ci)
        s = load_structure(cif, f"{seq}_c{ci}_s{seed_rank}")
        samp_coords.append(get_coords(s, mode))
        best_ranks.append(seed_rank)
        best_scores.append(score)

    print("\n[3/3] Running Procrustes permutation test ...")
    obs_stat, p, z, obs_disparities, null_arr = permutation_test(
        tmpl_coords, samp_coords)
    print(f"\n  mean self-disparity = {obs_stat:.5f}")
    print(f"  null mean           = {null_arr.mean():.5f}")
    print(f"  p = {p:.4f}   z = {z:.4f}")

    stem = f"procrustes_{seq}_{mode}_{DATETIME_TAG}"

    results_path = RESULTS_DIR / f"{stem}_results.txt"
    with open(results_path, "w") as f:
        f.write("Procrustes Analysis Results\n")
        f.write(f"Sequence          : {seq.upper()}\n")
        f.write(f"Mode              : {mode.upper()}\n")
        f.write(f"Permutations      : {N_PERMUTATIONS}\n")
        f.write(f"Tail              : lower\n")
        f.write(f"                    (significant = outputs closer to own\n")
        f.write(f"                     template than expected by chance)\n\n")
        f.write(f"mean self-disparity = {obs_stat:.6f}\n")
        f.write(f"null mean           = {null_arr.mean():.6f}\n")
        f.write(f"null std            = {null_arr.std():.6f}\n")
        f.write(f"p                   = {p:.6f}\n")
        f.write(f"z                   = {z:.6f}\n\n")
        f.write("Per-cluster self-disparity (output vs own template):\n")
        for ci, (d, rank, score) in enumerate(
                zip(obs_disparities, best_ranks, best_scores)):
            f.write(f"  cluster {ci}: disparity = {d:.5f}  "
                    f"seed = S{rank}  ranking_score = {score:.4f}\n")

    plot_heatmap(obs_disparities, null_arr, obs_stat, p, z,
                 best_ranks, best_scores, seq, mode,
                 RESULTS_DIR / f"{stem}_heatmap.png")

    print(f"  Results saved to: {results_path}")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procrustes analysis for AF3 template bias")
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQS,
                        help="Sequence names to process (default: all seqs)")
    parser.add_argument("--mode", default="allatom",
                        choices=["ca", "allatom"],
                        help="RMSD mode for seed selection (default: allatom)")
    args = parser.parse_args()

    for seq in args.seqs:
        try:
            run(seq, args.mode)
        except Exception as e:
            print(f"\n[ERROR] {seq} failed: {e}")
            import traceback; traceback.print_exc()
            continue
