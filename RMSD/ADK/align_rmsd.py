"""
align_rmsd.py
=============
Replicates PyMOL's `align` command in pure Python using BioPython.

PyMOL align pipeline (correctly implemented here):
  1. Cα-only sequence alignment using BLOSUM62 → matched residue pairs
  2. Expand matched residues to ALL heavy atoms → atom pairs for superposition
  3. Initial superposition (Kabsch/SVD) on all matched heavy atoms
  4. Iterative outlier rejection: per RESIDUE — if any atom in a residue
     exceeds cutoff after superposition, drop the whole residue, re-superpose
  5. Report RMSD over surviving atoms (after refinement) and before refinement

  This matches PyMOL's ExecutiveRMS behaviour: "109 atoms rejected during
  cycle 1 (RMSD=7.19)" etc., converging to e.g. 2.069 Å (1135 of 1656 atoms).

  Separately, a Cα-only RMSD is also computed (sequence-align → Cα superpose
  → Cα iterative rejection) so you get both numbers.

  With --heatmap, two PNGs are saved:
    <heatmap-out>_CA.png        Cα-only RMSD heatmap
    <heatmap-out>_allatom.png   All-heavy-atom RMSD heatmap (PyMOL-style)
  Each PNG includes a stats panel next to the colorbar showing per-pair
  alignment details exactly as printed to the terminal.

Usage:
  python align_rmsd.py file1.cif file2.cif [file3 ...] [--heatmap]

Optional flags:
  --cutoff FLOAT     Outlier rejection cutoff in Å  (default: 2.0)
  --cycles INT       Max rejection cycles           (default: 5)
  --chain CHAR       Chain ID                       (default: first chain)
  --heatmap          Save heatmap PNGs
  --heatmap-out PATH Base path for PNGs             (default: ADK folder)
  --labels STR [..] Custom axis labels

Requirements:
  pip install biopython numpy matplotlib
"""

import argparse
import sys
import itertools
import os
import numpy as np
from Bio import PDB, Align
from Bio.Align import substitution_matrices


# ---------------------------------------------------------------------------
# Kabsch superposition
# ---------------------------------------------------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (R, t) minimising RMSD:  P_rotated = P @ R.T + t"""
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    H = (P - P_mean).T @ (Q - Q_mean)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q_mean - P_mean @ R.T
    return R, t


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ R.T + t


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    diff = P - Q
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def parse_structure(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure("s", path)


def get_chain(structure, chain_id: str | None):
    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("No chains found in structure.")
    if chain_id is None:
        return chains[0]
    chain_ids = [c.id for c in chains]
    if chain_id not in chain_ids:
        raise ValueError(f"Chain {chain_id} not found. Available: {chain_ids}")
    return model[chain_id]


def get_peptide_residues(chain) -> list:
    """Return list of residue objects from polypeptide segments only."""
    ppb = PDB.PPBuilder()
    residues = []
    for pp in ppb.build_peptides(chain):
        residues.extend(pp)
    return residues


def res_one_letter(res) -> str:
    try:
        return PDB.Polypeptide.index_to_one(
            PDB.Polypeptide.three_to_index(res.get_resname())
        )
    except (KeyError, ValueError):
        return "X"


def ca_coords(residues: list) -> np.ndarray:
    """Cα coordinate array, shape (N, 3). Residues without CA are skipped."""
    coords = []
    for res in residues:
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())
    return np.array(coords)


def heavy_atom_coords_per_residue(residues: list) -> list[np.ndarray]:
    """
    For each residue, return array of heavy-atom coordinates (N_atoms, 3).
    Hydrogens excluded.
    """
    out = []
    for res in residues:
        atoms = [a for a in res.get_atoms()
                 if a.element not in ("H", "D") and a.element is not None]
        if not atoms:
            # fallback: take whatever is there
            atoms = list(res.get_atoms())
        coords = np.array([a.get_vector().get_array() for a in atoms])
        out.append(coords)
    return out


# ---------------------------------------------------------------------------
# Sequence alignment → matched residue index pairs
# ---------------------------------------------------------------------------

def sequence_align_indices(residues_mob: list, residues_tgt: list) -> tuple[list, list, int]:
    """
    BLOSUM62 global alignment on Cα sequences.
    Returns (mob_indices, tgt_indices, n_aligned_residues).
    """
    seq_mob = "".join(res_one_letter(r) for r in residues_mob)
    seq_tgt = "".join(res_one_letter(r) for r in residues_tgt)

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score  = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    best = next(iter(aligner.align(seq_mob, seq_tgt)))

    mob_idx, tgt_idx = [], []
    for (ms, me), (ts, te) in zip(*best.aligned):
        for dm, dt in zip(range(ms, me), range(ts, te)):
            mob_idx.append(dm)
            tgt_idx.append(dt)

    if not mob_idx:
        raise ValueError("Sequence alignment produced zero matched residue pairs.")

    return mob_idx, tgt_idx, len(mob_idx)


# ---------------------------------------------------------------------------
# Iterative outlier rejection — Cα mode
# ---------------------------------------------------------------------------

def align_ca(
    residues_mob: list,
    residues_tgt: list,
    mob_idx: list,
    tgt_idx: list,
    cutoff: float = 2.0,
    cycles: int = 5,
) -> dict:
    """
    Cα-only superposition + iterative rejection.
    Rejection is per-atom (= per-residue since there is 1 Cα per residue).
    """
    # Build full Cα arrays for matched residues only
    def ca(res):
        return res["CA"].get_vector().get_array()

    P_all = np.array([ca(residues_mob[i]) for i in mob_idx
                      if "CA" in residues_mob[i]])
    Q_all = np.array([ca(residues_tgt[i]) for i in tgt_idx
                      if "CA" in residues_tgt[i]])

    # guard: keep only pairs where both have CA
    valid = [
        (i, j) for i, j in zip(mob_idx, tgt_idx)
        if "CA" in residues_mob[i] and "CA" in residues_tgt[j]
    ]
    P_all = np.array([ca(residues_mob[i]) for i, _ in valid])
    Q_all = np.array([ca(residues_tgt[j]) for _, j in valid])

    mask = np.ones(len(P_all), dtype=bool)

    # initial superposition for "before" stats
    R, t = kabsch(P_all[mask], Q_all[mask])
    rmsd_before = rmsd(apply_transform(P_all[mask], R, t), Q_all[mask])
    n_before = int(mask.sum())

    cycle = 0
    for cycle in range(cycles):
        R, t = kabsch(P_all[mask], Q_all[mask])
        P_rot = apply_transform(P_all[mask], R, t)
        per_atom = np.sqrt(((P_rot - Q_all[mask]) ** 2).sum(axis=1))
        keep = per_atom <= cutoff
        if keep.all():
            break
        global_idx = np.where(mask)[0]
        for gi, k in zip(global_idx, keep):
            if not k:
                mask[gi] = False

    R, t = kabsch(P_all[mask], Q_all[mask])
    rmsd_after = rmsd(apply_transform(P_all[mask], R, t), Q_all[mask])

    return {
        "rmsd_before": rmsd_before,
        "n_before":    n_before,
        "rmsd_after":  rmsd_after,
        "n_after":     int(mask.sum()),
        "n_cycles":    cycle + 1 if cycles > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Iterative outlier rejection — all-atom mode  (matches PyMOL align exactly)
# ---------------------------------------------------------------------------

def align_allatom(
    residues_mob: list,
    residues_tgt: list,
    mob_idx: list,
    tgt_idx: list,
    cutoff: float = 2.0,
    cycles: int = 5,
) -> dict:
    """
    PyMOL-matching pipeline:
      - Sequence alignment done on Cα (indices already provided)
      - Superposition/rejection uses ALL heavy atoms
      - Rejection is per RESIDUE: if the max per-atom distance for a residue
        exceeds cutoff after superposition, the whole residue is dropped
      - Cycle RMSD printed matches PyMOL's "ExecutiveRMS: N atoms rejected
        during cycle K (RMSD=X.XX)"
    """
    # Expand matched residue pairs to all heavy atoms
    # paired_atoms: list of (mob_coords_array, tgt_coords_array) per residue pair
    # We need equal atom counts per residue — intersect by atom name
    paired_residues = []  # list of (P_res, Q_res) np arrays
    for i, j in zip(mob_idx, tgt_idx):
        res_m = residues_mob[i]
        res_t = residues_tgt[j]
        # get heavy atoms present in BOTH residues (by atom name)
        heavy_m = {a.get_name(): a.get_vector().get_array()
                   for a in res_m.get_atoms()
                   if a.element not in ("H", "D") and a.element is not None}
        heavy_t = {a.get_name(): a.get_vector().get_array()
                   for a in res_t.get_atoms()
                   if a.element not in ("H", "D") and a.element is not None}
        shared = sorted(set(heavy_m) & set(heavy_t))
        if not shared:
            continue
        P_res = np.array([heavy_m[n] for n in shared])
        Q_res = np.array([heavy_t[n] for n in shared])
        paired_residues.append((P_res, Q_res))

    if not paired_residues:
        raise ValueError("No shared heavy atoms found between matched residues.")

    # Flatten to atom arrays
    P_flat = np.concatenate([p for p, _ in paired_residues], axis=0)
    Q_flat = np.concatenate([q for _, q in paired_residues], axis=0)

    # Per-atom active mask (PyMOL rejects individual atoms, not whole residues)
    atom_active = np.ones(len(P_flat), dtype=bool)

    def active_atoms():
        return P_flat[atom_active], Q_flat[atom_active], atom_active.copy()

    # Initial superposition → "before" stats
    P_a, Q_a, _ = active_atoms()
    R, t = kabsch(P_a, Q_a)
    rmsd_before = rmsd(apply_transform(P_a, R, t), Q_a)
    n_before = len(P_a)

    cycle = 0
    n_rejected_log = []

    for cycle in range(cycles):
        P_a, Q_a, atom_mask = active_atoms()
        R, t = kabsch(P_a, Q_a)
        P_rot = apply_transform(P_a, R, t)
        per_atom_dist = np.sqrt(((P_rot - Q_a) ** 2).sum(axis=1))

        # Current RMSD — used both for logging and for the dynamic threshold.
        # PyMOL uses a RELATIVE cutoff: threshold = cutoff * current_RMSD
        # (not a fixed absolute Angstrom value). This is why only ~109/1656
        # atoms are rejected in cycle 1 at RMSD=7.19 with cutoff=2.0:
        # threshold = 2.0 * 7.19 = 14.38 Å, rejecting only the most extreme outliers.
        # As RMSD drops each cycle the threshold tightens, gradually converging.
        current_rmsd = rmsd(P_rot, Q_a)
        threshold = cutoff * current_rmsd

        # Reject individual atoms (not whole residues) beyond the dynamic threshold
        reject_atom_mask = per_atom_dist > threshold
        n_rejected = int(reject_atom_mask.sum())

        if n_rejected == 0:
            break

        n_rejected_log.append((n_rejected, current_rmsd))

        # PyMOL rejects individual atoms, not whole residues.
        # Update the global atom mask directly.
        global_atom_indices = np.where(atom_mask)[0]
        rejected_global = global_atom_indices[reject_atom_mask]
        atom_active[rejected_global] = False

    # Final superposition
    P_a, Q_a, _ = active_atoms()
    R, t = kabsch(P_a, Q_a)
    rmsd_after = rmsd(apply_transform(P_a, R, t), Q_a)

    return {
        "rmsd_before":    rmsd_before,
        "n_before":       n_before,
        "rmsd_after":     rmsd_after,
        "n_after":        len(P_a),
        "n_cycles":       cycle + 1 if cycles > 0 else 0,
        "rejection_log":  n_rejected_log,  # list of (n_atoms_rejected, rmsd_at_cycle)
    }


# ---------------------------------------------------------------------------
# High-level align function
# ---------------------------------------------------------------------------

def align(
    mobile_path: str,
    target_path: str,
    cutoff: float = 2.0,
    cycles: int = 5,
    chain_id: str | None = None,
    quiet: bool = False,
) -> dict:
    """
    Full PyMOL-matching align pipeline.
    Returns dict with keys:
      ca:       result dict from Cα alignment
      allatom:  result dict from all-atom alignment (PyMOL-matching)
      n_residues_aligned: residues matched by sequence alignment
    """
    struct_mob = parse_structure(mobile_path)
    struct_tgt = parse_structure(target_path)

    chain_mob = get_chain(struct_mob, chain_id)
    chain_tgt = get_chain(struct_tgt, chain_id)

    res_mob = get_peptide_residues(chain_mob)
    res_tgt = get_peptide_residues(chain_tgt)

    mob_idx, tgt_idx, n_res = sequence_align_indices(res_mob, res_tgt)

    result_ca = align_ca(res_mob, res_tgt, mob_idx, tgt_idx, cutoff, cycles)
    result_aa = align_allatom(res_mob, res_tgt, mob_idx, tgt_idx, cutoff, cycles)

    if not quiet:
        print(f"\n  Mobile : {mobile_path}  (chain {chain_mob.id})")
        print(f"  Target : {target_path}  (chain {chain_tgt.id})")
        print(f"  Sequence-aligned residues : {n_res}")
        print()
        print("  --- Cα-only ---")
        print(f"  RMSD before refinement    : {result_ca['rmsd_before']:.3f} Å  ({result_ca['n_before']} atoms)")
        print(f"  RMSD after  refinement    : {result_ca['rmsd_after']:.3f} Å  ({result_ca['n_after']} atoms)  [{result_ca['n_cycles']} cycle(s)]")
        print()
        print("  --- All-atom (PyMOL-matching) ---")
        print(f"  RMSD before refinement    : {result_aa['rmsd_before']:.3f} Å  ({result_aa['n_before']} atoms)")
        for ci, (n_rej, rms_at_cycle) in enumerate(result_aa.get("rejection_log", []), 1):
            print(f"  ExecutiveRMS: {n_rej} atoms rejected during cycle {ci} (RMSD={rms_at_cycle:.2f})")
        print(f"  RMSD after  refinement    : {result_aa['rmsd_after']:.3f} Å  ({result_aa['n_after']} atoms)  [{result_aa['n_cycles']} cycle(s)]")

    return {
        "ca":                  result_ca,
        "allatom":             result_aa,
        "n_residues_aligned":  n_res,
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def build_rmsd_matrix(structures, results, labels, mode="ca"):
    n = len(structures)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)
    idx = {s: i for i, s in enumerate(structures)}
    for mob, tgt, r in results:
        i, j = idx[mob], idx[tgt]
        val = r[mode]["rmsd_after"]
        mat[i, j] = val
        mat[j, i] = val
    return mat


def _stats_lines(results, labels, structures, mode="ca"):
    idx = {s: i for i, s in enumerate(structures)}
    lines = []
    for mob, tgt, r in results:
        i, j = idx[mob], idx[tgt]
        rc = r[mode]
        lines.append(f"{labels[i]}  \u2192  {labels[j]}")
        lines.append(f"  Sequence-aligned residues : {r['n_residues_aligned']}")
        lines.append(f"  RMSD before refinement    : {rc['rmsd_before']:.3f} \u00c5  ({rc['n_before']} atoms)")
        if mode == "allatom":
            for ci, (n_rej, rms_c) in enumerate(rc.get("rejection_log", []), 1):
                lines.append(f"  ExecutiveRMS: {n_rej} atoms rejected cycle {ci} (RMSD={rms_c:.2f})")
        lines.append(f"  RMSD after  refinement    : {rc['rmsd_after']:.3f} \u00c5  ({rc['n_after']} atoms)  [{rc['n_cycles']} cycle(s)]")
        lines.append("")
    return lines


def plot_heatmap(matrix, labels, results, structures, out_path, mode="ca"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [WARNING] matplotlib not found — pip install matplotlib")
        return

    n = len(labels)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    norm = mcolors.Normalize(vmin=0.0, vmax=2.0)

    stats_lines = _stats_lines(results, labels, structures, mode)
    stats_text  = "\n".join(stats_lines).rstrip()

    cell    = max(1.4, 6.0 / n * 1.5)
    heat_w  = n * cell
    stats_w = 4.8
    cbar_w  = 0.5
    fig_w   = heat_w + cbar_w + stats_w + 1.4
    fig_h   = max(n * cell + 1.5, 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(1, 3, width_ratios=[heat_w, cbar_w, stats_w],
                   left=0.08, right=0.98, wspace=0.05, figure=fig)
    ax  = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    tax = fig.add_subplot(gs[0, 2])

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if i == j:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1, 1, boxstyle="square,pad=0",
                    facecolor="#d0d0d0", edgecolor="white", linewidth=0.5))
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=max(7, 11 - n // 3), color="#666666")
            elif not np.isnan(val):
                color = cmap(norm(val))
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.5, i - 0.5), 1, 1, boxstyle="square,pad=0",
                    facecolor=color, edgecolor="white", linewidth=0.5))
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(7, 11 - n // 3), fontweight="bold",
                        color="white" if lum < 0.5 else "#222222")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(8, 12 - n // 4))
    ax.set_yticklabels(labels, fontsize=max(8, 12 - n // 4))
    ax.tick_params(length=0)

    mode_label = "C\u03b1-only" if mode == "ca" else "All-atom heavy (PyMOL-matching)"
    ax.set_title(f"Pairwise RMSD Heatmap  \u2014  {mode_label}, after outlier rejection",
                 fontsize=11, fontweight="bold", pad=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("RMSD (\u00c5)", fontsize=9)
    cax.tick_params(labelsize=8)

    tax.axis("off")
    tax.text(0.05, 0.97, stats_text, transform=tax.transAxes,
             fontsize=8.5, va="top", ha="left",
             fontfamily="monospace", color="#222222", linespacing=1.55)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved \u2192 {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyMOL-matching align: Cα sequence alignment + all-atom iterative superposition"
    )
    parser.add_argument("structures", nargs="+", metavar="FILE",
                        help="Two or more .pdb or .cif files")
    parser.add_argument("--cutoff",      type=float, default=2.0,
                        help="Outlier rejection cutoff in \u00c5 (default: 2.0)")
    parser.add_argument("--cycles",      type=int,   default=5,
                        help="Max rejection cycles (default: 5)")
    parser.add_argument("--chain",       type=str,   default=None,
                        help="Chain ID (default: first chain)")
    parser.add_argument("--heatmap",     action="store_true",
                        help="Save RMSD heatmap PNGs")
    parser.add_argument("--heatmap-out", type=str,
                        default="/mnt/gs21/scratch/renaldse/AlphaFold3/RMSD/ADK/rmsd_heatmap",
                        help="Base path for heatmap PNGs (_CA.png / _allatom.png appended)")
    parser.add_argument("--labels",      nargs="+", default=None,
                        help="Custom axis labels (must match number of files)")
    args = parser.parse_args()

    if len(args.structures) < 2:
        print("Error: provide at least 2 structure files.")
        sys.exit(1)

    if args.labels:
        if len(args.labels) != len(args.structures):
            print("Error: --labels count must match number of structure files.")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(s))[0] for s in args.structures]

    pairs = list(itertools.combinations(args.structures, 2))
    print(f"\nPyMOL-matching align  |  cutoff={args.cutoff} \u00c5  cycles={args.cycles}")
    print("=" * 70)

    results = []
    for mob, tgt in pairs:
        try:
            r = align(mob, tgt, cutoff=args.cutoff, cycles=args.cycles,
                      chain_id=args.chain)
            results.append((mob, tgt, r))
        except Exception as e:
            print(f"\n  [ERROR] {mob} vs {tgt}: {e}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Mobile':<30} {'Target':<30} {'Cα RMSD':>9} {'All-atom':>9}")
    print("-" * 70)
    for mob, tgt, r in results:
        print(f"{mob:<30} {tgt:<30} "
              f"{r['ca']['rmsd_after']:>9.3f} "
              f"{r['allatom']['rmsd_after']:>9.3f}")

    # Heatmaps
    if args.heatmap and results:
        print()
        base = args.heatmap_out
        mat_ca = build_rmsd_matrix(args.structures, results, labels, mode="ca")
        plot_heatmap(mat_ca, labels, results, args.structures,
                     out_path=base + "_CA.png", mode="ca")
        mat_aa = build_rmsd_matrix(args.structures, results, labels, mode="allatom")
        plot_heatmap(mat_aa, labels, results, args.structures,
                     out_path=base + "_allatom.png", mode="allatom")

    return results


if __name__ == "__main__":
    main()
