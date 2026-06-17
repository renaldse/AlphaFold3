#!/usr/bin/env python3
"""
rank_samples.py

Ranks AF3 output samples by ranking_score (from summary_confidences.json)
within each seed group, renaming seed-X_sample-Y dirs to model_0, model_1, ...
where model_0 is the best (highest ranking_score).

Usage:
    python3 rank_samples.py <pdbid_dir>

Where <pdbid_dir> is the inner directory containing the seed-*_sample-* folders, e.g.:
    /mnt/gs21/scratch/renaldse/AlphaFold3/outputs/seq1_cluster0_templated_apo_.../seq1_cluster0
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 rank_samples.py <pdbid_dir>", file=sys.stderr)
        sys.exit(1)

    pdbid_dir = Path(sys.argv[1])

    if not pdbid_dir.exists():
        print(f"ERROR: directory does not exist: {pdbid_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Ranking samples in: {pdbid_dir}", flush=True)

    # Group seed-X_sample-Y dirs by seed ID
    seed_groups = defaultdict(list)
    for d in sorted(pdbid_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("seed-"):
            continue
        # e.g. seed-1270035777_sample-3 -> seed ID is "1270035777"
        parts = d.name.split("_sample-")
        if len(parts) != 2:
            continue
        seed_id = parts[0]  # e.g. "seed-1270035777"
        seed_groups[seed_id].append(d)

    if not seed_groups:
        print(f"ERROR: no seed-*_sample-* directories found in {pdbid_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(seed_groups)} seed group(s).", flush=True)

    any_success = False
    for seed_id in sorted(seed_groups.keys()):
        sample_dirs = seed_groups[seed_id]
        print(f"\n  Processing {seed_id} ({len(sample_dirs)} samples)...", flush=True)

        candidates = []
        for sample_dir in sample_dirs:
            # Find summary_confidences.json directly inside the sample dir
            matches = list(sample_dir.glob("summary_confidences.json"))
            if not matches:
                print(f"    WARNING: no summary_confidences.json in {sample_dir.name}, skipping.", flush=True)
                continue
            summary = matches[0]
            try:
                with open(summary) as f:
                    data = json.load(f)
                score = float(data["ranking_score"])
                candidates.append((score, sample_dir))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f"    WARNING: could not read ranking_score from {summary}: {e}", flush=True)

        if not candidates:
            print(f"    WARNING: no samples with ranking_score found for {seed_id}, skipping.", flush=True)
            continue

        # Sort descending: highest score = model_0
        candidates.sort(key=lambda x: x[0], reverse=True)

        print(f"    Ranking:", flush=True)
        for rank, (score, sdir) in enumerate(candidates):
            print(f"      model_{rank}  score={score:.4f}  {sdir.name}", flush=True)

        # Two-pass rename to avoid collisions
        temp_names = {}
        for rank, (score, sdir) in enumerate(candidates):
            tmp = sdir.parent / f"_tmp_model_{rank}_{seed_id}"
            sdir.rename(tmp)
            temp_names[rank] = tmp

        for rank, tmp in temp_names.items():
            final = tmp.parent / f"{seed_id}_model_{rank}"
            tmp.rename(final)
            print(f"      Renamed -> {final.name}", flush=True)

        print(f"    Done. Best in {seed_id} (model_0) has ranking_score={candidates[0][0]:.4f}", flush=True)
        any_success = True

    if not any_success:
        print("ERROR: no samples were successfully ranked.", file=sys.stderr)
        sys.exit(1)

    print("\nRanking complete.", flush=True)


if __name__ == "__main__":
    main()
