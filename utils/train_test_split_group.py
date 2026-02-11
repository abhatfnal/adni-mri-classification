#!/usr/bin/env python3
"""
Per-class Grouped Split for ADNI-style datasets.

- Guarantees: the same group (e.g., subject RID) NEVER appears in both splits.
- Goal: make the TEST set's class distribution approximate the global distribution
        (i.e., per-class proportions close to `test_size` of each class),
        while honoring the no-leakage constraint.

This is done with a greedy group-selection algorithm:
1) Compute desired per-class counts for the test split.
2) Iteratively add whole groups to the test set if they help reduce the per-class
   "unmet need." If still under the overall test size, fill with remaining groups.

Notes:
- If a subject changes diagnosis across visits, ALL of that subject's scans travel
  together (group integrity). Class balance is therefore "best-effort".
- The script prints actual class counts and proportions for both splits.

Usage:
  python train_test_split_group.py \
      --input-csv /path/to/dataset.csv \
      --test-size 0.2 \
      --output-dir /path/to/out \
      --group-col rid \
      --label-col diagnosis \
      -r 42
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


def _greedy_per_class_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    test_size: float,
    random_state: int = 42,
):
    """
    Select a set of groups for TEST so that:
      - no group is split across train/test
      - per-class test counts are close to test_size * total_class_count
      - total test size is close to test_size * N

    Returns:
        test_groups: set of group ids chosen for test
    """
    assert 0.0 < test_size < 1.0, "test_size must be in (0, 1)."

    # Unique groups and a deterministic shuffle
    groups = df[group_col].unique().tolist()
    rng = np.random.RandomState(random_state)
    rng.shuffle(groups)

    # Per-sample and per-class totals
    labels = df[label_col].astype(int).to_numpy()
    classes = np.sort(df[label_col].astype(int).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    # Desired counts per class and in total
    total_N = len(df)
    desired_total_test = int(round(test_size * total_N))

    total_per_class = np.array([(labels == c).sum() for c in classes], dtype=int)
    desired_per_class = np.array(
        [int(round(test_size * cnt)) for cnt in total_per_class],
        dtype=int,
    )

    # Build per-group contributions: class counts + total size
    group_class_counts = {}
    group_sizes = {}
    for g in groups:
        g_labels = df.loc[df[group_col] == g, label_col].astype(int).to_numpy()
        cnts = np.zeros(n_classes, dtype=int)
        for c, k in Counter(g_labels).items():
            cnts[class_to_idx[c]] = k
        group_class_counts[g] = cnts
        group_sizes[g] = len(g_labels)

    # Greedy phase 1: add groups that reduce per-class unmet need the most
    test_groups = set()
    current_per_class = np.zeros(n_classes, dtype=int)
    current_total = 0

    remaining = groups.copy()
    improved = True
    while improved:
        improved = False
        best_g, best_gain = None, 0

        need = np.maximum(desired_per_class - current_per_class, 0)
        if need.sum() == 0:
            break  # met all per-class quotas

        for g in remaining:
            contrib = group_class_counts[g]
            gain = np.minimum(contrib, need).sum()  # how much unmet need this group satisfies
            if gain > best_gain:
                best_gain = gain
                best_g = g

        if best_g is not None and best_gain > 0:
            # Add the best group
            test_groups.add(best_g)
            current_per_class += group_class_counts[best_g]
            current_total += group_sizes[best_g]
            remaining.remove(best_g)
            improved = True

    # Greedy phase 2: if we are still under the desired total test size,
    # add random remaining groups until we get close.
    # (This keeps group integrity; class proportions may overshoot slightly.)
    remaining = [g for g in remaining if g not in test_groups]
    rng.shuffle(remaining)

    for g in remaining:
        if current_total >= desired_total_test:
            break
        test_groups.add(g)
        current_per_class += group_class_counts[g]
        current_total += group_sizes[g]

    return test_groups


def main():
    parser = argparse.ArgumentParser(
        description="Per-class Grouped Split: output trainval.csv and test.csv with no group leakage."
    )
    parser.add_argument("--input-csv", required=True, help="Path to dataset CSV.")
    parser.add_argument("--test-size", required=True, type=float, help="Test set ratio in (0,1).")
    parser.add_argument("--output-dir", required=True, help="Directory to write trainval.csv and test.csv.")
    parser.add_argument("--group-col", default="rid", help="Column name for group/subject ID (e.g., RID).")
    parser.add_argument("--label-col", default="diagnosis", help="Column name for class label.")
    parser.add_argument("-r", dest="random_state", default=42, type=int, help="Random seed.")
    args = parser.parse_args(sys.argv[1:])

    try:
        assert 0.0 < args.test_size < 1.0, "test_size must be in (0, 1)."

        df = pd.read_csv(args.input_csv)

        # Basic sanity checks
        for col in (args.group_col, args.label_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {args.input_csv}")
        # enforce int labels
        df[args.label_col] = pd.to_numeric(df[args.label_col], errors="raise").astype(int)

        # Compute test groups via greedy per-class grouped split
        test_groups = _greedy_per_class_group_split(
            df=df,
            group_col=args.group_col,
            label_col=args.label_col,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # Build masks
        is_test = df[args.group_col].isin(test_groups)
        test_df = df[is_test].reset_index(drop=True)
        trainval_df = df[~is_test].reset_index(drop=True)

        # Safety: ensure no leakage
        leak_groups = set(trainval_df[args.group_col]).intersection(set(test_df[args.group_col]))
        assert len(leak_groups) == 0, "Leakage detected: some groups appear in both splits."

        # Make output dir
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "trainval.csv")
        test_path = os.path.join(args.output_dir, "test.csv")
        trainval_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Report distributions
        def _counts(df_, name):
            vc = df_[args.label_col].value_counts().sort_index()
            total = len(df_)
            prop = (vc / max(total, 1)).round(4)
            return vc, prop, total, name

        tv_vc, tv_prop, tv_total, _ = _counts(trainval_df, "trainval")
        te_vc, te_prop, te_total, _ = _counts(test_df, "test")

        print("\n=== Split Summary (Per-class Grouped) ===")
        print(f"Total samples: {len(df)} | Desired test ratio: {args.test_size}")
        print(f"trainval: N={tv_total}\ncounts:\n{tv_vc}\nproportions:\n{tv_prop}\n")
        print(f"test:     N={te_total}\ncounts:\n{te_vc}\nproportions:\n{te_prop}\n")

        # Extra info: number of unique groups per split
        n_g_tv = trainval_df[args.group_col].nunique()
        n_g_te = test_df[args.group_col].nunique()
        print(f"Unique {args.group_col}s -> trainval: {n_g_tv}, test: {n_g_te}")

        # Warn if a class is missing in either split
        for klass in sorted(df[args.label_col].unique()):
            if klass not in tv_vc.index:
                print(f"WARNING: class {klass} missing from trainval split.")
            if klass not in te_vc.index:
                print(f"WARNING: class {klass} missing from test split.")

        print(f"\nWrote:\n  {train_path}\n  {test_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
