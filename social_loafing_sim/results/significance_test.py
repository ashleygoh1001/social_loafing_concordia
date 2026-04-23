"""
significance_test.py
====================
Tests whether perceived loafing (PLS) differs significantly across conditions.

Unit of analysis: group-level mean PLS (100 groups per condition).
Rationale: individual rater->target ratings within a group are not independent,
so we aggregate to the group first before any between-condition comparison.

Tests run
---------
1. Descriptive stats — mean, SD, SE, 95% CI per condition
2. Normality check — Shapiro-Wilk on each condition's group means
3. Levene's test — homogeneity of variance across conditions
4. One-way ANOVA — omnibus test across all 7 conditions
5. Kruskal-Wallis — non-parametric alternative (robust if normality fails)
6. Post-hoc pairwise comparisons vs control
     - Independent t-test (parametric) with Bonferroni correction
     - Mann-Whitney U (non-parametric) with Bonferroni correction
7. Effect sizes — Cohen's d (parametric) and rank-biserial r (non-parametric)

Usage
-----
  python significance_test.py                        # looks for CSVs in current dir
  python significance_test.py path/to/*.csv          # explicit files
  python significance_test.py --input-dir results/pl_scores
"""

import argparse
import glob
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def group_means(df):
    """Aggregate to group-level mean PLS — the unit of analysis."""
    return (
        df.dropna(subset=["PLS"])
        .groupby(["condition", "group_id"])["PLS"]
        .mean()
        .reset_index()
        .rename(columns={"PLS": "group_mean_PLS"})
    )


def cohens_d(a, b):
    """Cohen's d with pooled SD."""
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                         (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan


def rank_biserial_r(a, b):
    """Rank-biserial correlation from Mann-Whitney U."""
    stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2  = len(a), len(b)
    return 1 - (2 * stat) / (n1 * n2)


def ci95(arr):
    """95% confidence interval of the mean (t-based)."""
    n   = len(arr)
    se  = stats.sem(arr)
    t   = stats.t.ppf(0.975, df=n - 1)
    m   = np.mean(arr)
    return m - t * se, m + t * se


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def effect_label(d):
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def sep(char="-", width=72):
    print(char * width)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(df_raw, control_condition="control"):
    gm = group_means(df_raw)
    conditions = sorted(gm["condition"].unique())

    if control_condition not in conditions:
        print(f"WARNING: '{control_condition}' not found. Available: {conditions}")
        control_condition = conditions[0]

    groups_by_cond = {
        c: gm.loc[gm["condition"] == c, "group_mean_PLS"].values
        for c in conditions
    }

    # ------------------------------------------------------------------
    # 1. Descriptive stats
    # ------------------------------------------------------------------
    sep("=")
    print("1. DESCRIPTIVE STATISTICS  (unit = group-level mean PLS)")
    sep("=")
    rows = []
    for c in conditions:
        arr = groups_by_cond[c]
        lo, hi = ci95(arr)
        rows.append({
            "condition": c,
            "n_groups":  len(arr),
            "mean":      np.mean(arr),
            "SD":        np.std(arr, ddof=1),
            "SE":        stats.sem(arr),
            "95%_CI_lo": lo,
            "95%_CI_hi": hi,
            "min":       np.min(arr),
            "max":       np.max(arr),
        })
    desc = pd.DataFrame(rows).set_index("condition")
    print(desc.round(4).to_string())
    print()

    # ------------------------------------------------------------------
    # 2. Normality — Shapiro-Wilk
    # ------------------------------------------------------------------
    sep("=")
    print("2. NORMALITY CHECK  (Shapiro-Wilk on group means per condition)")
    sep("=")
    normal_flags = {}
    for c in conditions:
        arr = groups_by_cond[c]
        w, p = stats.shapiro(arr)
        normal_flags[c] = p > 0.05
        print(f"  {c:<30}  W={w:.4f}  p={p:.4f}  {'NORMAL' if p>0.05 else 'NON-NORMAL'}")
    all_normal = all(normal_flags.values())
    print(f"\n  All conditions normal? {'YES' if all_normal else 'NO — consider non-parametric tests'}")
    print()

    # ------------------------------------------------------------------
    # 3. Levene's test
    # ------------------------------------------------------------------
    sep("=")
    print("3. LEVENE'S TEST  (homogeneity of variance)")
    sep("=")
    lev_stat, lev_p = stats.levene(*groups_by_cond.values())
    print(f"  W={lev_stat:.4f}  p={lev_p:.4f}  "
          f"{'Equal variances assumed' if lev_p > 0.05 else 'Unequal variances — use Welch corrections'}")
    print()

    # ------------------------------------------------------------------
    # 4. One-way ANOVA
    # ------------------------------------------------------------------
    sep("=")
    print("4. ONE-WAY ANOVA  (omnibus test across all conditions)")
    sep("=")
    f_stat, anova_p = stats.f_oneway(*groups_by_cond.values())
    # Eta-squared
    all_vals = np.concatenate(list(groups_by_cond.values()))
    grand_mean = np.mean(all_vals)
    ss_between = sum(
        len(groups_by_cond[c]) * (np.mean(groups_by_cond[c]) - grand_mean)**2
        for c in conditions
    )
    ss_total = np.sum((all_vals - grand_mean)**2)
    eta_sq = ss_between / ss_total
    df_between = len(conditions) - 1
    df_within  = len(all_vals) - len(conditions)
    print(f"  F({df_between}, {df_within}) = {f_stat:.4f}  p = {anova_p:.4f} {stars(anova_p)}")
    print(f"  η² = {eta_sq:.4f}  ({effect_label(eta_sq**0.5)} effect)")
    print()

    # ------------------------------------------------------------------
    # 5. Kruskal-Wallis
    # ------------------------------------------------------------------
    sep("=")
    print("5. KRUSKAL-WALLIS TEST  (non-parametric omnibus)")
    sep("=")
    kw_stat, kw_p = stats.kruskal(*groups_by_cond.values())
    n_total = len(all_vals)
    k        = len(conditions)
    epsilon_sq = (kw_stat - k + 1) / (n_total - k)   # epsilon² effect size
    print(f"  H({df_between}) = {kw_stat:.4f}  p = {kw_p:.4f} {stars(kw_p)}")
    print(f"  ε² = {epsilon_sq:.4f}  ({effect_label(epsilon_sq**0.5)} effect)")
    print()

    # ------------------------------------------------------------------
    # 6 & 7. Pairwise comparisons vs control + effect sizes
    # ------------------------------------------------------------------
    sep("=")
    print(f"6 & 7. PAIRWISE COMPARISONS vs '{control_condition}'  "
          f"(Bonferroni-corrected, n_comparisons={len(conditions)-1})")
    sep("=")

    non_control = [c for c in conditions if c != control_condition]
    n_comp      = len(non_control)
    ctrl        = groups_by_cond[control_condition]

    rows = []
    for c in non_control:
        arr = groups_by_cond[c]

        # Parametric: Welch t-test
        t_stat, t_p_raw = stats.ttest_ind(ctrl, arr, equal_var=False)
        t_p_bon = min(t_p_raw * n_comp, 1.0)
        d       = cohens_d(ctrl, arr)

        # Non-parametric: Mann-Whitney U
        mw_stat, mw_p_raw = stats.mannwhitneyu(ctrl, arr, alternative="two-sided")
        mw_p_bon = min(mw_p_raw * n_comp, 1.0)
        r        = rank_biserial_r(ctrl, arr)

        rows.append({
            "condition":        c,
            "mean_diff (ctrl-cond)": round(np.mean(ctrl) - np.mean(arr), 4),
            # t-test
            "t":                round(t_stat, 3),
            "p_raw (t)":        round(t_p_raw, 4),
            "p_bonf (t)":       round(t_p_bon, 4),
            "sig (t)":          stars(t_p_bon),
            "Cohen's d":        round(d, 3),
            "effect (d)":       effect_label(d),
            # Mann-Whitney
            "U":                round(mw_stat, 1),
            "p_raw (MW)":       round(mw_p_raw, 4),
            "p_bonf (MW)":      round(mw_p_bon, 4),
            "sig (MW)":         stars(mw_p_bon),
            "rank-biserial r":  round(r, 3),
            "effect (r)":       effect_label(r * 2),   # r ~ d/2 rule-of-thumb
        })

    pw = pd.DataFrame(rows).set_index("condition")

    print("\n  — Parametric (Welch t-test + Cohen's d) —")
    t_cols = ["mean_diff (ctrl-cond)", "t", "p_raw (t)", "p_bonf (t)", "sig (t)",
              "Cohen's d", "effect (d)"]
    print(pw[t_cols].to_string())

    print("\n  — Non-parametric (Mann-Whitney U + rank-biserial r) —")
    mw_cols = ["mean_diff (ctrl-cond)", "U", "p_raw (MW)", "p_bonf (MW)", "sig (MW)",
               "rank-biserial r", "effect (r)"]
    print(pw[mw_cols].to_string())
    print()

    sep("=")
    print("LEGEND")
    sep("=")
    print("  *** p < 0.001    ** p < 0.01    * p < 0.05    ns = not significant")
    print("  Cohen's d:  |d| < 0.2 negligible, 0.2 small, 0.5 medium, 0.8 large")
    print("  All p-values for pairwise tests are Bonferroni-corrected.")
    print("  Unit of analysis: group-level mean PLS (N=100 per condition).")
    print()

    return gm, desc, pw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Significance testing for perceived loafing PLS scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python significance_test.py results/pl_scores/*.csv
  python significance_test.py --input-dir results/pl_scores
  python significance_test.py *.csv --control meaningful_feedback
        """,
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="CSV files output by perceived_loafing.py"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory containing CSV files (used if no positional args given)"
    )
    parser.add_argument(
        "--control", default="control",
        help="Name of the control condition (default: 'control')"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save results table as CSV"
    )
    args = parser.parse_args()

    paths = list(args.csv_files)
    if not paths and args.input_dir:
        paths = glob.glob(os.path.join(args.input_dir, "*.csv"))
    if not paths:
        # Try current directory as fallback
        paths = glob.glob("*.csv")
    if not paths:
        sys.exit("ERROR: No CSV files found. Pass file paths or --input-dir.")

    print(f"\nLoading {len(paths)} file(s):")
    for p in sorted(paths):
        print(f"  {p}")
    print()

    df_raw = load_all(paths)
    print(f"Total rows: {len(df_raw)}  |  Conditions: {sorted(df_raw.condition.unique())}\n")

    gm, desc, pw = run_analysis(df_raw, control_condition=args.control)

    if args.output:
        gm.to_csv(args.output, index=False)
        print(f"Group means saved to {args.output}")


if __name__ == "__main__":
    main()