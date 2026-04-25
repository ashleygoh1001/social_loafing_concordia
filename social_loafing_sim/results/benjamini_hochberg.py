"""
Peer Learning Score (PLS) Analysis
===================================
Design:  Between-subjects experiment with 7 conditions.
         Each condition has ~100 groups; within each group, every student
         rates every other student (round-robin peer evaluations).
Outcome: PLS (Peer Learning Score, continuous)
Model:   OLS with cluster-robust standard errors (cluster = group_id)
Post-hoc: Estimated marginal means, all pairwise contrasts,
          BH-corrected p-values
"""

import glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ── 1. Load & combine all condition files ────────────────────────────────────

files = {
    "agile":        "pl_outputs/agile_pl_results.csv",
    "control":      "pl_outputs/control_pl_score.csv",
    "indiv_contrib":"pl_outputs/indiv_contrib_pl_results.csv",
    "meaning_fback":"pl_outputs/meaning_fback_pl_results.csv",
    "peer_eval":    "pl_outputs/peer_eval_pl_results.csv",
    "task_visib":   "pl_outputs/task_visib_pl_results.csv",
    "weekly_log":   "pl_outputs/weekly_log_pl_results.csv",
}

dfs = []
for condition_label, fname in files.items():
    df = pd.read_csv(fname)
    # Overwrite condition with our canonical label in case file spellings differ
    df["condition"] = condition_label
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Make condition a categorical with 'control' as the reference level
REFERENCE = "control"
conditions = [REFERENCE] + [c for c in sorted(data["condition"].unique()) if c != REFERENCE]
data["condition"] = pd.Categorical(data["condition"], categories=conditions, ordered=False)

print("=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)
print(data.groupby("condition")[["group_id", "PLS"]].agg(
    n_obs=("PLS", "count"),
    n_groups=("group_id", "nunique"),
    mean_PLS=("PLS", "mean"),
    sd_PLS=("PLS", "std"),
).round(3))
print()

# ── 2. OLS with cluster-robust SEs ───────────────────────────────────────────
#   Cluster on group_id: observations within a group share rater / social context

model = smf.ols("PLS ~ C(condition, Treatment(reference='control'))", data=data)
fit = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": data["group_id"]},
)

print("=" * 65)
print("OLS WITH CLUSTER-ROBUST SEs  (reference = control)")
print("=" * 65)
print(fit.summary2(float_format="%.4f"))
print()

# ── 3. Estimated marginal means (EMMs) ───────────────────────────────────────
#   For a one-way model, EMM of each condition = intercept + condition coef.
#   SE via delta method (sqrt of variance of linear combination).

coef   = fit.params
vcov   = fit.cov_params()
intercept = coef["Intercept"]

emm_rows = []
for cond in conditions:
    if cond == REFERENCE:
        contrast_vec = np.zeros(len(coef))
        contrast_vec[coef.index.get_loc("Intercept")] = 1.0
    else:
        param = f"C(condition, Treatment(reference='control'))[T.{cond}]"
        contrast_vec = np.zeros(len(coef))
        contrast_vec[coef.index.get_loc("Intercept")] = 1.0
        contrast_vec[coef.index.get_loc(param)] = 1.0

    emm   = contrast_vec @ coef.values
    se    = np.sqrt(contrast_vec @ vcov.values @ contrast_vec)
    t_val = emm / se
    # Two-sided p from t with robust df ≈ n_clusters - 1
    from scipy.stats import t as t_dist
    n_clusters = data.groupby("condition")["group_id"].nunique().sum()
    df_resid   = n_clusters - len(coef)
    p_val = 2 * t_dist.sf(abs(t_val), df=df_resid)
    emm_rows.append({
        "condition": cond,
        "EMM": emm,
        "SE": se,
        "CI_low": emm - 1.96 * se,
        "CI_high": emm + 1.96 * se,
        "t": t_val,
        "p": p_val,
    })

emm_df = pd.DataFrame(emm_rows).set_index("condition")

print("=" * 65)
print("ESTIMATED MARGINAL MEANS")
print("=" * 65)
print(emm_df.round(4))
print()

# ── 4. All pairwise contrasts with BH correction ─────────────────────────────

contrast_rows = []
for c1, c2 in combinations(conditions, 2):
    # Build contrast vector: EMM(c1) - EMM(c2)
    def _vec(cond):
        v = np.zeros(len(coef))
        v[coef.index.get_loc("Intercept")] = 1.0
        if cond != REFERENCE:
            param = f"C(condition, Treatment(reference='control'))[T.{cond}]"
            v[coef.index.get_loc(param)] = 1.0
        return v

    diff_vec = _vec(c1) - _vec(c2)
    diff     = diff_vec @ coef.values
    se_diff  = np.sqrt(diff_vec @ vcov.values @ diff_vec)
    t_val    = diff / se_diff
    p_raw    = 2 * t_dist.sf(abs(t_val), df=df_resid)

    contrast_rows.append({
        "contrast": f"{c1} − {c2}",
        "estimate": diff,
        "SE": se_diff,
        "t": t_val,
        "p_raw": p_raw,
        "CI_low": diff - 1.96 * se_diff,
        "CI_high": diff + 1.96 * se_diff,
    })

contrasts_df = pd.DataFrame(contrast_rows)

# BH correction across all pairwise p-values
reject, p_bh, _, _ = multipletests(contrasts_df["p_raw"], method="fdr_bh")
contrasts_df["p_BH"]    = p_bh
contrasts_df["sig_BH"]  = reject  # True = significant at α=0.05 after BH

contrasts_df = contrasts_df.sort_values("p_BH").reset_index(drop=True)

print("=" * 65)
print("PAIRWISE CONTRASTS  (BH-corrected, sorted by p_BH)")
print("=" * 65)
with pd.option_context("display.max_rows", None, "display.width", 120):
    print(contrasts_df[
        ["contrast", "estimate", "SE", "CI_low", "CI_high", "t", "p_raw", "p_BH", "sig_BH"]
    ].round(4).to_string(index=False))
print()

# ── 5. Compact significance summary ──────────────────────────────────────────

sig = contrasts_df[contrasts_df["sig_BH"]].copy()
print("=" * 65)
print(f"SIGNIFICANT CONTRASTS AFTER BH CORRECTION  (n = {len(sig)} / {len(contrasts_df)})")
print("=" * 65)
if sig.empty:
    print("  None")
else:
    print(sig[["contrast", "estimate", "p_BH"]].round(4).to_string(index=False))
print()

# ── 6. Save outputs ──────────────────────────────────────────────────────────

emm_df.round(4).to_csv("emm_results.csv")
contrasts_df.round(4).to_csv("pairwise_contrasts_bh.csv", index=False)
print("Results saved: emm_results.csv  |  pairwise_contrasts_bh.csv")