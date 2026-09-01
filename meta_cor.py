#%%
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from pymare import Dataset
from pymare.estimators import WeightedLeastSquares
from pymare.estimators import DerSimonianLaird


def plot_correlation(df, x_col='sex_arousal_after', y_col='threshold'):
    r, p = st.pearsonr(df[x_col], df[y_col])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x_col], df[y_col], alpha=0.6)
    
    m, b = np.polyfit(df[x_col], df[y_col], 1)
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(x_line, m*x_line + b, color='red')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'r = {r:.3f}, p = {p:.3f}')
    
    plt.tight_layout()
    plt.show()
    
    return r, p



CLEAN_DATA = {
    1: 'exp1Clean.csv',
    2: 'exp2Clean.csv',
    3: 'exp3Clean.csv',
    4: 'exp4Clean.csv',
    5: 'exp5Clean.csv',
    6: 'Exp6_cleanData.csv',
    7: 'exp7cleanData.csv',
    8: 'exp8cleanData_final.csv',
    9: 'exp9cleanData.csv'
}

def comp_pooled_d(n1,n2,sd1,sd2,mean1,mean2):
    pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
    d = (mean1 - mean2) / pooled_sd
    return d, pooled_sd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

def morphed_faces_threshold_comparison(
        df_path, pred, targ, choices,
        cond_col='Condition', id_col='ProlificID', sex_col='sex_arousal_after',
        char_col='Character_num', condition=None, *args, verbose=False):
    """
    Per participant, fits a no-penalty trial-level binomial GLM (statsmodels)
    for threshold (PSE) extraction. On singular-matrix failures, falls back
    to a weighted-midpoint threshold if the two response classes are cleanly
    separated on x. Plots raw per-trial responses when the fit is not ok:
    one scatter plot per participant, response=1 trials above the midline,
    response=0 trials below, one row per character, colored by character.
    Diagnostics (summary prints and plots) are only produced when verbose=True.
    """

    singular_resolved = []

    def fit_threshold(group, pid):
        result = {
            "threshold": np.nan,
            "se": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "prsquared": np.nan,
            "converged": False,
            "n_trials": len(group),
            "n_levels": group[pred].nunique(),
            "fit_method": None,
            "status": None,
            "error": None,
        }

        x = group[pred].values
        y = group["Parsed_Target"].values

        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            result["status"] = "insufficient_variation"
            return result

            
        #These cases were filtered later. Still, this code is kept in case it will be of use in the future.
        def try_singular_fallback():
            x0, x1 = x[y == 0], x[y == 1]
            if len(x0) == 0 or len(x1) == 0:
                return

            max0, min0 = x0.max(), x0.min()
            max1, min1 = x1.max(), x1.min()

            if max0 < min1:
                n0 = np.sum(x0 == max0)
                n1 = np.sum(x1 == min1)
                result["threshold"] = (max0 * n0 + min1 * n1) / (n0 + n1)
            elif max1 < min0:
                n1 = np.sum(x1 == max1)
                n0 = np.sum(x0 == min0)
                result["threshold"] = (max1 * n1 + min0 * n0) / (n0 + n1)
            elif max0 == min1 or max1 == min0:
                m = max0 if max0 == min1 else max1
                below_vals = x[x < m]
                above_vals = x[x > m]
                if len(below_vals) == 0 or len(above_vals) == 0:
                    result["status"] = "singular_matrix"
                    return
                x_below = below_vals.max()
                x_above = above_vals.min()
                n0_m = np.sum((x == m) & (y == 0))
                n1_m = np.sum((x == m) & (y == 1))
                result["threshold"] = (x_below * n1_m + x_above * n0_m) / (n0_m + n1_m)
            else:
                result["status"] = "singular_matrix"
                return

            result["status"] = "singular_matrix_resolved"
            singular_resolved.append(pid)

        X_design = sm.add_constant(x)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", PerfectSeparationWarning)
                fit = sm.Logit(y, X_design).fit(disp=0)
                separated = any(issubclass(wi.category, PerfectSeparationWarning) for wi in w)

            result["fit_method"] = "logreg_statsmodels_trial"
            result["converged"] = bool(fit.mle_retvals.get("converged", False)) and not separated

            b0, b1 = fit.params
            result["slope"] = b1
            result["intercept"] = b0
            result["prsquared"] = fit.prsquared

            if separated:
                result["status"] = "separation"
                return result

            if np.isclose(b1, 0):
                result["status"] = "zero_slope"
                return result

            threshold = -b0 / b1
            result["threshold"] = threshold

            cov = fit.cov_params()
            grad_t = np.array([-1 / b1, b0 / b1 ** 2])
            var_threshold = grad_t @ cov @ grad_t

            if np.isfinite(var_threshold) and var_threshold >= 0:
                result["se"] = np.sqrt(var_threshold)
            else:
                result["status"] = "invalid_variance"

        except np.linalg.LinAlgError as e:
            result["error"] = str(e)
            try_singular_fallback()
            return result
        except Exception as e:
            result["status"] = "fit_failed"
            result["error"] = str(e)
            return result

        if result["status"] is None:
            result["status"] = "ok"

        return result

    def plot_participant(group, pid, result):
        chars = sorted(group[char_col].unique())
        colors = cm.tab10.colors
        color_map = {c: colors[i % len(colors)] for i, c in enumerate(chars)}

        fig, ax = plt.subplots(figsize=(7, 1.2 + 0.6 * len(chars)))

        for i, char_val in enumerate(chars):
            row = i + 1
            sub1 = group[(group[char_col] == char_val) & (group["Parsed_Target"] == 1)]
            sub0 = group[(group[char_col] == char_val) & (group["Parsed_Target"] == 0)]
            ax.scatter(sub1[pred], np.full(len(sub1), row),
                       color=color_map[char_val], alpha=0.6,
                       label=f"{char_col}={char_val}")
            ax.scatter(sub0[pred], np.full(len(sub0), -row),
                       color=color_map[char_val], alpha=0.6)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_yticks([])
        ax.set_xlabel(pred)
        ax.set_title(f"{pid} | status={result['status']}\n"
                     f"top: {targ}={choices[1]}, bottom: {targ}={choices[0]}")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        plt.tight_layout()
        plt.show()

    def process_participant(group):
        pid = group.name
        result = fit_threshold(group, pid)

        if verbose and result["status"] != "ok":
            plot_participant(group, pid, result)

        return pd.Series(result)

    df = pd.read_csv(df_path)
    df = df[[id_col, cond_col, pred, targ, char_col, sex_col, *args]]
    df["Parsed_Target"] = df[targ].map({choices[1]: 1, choices[0]: 0})

    if condition is not None:
        df = df[df[cond_col] == condition]

    res = df.groupby(id_col).apply(process_participant).reset_index()

    cond_map = df.groupby(id_col)[cond_col].first()
    res[cond_col] = res[id_col].map(cond_map)

    emotions_map = df.groupby(id_col)[sex_col].first()
    res[sex_col] = res[id_col].map(emotions_map)

    cols = ['ProlificID', 'threshold', 'se', 'slope', 'intercept', 'prsquared',
            'converged', 'n_trials', 'n_levels', 'fit_method', 'status', 'error',
            cond_col, sex_col]
    res = res[cols]

    if verbose:
        print("mean threshold:", res["threshold"].mean())
        print("n status != ok:", (res["status"] != "ok").sum())
        print(res["fit_method"].value_counts(dropna=False))
        print("singular matrix resolved via midpoint for:", singular_resolved)

        not_ok = res[res["status"] != "ok"]
        print(not_ok)

    return res


def filter_thresholds_in_range(res, low=0, high=1, threshold_col='threshold', status_col='status'):
    """
    Removes participants whose threshold falls outside [low, high]
    (default [0, 1]). Keeps NaN thresholds out as well, since they're
    not valid estimates to begin with. Also removes participants whose
    threshold came from a singular-matrix fallback, resolved or not.
    """
    mask = res[threshold_col].between(low, high)
    mask &= ~res[status_col].isin(["singular_matrix", "singular_matrix_resolved"])
    removed = res[~mask]
    print(f"Removed {len(removed)} participants:")
    print(removed)
    return res[mask].reset_index(drop=True)



#%%
mods1 = ['rmte','vid-between', 'gender-within', 'mood-end', 'stimrand-no']
df1 = pd.read_csv(CLEAN_DATA[1])
r_stat1, p_val = st.pearsonr(df1['sex_arousal'], df1['RMET_mean_all'])

n1 = len(df1)

print(f"r: {r_stat1:.4f}, p-value: {p_val:.4f}")
# %%
mods2 = ['rmte','vid-within', 'gender-onlyf', 'mood-end', 'stimrand-no']

df2 = pd.read_csv(CLEAN_DATA[2])
r_stat2, p_val = st.pearsonr(df2['sex_arosal_2nd'], df2['RMET_mean_all'])
n2 = len(df2)

print(f"r: {r_stat2:.4f}, p-value: {p_val:.4f}")

# %%

mods3 = ['rmte','vid-neutral', 'gender-within', 'mood-end', 'stimrand-yes']

df3 = pd.read_csv(CLEAN_DATA[3])
r_stat3, p_val = st.pearsonr(df3['sex_arousal'], df3['RMET_mean_all'])
n3 = len(df3)



print(f"r: {r_stat3:.4f}, p-value: {p_val:.4f}")

# %%

mods4 = ['rmte','vid-within', 'gender-within', 'mood-end', 'stimrand-no']

df4 = pd.read_csv(CLEAN_DATA[4])
r_stat4, p_val = st.pearsonr(df4['sex_arosal_2nd'], df4['RMET_mean_all'])
n4 = len(df4)



# %%
mods5 = ['rmte','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']

df5 = pd.read_csv(CLEAN_DATA[5])
r_stat5, p_val = st.pearsonr(df5['emotions_after_sex_arousal'], df5['RMET_mean_all'])
n5 = len(df5)



print(f"r: {r_stat5:.4f}, p-value: {p_val:.4f}")

# %%
mods6 = ['morph','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']

df6 = pd.read_csv(CLEAN_DATA[6])
res6 = morphed_faces_threshold_comparison(
    df_path='Exp6_cleanData.csv',
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='Condition',
    id_col='ProlificID',
    char_col='Character_num',
)
res6 = filter_thresholds_in_range(res6)
n6 = len(res6)


r_stat6, p_val = st.pearsonr(res6['sex_arousal_after'], res6['threshold'])

# %%
mods7 = ['morph','vid-between', 'gender-between', 'mood-both', 'stimrand-yes']

res7 = morphed_faces_threshold_comparison(
    df_path=CLEAN_DATA[7],
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='conditionVid',
    id_col='ProlificID',
    char_col='Character_num',
    sex_col='sex_arousal_after'
)
res7_c = res7.dropna(subset=['threshold'])
res7_c = filter_thresholds_in_range(res7_c)

r_stat7, p_val = st.pearsonr(res7_c['sex_arousal_after'], res7_c['threshold'])
n7 = len(res7_c)



# %%
mods8 = ['morph','vid-sexual', 'gender-onlyf', 'mood-both', 'stimrand-yes']

res8 = morphed_faces_threshold_comparison(
    df_path=CLEAN_DATA[8],
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='Condition',
    id_col='ProlificID',
    char_col='Character_num',
    sex_col='sex_arousal_after'
)
res8_c = res8.dropna(subset=['threshold'])
res8_c = filter_thresholds_in_range(res8_c)

r_stat8, p_val = st.pearsonr(res8_c['sex_arousal_after'], res8_c['threshold'])
n8 = len(res8_c)


# %%
mods9 = ['morph','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']

res9 = morphed_faces_threshold_comparison(
    df_path='cleanData9.csv',
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='conditionVid',
    id_col='ProlificID',
    char_col='Character_num',
    sex_col='sex_arousal_after'
)

res9_c = res9.dropna(subset=['threshold', 'sex_arousal_after'])
res9_c = filter_thresholds_in_range(res9_c)


r_stat9, p_val = st.pearsonr(res9_c['sex_arousal_after'], res9_c['threshold'])
n9 = len(res9_c)



# %%

mod_names = ['task', 'video', 'gender', 'mood', 'stimrand']  # rename to your 5 actual moderator names, matching list order

all_mods = [mods1, mods2, mods3, mods4, mods5, mods6, mods7, mods8, mods9]

meta_df = pd.DataFrame(all_mods, columns=mod_names)
meta_reg_input = pd.get_dummies(meta_df, columns=mod_names)

# %% Custon code random effects

def cohens_d_from_r(r):
    r = np.asarray(r, dtype=float)
    return 2 * r / np.sqrt(1 - r**2)

def pooled_cohens_d_random_effects(spearman_rs, ns):
    r = np.asarray(spearman_rs, dtype=float)
    n = np.asarray(ns, dtype=float)
    k = len(r)
    
    z = np.arctanh(r)
    v = 1 / (n - 3)                 # variance of each z
    w_fixed = 1 / v
    
    # Fixed-effect pooled estimate (needed for Q statistic)
    z_fixed = np.sum(w_fixed * z) / np.sum(w_fixed)
    Q = np.sum(w_fixed * (z - z_fixed)**2)
    df = k - 1
    
    C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0.0
    
    w_random = 1 / (v + tau2)
    z_pooled = np.sum(w_random * z) / np.sum(w_random)
    se_pooled = np.sqrt(1 / np.sum(w_random))
    
    r_pooled = np.tanh(z_pooled)
    d_pooled = cohens_d_from_r(r_pooled)
    
    z_stat = z_pooled / se_pooled
    p_value = 2 * (1 - st.norm.cdf(abs(z_stat)))
    
    return {
        "pooled_d": d_pooled,
        "pooled_r": r_pooled,
        "tau2": tau2,
        "Q": Q,
        "z_stat": z_stat,
        "p_value": p_value
    }

spearman_rs = [r_stat1,
    r_stat2,
    r_stat3,
    r_stat4,
    r_stat5,
    r_stat6,
    r_stat7,
    r_stat8,
    r_stat9]
print(pooled_cohens_d_random_effects(spearman_rs,  [n1,n2,n3,n4,n5,n6,n7,n8,n9]))


# %% Meta regression no moderators (to verify previous tests)

r = np.array([r_stat1,
    r_stat2,
    r_stat3,
    r_stat4,
    r_stat5,
    r_stat6,
    r_stat7,
    r_stat8,
    r_stat9])   # correlation values
n = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9])   # sample size per study

z = np.arctanh(r)
v = 1 / (n - 3)


dset_pooled = Dataset(y=z, v=v, add_intercept=True)
est_pooled = DerSimonianLaird()
est_pooled.fit_dataset(dset_pooled)
print(est_pooled.summary().to_df())
print(est_pooled.summary().get_heterogeneity_stats())
print(est_pooled.summary().get_re_stats())

# %% Meta regression

mods = meta_reg_input.astype(int).to_numpy()

r = np.array([r_stat1, r_stat2, r_stat3, r_stat4, r_stat5,
              r_stat6, r_stat7, r_stat8, r_stat9])   # correlation values
n = np.array([n1, n2, n3, n4, n5, n6, n7, n8, n9])   # sample size per study

for c in meta_reg_input.columns:
    X = meta_reg_input[c].astype(int).to_numpy()  # moderators

    z = np.arctanh(r)
    v = 1 / (n - 3)

    dset = Dataset(y=z, v=v, X=X, add_intercept=True)
    est = DerSimonianLaird()
    est.fit_dataset(dset)
    results = est.summary()

    df = results.to_df()

    # back-transform Fisher z estimates (and CIs, if present) to Pearson r
    df_r = df.copy()
    for col in ["estimate", "ci_0.025", "ci_0.975"]:
        if col in df_r.columns:
            df_r[col] = np.tanh(df_r[col])

    print(f"===== Moderator: {c} =====")
    print("\n-- Fisher z-scale estimates --")
    print(df)
    print("\n-- Back-transformed to Pearson r --")
    print(df_r)
    print("\n-- Heterogeneity stats --")
    print(results.get_heterogeneity_stats())
    print("\n-- Random-effects stats --")
    print(results.get_re_stats())
    print()

from matplotlib.patches import Polygon

def forest_plot(r, n, labels=None, model_result=None, alpha=0.05, ax=None):
    """
    r: array of correlation coefficients per study
    n: array of sample sizes per study
    labels: study labels (defaults to Study 1, Study 2, ...)
    model_result: optional PyMARE MetaRegressionResults (e.g. from est.summary())
                  used to plot the pooled/overall effect as a diamond
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    k = len(r)

    if labels is None:
        labels = [f"Study {i+1}" for i in range(k)]

    z = np.arctanh(r)
    se = np.sqrt(1 / (n - 3))
    z_crit = norm.ppf(1 - alpha / 2)

    lo_z = z - z_crit * se
    hi_z = z + z_crit * se

    r_lo = np.tanh(lo_z)
    r_hi = np.tanh(hi_z)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.3 * k + 1), dpi=900)

    y_pos = np.arange(k, 0, -1)

    ax.errorbar(
        r, y_pos,
        xerr=[r - r_lo, r_hi - r],
        fmt='o', color='black', ecolor='gray', capsize=3
    )

    if model_result is not None:
        df = model_result.to_df()
        est_z = df.loc[df.index[0], "estimate"] if "estimate" in df.columns else df.iloc[0, 0]
        se_z = df.loc[df.index[0], "se"] if "se" in df.columns else df.iloc[0, 1]
        est_r = np.tanh(est_z)
        est_lo = np.tanh(est_z - z_crit * se_z)
        est_hi = np.tanh(est_z + z_crit * se_z)

        h = 0.3  # half-height of the diamond
        ax.add_patch(Polygon(
            [[est_lo, 0], [est_r, h], [est_hi, 0], [est_r, -h]],
            closed=True, facecolor='red', edgecolor='red', zorder=3
        ))
        labels = list(labels) + ["Overall"]
        y_pos = np.append(y_pos, 0)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Correlation (r)")
    ax.set_ylim(min(y_pos) - 0.5, max(y_pos) + 0.5)
    plt.tight_layout()
    return ax

forest_plot(r, n, model_result=est_pooled.summary())

# %% Meta regression no moderators (to verify previous tests)

r = np.array([r_stat1,
    r_stat2,
    r_stat4,
    r_stat5,
    r_stat6,
    r_stat7,
    r_stat8,
    r_stat9])   # correlation values
n = np.array([n1,n2,n4,n5,n6,n7,n8,n9])   # sample size per study
X = mods  # moderator(s)

z = np.arctanh(r)
v = 1 / (n - 3)


dset_pooled = Dataset(y=z, v=v, add_intercept=True)
est_pooled = DerSimonianLaird()
est_pooled.fit_dataset(dset_pooled)
print(est_pooled.summary().to_df())
print(est_pooled.summary().get_heterogeneity_stats())
print(est_pooled.summary().get_re_stats())


# %%  Pooled fixed effects model only morph studies
def cohens_d_from_r(r):
    r = np.asarray(r, dtype=float)
    return 2 * r / np.sqrt(1 - r**2)

def pooled_cohens_d(spearman_rs, ns, alpha=0.05):
    """
    Compute pooled Cohen's d and p-value from Spearman correlations,
    using Fisher z inverse-variance weighting.
    
    Parameters
    ----------
    spearman_rs : array-like
        Spearman correlation coefficients.
    ns : array-like
        Sample sizes corresponding to each correlation.
    alpha : float
        Significance level for confidence intervals.
    
    Returns
    -------
    dict with pooled_d, pooled_r, z_stat, p_value, ci_r, ci_d
    """
    r = np.asarray(spearman_rs, dtype=float)
    n = np.asarray(ns, dtype=float)
    
    z = np.arctanh(r)                      # Fisher z transform
    w = n - 3                              # inverse-variance weights
    z_pooled = np.sum(w * z) / np.sum(w)
    se_pooled = np.sqrt(1 / np.sum(w))
    
    r_pooled = np.tanh(z_pooled)
    d_pooled = cohens_d_from_r(r_pooled)
    
    z_stat = z_pooled / se_pooled
    p_value = 2 * (1 - st.norm.cdf(abs(z_stat)))
    
    z_crit = st.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z_pooled - z_crit * se_pooled, z_pooled + z_crit * se_pooled
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)
    d_lo, d_hi = cohens_d_from_r(r_lo), cohens_d_from_r(r_hi)
    
    return {
        "pooled_d": d_pooled,
        "pooled_r": r_pooled,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_r": (r_lo, r_hi),
        "ci_d": (d_lo, d_hi)
    }


spearman_rs = [
    r_stat6,
    r_stat7,
    r_stat8,
    r_stat9]
print(pooled_cohens_d(spearman_rs, [n6,n7,n8,n9]))


spearman_rs = [
    r_stat1,
    r_stat2,
    r_stat3,
    r_stat4,
    r_stat5]
print(pooled_cohens_d(spearman_rs, [n1,n2,n3,n4,n5]))


# %%
r = np.array([r_stat6,
    r_stat7,
    r_stat8,
    r_stat9
    ])   # correlation values
n = np.array([n6,n7,n8,n9])   # sample size per study

z = np.arctanh(r)
v = 1 / (n - 3)

dset_pooled = Dataset(y=z, v=v, add_intercept=True)
est_pooled = DerSimonianLaird()
est_pooled.fit_dataset(dset_pooled)
print(est_pooled.summary().to_df())
print(est_pooled.summary().get_heterogeneity_stats())
print(est_pooled.summary().get_re_stats())

def pooled_r_and_d(results):
    df = results.to_df()
    z_pooled = df["estimate"].iloc[0]
    z_lo, z_hi = df["ci_0.025"].iloc[0], df["ci_0.975"].iloc[0]

    r_pooled = np.tanh(z_pooled)
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

    d_pooled = cohens_d_from_r(r_pooled)
    d_lo, d_hi = cohens_d_from_r(r_lo), cohens_d_from_r(r_hi)

    return {
        "pooled_r": r_pooled, "ci_r": (r_lo, r_hi),
        "pooled_d": d_pooled, "ci_d": (d_lo, d_hi),
    }

pooled_r_and_d(est_pooled.summary())

# %%

cols = ["gender_gender-within", "task_rmte", "mood_mood-end"]
corr = meta_reg_input[cols].corr()

with pd.option_context(
    "display.max_rows", None,
    "display.max_colwidth", None,
    "display.float_format", "{:.6f}".format,
):
    print(corr)

# %%


def pymare_fixed_effect(spearman_rs, ns, alpha=0.05):
    r = np.asarray(spearman_rs, dtype=float)
    n = np.asarray(ns, dtype=float)

    z = np.arctanh(r)
    v = 1 / (n - 3)

    dset = Dataset(y=z, v=v)
    results = WeightedLeastSquares(tau2=0).fit_dataset(dset).summary()
    df = results.to_df()

    z_pooled = df["estimate"].iloc[0]
    z_lo, z_hi = df["ci_0.025"].iloc[0], df["ci_0.975"].iloc[0]

    r_pooled = np.tanh(z_pooled)
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

    return {
        "pooled_d": cohens_d_from_r(r_pooled),
        "pooled_r": r_pooled,
        "z_stat": df["z-score"].iloc[0],
        "p_value": df["p-value"].iloc[0],
        "ci_r": (r_lo, r_hi),
        "ci_d": (cohens_d_from_r(r_lo), cohens_d_from_r(r_hi)),
    }


print(pymare_fixed_effect([r_stat6, r_stat7, r_stat8, r_stat9], [n6, n7, n8, n9]))
print(pymare_fixed_effect([r_stat1, r_stat2, r_stat3, r_stat4, r_stat5], [n1, n2, n3, n4, n5]))
# %%


#%% Violin plots of end-of-session sexual arousal, by condition
# ---------------------------------------------------------------------------
PLOT_CONFIG = {
    1: dict(kind='between', data=lambda: df1,
            arousal='sex_arousal', cond='condition'),
    2: dict(kind='within', data=lambda: df2,
            cols=('sex_arousal_neutral', 'sex_arousal_sex'),
            labels=('Neutral', 'Sexual')),
    4: dict(kind='within', data=lambda: df4,
            cols=('sex_arousal_neutral', 'sex_arousal_sex'),
            labels=('Neutral', 'Sexual')),
    5: dict(kind='between', data=lambda: df5,
            arousal='emotions_after_sex_arousal', cond='condBetween'),
    6: dict(kind='between', data=lambda: res6,
            arousal='sex_arousal_after', cond='Condition'),
    7: dict(kind='between', data=lambda: res7_c,
            arousal='sex_arousal_after', cond='conditionVid'),
    9: dict(kind='between', data=lambda: res9_c,
            arousal='sex_arousal_after', cond='conditionVid'),
}
 
 
def _get_groups(cfg):
    """Returns ([values_cond1, values_cond2], [label1, label2])."""
    df = cfg['data']()
 
    if cfg['kind'] == 'between':
        d = df.dropna(subset=[cfg['arousal'], cfg['cond']])
        levels = sorted(d[cfg['cond']].unique(), key=str)
        if len(levels) != 2:
            raise ValueError(f"expected 2 conditions, found {levels}")
        sexual = [lv for lv in levels if 'sex' in str(lv).lower()]
        if len(sexual) != 1:
            raise ValueError(f"could not identify the sexual condition in {levels}")
        ordered = [lv for lv in levels if lv != sexual[0]] + sexual
        vals = [d.loc[d[cfg['cond']] == lv, cfg['arousal']].astype(float).to_numpy()
                for lv in ordered]
        return vals, ['Neutral', 'Sexual']
     
    d = df.dropna(subset=list(cfg['cols']))            # listwise, keeps pairing
    vals = [d[c].astype(float).to_numpy() for c in cfg['cols']]
    return vals, list(cfg.get('labels', cfg['cols']))
 
 
def _compare(cfg, vals):
    """Two groups only -> t test. Between: Welch. Within: paired."""
    if cfg['kind'] == 'between':
        t, p = st.ttest_ind(vals[0], vals[1], equal_var=False)
        n1, n2 = len(vals[0]), len(vals[1])
        s1 = vals[0].var(ddof=1) / n1
        s2 = vals[1].var(ddof=1) / n2
        dof = (s1 + s2) ** 2 / (s1 ** 2 / (n1 - 1) + s2 ** 2 / (n2 - 1))
        return dict(test="Welch t", t=t, df=dof, p=p,
                    label=f"Welch t({dof:.1f}) = {t:.2f}, p = {p:.3f}")
 
    t, p = st.ttest_rel(vals[0], vals[1])
    dof = len(vals[0]) - 1
    return dict(test="paired t", t=t, df=dof, p=p,
                label=f"paired t({dof}) = {t:.2f}, p = {p:.3f}")
 
 
FIGURES = {
    "RMET experiments": [1, 2, 4, 5],
    "Morphing experiments": [6, 7, 9],
}
 
# Both figures use the same canvas and the same number of columns, so the two
# rows come out identical in width, height and panel size (unused slots hidden).
N_COLS = max(len(e) for e in FIGURES.values())
FIG_W, FIG_H, DPI = 24.0, 7.5, 800
 
stats_rows = []
 
for fig_title, exps in FIGURES.items():
    fig, axes = plt.subplots(1, len(exps), figsize=(FIG_W, FIG_H), dpi=DPI)
    axes = np.atleast_1d(axes)
 
    for ax in axes[len(exps):]:
        ax.axis('off')
 
    for ax, exp in zip(axes, exps):
        cfg = PLOT_CONFIG[exp]
 
        vals, labels = _get_groups(cfg)
        res = _compare(cfg, vals)
 
        parts = ax.violinplot(vals, positions=[1, 2], widths=0.8,
                              showmeans=False, showmedians=False, showextrema=True)
        for body in parts['bodies']:
            body.set_alpha(0.45)
 
        means = [v.mean() for v in vals]
        ax.scatter([1, 2], means, marker='D', s=110, color='black', zorder=3)
        for x, m in zip([1, 2], means):
            ax.annotate(f"M = {m:.2f}", (x, m), textcoords="offset points",
                        xytext=(14, 0), va='center', fontsize=20)
 
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{lab}\n(n = {len(v)})" for lab, v in zip(labels, vals)],
                           fontsize=21)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel("sexual arousal (end)", fontsize=21)
        ax.set_title(f"Exp {exp}", fontsize=22)
 
        stats_rows.append({"exp": exp, "design": cfg['kind'], "test": res['test'],
                           "group1": labels[0], "mean1": means[0], "n1": len(vals[0]),
                           "group2": labels[1], "mean2": means[1], "n2": len(vals[1]),
                           "t": res['t'], "df": res['df'], "p": res['p']})
 
    fig.suptitle(fig_title, fontsize=23)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
 
stats_table = pd.DataFrame(stats_rows)
print(stats_table.to_string(index=False))
# %%
