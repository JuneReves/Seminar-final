#%% Preloading
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

def comp_pooled_d(n1, n2, sd1, sd2, mean1, mean2):
    pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
    d = (mean1 - mean2) / pooled_sd
    v = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2))
    return d, v, pooled_sd

def pooled_d_helper(c,n,sd,m):
    return comp_pooled_d(
        n[c[0]], n[c[1]],
        sd[c[0]], sd[c[1]],
        m[c[0]], m[c[1]])


def comp_paired_d(n, sd_diff, mean_diff):
    d = mean_diff / sd_diff
    v = 1/n + (d**2) / (2*n)
    return d, v, sd_diff



def morphed_faces_threshold_comparison(
        df_path, pred, targ, choices,
        cond_col='Condition', id_col='ProlificID',
        char_col='Character_num', condition=None, *args, verbose=False):
    """
    Per participant, fits a no-penalty trial-level binomial GLM (statsmodels)
    for threshold (PSE) extraction. On singular-matrix failures, falls back
    to a weighted-midpoint threshold if the two response classes are cleanly
    separated on x. Plots raw per-trial responses when the fit is not ok:
    one scatter plot per participant, response=1 trials above the midline,
    response=0 trials below, one row per character, colored by character.
    """

    singular_resolved = []

    def fit_threshold(group, pid):
        result = {
            "threshold": np.nan,
            "se": np.nan,
            "slope": np.nan,
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
    df = df[[id_col, cond_col, pred, targ, char_col, *args]]
    df["Parsed_Target"] = df[targ].map({choices[1]: 1, choices[0]: 0})

    if condition is not None:
        df = df[df[cond_col] == condition]

    res = df.groupby(id_col).apply(process_participant).reset_index()

    cond_map = df.groupby(id_col)[cond_col].first()
    res[cond_col] = res[id_col].map(cond_map)

    cols = ['ProlificID', 'threshold', 'se', 'slope', 'prsquared', 'converged',
            'n_trials', 'n_levels', 'fit_method', 'status', 'error',
            cond_col]
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


#%% Exp 1
cdf1 = pd.read_csv(CLEAN_DATA[1])

c1 = ['RMTE_Neutral', 'RMTE_Sexual']

m1 = cdf1.groupby('condition')['RMET_mean_all'].mean()
sd1 = cdf1.groupby('condition')['RMET_mean_all'].std()

print(cdf1['ProlificID'].nunique())
n1 = cdf1.groupby('condition')['ProlificID'].nunique()

d1, v1, p_sd1 = pooled_d_helper(c1,n1,sd1,m1)

#%% exp 2
ldf2 = pd.read_csv('exp2long.csv')
# Keep relevant columns
ldf2 = ldf2[["ProlificID", "withinCondition", "item", "correctAnswer"]]

# Convert to wide format (one row per participant)
accuracy = (
ldf2.groupby(["ProlificID", "withinCondition"])["correctAnswer"]
.mean()
.reset_index()
)

accuracy_wide2 = (
accuracy.pivot(
index="ProlificID",
columns="withinCondition",
values="correctAnswer"
)
.reset_index()
)
n2 = accuracy_wide2['ProlificID'].nunique()
accuracy_wide2['diff'] = -accuracy_wide2['Sex_clip']+accuracy_wide2['Neutral_clip']

d2, v2, p_sd2 = comp_paired_d(n2, accuracy_wide2['diff'].std(), accuracy_wide2['diff'].mean())

#%%
ldf4 = pd.read_csv('exp4long.csv')
# Keep relevant columns
ldf4 = ldf4[["ProlificID", "withinCondition", "item", "correctAnswer"]]
# Convert to wide format (one row per participant)
accuracy = (
ldf4.groupby(["ProlificID", "withinCondition"])["correctAnswer"]
.mean()
.reset_index()
)

accuracy_wide4 = (
accuracy.pivot(
index="ProlificID",
columns="withinCondition",
values="correctAnswer"
)
.reset_index()
)

accuracy_wide4['diff'] = -accuracy_wide4['Sex_clip']+accuracy_wide4['Neutral_clip']

n4 = accuracy_wide2['ProlificID'].nunique()
accuracy_wide2['diff'] = -accuracy_wide4['Sex_clip']+accuracy_wide4['Neutral_clip']

d4, v4, p_sd4 = comp_paired_d(n2, accuracy_wide4['diff'].std(), accuracy_wide4['diff'].mean())

#%%
cdf5 = pd.read_csv(CLEAN_DATA[5])

c5 = ['Neutral', 'Sexual']
# Keep relevant columns
# cdf5 = cdf5[["ProlificID", "condBetween", "item", "correctAnswer"]]
print(cdf5.groupby('condBetween')['RMET_mean_all'].mean())
print(cdf5.groupby('condBetween')['RMET_mean_all'].std())

n5 = cdf5.groupby('condBetween')['ProlificID'].nunique()
m5 = cdf5.groupby('condBetween')['RMET_mean_all'].mean()
sd5 = cdf5.groupby('condBetween')['RMET_mean_all'].std()


d5, v5, p_sd5 = pooled_d_helper(c5,n5,sd5,m5)
#%%
c6 = ['Neutral', 'Sexual']

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

n6 = res6.groupby('Condition')['ProlificID'].nunique()
m6 = res6.groupby('Condition')['threshold'].mean()
sd6 = res6.groupby('Condition')['threshold'].std()


d6, v6, p_sd6 = pooled_d_helper(c6,n6,sd6,m6)
#%%
res7 = morphed_faces_threshold_comparison(
    df_path=CLEAN_DATA[7],
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='conditionVid',
    id_col='ProlificID',
    char_col='Character_num',
)
res7 = res7.dropna(subset=['threshold'])
res7 = filter_thresholds_in_range(res7)



n7 = res7.groupby('conditionVid')['ProlificID'].nunique()
m7 = res7.groupby('conditionVid')['threshold'].mean()
sd7 = res7.groupby('conditionVid')['threshold'].std()
c7 = ['Neutral', 'Sexual']

d7, v7, p_sd7 = pooled_d_helper(c7,n7,sd7,m7)


mods9 = ['morph','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']

res9 = morphed_faces_threshold_comparison(
    df_path='cleanData9.csv',
    pred='MorphLevel',
    targ='choice',
    choices=['Neutral', 'Positive'],
    cond_col='conditionVid',
    id_col='ProlificID',
    char_col='Character_num',
)
res9_c = res9.dropna(subset=['threshold'])
res9_c = filter_thresholds_in_range(res9_c)



n9 = res9.groupby('conditionVid')['ProlificID'].nunique()
m9 = res9.groupby('conditionVid')['threshold'].mean()
sd9 = res9.groupby('conditionVid')['threshold'].std()
c9 = ['Control','Sexual']

d9, v9, p_sd9 = pooled_d_helper(c9,n9,sd9,m9)


# %%

#%%
from pymare import Dataset
from pymare.estimators import DerSimonianLaird

mod_names = ['task', 'video', 'gender', 'mood', 'stimrand']  # rename to your 5 actual moderator names, matching list order

mods1 = ['rmte','vid-between', 'gender-within', 'mood-end', 'stimrand-no']
mods2 = ['rmte','vid-within', 'gender-onlyf', 'mood-end', 'stimrand-no']
mods4 = ['rmte','vid-within', 'gender-within', 'mood-end', 'stimrand-no']
mods5 = ['rmte','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']
mods6 = ['morph','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']
mods7 = ['morph','vid-between', 'gender-between', 'mood-both', 'stimrand-yes']
mods9 = ['morph','vid-between', 'gender-onlyf', 'mood-both', 'stimrand-yes']

all_mods = [mods1,mods2,mods4,mods5,mods6,mods7,mods9]

meta_df = pd.DataFrame(all_mods, columns=mod_names)
meta_reg_input = pd.get_dummies(meta_df, columns=mod_names)
mods = meta_reg_input.astype(int).to_numpy()


# Inputs
d = np.array([
d1,d2,d4,d5,d6,d7,d9
])  
v = np.array([v1,v2,v4,v5,v6,v7,v9])   


for c in meta_reg_input.columns:
    X = meta_reg_input[c].astype(int).to_numpy()  # moderators
    # Variance of d (Hedges & Olkin approximation)

    dset = Dataset(y=d, v=v, X=X, add_intercept=True)
    est = DerSimonianLaird()
    est.fit_dataset(dset)
    results = est.summary()

    print('Moderator: ', c)
    print(results.to_df())
# %%
dset_pooled = Dataset(y=d, v=v, add_intercept=True)
est_pooled = DerSimonianLaird()
est_pooled.fit_dataset(dset_pooled)
print(est_pooled.summary().to_df())

# Full study set (matches original d, v arrays)
labels = ['Study 1', 'Study 2', 'Study 4', 'Study 5', 'Study 6', 'Study 7', 'Study 9']
effects = d  # np.array([d1,d2,d4,d5,d6,d7,d9])
ses = np.sqrt(v)
ci_low = effects - 1.96 * ses
ci_high = effects + 1.96 * ses

# Pooled random-effects estimate (no moderators)
dset_overall = Dataset(y=d, v=v, add_intercept=True)
est_overall = DerSimonianLaird()
est_overall.fit_dataset(dset_overall)
pooled_row = est_overall.summary().to_df().iloc[0]
pooled_est, pooled_lo, pooled_hi = pooled_row['estimate'], pooled_row['ci_0.025'], pooled_row['ci_0.975']

fig, ax = plt.subplots(figsize=(8,4), dpi=800)

y_pos = np.arange(len(labels))[::-1]

ax.errorbar(effects, y_pos, xerr=[effects - ci_low, ci_high - effects],
            fmt='s', color='black', ecolor='black', capsize=3, markersize=6)

def diamond(ax, est, lo, hi, y, color):
    ax.plot([lo, est, hi, est, lo], [y, y + 0.15, y, y - 0.15, y], color=color)

pooled_y = -1
diamond(ax, pooled_est, pooled_lo, pooled_hi, pooled_y, 'tab:blue')

all_y = list(y_pos) + [pooled_y]
all_labels = labels + ['Pooled']

ax.set_yticks(all_y)
ax.set_yticklabels(all_labels)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel("Effect size (d)")
ax.set_title("")

plt.tight_layout()
plt.show()
# %%

# %%

#%%
from pymare import Dataset
from pymare.estimators import DerSimonianLaird, WeightedLeastSquares

# Subset: studies 6, 7, 9
d_sub = np.array([d6, d7, d9])
v_sub = np.array([v6, v7, v9])

# 1. Mixed/random-effects, no moderators (intercept-only)
dset_re = Dataset(y=d_sub, v=v_sub)  # add_intercept=True by default
est_re = DerSimonianLaird()
est_re.fit_dataset(dset_re)
results_re = est_re.summary()

print('=== Mixed-effects (DerSimonianLaird), studies 6,7,9 ===')
print(results_re.to_df())
print('params:', est_re.params_)  # includes tau2

# 2. Fixed-effects, no moderators — tau2 fixed at 0 (the only correction needed;
# v already carries the correct sampling variance, so no re-weighting is required)
dset_fe = Dataset(y=d_sub, v=v_sub)
est_fe = WeightedLeastSquares(tau2=0)
est_fe.fit_dataset(dset_fe)
results_fe = est_fe.summary()

print('=== Fixed-effects (WeightedLeastSquares, tau2=0), studies 6,7,9 ===')
print(results_fe.to_df())
# %%