/* Code files for the landing page — full reproducible source */

const CODE_FILES = [
  {
    id: "readme",
    name: "README.md",
    lang: "markdown",
    desc: "How to reproduce the experiments.",
    code: `# Reproducing the Benchmarking Experiments

## Paper
**Toward a Standardized Framework for Missing Data Imputation Benchmarking**
Ricardo Vicente and Francisco Antunes
Coimbra Business School — ISCAC, Polytechnic Institute of Coimbra

## Quick Start

1. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

2. Keep \`utils.py\` and \`nb_master_compute.py\` in the same directory.

3. Run the experiment:
   \`\`\`
   python nb_master_compute.py
   \`\`\`
   The run produces a \`results/\` directory with one JSON checkpoint per
   condition (e.g., \`master_uniform_0.20_all_100.json\`).

   Runtime is roughly 8–12 hours on the reference machine (12th Gen i7, 16 GB RAM).
   Checkpoints are written as each condition finishes, so an interrupted run
   resumes from where it stopped — existing checkpoints are skipped.

   The experiment scope is controlled from \`utils.py\`: the number of replicates
   (N_REPS), the dataset registry (DATASET_LOADERS) and the method registry
   (METHODS). Drop an entry from either registry to leave it out of the sweep.

## Files

| File | Description |
|------|-------------|
| \`utils.py\` | Core library: dataset loaders (12 datasets), 6 imputation methods, 5 amputation algorithms, type-aware metrics, experiment runner, statistical tests |
| \`nb_master_compute.py\` | Experiment driver: defines all conditions and runs the full sweep |
| \`requirements.txt\` | Python dependencies with pinned versions |

## What Gets Computed

The experiment computes 11 unique conditions:

| Condition | Amputation | Rate | Methods | Replicates |
|-----------|------------|------|---------|------------|
| Base      | uniform    |  20% |   all 6 |   100      |
| Amputation variants | by-variable, blockwise | 20% | all 6 | 100 |
| MAR | mar-aggregate, mar-single | 20% | all 6 | 100 |
| Degradation | uniform | 5%, 10%, 30%, 40%, 50% | all 6 | 100 |
| Convergence | uniform | 20% | k-NN, SoftImpute | 200 |

Total: ~105,600 individual imputations across 12 datasets.

## Output Format

Each checkpoint file is a JSON dict:
\`\`\`
{
  "DatasetName": {
    "MethodName": [
      {  // replicate 0
        "nrmse_sd_mean": 0.467,
        "nrmse_sd_wmean": 0.466,
        "nrmse_range_mean": 0.112,
        ...
        "pfc_mean": null,       // null for continuous-only datasets
        "rmse_full": 0.174,
        "rmse_imponly": 0.392,
        "miss_rate_actual": 0.199
      },
      // ... 99 more replicates
    ]
  }
}
\`\`\`

## Datasets

The 12 datasets are loaded automatically from scikit-learn, seaborn, and UCI
(via the \`ucimlrepo\` package). No manual downloads required.

| Dataset | n | p | Source |
|---------|---|---|--------|
| Iris | 150 | 4 | scikit-learn |
| Wine | 178 | 13 | scikit-learn |
| Penguins | 333 | 4 | seaborn |
| Heart Failure | 299 | 12 | UCI (id=519) |
| Heart Disease | 297 | 13 | UCI (id=45) |
| Ionosphere | 351 | 34 | UCI (id=52) |
| Auto MPG | 392 | 7 | UCI (id=9) |
| Diabetes | 442 | 10 | scikit-learn |
| Breast Cancer | 569 | 30 | scikit-learn |
| Titanic | 712 | 7 | seaborn |
| Abalone | 4177 | 8 | UCI (id=1) |
| California Housing | 20640 | 8 | scikit-learn |

## Reproducibility

All experiments use fixed seeds (base seed = 42, replicate r uses seed 42 + r).
Within each replicate, the same amputation mask is applied to all methods.
Results should be identical on the same platform with the same package versions.

## Environment

Original experiments were run on:
- Python 3.12.3, Windows 10 Pro
- 12th Gen Intel i7-1260P, 16 GB RAM
`,
  },
  {
    id: "utils",
    name: "utils.py",
    lang: "python",
    desc: "Core library: loaders, methods, amputation, metrics, runner, stats (980 lines).",
    code: `"""
utils.py — shared functions for the imputation benchmarking experiments.

Sections:
  1. Imports & configuration
  2. Dataset loaders
  3. Dataset profiling
  4. Type-aware imputation methods (including missForest)
  5. Amputation algorithms
  6. Type-aware metrics
  7. Experiment runner
  8. Statistical tests
  9. Helpers
"""

# ============================================================
# 1. IMPORTS & CONFIGURATION
# ============================================================
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.special import expit
from scipy import stats
from scipy.stats import skew as scipy_skew
from concurrent.futures import ThreadPoolExecutor
import os
import warnings
import time

warnings.filterwarnings('ignore')

N_REPS = 100
BASE_SEED = 42
N_WORKERS = min(os.cpu_count(), 10)


# ============================================================
# 2. DATASET LOADERS
# ============================================================
# Twelve datasets in total: a mix of small/large n, low/high p, and
# continuous-only vs. mixed-type, so the benchmark isn't tied to one regime.

def load_iris_ds():
    d = load_iris()
    types = ['cont'] * 4
    print(f"Iris: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_wine_ds():
    d = load_wine()
    types = ['cont'] * 13
    print(f"Wine: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_diabetes_ds():
    d = load_diabetes(scaled=False)
    names = list(d.feature_names)
    types = ['cat' if n == 'sex' else 'cont' for n in names]
    print(f"Diabetes: {d.data.shape[0]}x{d.data.shape[1]}, 9 cont + 1 cat (raw scale)")
    return d.data, names, types

def load_breastcancer_ds():
    d = load_breast_cancer()
    types = ['cont'] * 30
    print(f"Breast Cancer: {d.data.shape[0]}x{d.data.shape[1]}, all continuous")
    return d.data, list(d.feature_names), types

def load_heart_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    cat_vars = {'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'}
    types = ['cat' if n in cat_vars else 'cont' for n in names]
    print(f"Heart Disease: {X.shape[0]}x{X.shape[1]}, 5 cont + 8 cat")
    return X, names, types

def load_abalone_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=1)
    df = ds.data.features.copy()
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cat' if n == 'Sex' else 'cont' for n in names]
    print(f"Abalone: {X.shape[0]}x{X.shape[1]}, 7 cont + 1 cat")
    return X, names, types

def load_california_housing_ds():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    X = d.data.astype(float)
    names = list(d.feature_names)
    types = ['cont'] * X.shape[1]
    print(f"California Housing: {X.shape[0]}x{X.shape[1]}, all continuous (large-n real-world)")
    return X, names, types

def load_penguins_ds():
    import seaborn as sns
    df = sns.load_dataset("penguins").dropna()
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values.astype(float)
    names = ['bill_length', 'bill_depth', 'flipper_length', 'body_mass']
    types = ['cont'] * 4
    print(f"Penguins: {X.shape[0]}x{X.shape[1]}, all continuous (small, clean)")
    return X, names, types

def load_titanic_ds():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df = df[['age', 'fare', 'pclass', 'sibsp', 'parch', 'sex', 'embarked']].dropna()
    df['sex'] = df['sex'].map({'male':0, 'female':1})
    df['embarked'] = df['embarked'].map({'C':0, 'Q':1, 'S':2})
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cont', 'cont', 'cat', 'cont', 'cont', 'cat', 'cat']
    print(f"Titanic: {X.shape[0]}x{X.shape[1]}, mixed-type (classic)")
    return X, names, types

def load_auto_mpg_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=9)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    types = ['cont'] * X.shape[1]
    print(f"Auto MPG: {X.shape[0]}x{X.shape[1]}, continuous (small)")
    return X, names, types

def load_ionosphere_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=52)
    X = ds.data.features.values.astype(float)
    names = list(ds.data.features.columns)
    types = ['cont'] * X.shape[1]
    print(f"Ionosphere: {X.shape[0]}x{X.shape[1]}, high-dim small-n continuous")
    return X, names, types

def load_heart_failure_ds():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=519)
    df = ds.data.features.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    X = df.values.astype(float)
    names = list(df.columns)
    cat_vars = {'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'}
    types = ['cat' if n in cat_vars else 'cont' for n in names]
    print(f"Heart Failure: {X.shape[0]}x{X.shape[1]}, mixed-type (clinical)")
    return X, names, types


# ====================== DATASET REGISTRY (12 datasets) ======================
DATASET_LOADERS = {
    'Iris':                load_iris_ds,
    'Wine':                load_wine_ds,
    'Diabetes':            load_diabetes_ds,
    'Breast Cancer':       load_breastcancer_ds,
    'Heart Disease':       load_heart_ds,
    'Abalone':             load_abalone_ds,
    'California Housing':  load_california_housing_ds,
    'Penguins':            load_penguins_ds,
    'Titanic':             load_titanic_ds,
    'Auto MPG':            load_auto_mpg_ds,
    'Ionosphere':          load_ionosphere_ds,
    'Heart Failure':       load_heart_failure_ds,
}

def load_available_datasets():
    loaded = {}
    for name, fn in DATASET_LOADERS.items():
        try:
            X, names, types = fn()
            loaded[name] = {'X': X, 'names': names, 'types': types}
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    return loaded

def get_type_indices(var_types):
    cont_idx = [j for j, t in enumerate(var_types) if t == 'cont']
    cat_idx = [j for j, t in enumerate(var_types) if t == 'cat']
    return cont_idx, cat_idx


# ============================================================
# 3. DATASET PROFILING
# ============================================================
def profile_dataset(X, names, types, dataset_name):
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(types)
    sd = np.std(X, axis=0, ddof=1)
    rng = np.ptp(X, axis=0)
    ratio = np.where(sd > 0, rng / sd, np.nan)
    corr = np.corrcoef(X, rowvar=False)
    upper = corr[np.triu_indices(p, k=1)]
    sk = scipy_skew(X, axis=0)

    print(f"\\n{'='*70}")
    print(f"{dataset_name}: {n}x{p} ({len(cont_idx)} cont, {len(cat_idx)} cat)")
    print(f"  Correlation: mean|r|={np.mean(np.abs(upper)):.3f}, max|r|={np.max(np.abs(upper)):.3f}")
    print(f"  Skewness: mean|skew|={np.mean(np.abs(sk)):.2f}, max|skew|={np.max(np.abs(sk)):.2f}")
    if len(cont_idx) > 0:
        r_cont = ratio[cont_idx]
        print(f"  Range/SD (cont only): min={np.nanmin(r_cont):.2f}, max={np.nanmax(r_cont):.2f}, "
              f"CV={np.nanstd(r_cont)/np.nanmean(r_cont):.2f}")
    print(f"{'='*70}")
    print(f"  {'Variable':<24} {'Type':<5} {'Mean':>8} {'SD':>8} {'Range':>8} {'Skew':>6} {'R/SD':>6}")
    print(f"  {'-'*68}")
    for j in range(p):
        nm = names[j][:24]
        print(f"  {nm:<24} {types[j]:<5} {np.mean(X[:,j]):>8.2f} {sd[j]:>8.2f} "
              f"{rng[j]:>8.2f} {sk[j]:>6.2f} {ratio[j]:>6.2f}")


# ============================================================
# 4. TYPE-AWARE IMPUTATION METHODS
# ============================================================
def _get_unique_cats(X, cat_idx):
    """Get unique category values per categorical variable (for rounding)."""
    cats = {}
    for j in cat_idx:
        vals = X[:, j]
        cats[j] = np.unique(vals[~np.isnan(vals)])
    return cats

def _round_to_categories(X_imp, cat_idx, cat_vals):
    """Round imputed categorical values to nearest valid category (vectorized)."""
    X = X_imp.copy()
    for j in cat_idx:
        if j in cat_vals and len(cat_vals[j]) > 0:
            cats = cat_vals[j]
            col = X[:, j]
            diffs = np.abs(col[:, None] - cats[None, :])
            X[:, j] = cats[np.argmin(diffs, axis=1)]
    return X


def impute_mean_mode(X_inc, var_types):
    """Mean for continuous, mode for categorical."""
    X = X_inc.copy()
    cont_idx, cat_idx = get_type_indices(var_types)
    for j in cont_idx:
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.nanmean(X[:, j])
    for j in cat_idx:
        m = np.isnan(X[:, j])
        if m.any():
            obs = X[~m, j]
            if len(obs) > 0:
                vals, counts = np.unique(obs, return_counts=True)
                X[m, j] = vals[np.argmax(counts)]
    return X


def impute_knn(X_inc, var_types, k=5):
    """k-NN with a mixed distance: standardised Euclidean on continuous
    variables plus overlap (0/1) on categoricals.

    The donor search and distance computation are vectorised with NumPy.
    The remaining Python loop runs over the missing cells, which has to
    stay element-wise because each cell can have a different set of usable
    predictor columns."""
    X = X_inc.copy()
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(var_types)
    cont_idx = np.asarray(cont_idx)
    cat_idx = np.asarray(cat_idx)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0

    # Observed mask, refreshed after each column. Columns are filled left to
    # right, so values imputed earlier become valid donors for later columns.
    is_obs = ~np.isnan(X)

    for j in range(p):
        miss = np.flatnonzero(~is_obs[:, j])
        if len(miss) == 0:
            continue

        obs = np.flatnonzero(is_obs[:, j])

        if len(obs) < k:
            if var_types[j] == 'cont':
                X[miss, j] = mu[j]
            else:
                vals_obs = X[obs, j]
                if len(vals_obs) > 0:
                    uv, uc = np.unique(vals_obs, return_counts=True)
                    X[miss, j] = uv[np.argmax(uc)]
                else:
                    X[miss, j] = mu[j]
            is_obs[miss, j] = True
            continue

        other_idx = np.delete(np.arange(p), j)

        for i in miss:
            # Columns observed in row i, excluding the target column j
            usable = other_idx[is_obs[i, other_idx]]
            if len(usable) == 0:
                X[i, j] = mu[j]
                continue

            # Keep only donors that are themselves observed on every usable column
            donor_mask = np.all(is_obs[np.ix_(obs, usable)], axis=1)
            donors = obs[donor_mask]

            if len(donors) < k:
                X[i, j] = mu[j]
                continue

            d = np.zeros(len(donors), dtype=float)

            # Continuous contribution
            usable_is_cont = np.isin(usable, cont_idx)
            usable_cont = usable[usable_is_cont]
            if len(usable_cont) > 0:
                vi = (X[i, usable_cont] - mu[usable_cont]) / sd[usable_cont]
                vd = (X[np.ix_(donors, usable_cont)] - mu[usable_cont]) / sd[usable_cont]
                d += np.sum((vd - vi)**2, axis=1)

            # Categorical contribution
            usable_cat = usable[~usable_is_cont]
            if len(usable_cat) > 0:
                ci = X[i, usable_cat]
                cd = X[np.ix_(donors, usable_cat)]
                d += np.sum(cd != ci, axis=1)

            # k nearest donors
            nn = donors[np.argsort(d)[:k]]

            if var_types[j] == 'cont':
                X[i, j] = np.mean(X[nn, j])
            else:
                vals_nn = X[nn, j]
                uv, uc = np.unique(vals_nn, return_counts=True)
                X[i, j] = uv[np.argmax(uc)]

        # Make this column's freshly imputed values available to later columns
        is_obs[miss, j] = True

    return X


def impute_iterreg(X_inc, var_types, cycles=1):
    """Iterative regression: LinearRegression for cont, LogisticRegression for cat."""
    X = X_inc.copy()
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(var_types)
    for j in cont_idx:
        m = np.isnan(X[:, j])
        X[m, j] = np.nanmean(X_inc[:, j])
    for j in cat_idx:
        m = np.isnan(X[:, j])
        obs = X_inc[~m, j]
        if len(obs) > 0:
            uv, uc = np.unique(obs, return_counts=True)
            X[m, j] = uv[np.argmax(uc)]
    for _ in range(cycles):
        for j in range(p):
            m = np.isnan(X_inc[:, j])
            if not m.any():
                continue
            oc = [c for c in range(p) if c != j]
            obs = ~m
            if obs.sum() < 5:
                continue
            try:
                if var_types[j] == 'cont':
                    lr = LinearRegression().fit(X[obs][:, oc], X_inc[obs, j])
                    X[m, j] = lr.predict(X[m][:, oc])
                else:
                    y_obs = X_inc[obs, j].astype(int)
                    if len(np.unique(y_obs)) < 2:
                        continue
                    lr = LogisticRegression(max_iter=500, solver='lbfgs',
                                            multi_class='auto').fit(X[obs][:, oc], y_obs)
                    X[m, j] = lr.predict(X[m][:, oc]).astype(float)
            except Exception:
                pass
    return X


def impute_mice(X_inc, var_types, max_iter=10, random_state=0):
    """MICE via IterativeImputer. Post-hoc rounding for categoricals."""
    cat_idx = get_type_indices(var_types)[1]
    cat_vals = _get_unique_cats(X_inc, cat_idx) if cat_idx else {}
    imp = IterativeImputer(max_iter=max_iter, random_state=random_state,
                           sample_posterior=False)
    X_imp = imp.fit_transform(X_inc)
    if cat_idx:
        X_imp = _round_to_categories(X_imp, cat_idx, cat_vals)
    return X_imp


def impute_softimpute(X_inc, var_types, max_iter=50, lam=None, tol=1e-5):
    """SoftImpute: iterative SVD thresholding. Mazumder, Hastie & Tibshirani (2010)."""
    cat_idx = get_type_indices(var_types)[1]
    cat_vals = _get_unique_cats(X_inc, cat_idx) if cat_idx else {}
    X = X_inc.copy()
    mask = np.isnan(X)
    for j in range(X.shape[1]):
        mj = mask[:, j]
        if not mj.any():
            continue
        if var_types[j] == 'cont':
            X[mj, j] = np.nanmean(X_inc[:, j])
        else:
            obs = X_inc[~mj, j]
            if len(obs) > 0:
                uv, uc = np.unique(obs, return_counts=True)
                X[mj, j] = uv[np.argmax(uc)]
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    if lam is None:
        s0 = np.linalg.svd(Z, compute_uv=False)
        lam = s0[0] * 0.1
    for it in range(max_iter):
        Z_old = Z.copy()
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s_thresh = np.maximum(s - lam, 0)
        Z_new = U * s_thresh @ Vt
        Z[mask] = Z_new[mask]
        change = np.sqrt(np.sum((Z[mask] - Z_old[mask])**2)) / (np.sqrt(np.sum(Z[mask]**2)) + 1e-10)
        if change < tol:
            break
    X_imp = Z * sd + mu
    if cat_idx:
        X_imp = _round_to_categories(X_imp, cat_idx, cat_vals)
    return X_imp


def impute_missforest(X_inc, var_types, max_iter=10, n_trees=100, random_state=0):
    """
    missForest: iterative random forest imputation for mixed-type data.
    Stekhoven & Bühlmann (2012).
    
    Uses RandomForestRegressor for continuous and RandomForestClassifier 
    for categorical variables. Iterates until convergence or max_iter.
    """
    X = X_inc.copy()
    n, p = X.shape
    cont_idx, cat_idx = get_type_indices(var_types)
    
    # Initial fill: mean for continuous, mode for categorical
    for j in cont_idx:
        m = np.isnan(X[:, j])
        if m.any():
            X[m, j] = np.nanmean(X_inc[:, j])
    for j in cat_idx:
        m = np.isnan(X[:, j])
        if m.any():
            obs = X_inc[~m, j]
            if len(obs) > 0:
                uv, uc = np.unique(obs, return_counts=True)
                X[m, j] = uv[np.argmax(uc)]
    
    # Sort variables by number of missing values (ascending)
    miss_count = np.isnan(X_inc).sum(axis=0)
    var_order = np.argsort(miss_count)
    
    # Track previous imputation for convergence
    X_prev = X.copy()
    
    for iteration in range(max_iter):
        for j in var_order:
            m = np.isnan(X_inc[:, j])
            if not m.any():
                continue
            
            obs_idx = ~m
            miss_idx = m
            
            # Predictors: all other variables
            oc = [c for c in range(p) if c != j]
            
            if obs_idx.sum() < 5:
                continue
            
            try:
                if var_types[j] == 'cont':
                    rf = RandomForestRegressor(
                        n_estimators=n_trees,
                        random_state=random_state,
                        n_jobs=-1  # parallelise the forest across cores
                    )
                    rf.fit(X[obs_idx][:, oc], X_inc[obs_idx, j])
                    X[miss_idx, j] = rf.predict(X[miss_idx][:, oc])
                else:
                    y_obs = X_inc[obs_idx, j].astype(int)
                    if len(np.unique(y_obs)) < 2:
                        continue
                    rf = RandomForestClassifier(
                        n_estimators=n_trees,
                        random_state=random_state,
                        n_jobs=-1  # parallelise the forest across cores
                    )
                    rf.fit(X[obs_idx][:, oc], y_obs)
                    X[miss_idx, j] = rf.predict(X[miss_idx][:, oc]).astype(float)
            except Exception:
                pass
        
        # Convergence: relative change in the imputed continuous cells,
        # sum((X_new - X_old)^2) / sum(X_new^2), following Stekhoven & Bühlmann.
        cont_mask = np.isnan(X_inc)
        
        if cont_idx:
            cont_miss = cont_mask[:, cont_idx]
            if cont_miss.any():
                diff_cont = np.sum((X[cont_miss[:, :len(cont_idx)].any(axis=1)][:, cont_idx] - 
                                   X_prev[cont_miss[:, :len(cont_idx)].any(axis=1)][:, cont_idx])**2)
                norm_cont = np.sum(X[:, cont_idx]**2) + 1e-10
                delta_cont = diff_cont / norm_cont
            else:
                delta_cont = 0
        else:
            delta_cont = 0
            
        if iteration > 0 and delta_cont < 1e-4:
            break
            
        X_prev = X.copy()
    
    return X


# Method registry
METHODS = {
    'Mean/Mode':   impute_mean_mode,
    'k-NN':        impute_knn,
    'IterReg':     impute_iterreg,
    'MICE':        impute_mice,
    'missForest':  impute_missforest,
    'SoftImpute':  impute_softimpute,
}


# ============================================================
# 5. AMPUTATION ALGORITHMS
# ============================================================
def ampute_uniform(X, prop, rng):
    mask = rng.random(X.shape) < prop
    for i in range(X.shape[0]):
        if mask[i].all():
            mask[i, rng.integers(0, X.shape[1])] = False
    Xa = X.copy().astype(float); Xa[mask] = np.nan
    return Xa, mask

def ampute_by_variable(X, prop, rng):
    n, p = X.shape
    mask = np.zeros_like(X, dtype=bool)
    n_miss = int(np.floor(prop * n))
    for j in range(p):
        mask[rng.choice(n, size=n_miss, replace=False), j] = True
    for i in range(n):
        if mask[i].all():
            mask[i, rng.integers(0, p)] = False
    Xa = X.copy().astype(float); Xa[mask] = np.nan
    return Xa, mask

def ampute_blockwise(X, prop, rng):
    n, p = X.shape
    mask = np.zeros_like(X, dtype=bool)
    n_rows = min(int(np.floor(prop * n * 2)), n)
    selected = rng.choice(n, size=n_rows, replace=False)
    for i in selected:
        nv = rng.integers(max(1, int(0.3*p)), max(2, int(0.7*p)) + 1)
        mask[i, rng.choice(p, size=nv, replace=False)] = True
    for i in range(n):
        if mask[i].all():
            mask[i, rng.integers(0, p)] = False
    Xa = X.copy().astype(float); Xa[mask] = np.nan
    return Xa, mask

def ampute_mar(X, prop, rng):
    """MAR-aggregate: P(Xj missing | X_{-j}) via logistic on aggregate score."""
    n, p = X.shape
    mask = np.zeros_like(X, dtype=bool)
    mu = X.mean(0); sd = X.std(0); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    for j in range(p):
        others = [c for c in range(p) if c != j]
        score = Z[:, others].sum(1)
        lo, hi = -10.0, 10.0
        for _ in range(50):
            mid = (lo+hi)/2
            if expit(mid + score).mean() < prop: lo = mid
            else: hi = mid
        mask[:, j] = rng.random(n) < expit(mid + score)
    for i in range(n):
        if mask[i].all():
            mask[i, rng.integers(0, p)] = False
    Xa = X.copy().astype(float); Xa[mask] = np.nan
    return Xa, mask

def ampute_mar_single(X, prop, rng):
    """MAR-single: P(Xj missing | X_k) where k is the most correlated variable."""
    n, p = X.shape
    mask = np.zeros_like(X, dtype=bool)
    mu = X.mean(0); sd = X.std(0); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    corr = np.corrcoef(X.T)
    for j in range(p):
        corr_j = np.abs(corr[j, :])
        corr_j[j] = 0
        corr_j[np.isnan(corr_j)] = 0
        if corr_j.max() == 0:
            mask[:, j] = rng.random(n) < prop
            continue
        k = np.argmax(corr_j)
        score = Z[:, k]
        lo, hi = -10.0, 10.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if expit(mid + score).mean() < prop: lo = mid
            else: hi = mid
        probs = expit(mid + score)
        mask[:, j] = rng.random(n) < probs
    for i in range(n):
        if mask[i].all():
            mask[i, rng.integers(0, p)] = False
    Xa = X.copy().astype(float)
    Xa[mask] = np.nan
    return Xa, mask

AMPUTE_METHODS = {
    'uniform':     ampute_uniform,
    'by_variable': ampute_by_variable,
    'blockwise':   ampute_blockwise,
    'mar':         ampute_mar,
    'mar_single':  ampute_mar_single,
}


# ============================================================
# 6. TYPE-AWARE METRICS
# ============================================================
def var_stats(X):
    """Per-variable scale stats, guarded against empty or all-NaN columns."""
    X = np.asarray(X, dtype=float)
    if X.size == 0 or X.shape[1] == 0:
        p = X.shape[1] if X.ndim > 1 else 0
        return {'sd': np.zeros(p), 'range': np.ones(p), 'iqr': np.ones(p), 'mean': np.ones(p)}
    
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    
    col_max = np.nanmax(X, axis=0)
    col_min = np.nanmin(X, axis=0)
    range_vals = np.where(np.isnan(col_max - col_min), 1.0, col_max - col_min)
    
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    iqr_vals = np.where(np.isnan(q75 - q25), 1.0, q75 - q25)
    
    return {
        'sd':    sd,
        'range': range_vals,
        'iqr':   iqr_vals,
        'mean':  np.abs(np.nanmean(X, axis=0)),
    }

def per_var_metrics(X_true, X_imp, mask, st, var_types):
    """Compute per-variable metrics, type-aware."""
    p = X_true.shape[1]
    res = {}
    for j in range(p):
        mj = mask[:, j]
        if not mj.any():
            continue
        n_miss = int(mj.sum())
        if var_types[j] == 'cont':
            err = X_true[mj, j] - X_imp[mj, j]
            rmse = np.sqrt(np.mean(err**2))
            mae = np.mean(np.abs(err))
            res[j] = {
                'type': 'cont', 'rmse': rmse, 'mae': mae,
                'nrmse_sd':    rmse / st['sd'][j]    if st['sd'][j] > 0    else np.nan,
                'nrmse_range': rmse / st['range'][j] if st['range'][j] > 0 else np.nan,
                'nrmse_iqr':   rmse / st['iqr'][j]   if st['iqr'][j] > 0   else np.nan,
                'nrmse_mean':  rmse / st['mean'][j]  if st['mean'][j] > 0  else np.nan,
                'pfc': np.nan,
                'n_miss': n_miss,
            }
        else:
            pfc = np.mean(X_true[mj, j] != X_imp[mj, j])
            res[j] = {
                'type': 'cat', 'rmse': np.nan, 'mae': np.nan,
                'nrmse_sd': np.nan, 'nrmse_range': np.nan,
                'nrmse_iqr': np.nan, 'nrmse_mean': np.nan,
                'pfc': pfc,
                'n_miss': n_miss,
            }
    return res

def aggregate_by_type(per_var, metric_key, strategy='mean'):
    """Aggregate, filtering to variables that have valid values for metric_key."""
    vals = [per_var[j][metric_key] for j in per_var
            if not np.isnan(per_var[j][metric_key])]
    weights = [per_var[j]['n_miss'] for j in per_var
               if not np.isnan(per_var[j][metric_key])]
    if not vals:
        return np.nan
    if strategy == 'mean':
        return np.mean(vals)
    elif strategy == 'weighted':
        return np.average(vals, weights=weights)
    raise ValueError(strategy)

def full_matrix_rmse(X_true, X_imp, var_types):
    """RMSE on continuous variables only, full matrix."""
    cont = [j for j, t in enumerate(var_types) if t == 'cont']
    if not cont:
        return np.nan
    return np.sqrt(np.mean((X_true[:, cont] - X_imp[:, cont])**2))

def imputed_only_rmse(X_true, X_imp, mask, var_types):
    """RMSE on continuous imputed cells only."""
    cont = [j for j, t in enumerate(var_types) if t == 'cont']
    if not cont:
        return np.nan
    cont_mask = mask.copy()
    cat_cols = [j for j, t in enumerate(var_types) if t == 'cat']
    cont_mask[:, cat_cols] = False
    if not cont_mask.any():
        return np.nan
    return np.sqrt(np.mean((X_true[cont_mask] - X_imp[cont_mask])**2))


# ============================================================
# 7. EXPERIMENT RUNNER
# ============================================================
def run_single(X, method_fn, ampute_fn, prop, var_types, rng_seed):
    rng = np.random.default_rng(rng_seed)
    X_amp, mask = ampute_fn(X, prop, rng)
    st = var_stats(X)
    X_imp = method_fn(X_amp, var_types)
    pv = per_var_metrics(X, X_imp, mask, st, var_types)

    result = {}
    for mk in ['nrmse_sd', 'nrmse_range', 'nrmse_iqr', 'nrmse_mean', 'rmse', 'mae']:
        result[f'{mk}_mean'] = aggregate_by_type(pv, mk, 'mean')
        result[f'{mk}_wmean'] = aggregate_by_type(pv, mk, 'weighted')
    result['pfc_mean'] = aggregate_by_type(pv, 'pfc', 'mean')
    result['pfc_wmean'] = aggregate_by_type(pv, 'pfc', 'weighted')
    result['rmse_full'] = full_matrix_rmse(X, X_imp, var_types)
    result['rmse_imponly'] = imputed_only_rmse(X, X_imp, mask, var_types)
    result['miss_rate_actual'] = mask.sum() / mask.size
    result['per_var'] = pv
    return result


from joblib import Parallel, delayed

# missForest already sets n_jobs=-1 inside each fit, so its replicates run
# sequentially to avoid oversubscribing the cores. Everything else is light
# enough to parallelise across replicates.
_HEAVY_METHODS = {'missForest'}

def run_mc(X, method_fn, ampute_fn, prop, var_types, n_reps, base_seed=BASE_SEED):
    """Monte Carlo over n_reps replicates.

    missForest runs its replicates sequentially (the forests already use every
    core); the lighter methods spread their replicates across cores via joblib.
    """
    method_name = None
    for name, fn in METHODS.items():
        if fn is method_fn:
            method_name = name
            break

    is_heavy = method_name in _HEAVY_METHODS

    if is_heavy:
        results = []
        for r in range(n_reps):
            results.append(run_single(X, method_fn, ampute_fn, prop, var_types, base_seed + r))
            if (r + 1) % 10 == 0:
                print(f"    [{method_name}] rep {r+1}/{n_reps}")
        return results
    else:
        def single_rep(r):
            return run_single(X, method_fn, ampute_fn, prop, var_types, base_seed + r)
        return Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(single_rep)(r) for r in range(n_reps)
        )

def summarise(results, key):
    vals = [r[key] for r in results if not np.isnan(r[key])]
    if not vals:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals)


# ============================================================
# 8. STATISTICAL TESTS
# ============================================================
def friedman_test(results_dict, n_reps, metric_key):
    methods = list(results_dict.keys())
    matrix = np.column_stack([
        [results_dict[m][r][metric_key] for r in range(n_reps)]
        for m in methods
    ])
    valid = ~np.isnan(matrix).any(axis=0)
    if valid.sum() < 2:
        return np.nan, np.nan, {}
    matrix = matrix[:, valid]
    valid_methods = [m for m, v in zip(methods, valid) if v]
    ranks = np.zeros_like(matrix)
    for r in range(n_reps):
        order = np.argsort(matrix[r, :])
        for rank, idx in enumerate(order, 1):
            ranks[r, idx] = rank
    mean_ranks = {m: ranks[:, i].mean() for i, m in enumerate(valid_methods)}
    stat, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    return stat, p, mean_ranks

def wilcoxon_paired(results_a, results_b, metric_key, n_reps):
    a = np.array([results_a[r][metric_key] for r in range(n_reps)])
    b = np.array([results_b[r][metric_key] for r in range(n_reps)])
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 10 or np.allclose(a, b):
        return np.nan, 1.0, 0.0
    stat, p = stats.wilcoxon(a, b, alternative='two-sided')
    return stat, p, a.mean() - b.mean()

def nemenyi_cd(k, n, alpha=0.05):
    q = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:3.006}.get(k, 3.006)
    return q * np.sqrt(k * (k+1) / (6*n))

def print_stats(results_dict, metric_key, n_reps):
    methods = list(results_dict.keys())
    stat, p, mean_ranks = friedman_test(results_dict, n_reps, metric_key)
    if not mean_ranks:
        print("  (metric not available for enough methods)")
        return
    print(f"  Friedman: chi2={stat:.2f}, p={p:.2e}")
    ranked = sorted(mean_ranks, key=mean_ranks.get)
    print(f"  Ranks: " + "  ".join(f"{m}={mean_ranks[m]:.2f}" for m in ranked))
    cd = nemenyi_cd(len(mean_ranks), n_reps)
    print(f"  Nemenyi CD (a=0.05): {cd:.3f}")
    for i in range(len(ranked)-1):
        m1, m2 = ranked[i], ranked[i+1]
        _, wp, diff = wilcoxon_paired(results_dict[m1], results_dict[m2], metric_key, n_reps)
        sig = "***" if wp<0.001 else "**" if wp<0.01 else "*" if wp<0.05 else "ns"
        print(f"    {m1} vs {m2}: D={diff:+.4f}, p={wp:.4f} {sig}")


# ============================================================
# 9. HELPERS
# ============================================================
def run_for_all_datasets(demo_fn, all_data, n_reps=N_REPS, **kwargs):
    """Run a demo function across all loaded datasets."""
    all_results = {}
    for ds_name, ds in all_data.items():
        t0 = time.time()
        res = demo_fn(ds['X'], ds['names'], ds['types'], ds_name, n_reps, **kwargs)
        elapsed = time.time() - t0
        print(f"  [{ds_name}: {elapsed:.0f}s]")
        all_results[ds_name] = res
    return all_results


# ============================================================
# 10. RESULTS EXPORT / IMPORT
# ============================================================
import json

RESULTS_DIR = "results"

def _make_serialisable(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def save_results(section_name, results_dict):
    """Write results to results/<section_name>.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{section_name}.json")
    clean = _make_serialisable(results_dict)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {path}")

def load_results(section_name):
    """Load results from results/<section_name>.json."""
    path = os.path.join(RESULTS_DIR, f"{section_name}.json")
    with open(path, 'r') as f:
        return json.load(f)

def load_all_results():
    """Load all result files from results/ directory."""
    all_res = {}
    if not os.path.exists(RESULTS_DIR):
        print(f"No {RESULTS_DIR}/ directory found.")
        return all_res
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            key = fname.replace('.json', '')
            with open(os.path.join(RESULTS_DIR, fname), 'r') as f:
                all_res[key] = json.load(f)
            print(f"  Loaded {fname}")
    return all_res


def extract_summary_table(results, metric_key='nrmse_sd_mean'):
    """
    From a {method: [replicate_results]} dict, extract a summary table.
    Returns dict of {method: (mean, sd)}.
    """
    summary = {}
    for mname, reps in results.items():
        if isinstance(reps, list):
            vals = [r[metric_key] for r in reps if r.get(metric_key) is not None]
            if vals:
                summary[mname] = (np.mean(vals), np.std(vals))
    return summary
`,
  },
  {
    id: "compute",
    name: "nb_master_compute.py",
    lang: "python",
    desc: "Experiment driver: all conditions, checkpoint save/resume.",
    code: `# Master compute script.
# Runs every unique (dataset x amputation x rate x method) combination once
# and writes the per-replicate results to JSON. Conditions that are shared
# across analyses (e.g. uniform MCAR at 20%) are therefore computed a single
# time rather than repeated.

import time
from utils import *

ALL_DATA = load_available_datasets()
print(f"\\nDatasets loaded: {len(ALL_DATA)}")
for name, ds in ALL_DATA.items():
    print(f"  {name}: {ds['X'].shape}")

# Each job is (amputation_key, rate, method_set, n_reps).
# method_set: 'all' = the six methods; 'convergence' = k-NN + SoftImpute only.

JOBS = [
    # Baseline: uniform MCAR at 20% (the most reused condition)
    ('uniform', 0.20, 'all', N_REPS),

    # Other amputation mechanisms at 20%
    ('by_variable', 0.20, 'all', N_REPS),
    ('blockwise', 0.20, 'all', N_REPS),
    ('mar', 0.20, 'all', N_REPS),
    ('mar_single', 0.20, 'all', N_REPS),

    # Rate degradation under uniform (20% already covered above)
    ('uniform', 0.05, 'all', N_REPS),
    ('uniform', 0.10, 'all', N_REPS),
    ('uniform', 0.30, 'all', N_REPS),
    ('uniform', 0.40, 'all', N_REPS),
    ('uniform', 0.50, 'all', N_REPS),

    # Convergence study: uniform 20%, two methods, 200 replicates
    ('uniform', 0.20, 'convergence', 200),
]

# Count the unique imputations the sweep will run
total = 0
for amp, rate, mset, nreps in JOBS:
    n_methods = 2 if mset == 'convergence' else len(METHODS)
    n_imps = len(ALL_DATA) * n_methods * nreps
    total += n_imps
    print(f"  {amp:15s} {rate:4.0%} {mset:>12s} {nreps:>4d} reps → {n_imps:>7,d} imputations")
print(f"\\n  TOTAL: {total:>10,d} imputations")

# Compute loop

import os, json
from utils import _make_serialisable

os.makedirs(RESULTS_DIR, exist_ok=True)

for amp_key, rate, method_set, n_reps in JOBS:
    if method_set == 'all':
        methods_to_run = METHODS
    elif method_set == 'convergence':
        methods_to_run = {m: METHODS[m] for m in ['k-NN', 'SoftImpute']}
    else:
        methods_to_run = METHODS
    
    job_id = f"{amp_key}_{rate:.2f}_{method_set}_{n_reps}"
    checkpoint = os.path.join(RESULTS_DIR, f"master_{job_id}.json")
    
    if os.path.exists(checkpoint):
        print(f"SKIP {job_id}: checkpoint exists")
        continue
    
    print(f"\\n{'='*70}")
    print(f"JOB: {amp_key} @ {rate:.0%}, {len(methods_to_run)} methods, {n_reps} reps")
    print(f"{'='*70}")
    
    amp_fn = AMPUTE_METHODS[amp_key]
    job_results = {}
    
    for ds_name, ds in ALL_DATA.items():
        X = ds['X']
        var_types = ds['types']
        
        job_results[ds_name] = {}
        
        for mname, mfn in methods_to_run.items():
            t0 = time.time()
            results = run_mc(X, mfn, amp_fn, rate, var_types, n_reps)
            elapsed = time.time() - t0
            
            mu, sd = summarise(results, 'nrmse_sd_mean')
            print(f"  {ds_name:25s} | {mname:12s} | NRMSE_sd={mu:.4f}±{sd:.3f} | {elapsed:.1f}s")
            
            # Drop per_var from the stored output to keep the JSON small
            job_results[ds_name][mname] = [
                {k: v for k, v in r.items() if k != 'per_var'} for r in results
            ]
    
    # Save checkpoint
    clean = _make_serialisable(job_results)
    with open(checkpoint, 'w') as f:
        json.dump(clean, f)
    print(f"  Saved: {checkpoint}")

print("\\n\\nALL JOBS COMPLETE!")

# Quick inventory of what landed on disk
n_files = 0
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.startswith('master_') and f.endswith('.json'):
        size_mb = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)")
        n_files += 1
print(f"\\nTotal: {n_files} checkpoint files")
`,
  },
  {
    id: "requirements",
    name: "requirements.txt",
    lang: "text",
    desc: "Python dependencies (pinned versions).",
    code: `# Requirements for reproducing the benchmarking experiments
# Tested with Python 3.12.3

numpy==1.26.4
pandas==2.2.3
scikit-learn==1.4.2
scipy==1.13.1
joblib>=1.3.0
seaborn>=0.13.0
ucimlrepo>=0.0.7
matplotlib>=3.8.0
`,
  },
];

window.CODE_FILES = CODE_FILES;
