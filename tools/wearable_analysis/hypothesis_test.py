"""
Literature-grounded hypothesis testing for N=1 wearable data.
Tests pre-registered hypotheses against personal data using
Bayesian updating with literature priors.

Performance: pre-computes dropna pairs, caches correlation matrices,
uses analytical CIs instead of bootstrap where possible.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import yaml
import os
import logging
import warnings
from .config import resolve_column

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_BOOTSTRAP_N = 500   # reduced from 1000; analytical CI is primary
_P_THRESHOLD = 0.05
_EFFECT_THRESHOLDS = {
    'small': 0.10,
    'medium': 0.30,
    'large': 0.50,
}


# ═══════════════════════════════════════════════════════════════════════════
# Pre-computation cache for DataFrame operations
# ═══════════════════════════════════════════════════════════════════════════

class _HypothesisCache:
    """Pre-computes and caches expensive DataFrame operations.

    Avoids repeated dropna/copy per hypothesis by storing pairwise
    valid-mask intersections and pre-extracted numpy arrays.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        # Pre-compute non-null masks for all numeric columns
        self._masks = {col: df[col].notna().values for col in self._numeric_cols}
        # Pre-extract numpy arrays for all numeric columns
        self._arrays = {col: df[col].values for col in self._numeric_cols}
        # Cache for pairwise clean data: (col1, col2) -> (x, y, n)
        self._pair_cache = {}
        # Cache for Spearman correlation matrix (computed lazily)
        self._spearman_matrix = None
        self._spearman_cols = None

    def get_pair(self, col1: str, col2: str):
        """Get clean (no-NaN) paired arrays for two columns.

        Returns (x_array, y_array, n) or (None, None, 0) if insufficient.
        """
        key = (col1, col2)
        if key in self._pair_cache:
            return self._pair_cache[key]

        if col1 not in self._masks or col2 not in self._masks:
            result = (None, None, 0)
        else:
            mask = self._masks[col1] & self._masks[col2]
            n = mask.sum()
            if n < 5:
                result = (None, None, n)
            else:
                result = (self._arrays[col1][mask], self._arrays[col2][mask], n)

        self._pair_cache[key] = result
        return result

    def get_triple(self, col1: str, col2: str, col3: str):
        """Get clean paired arrays for three columns."""
        if col1 not in self._masks or col2 not in self._masks or col3 not in self._masks:
            return None, None, None, 0
        mask = self._masks[col1] & self._masks[col2] & self._masks[col3]
        n = mask.sum()
        if n < 5:
            return None, None, None, n
        return (self._arrays[col1][mask], self._arrays[col2][mask],
                self._arrays[col3][mask], n)

    def get_single(self, col: str):
        """Get clean array for a single column."""
        if col not in self._masks:
            return None, 0
        mask = self._masks[col]
        vals = self._arrays[col][mask]
        return vals, len(vals)


# Global cache instance — set per run_all_hypotheses call
_cache: _HypothesisCache = None


def _effect_size_label(r: float) -> str:
    """Classify |r| into small / medium / large (Cohen 1988)."""
    ar = abs(r)
    if ar >= _EFFECT_THRESHOLDS['large']:
        return 'large'
    if ar >= _EFFECT_THRESHOLDS['medium']:
        return 'medium'
    if ar >= _EFFECT_THRESHOLDS['small']:
        return 'small'
    return 'negligible'


def _cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD denominator)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load hypotheses from YAML
# ═══════════════════════════════════════════════════════════════════════════

def load_hypotheses(hypotheses_dir: str, domains: list = None) -> list:
    """Load hypothesis definitions from YAML files in *hypotheses_dir*.

    Each YAML file is expected to contain a top-level list of hypothesis dicts.
    Optionally filter by *domains* (e.g. ``['sleep', 'recovery']``).
    """
    hyp_path = Path(hypotheses_dir)
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hypotheses directory not found: {hypotheses_dir}")

    all_hyps: list[dict] = []
    for fpath in sorted(hyp_path.glob('*.yaml')):
        with open(fpath, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            all_hyps.extend(data)
        elif isinstance(data, dict) and 'hypotheses' in data:
            all_hyps.extend(data['hypotheses'])

    if domains:
        domains_lower = [d.lower() for d in domains]
        all_hyps = [h for h in all_hyps if h.get('domain', '').lower() in domains_lower]

    return all_hyps


# ═══════════════════════════════════════════════════════════════════════════
# 2. Correlation test
# ═══════════════════════════════════════════════════════════════════════════

def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    """Analytical CI for correlation using Fisher z-transformation.

    ~100x faster than bootstrap with equivalent accuracy for n > 25.
    """
    if abs(r) >= 1.0 or n < 5:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


def test_correlation(df: pd.DataFrame, var1: str, var2: str,
                     method: str = 'pearson') -> dict:
    """Test bivariate correlation with analytical Fisher CI.

    Parameters
    ----------
    method : 'pearson' | 'spearman' | 'kendall'

    Returns dict with r, p, ci_low, ci_high, n, method, effect_size.
    """
    global _cache

    # Use cache if available, otherwise fall back to DataFrame ops
    if _cache is not None:
        x, y, n = _cache.get_pair(var1, var2)
        if x is None:
            return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'n': n, 'method': method, 'effect_size': 'insufficient_data'}
    else:
        sub = df[[var1, var2]].dropna()
        n = len(sub)
        if n < 5:
            return {'r': np.nan, 'p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'n': n, 'method': method, 'effect_size': 'insufficient_data'}
        x, y = sub[var1].values, sub[var2].values

    if method == 'spearman':
        r, p = stats.spearmanr(x, y)
    elif method == 'kendall':
        r, p = stats.kendalltau(x, y)
    else:
        r, p = stats.pearsonr(x, y)

    # Analytical Fisher z CI (fast, accurate for n > 25)
    ci_low, ci_high = _fisher_ci(float(r), n)

    return {
        'r': round(float(r), 4),
        'p': round(float(p), 6),
        'ci_low': round(ci_low, 4) if not np.isnan(ci_low) else np.nan,
        'ci_high': round(ci_high, 4) if not np.isnan(ci_high) else np.nan,
        'n': n,
        'method': method,
        'effect_size': _effect_size_label(r),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dose-response test
# ═══════════════════════════════════════════════════════════════════════════

def test_dose_response(df: pd.DataFrame, x: str, y: str,
                       bins: list = None, n_bins: int = 8) -> dict:
    """Test dose-response relationship by binning *x* and computing *y* stats
    per bin, then fitting linear and quadratic curves to detect shape.

    Returns dict with bin statistics, linearity test, optimal range, and shape.
    """
    global _cache

    if _cache is not None:
        xv_raw, yv_raw, n_raw = _cache.get_pair(x, y)
        if xv_raw is None or n_raw < 20:
            return {'bins': [], 'means': [], 'ns': [], 'ci_lows': [], 'ci_highs': [],
                    'linear_r': np.nan, 'linear_p': np.nan,
                    'optimal_range': (np.nan, np.nan), 'shape': 'insufficient_data'}
        # Need a DataFrame for pd.cut/qcut — build minimal one
        sub = pd.DataFrame({x: xv_raw, y: yv_raw})
        xv, yv = xv_raw, yv_raw
    else:
        sub = df[[x, y]].dropna()
        if len(sub) < 20:
            return {'bins': [], 'means': [], 'ns': [], 'ci_lows': [], 'ci_highs': [],
                    'linear_r': np.nan, 'linear_p': np.nan,
                    'optimal_range': (np.nan, np.nan), 'shape': 'insufficient_data'}
        xv, yv = sub[x].values, sub[y].values

    # Create bins
    if bins is not None:
        sub = sub.copy()
        sub['_bin'] = pd.cut(sub[x], bins=bins, include_lowest=True)
    else:
        sub = sub.copy()
        try:
            sub['_bin'] = pd.qcut(sub[x], q=n_bins, duplicates='drop')
        except ValueError:
            sub['_bin'] = pd.cut(sub[x], bins=n_bins, duplicates='drop')

    grouped = sub.groupby('_bin', observed=True)[y]
    bin_labels = []
    means, ns, ci_lows, ci_highs = [], [], [], []
    for name, group in grouped:
        vals = group.values
        if len(vals) < 2:
            continue
        m = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        bin_labels.append(str(name))
        means.append(round(m, 4))
        ns.append(len(vals))
        ci_lows.append(round(m - 1.96 * se, 4))
        ci_highs.append(round(m + 1.96 * se, 4))

    # Linear fit
    r_lin, p_lin = stats.pearsonr(xv, yv) if len(xv) >= 5 else (np.nan, np.nan)

    # Determine shape: compare linear vs quadratic R2
    from numpy.polynomial import polynomial as P
    shape = 'linear'
    if len(xv) >= 10:
        # Linear R2
        slope, intercept, _, _, _ = stats.linregress(xv, yv)
        pred_lin = intercept + slope * xv
        ss_res_lin = np.sum((yv - pred_lin) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0

        # Quadratic fit
        coeffs = np.polyfit(xv, yv, 2)
        pred_quad = np.polyval(coeffs, xv)
        ss_res_quad = np.sum((yv - pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

        # Classify shape
        if r2_quad - r2_lin > 0.02 and coeffs[0] < 0:
            shape = 'inverted-U'
        elif r2_quad - r2_lin > 0.02 and coeffs[0] > 0:
            shape = 'U-shaped'
        elif r_lin > 0 and p_lin < 0.05:
            shape = 'linear_positive'
        elif r_lin < 0 and p_lin < 0.05:
            shape = 'linear_negative'
        else:
            # Check for plateau using piecewise comparison
            if len(means) >= 3:
                first_half = means[:len(means) // 2]
                second_half = means[len(means) // 2:]
                diff_first = max(first_half) - min(first_half) if first_half else 0
                diff_second = max(second_half) - min(second_half) if second_half else 0
                if diff_first > 2 * diff_second and diff_second < 2:
                    shape = 'plateau'
                else:
                    shape = 'flat'
            else:
                shape = 'flat'

    # Optimal range: bin with highest mean y
    if means:
        best_idx = int(np.argmax(means))
        optimal_range = (bin_labels[best_idx], means[best_idx])
    else:
        optimal_range = (None, np.nan)

    return {
        'bins': bin_labels,
        'means': means,
        'ns': ns,
        'ci_lows': ci_lows,
        'ci_highs': ci_highs,
        'linear_r': round(float(r_lin), 4) if not np.isnan(r_lin) else np.nan,
        'linear_p': round(float(p_lin), 6) if not np.isnan(p_lin) else np.nan,
        'optimal_range': optimal_range,
        'shape': shape,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Threshold test
# ═══════════════════════════════════════════════════════════════════════════

def test_threshold(df: pd.DataFrame, x: str, y: str,
                   threshold: float) -> dict:
    """Test threshold effect: compare *y* when *x* is above vs below *threshold*.

    Returns dict with group means, difference, Cohen's d, p-value, and counts.
    """
    global _cache

    if _cache is not None:
        xv, yv, n_pair = _cache.get_pair(x, y)
        if xv is None:
            return {'below_mean': np.nan, 'above_mean': np.nan, 'diff': np.nan,
                    'cohen_d': np.nan, 'p_value': np.nan,
                    'n_below': 0, 'n_above': 0}
        below = yv[xv < threshold]
        above = yv[xv >= threshold]
    else:
        sub = df[[x, y]].dropna()
        below = sub.loc[sub[x] < threshold, y].values
        above = sub.loc[sub[x] >= threshold, y].values

    n_below, n_above = len(below), len(above)
    if n_below < 3 or n_above < 3:
        return {'below_mean': np.nan, 'above_mean': np.nan, 'diff': np.nan,
                'cohen_d': np.nan, 'p_value': np.nan,
                'n_below': n_below, 'n_above': n_above}

    below_mean = float(below.mean())
    above_mean = float(above.mean())
    diff = above_mean - below_mean
    d = _cohen_d(above, below)

    # Use Mann-Whitney U for robustness (N=1 data is often non-normal)
    _, p_mw = stats.mannwhitneyu(below, above, alternative='two-sided')
    # Also provide t-test p for reference
    _, p_t = stats.ttest_ind(below, above, equal_var=False)

    return {
        'below_mean': round(below_mean, 4),
        'above_mean': round(above_mean, 4),
        'diff': round(diff, 4),
        'cohen_d': round(d, 4),
        'p_value': round(float(min(p_mw, p_t)), 6),
        'p_mannwhitney': round(float(p_mw), 6),
        'p_welch_t': round(float(p_t), 6),
        'n_below': n_below,
        'n_above': n_above,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Granger causality (causal lag)
# ═══════════════════════════════════════════════════════════════════════════

def test_causal_lag(df: pd.DataFrame, cause: str, effect: str,
                    max_lag: int = 3) -> dict:
    """Granger-causality test: does *cause* at lag 1..max_lag predict *effect*?

    Uses statsmodels if available, otherwise a manual F-test approach.
    Returns dict with per-lag F-stats, p-values, best lag, and direction.
    """
    global _cache

    if _cache is not None:
        xv, yv, n_pair = _cache.get_pair(cause, effect)
        if xv is None or n_pair < max_lag + 10:
            return {'lags': [], 'f_stats': [], 'p_values': [],
                    'best_lag': np.nan, 'is_causal': False, 'direction': 'insufficient_data'}
        # Reconstruct as DataFrame for grangercausalitytests
        sub = pd.DataFrame({cause: xv, effect: yv})
    else:
        sub = df[[cause, effect]].dropna()
    if len(sub) < max_lag + 10:
        return {'lags': [], 'f_stats': [], 'p_values': [],
                'best_lag': np.nan, 'is_causal': False, 'direction': 'insufficient_data'}

    lags = list(range(1, max_lag + 1))
    f_stats, p_values = [], []

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        data_gc = sub[[effect, cause]].values  # grangercausalitytests expects [y, x]
        gc_results = grangercausalitytests(data_gc, maxlag=max_lag, verbose=False)
        for lag in lags:
            # Extract the F-test result (ssr based F test)
            f_val = gc_results[lag][0]['ssr_ftest'][0]
            p_val = gc_results[lag][0]['ssr_ftest'][1]
            f_stats.append(round(float(f_val), 4))
            p_values.append(round(float(p_val), 6))
    except (ImportError, Exception):
        # Manual approach: compare AR(p) model with and without cause lags
        from sklearn.linear_model import LinearRegression
        y = sub[effect].values
        x = sub[cause].values
        for lag in lags:
            n = len(y) - lag
            if n < lag + 5:
                f_stats.append(np.nan)
                p_values.append(np.nan)
                continue
            # Build lagged matrices
            Y = y[lag:]
            # Restricted model: only effect's own lags
            X_restricted = np.column_stack([y[lag - l:len(y) - l] for l in range(1, lag + 1)])
            # Unrestricted: effect lags + cause lags
            X_full = np.column_stack([
                X_restricted,
                *[x[lag - l:len(x) - l] for l in range(1, lag + 1)]
            ])

            lr_r = LinearRegression().fit(X_restricted, Y)
            lr_f = LinearRegression().fit(X_full, Y)

            ss_r = np.sum((Y - lr_r.predict(X_restricted)) ** 2)
            ss_f = np.sum((Y - lr_f.predict(X_full)) ** 2)

            df_num = lag  # extra parameters
            df_den = n - 2 * lag - 1
            if df_den <= 0 or ss_f <= 0:
                f_stats.append(np.nan)
                p_values.append(np.nan)
                continue

            f_val = ((ss_r - ss_f) / df_num) / (ss_f / df_den)
            p_val = 1 - stats.f.cdf(f_val, df_num, df_den)
            f_stats.append(round(float(f_val), 4))
            p_values.append(round(float(p_val), 6))

    # Determine best lag (lowest p-value)
    valid_p = [(l, p) for l, p in zip(lags, p_values) if not np.isnan(p)]
    if valid_p:
        best_lag, best_p = min(valid_p, key=lambda x: x[1])
        is_causal = best_p < _P_THRESHOLD
    else:
        best_lag, is_causal = np.nan, False

    # Direction: correlation sign at best lag
    direction = 'none'
    if not np.isnan(best_lag):
        shifted = sub[cause].shift(int(best_lag))
        valid = sub[effect].notna() & shifted.notna()
        if valid.sum() >= 5:
            r_dir, _ = stats.spearmanr(shifted[valid], sub.loc[valid, effect])
            direction = 'positive' if r_dir > 0 else 'negative'

    return {
        'lags': lags,
        'f_stats': f_stats,
        'p_values': p_values,
        'best_lag': int(best_lag) if not np.isnan(best_lag) else None,
        'is_causal': is_causal,
        'direction': direction,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Mediation analysis (Baron-Kenny)
# ═══════════════════════════════════════════════════════════════════════════

def test_mediation(df: pd.DataFrame, x: str, mediator: str, y: str) -> dict:
    """Baron-Kenny 4-step mediation analysis: X -> M -> Y.

    Steps
    -----
    1. c  path: X -> Y  (total effect)
    2. a  path: X -> M
    3. b  path: M -> Y controlling for X
    4. c' path: X -> Y controlling for M (direct effect)

    Indirect effect = a * b.  Sobel test for significance.
    """
    from sklearn.linear_model import LinearRegression
    global _cache

    if _cache is not None:
        xv, mv, yv, n = _cache.get_triple(x, mediator, y)
        if xv is None or n < 10:
            return {'total_effect': np.nan, 'direct_effect': np.nan,
                    'indirect_effect': np.nan, 'proportion_mediated': np.nan,
                    'sobel_z': np.nan, 'sobel_p': np.nan}
        X_vals = xv.reshape(-1, 1)
        M_vals = mv.reshape(-1, 1)
        Y_vals = yv
        # Create a proxy sub for the attribute accesses below
        sub = pd.DataFrame({x: xv, mediator: mv, y: yv})
    else:
        sub = df[[x, mediator, y]].dropna()
        n = len(sub)
        if n < 10:
            return {'total_effect': np.nan, 'direct_effect': np.nan,
                    'indirect_effect': np.nan, 'proportion_mediated': np.nan,
                    'sobel_z': np.nan, 'sobel_p': np.nan}
        X_vals = sub[x].values.reshape(-1, 1)
        M_vals = sub[mediator].values.reshape(-1, 1)
        Y_vals = sub[y].values

    # Step 1: total effect c (X -> Y)
    lr_c = LinearRegression().fit(X_vals, Y_vals)
    c = float(lr_c.coef_[0])

    # Step 2: a path (X -> M)
    lr_a = LinearRegression().fit(X_vals, sub[mediator].values)
    a = float(lr_a.coef_[0])
    # SE of a
    resid_a = sub[mediator].values - lr_a.predict(X_vals)
    mse_a = np.sum(resid_a ** 2) / (n - 2)
    x_var = np.sum((X_vals.ravel() - X_vals.mean()) ** 2)
    se_a = np.sqrt(mse_a / x_var) if x_var > 0 else np.nan

    # Step 3 + 4: Y ~ X + M (gives b and c')
    XM = np.column_stack([X_vals.ravel(), M_vals.ravel()])
    lr_bc = LinearRegression().fit(XM, Y_vals)
    c_prime = float(lr_bc.coef_[0])   # direct effect
    b = float(lr_bc.coef_[1])         # b path
    # SE of b
    resid_bc = Y_vals - lr_bc.predict(XM)
    mse_bc = np.sum(resid_bc ** 2) / (n - 3)
    # Approximate SE of b from the covariance matrix of XM
    XM_centered = XM - XM.mean(axis=0)
    try:
        cov_inv = np.linalg.inv(XM_centered.T @ XM_centered)
        se_b = np.sqrt(mse_bc * cov_inv[1, 1])
    except np.linalg.LinAlgError:
        se_b = np.nan

    # Indirect effect
    indirect = a * b

    # Proportion mediated
    proportion = indirect / c if abs(c) > 1e-10 else np.nan

    # Sobel test: z = a*b / sqrt(b^2 * se_a^2 + a^2 * se_b^2)
    if not (np.isnan(se_a) or np.isnan(se_b)):
        sobel_se = np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2)
        sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
        sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z)))) if not np.isnan(sobel_z) else np.nan
    else:
        sobel_z, sobel_p = np.nan, np.nan

    return {
        'total_effect': round(c, 6),
        'direct_effect': round(c_prime, 6),
        'indirect_effect': round(indirect, 6),
        'proportion_mediated': round(float(proportion), 4) if not np.isnan(proportion) else np.nan,
        'sobel_z': round(float(sobel_z), 4) if not np.isnan(sobel_z) else np.nan,
        'sobel_p': round(float(sobel_p), 6) if not np.isnan(sobel_p) else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Interaction test
# ═══════════════════════════════════════════════════════════════════════════

def test_interaction(df: pd.DataFrame, x1: str, x2: str, y: str) -> dict:
    """Test interaction effect: does x1 * x2 add predictive value for y
    beyond the main effects?

    Fits two OLS models:
    - without interaction: y ~ x1 + x2
    - with interaction:    y ~ x1 + x2 + x1*x2

    Returns main effects, interaction coefficient, p-value, and R2 change.
    """
    from sklearn.linear_model import LinearRegression
    global _cache

    if _cache is not None:
        x1v, x2v, yv, n = _cache.get_triple(x1, x2, y)
        if x1v is None or n < 15:
            return {'main_x1': np.nan, 'main_x2': np.nan, 'interaction': np.nan,
                    'interaction_p': np.nan, 'r2_without': np.nan, 'r2_with': np.nan,
                    'r2_change': np.nan}
        X1, X2, Y = x1v, x2v, yv
    else:
        sub = df[[x1, x2, y]].dropna()
        n = len(sub)
        if n < 15:
            return {'main_x1': np.nan, 'main_x2': np.nan, 'interaction': np.nan,
                    'interaction_p': np.nan, 'r2_without': np.nan, 'r2_with': np.nan,
                    'r2_change': np.nan}
        X1 = sub[x1].values
        X2 = sub[x2].values
        Y = sub[y].values

    # Standardize for interpretability
    X1_z = (X1 - X1.mean()) / X1.std() if X1.std() > 0 else X1 - X1.mean()
    X2_z = (X2 - X2.mean()) / X2.std() if X2.std() > 0 else X2 - X2.mean()
    X_inter = X1_z * X2_z

    # Model without interaction
    X_no_int = np.column_stack([X1_z, X2_z])
    lr_no = LinearRegression().fit(X_no_int, Y)
    pred_no = lr_no.predict(X_no_int)
    ss_res_no = np.sum((Y - pred_no) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_without = 1 - ss_res_no / ss_tot if ss_tot > 0 else 0

    # Model with interaction
    X_with_int = np.column_stack([X1_z, X2_z, X_inter])
    lr_with = LinearRegression().fit(X_with_int, Y)
    pred_with = lr_with.predict(X_with_int)
    ss_res_with = np.sum((Y - pred_with) ** 2)
    r2_with = 1 - ss_res_with / ss_tot if ss_tot > 0 else 0

    # F-test for the interaction term (1 df added)
    df_num = 1
    df_den = n - 4  # 3 predictors + intercept
    if df_den > 0 and ss_res_with > 0:
        f_val = ((ss_res_no - ss_res_with) / df_num) / (ss_res_with / df_den)
        p_int = 1 - stats.f.cdf(f_val, df_num, df_den)
    else:
        f_val, p_int = np.nan, np.nan

    return {
        'main_x1': round(float(lr_with.coef_[0]), 6),
        'main_x2': round(float(lr_with.coef_[1]), 6),
        'interaction': round(float(lr_with.coef_[2]), 6),
        'interaction_p': round(float(p_int), 6) if not np.isnan(p_int) else np.nan,
        'r2_without': round(float(r2_without), 4),
        'r2_with': round(float(r2_with), 4),
        'r2_change': round(float(r2_with - r2_without), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. Temporal trend test
# ═══════════════════════════════════════════════════════════════════════════

def test_temporal(df: pd.DataFrame, var: str, date_col: str = 'date') -> dict:
    """Test temporal trend using Mann-Kendall and OLS.

    Returns dict with slope_per_month, MK p-value, OLS p-value, direction, R2.
    """
    # Temporal test needs date alignment, so always use DataFrame
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        values = df[var]
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
        values = df[var]
    else:
        dates = None
        values = df[var]

    v = values.dropna()
    if len(v) < 10:
        return {'slope_per_month': np.nan, 'p_mk': np.nan, 'p_ols': np.nan,
                'direction': 'insufficient_data', 'r2': np.nan}

    y = v.values
    n = len(y)

    # Mann-Kendall test (vectorized — O(n) memory, much faster than O(n^2) loop)
    # For each element, count how many following elements are greater/lesser
    s = 0
    for i in range(n - 1):
        diff = y[i + 1:] - y[i]
        s += int(np.sum(diff > 0)) - int(np.sum(diff < 0))

    # Variance of S (no ties correction for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z_mk = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z_mk = (s + 1) / np.sqrt(var_s)
    else:
        z_mk = 0
    p_mk = float(2 * (1 - stats.norm.cdf(abs(z_mk))))

    # OLS trend
    if dates is not None:
        d = dates.loc[v.index] if hasattr(dates, 'loc') else dates[v.index]
        t_days = (d - d.min()).dt.total_seconds() / 86400
        t = t_days.values
    else:
        t = np.arange(n, dtype=float)

    slope, intercept, r_val, p_ols, _ = stats.linregress(t, y)
    r2 = r_val ** 2

    # Convert slope to per-month (30.44 days)
    slope_per_month = slope * 30.44

    # Direction
    if p_mk < 0.05:
        direction = 'increasing' if s > 0 else 'decreasing'
    else:
        direction = 'no_trend'

    return {
        'slope_per_month': round(float(slope_per_month), 6),
        'p_mk': round(p_mk, 6),
        'p_ols': round(float(p_ols), 6),
        'direction': direction,
        'r2': round(float(r2), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. Bayesian updating
# ═══════════════════════════════════════════════════════════════════════════

def bayesian_update(observed_r: float, observed_se: float,
                    prior_mean: float, prior_sd: float) -> dict:
    """Bayesian conjugate Normal-Normal update.

    Prior:      theta ~ N(prior_mean, prior_sd^2)
    Likelihood: observed_r | theta ~ N(theta, observed_se^2)
    Posterior:  theta | data ~ N(posterior_mean, posterior_sd^2)

    Also computes a Savage-Dickey Bayes Factor (BF10 vs H0: theta = 0) and
    a concordance label.
    """
    if np.isnan(observed_r) or np.isnan(observed_se) or observed_se <= 0:
        return {'posterior_mean': np.nan, 'posterior_sd': np.nan,
                'ci_low': np.nan, 'ci_high': np.nan,
                'bayes_factor': np.nan, 'concordance': 'unavailable'}

    prior_prec = 1.0 / (prior_sd ** 2)
    lik_prec = 1.0 / (observed_se ** 2)
    posterior_prec = prior_prec + lik_prec
    posterior_var = 1.0 / posterior_prec
    posterior_sd = np.sqrt(posterior_var)
    posterior_mean = (prior_mean * prior_prec + observed_r * lik_prec) / posterior_prec

    # 89% credible interval (Bayesian convention following McElreath)
    z_89 = stats.norm.ppf(0.945)  # 89% CI -> 5.5% in each tail
    ci_low = posterior_mean - z_89 * posterior_sd
    ci_high = posterior_mean + z_89 * posterior_sd

    # Savage-Dickey BF10: density of H0 (theta=0) under prior / under posterior
    prior_at_zero = stats.norm.pdf(0, loc=prior_mean, scale=prior_sd)
    posterior_at_zero = stats.norm.pdf(0, loc=posterior_mean, scale=posterior_sd)
    bf10 = prior_at_zero / posterior_at_zero if posterior_at_zero > 1e-300 else np.inf

    # Concordance: does observed match the sign and rough magnitude of the prior?
    if np.sign(observed_r) == np.sign(prior_mean) and abs(observed_r) <= 2 * abs(prior_mean):
        concordance = 'concordant'
    elif np.sign(observed_r) == np.sign(prior_mean):
        concordance = 'concordant_magnitude_differs'
    elif abs(observed_r) < 0.05:
        concordance = 'null_effect'
    else:
        concordance = 'discordant'

    return {
        'posterior_mean': round(float(posterior_mean), 4),
        'posterior_sd': round(float(posterior_sd), 4),
        'ci_low': round(float(ci_low), 4),
        'ci_high': round(float(ci_high), 4),
        'bayes_factor': round(float(bf10), 4) if not np.isinf(bf10) else float('inf'),
        'concordance': concordance,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. Master hypothesis router
# ═══════════════════════════════════════════════════════════════════════════

def _parse_population_effect(pop_str: str) -> tuple:
    """Extract a rough (mean, sd) from the population_effect text.

    Heuristic: look for r = X.XX or numbers in a known pattern.
    Returns (mean, sd) or (None, None) if unparseable.
    """
    import re
    # Try patterns like "r = 0.60-0.80" or "r = 0.40"
    m = re.search(r'r\s*[=≈]\s*([-\d.]+)\s*[-–]\s*([-\d.]+)', pop_str)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return ((lo + hi) / 2, (hi - lo) / 2)

    m = re.search(r'r\s*[=≈]\s*([-\d.]+)', pop_str)
    if m:
        val = float(m.group(1))
        return (val, abs(val) * 0.3 + 0.05)  # rough SD

    # Try percentage patterns like "20-40%"
    m = re.search(r'([-\d.]+)\s*[-–]\s*([-\d.]+)\s*%', pop_str)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        # Interpret as a relative effect; normalize to a correlation-like scale
        mid = (lo + hi) / 200  # rough mapping
        sd = (hi - lo) / 400
        return (mid, max(sd, 0.05))

    return (None, None)


def test_hypothesis(hypothesis: dict, df: pd.DataFrame) -> dict:
    """Route a single hypothesis to the appropriate test function.

    Returns a result dict augmented with Bayesian posterior when
    ``population_effect`` is available.
    """
    h_id = hypothesis.get('id', 'unknown')
    test_type = hypothesis.get('test_type', 'correlation')
    variables = hypothesis.get('variables', [])

    # Map hypothesis variable names to DataFrame columns (try direct and common aliases)
    # Use central alias resolver from schema.yaml (single source of truth)
    mapped = [resolve_column(v, df.columns.tolist()) for v in variables]
    available = [m for m in mapped if m is not None]
    missing_vars = [v for v, m in zip(variables, mapped) if m is None]

    result = {
        'hypothesis_id': h_id,
        'test_type': test_type,
        'variables_requested': variables,
        'variables_mapped': available,
        'variables_missing': missing_vars,
    }

    # Bail early if key variables unavailable
    min_vars = 1 if test_type in ('temporal',) else 2
    if len(available) < min_vars:
        result['status'] = 'skipped_missing_variables'
        return result

    # ---- Dispatch by test type ----
    try:
        if test_type == 'correlation':
            var1, var2 = available[0], available[1]
            test_res = test_correlation(df, var1, var2, method='spearman')
            result.update(test_res)

        elif test_type == 'dose_response':
            var_x, var_y = available[0], available[1]
            test_res = test_dose_response(df, var_x, var_y)
            result.update(test_res)

        elif test_type == 'threshold':
            var_x, var_y = available[0], available[1]
            # Try to extract a threshold from the hypothesis text
            import re
            threshold_val = None
            pred_text = hypothesis.get('prediction', '') + ' ' + hypothesis.get('test_spec', '')
            m = re.search(r'threshold[=:]\s*([\d.]+)', pred_text, re.IGNORECASE)
            if m:
                threshold_val = float(m.group(1))
            else:
                m = re.search(r'[<>]=?\s*([\d.]+)', pred_text)
                if m:
                    threshold_val = float(m.group(1))
            if threshold_val is None:
                # Use median as fallback
                threshold_val = float(df[var_x].median())
            test_res = test_threshold(df, var_x, var_y, threshold=threshold_val)
            result['threshold_used'] = threshold_val
            result.update(test_res)

        elif test_type == 'causal_lag':
            var_cause, var_effect = available[0], available[1]
            test_res = test_causal_lag(df, var_cause, var_effect, max_lag=3)
            result.update(test_res)

        elif test_type == 'interaction':
            if len(available) >= 3:
                x1, x2, y_var = available[0], available[1], available[2]
                test_res = test_interaction(df, x1, x2, y_var)
                result.update(test_res)
            else:
                result['status'] = 'skipped_need_3_variables'
                return result

        elif test_type == 'temporal':
            var_t = available[0]
            test_res = test_temporal(df, var_t)
            result.update(test_res)

        elif test_type == 'mediation':
            if len(available) >= 3:
                test_res = test_mediation(df, available[0], available[1], available[2])
                result.update(test_res)
            else:
                result['status'] = 'skipped_need_3_variables'
                return result

        else:
            # Default to correlation
            test_res = test_correlation(df, available[0], available[1], method='spearman')
            result.update(test_res)

        result['status'] = 'tested'

    except Exception as exc:
        logger.debug(f"Hypothesis {h_id} error: {exc}")
        result['status'] = 'skipped_error'
        result['error_detail'] = str(exc)
        return result

    # ---- Bayesian updating (if population effect is available) ----
    pop_effect_str = hypothesis.get('population_effect', '')
    if pop_effect_str:
        prior_mean, prior_sd = _parse_population_effect(pop_effect_str)
        if prior_mean is not None and prior_sd is not None:
            # Determine the observed effect and its SE
            observed_r = result.get('r', result.get('linear_r', result.get('cohen_d', None)))
            if observed_r is not None and not np.isnan(observed_r):
                n_obs = result.get('n', result.get('n_below', 0) + result.get('n_above', 0))
                if n_obs < 5:
                    n_obs = len(df)
                observed_se = 1.0 / np.sqrt(max(n_obs - 3, 1))
                # Widen prior for N=1 context
                prior_sd_widened = prior_sd * 2.0
                bayes = bayesian_update(float(observed_r), observed_se,
                                        prior_mean, prior_sd_widened)
                result['bayesian'] = bayes
                result['concordance'] = bayes.get('concordance', 'unknown')

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 11. Run all hypotheses with FDR correction
# ═══════════════════════════════════════════════════════════════════════════

def run_all_hypotheses(hypotheses: list, df: pd.DataFrame,
                       fdr_method: str = 'fdr_bh') -> pd.DataFrame:
    """Test all hypotheses, apply FDR correction for multiple comparisons.

    Pre-computes a cache of clean arrays and masks to avoid redundant
    DataFrame operations per hypothesis (~5-10x speedup).

    Returns a DataFrame with one row per hypothesis including test results
    and corrected p-values.
    """
    global _cache

    # Guard: empty DataFrame or no hypotheses
    if df.empty or not hypotheses:
        logger.warning("run_all_hypotheses: empty DataFrame or no hypotheses")
        return pd.DataFrame()

    _cache = _HypothesisCache(df)

    results = []
    for h in hypotheses:
        r = test_hypothesis(h, df)
        # Carry forward useful metadata from the hypothesis
        r['domain'] = h.get('domain', '')
        r['hypothesis_text'] = h.get('hypothesis', '')
        r['priority'] = h.get('priority', 'medium')
        r['actionable'] = h.get('actionable', False)
        r['action_if_confirmed'] = h.get('action_if_confirmed', '')
        r['action_if_refuted'] = h.get('action_if_refuted', '')
        results.append(r)

    results_df = pd.DataFrame(results)

    # Merge p-values from different test types into unified column
    p_candidates = ['p', 'p_value', 'linear_p', 'p_mk', 'interaction_p', 'sobel_p', 'p_welch_t']
    results_df['p_unified'] = np.nan
    for pc in p_candidates:
        if pc in results_df.columns:
            results_df['p_unified'] = results_df['p_unified'].fillna(results_df[pc])
    # For causal lag tests, use the best lag p-value
    if 'p_values' in results_df.columns:
        for idx, row in results_df.iterrows():
            if pd.isna(results_df.at[idx, 'p_unified']) and isinstance(row.get('p_values'), list):
                valid_p = [p for p in row['p_values'] if not np.isnan(p)]
                if valid_p:
                    results_df.at[idx, 'p_unified'] = min(valid_p)

    p_col = 'p_unified'
    if results_df[p_col].notna().sum() > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            # Collect all valid p-values
            valid_mask = results_df[p_col].notna()
            if valid_mask.sum() >= 2:
                raw_p = results_df.loc[valid_mask, p_col].values.astype(float)
                reject, corrected, _, _ = multipletests(raw_p, method=fdr_method)
                results_df.loc[valid_mask, 'p_fdr'] = corrected
                results_df.loc[valid_mask, 'significant_fdr'] = reject
            else:
                results_df['p_fdr'] = results_df.get(p_col, np.nan)
                results_df['significant_fdr'] = results_df.get(p_col, 1.0) < _P_THRESHOLD
        except ImportError:
            # Manual Benjamini-Hochberg
            valid_mask = results_df[p_col].notna()
            if valid_mask.sum() >= 2:
                raw_p = results_df.loc[valid_mask, p_col].values.astype(float)
                m = len(raw_p)
                sorted_idx = np.argsort(raw_p)
                ranks = np.empty(m, dtype=float)
                ranks[sorted_idx] = np.arange(1, m + 1)
                corrected = np.minimum(1.0, raw_p * m / ranks)
                # Enforce monotonicity
                sorted_corrected = corrected[sorted_idx]
                for i in range(m - 2, -1, -1):
                    sorted_corrected[i] = min(sorted_corrected[i], sorted_corrected[i + 1])
                corrected[sorted_idx] = sorted_corrected
                results_df.loc[valid_mask, 'p_fdr'] = corrected
                results_df.loc[valid_mask, 'significant_fdr'] = corrected < _P_THRESHOLD

    # Sort by priority then p-value
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    results_df['_priority_rank'] = results_df['priority'].map(priority_order).fillna(1)
    sort_cols = ['_priority_rank']
    if 'p_fdr' in results_df.columns:
        sort_cols.append('p_fdr')
    results_df = results_df.sort_values(sort_cols).drop(columns=['_priority_rank'])
    results_df = results_df.reset_index(drop=True)

    # Clean up cache
    _cache = None

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# 12. Hypothesis report generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_hypothesis_report(results: pd.DataFrame,
                               output_dir: str) -> str:
    """Generate a markdown hypothesis testing report with summary tables
    and per-hypothesis detail blocks.

    Returns path to ``hypothesis_results.md``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)

    lines: list[str] = ['# Hypothesis Testing Results\n']
    lines.append(f'**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append(f'**Hypotheses tested:** {len(results)}\n\n')

    # --- Summary statistics ---
    tested = results[results.get('status', pd.Series(dtype=str)) == 'tested'] if 'status' in results.columns else results
    n_tested = len(tested)
    n_sig = int(tested['significant_fdr'].sum()) if 'significant_fdr' in tested.columns else 0
    lines.append(f'**Successfully tested:** {n_tested} | '
                 f'**Significant (FDR-corrected):** {n_sig}\n\n')

    # Concordance summary
    if 'concordance' in results.columns:
        conc_counts = results['concordance'].value_counts()
        lines.append('## Concordance with Literature\n')
        lines.append('| Category | Count |\n|---|---|\n')
        for cat, cnt in conc_counts.items():
            lines.append(f'| {cat} | {cnt} |\n')
        lines.append('\n')

    # --- Summary table ---
    lines.append('## Summary Table\n')
    display_cols = ['hypothesis_id', 'domain', 'test_type', 'status', 'priority']
    # Add the main effect-size column
    for eff_col in ['r', 'cohen_d', 'linear_r', 'slope_per_month']:
        if eff_col in results.columns:
            display_cols.append(eff_col)
            break
    for p_col in ['p', 'p_value', 'linear_p', 'p_mk']:
        if p_col in results.columns:
            display_cols.append(p_col)
            break
    if 'p_fdr' in results.columns:
        display_cols.append('p_fdr')
    if 'significant_fdr' in results.columns:
        display_cols.append('significant_fdr')
    if 'concordance' in results.columns:
        display_cols.append('concordance')

    display_cols = [c for c in display_cols if c in results.columns]
    lines.append(results[display_cols].to_markdown(index=False) + '\n\n')

    # --- Per-hypothesis details ---
    lines.append('## Detailed Results\n')
    for _, row in results.iterrows():
        h_id = row.get('hypothesis_id', '?')
        lines.append(f'### {h_id}\n')
        lines.append(f'**Domain:** {row.get("domain", "")} | '
                     f'**Type:** {row.get("test_type", "")} | '
                     f'**Priority:** {row.get("priority", "")}\n\n')
        lines.append(f'> {row.get("hypothesis_text", "")}\n\n')

        status = row.get('status', '')
        if status != 'tested':
            lines.append(f'*Status: {status}*\n\n')
            continue

        # Key metrics
        lines.append('**Results:**\n\n')
        skip_keys = {'hypothesis_id', 'test_type', 'variables_requested',
                     'variables_mapped', 'variables_missing', 'status',
                     'domain', 'hypothesis_text', 'priority', 'actionable',
                     'action_if_confirmed', 'action_if_refuted',
                     'bins', 'means', 'ns', 'ci_lows', 'ci_highs',
                     'lags', 'f_stats', 'p_values', 'bayesian'}
        for key, val in row.items():
            if key in skip_keys or pd.isna(val):
                continue
            if isinstance(val, float):
                lines.append(f'- **{key}:** {val:.4f}\n')
            else:
                lines.append(f'- **{key}:** {val}\n')

        # Bayesian posterior
        bayes = row.get('bayesian', None)
        if isinstance(bayes, dict):
            lines.append('\n**Bayesian posterior:**\n')
            lines.append(f'- Posterior mean: {bayes.get("posterior_mean", "N/A")}\n')
            lines.append(f'- Posterior SD: {bayes.get("posterior_sd", "N/A")}\n')
            lines.append(f'- 89% CI: [{bayes.get("ci_low", "N/A")}, {bayes.get("ci_high", "N/A")}]\n')
            lines.append(f'- BF10: {bayes.get("bayes_factor", "N/A")}\n')
            lines.append(f'- Concordance: {bayes.get("concordance", "N/A")}\n')

        # Actions
        if row.get('actionable'):
            sig = row.get('significant_fdr', row.get('p', 1.0))
            is_sig = sig is True or (isinstance(sig, (float, int)) and sig < _P_THRESHOLD)
            if is_sig:
                lines.append(f'\n**Action (confirmed):** {row.get("action_if_confirmed", "")}\n')
            else:
                lines.append(f'\n**Action (refuted/inconclusive):** {row.get("action_if_refuted", "")}\n')

        lines.append('\n---\n\n')

    # --- Volcano-style plot: effect size vs -log10(p) ---
    try:
        eff_col = None
        for candidate in ['r', 'cohen_d', 'linear_r']:
            if candidate in results.columns:
                eff_col = candidate
                break
        p_col_plot = None
        for candidate in ['p_fdr', 'p', 'p_value', 'linear_p']:
            if candidate in results.columns:
                p_col_plot = candidate
                break

        if eff_col and p_col_plot:
            plot_df = results[[eff_col, p_col_plot, 'hypothesis_id']].dropna()
            if len(plot_df) >= 3:
                fig, ax = plt.subplots(figsize=(8, 6))
                x_vals = plot_df[eff_col].values
                y_vals = -np.log10(plot_df[p_col_plot].values.clip(1e-10))
                colors = ['red' if p < 0.05 else 'grey'
                          for p in plot_df[p_col_plot].values]
                ax.scatter(x_vals, y_vals, c=colors, alpha=0.7, s=50)
                ax.axhline(-np.log10(0.05), color='blue', linestyle='--',
                           alpha=0.5, label='p = 0.05')
                ax.axvline(0.3, color='green', linestyle=':', alpha=0.4)
                ax.axvline(-0.3, color='green', linestyle=':', alpha=0.4)
                for _, row_p in plot_df.iterrows():
                    ax.annotate(row_p['hypothesis_id'],
                                (row_p[eff_col], -np.log10(max(row_p[p_col_plot], 1e-10))),
                                fontsize=6, alpha=0.7)
                ax.set_xlabel(f'Effect size ({eff_col})')
                ax.set_ylabel(f'-log10({p_col_plot})')
                ax.set_title('Hypothesis Testing: Effect Size vs Significance')
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / 'hypothesis_volcano.png', dpi=150)
                plt.close(fig)
                lines.append('## Volcano Plot\n')
                lines.append('![Volcano plot](figures/hypothesis_volcano.png)\n\n')
    except Exception as exc:
        lines.append(f'*Volcano plot failed: {exc}*\n\n')

    # --- Write ---
    report_path = out / 'hypothesis_results.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return str(report_path)
