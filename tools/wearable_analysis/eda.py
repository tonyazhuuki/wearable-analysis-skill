"""
Automated EDA for wearable data.
Produces: univariate profiles, correlation matrices (with lags),
SHAP-based interaction detection, time series decomposition, changepoint detection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame, cols: list = None) -> list:
    """Return the list of numeric columns, optionally filtered by *cols*."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is not None:
        num = [c for c in cols if c in num]
    return num


def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """If the frame has a 'date' column, set it as a DatetimeIndex."""
    out = df.copy()
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'])
        out = out.set_index('date').sort_index()
    elif not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
        except Exception:
            pass
    return out


def _iqr_outlier_count(series: pd.Series) -> int:
    """Count values outside 1.5 * IQR fences."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


# ---------------------------------------------------------------------------
# 1. Univariate profile
# ---------------------------------------------------------------------------

def univariate_profile(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Compute mean, median, SD, min, max, missing%, skewness, kurtosis for each
    numeric column.  Returns a DataFrame with one row per variable."""
    num = _numeric_cols(df, cols)
    rows = []
    for c in num:
        s = df[c]
        n_total = len(s)
        n_missing = int(s.isna().sum())
        s_clean = s.dropna()
        n_valid = len(s_clean)
        if n_valid == 0:
            rows.append({
                'variable': c, 'n': 0, 'missing_pct': 100.0,
                'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'min': np.nan, 'max': np.nan, 'skew': np.nan,
                'kurtosis': np.nan, 'outliers_iqr': 0, 'outliers_3sd': 0,
            })
            continue
        mean_val = float(s_clean.mean())
        std_val = float(s_clean.std())
        rows.append({
            'variable': c,
            'n': n_valid,
            'missing_pct': round(100.0 * n_missing / n_total, 2),
            'mean': round(mean_val, 4),
            'median': round(float(s_clean.median()), 4),
            'std': round(std_val, 4),
            'min': round(float(s_clean.min()), 4),
            'max': round(float(s_clean.max()), 4),
            'skew': round(float(s_clean.skew()), 4),
            'kurtosis': round(float(s_clean.kurtosis()), 4),
            'outliers_iqr': _iqr_outlier_count(s_clean),
            'outliers_3sd': int(((s_clean - mean_val).abs() > 3 * std_val).sum()) if std_val > 0 else 0,
        })
    return pd.DataFrame(rows).set_index('variable')


# ---------------------------------------------------------------------------
# 2. Correlation matrix with lags
# ---------------------------------------------------------------------------

def correlation_matrix_with_lags(df: pd.DataFrame, target: str = 'recovery',
                                 cols: list = None, max_lag: int = 3) -> dict:
    """Compute Spearman correlation of each variable with *target* at lags 0..max_lag.

    Returns ``{lag_int: DataFrame}`` where each DataFrame has columns
    ``['variable', 'r', 'p', 'n', 'significant']`` (significant when |r| > 0.3
    *and* p < 0.05).
    """
    num = _numeric_cols(df, cols)
    if target in num:
        num.remove(target)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    result: dict[int, pd.DataFrame] = {}
    for lag in range(max_lag + 1):
        rows = []
        target_series = df[target]
        for c in num:
            shifted = df[c].shift(lag)
            valid = pd.notna(target_series) & pd.notna(shifted)
            n = int(valid.sum())
            if n < 5:
                rows.append({'variable': c, 'r': np.nan, 'p': np.nan, 'n': n,
                             'significant': False})
                continue
            r, p = stats.spearmanr(shifted[valid], target_series[valid])
            rows.append({
                'variable': c,
                'r': round(float(r), 4),
                'p': round(float(p), 6),
                'n': n,
                'significant': abs(r) > 0.3 and p < 0.05,
            })
        result[lag] = pd.DataFrame(rows)
    return result


# ---------------------------------------------------------------------------
# 3. Partial correlations
# ---------------------------------------------------------------------------

def partial_correlations(df: pd.DataFrame, target: str, predictors: list,
                         controls: list) -> pd.DataFrame:
    """Compute partial correlations of each predictor with *target* controlling
    for *controls* using the regression-residual method.

    Returns a DataFrame with ``['predictor', 'partial_r', 'p', 'n']``.
    """
    from sklearn.linear_model import LinearRegression

    sub = df[[target] + predictors + controls].dropna()
    n = len(sub)
    if n < len(controls) + 3:
        return pd.DataFrame(columns=['predictor', 'partial_r', 'p', 'n'])

    X_ctrl = sub[controls].values
    rows = []
    for pred in predictors:
        # Residualize target on controls
        lr_t = LinearRegression().fit(X_ctrl, sub[target].values)
        res_t = sub[target].values - lr_t.predict(X_ctrl)
        # Residualize predictor on controls
        lr_p = LinearRegression().fit(X_ctrl, sub[pred].values)
        res_p = sub[pred].values - lr_p.predict(X_ctrl)
        # Correlate residuals
        r, p = stats.pearsonr(res_t, res_p)
        rows.append({'predictor': pred, 'partial_r': round(float(r), 4),
                      'p': round(float(p), 6), 'n': n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. SHAP feature importance
# ---------------------------------------------------------------------------

def shap_feature_importance(df: pd.DataFrame, target: str = 'recovery',
                            features: list = None,
                            n_background: int = 100) -> dict:
    """Train a Random Forest on *features* -> *target*, compute SHAP values.

    Returns a dict with keys ``importance``, ``shap_values``, ``interactions``,
    ``model_performance``.  Falls back gracefully when *shap* is not installed.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    num = _numeric_cols(df, features)
    if target in num:
        num.remove(target)
    sub = df[num + [target]].dropna()
    if len(sub) < 30:
        return {'importance': pd.DataFrame(), 'shap_values': None,
                'interactions': [], 'model_performance': {}}

    X = sub[num].values
    y = sub[target].values

    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Model performance via 5-fold CV
    cv_r2 = cross_val_score(rf, X, y, cv=min(5, len(sub)), scoring='r2')
    cv_mae = -cross_val_score(rf, X, y, cv=min(5, len(sub)), scoring='neg_mean_absolute_error')
    perf = {
        'r2_cv_mean': round(float(cv_r2.mean()), 4),
        'r2_cv_std': round(float(cv_r2.std()), 4),
        'mae_cv_mean': round(float(cv_mae.mean()), 4),
    }

    # Attempt SHAP
    shap_vals = None
    importance_df = pd.DataFrame()
    interactions = []

    try:
        import shap
        bg = shap.sample(sub[num], min(n_background, len(sub)))
        explainer = shap.TreeExplainer(rf, data=bg)
        sv = explainer.shap_values(X)
        shap_vals = sv

        mean_abs = np.abs(sv).mean(axis=0)
        importance_df = (pd.DataFrame({'feature': num, 'mean_abs_shap': mean_abs})
                         .sort_values('mean_abs_shap', ascending=False)
                         .reset_index(drop=True))

        # Interaction values (top-10 pairs by mean abs interaction)
        try:
            interaction_vals = explainer.shap_interaction_values(X)
            n_feat = len(num)
            pair_importance = []
            for i in range(n_feat):
                for j in range(i + 1, n_feat):
                    pair_importance.append((
                        num[i], num[j],
                        float(np.abs(interaction_vals[:, i, j]).mean())
                    ))
            pair_importance.sort(key=lambda x: x[2], reverse=True)
            interactions = pair_importance[:10]
        except Exception:
            interactions = []

    except ImportError:
        # Fallback: use sklearn feature_importances_
        fi = rf.feature_importances_
        importance_df = (pd.DataFrame({'feature': num, 'mean_abs_shap': fi})
                         .sort_values('mean_abs_shap', ascending=False)
                         .reset_index(drop=True))
        importance_df.rename(columns={'mean_abs_shap': 'rf_importance'}, inplace=True)

    return {
        'importance': importance_df,
        'shap_values': shap_vals,
        'interactions': interactions,
        'model_performance': perf,
    }


# ---------------------------------------------------------------------------
# 5. Time series decomposition
# ---------------------------------------------------------------------------

def time_series_decomposition(series: pd.Series, period: int = 7) -> dict:
    """STL decomposition of a time series.

    Returns dict with ``trend``, ``seasonal``, ``residual`` Series plus
    ``trend_slope``, ``trend_p``, ``seasonal_strength``.
    """
    from statsmodels.tsa.seasonal import STL

    s = series.dropna()
    if len(s) < 2 * period + 1:
        return {'trend': pd.Series(dtype=float), 'seasonal': pd.Series(dtype=float),
                'residual': pd.Series(dtype=float), 'trend_slope': np.nan,
                'trend_p': np.nan, 'seasonal_strength': np.nan}

    stl = STL(s, period=period, robust=True)
    result = stl.fit()

    # Trend slope via OLS on integer index
    t = np.arange(len(result.trend))
    slope, intercept, r, p, se = stats.linregress(t, result.trend.values)

    # Seasonal strength: 1 - Var(residual) / Var(seasonal + residual)
    var_resid = np.var(result.resid.values)
    var_seas_resid = np.var(result.seasonal.values + result.resid.values)
    seasonal_strength = max(0.0, 1.0 - var_resid / var_seas_resid) if var_seas_resid > 0 else 0.0

    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'trend_slope': round(float(slope), 6),
        'trend_p': round(float(p), 6),
        'seasonal_strength': round(float(seasonal_strength), 4),
    }


# ---------------------------------------------------------------------------
# 6. Changepoint detection
# ---------------------------------------------------------------------------

def detect_changepoints(series: pd.Series, method: str = 'cusum',
                        min_segment: int = 14) -> list:
    """Detect changepoints in a time series.

    Methods
    -------
    cusum : cumulative sum of deviations from the running mean.
    pelt  : PELT algorithm via the ``ruptures`` library (optional).

    Returns a list of ``(date_or_index, direction, magnitude)`` tuples.
    """
    s = series.dropna()
    if len(s) < 2 * min_segment:
        return []

    if method == 'pelt':
        try:
            import ruptures as rpt
            algo = rpt.Pelt(model='rbf', min_size=min_segment).fit(s.values.reshape(-1, 1))
            bkps = algo.predict(pen=3)
            results = []
            for bp in bkps[:-1]:  # last element is len(s)
                if bp < len(s):
                    before = s.iloc[max(0, bp - min_segment):bp].mean()
                    after = s.iloc[bp:min(len(s), bp + min_segment)].mean()
                    direction = 'increase' if after > before else 'decrease'
                    magnitude = float(after - before)
                    idx_label = s.index[bp] if bp < len(s) else s.index[-1]
                    results.append((idx_label, direction, round(magnitude, 4)))
            return results
        except ImportError:
            # Fall through to CUSUM
            pass

    # --- CUSUM implementation ---
    values = s.values.astype(float)
    global_mean = values.mean()
    global_std = values.std()
    if global_std == 0:
        return []

    # Normalized cumulative sum
    normalized = (values - global_mean) / global_std
    cusum_pos = np.zeros(len(values))
    cusum_neg = np.zeros(len(values))
    threshold = 4.0  # standard CUSUM threshold in SD units

    changepoints = []
    last_cp = 0

    for i in range(1, len(values)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + normalized[i] - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - normalized[i] - 0.5)

        crossed = False
        direction = ''
        if cusum_pos[i] > threshold:
            crossed = True
            direction = 'increase'
            cusum_pos[i] = 0
        elif cusum_neg[i] > threshold:
            crossed = True
            direction = 'decrease'
            cusum_neg[i] = 0

        if crossed and (i - last_cp) >= min_segment:
            seg_before = values[max(0, i - min_segment):i]
            seg_after = values[i:min(len(values), i + min_segment)]
            if len(seg_after) > 0 and len(seg_before) > 0:
                magnitude = float(seg_after.mean() - seg_before.mean())
                idx_label = s.index[i]
                changepoints.append((idx_label, direction, round(magnitude, 4)))
                last_cp = i

    return changepoints


# ---------------------------------------------------------------------------
# 7. Missing data analysis
# ---------------------------------------------------------------------------

def missing_data_analysis(df: pd.DataFrame) -> dict:
    """Analyze missing data patterns.

    Returns dict with ``missing_pct``, ``missing_matrix``, ``pattern_type``,
    and ``little_test_p`` (approximation).
    """
    num = _numeric_cols(df)
    missing_pct = (df[num].isna().sum() / len(df) * 100).round(2)

    # Missing indicator matrix
    missing_matrix = df[num].isna()

    # --- Pattern type heuristic ---
    # Approximate Little's MCAR test: correlate missingness of each variable
    # with observed values of other variables. If many significant correlations,
    # data is likely MAR or MNAR.
    indicator_cols = missing_matrix.columns[missing_matrix.any()]
    n_tests = 0
    n_significant = 0
    for ic in indicator_cols:
        for vc in num:
            if vc == ic:
                continue
            valid = df[vc].notna()
            if valid.sum() < 10:
                continue
            groups = df.loc[valid, vc].groupby(missing_matrix.loc[valid, ic])
            if len(groups) < 2:
                continue
            group_vals = [g.values for _, g in groups]
            if any(len(g) < 3 for g in group_vals):
                continue
            _, p_val = stats.mannwhitneyu(*group_vals, alternative='two-sided')
            n_tests += 1
            if p_val < 0.05:
                n_significant += 1

    if n_tests == 0:
        pattern_type = 'MCAR'
        little_p = 1.0
    else:
        sig_ratio = n_significant / n_tests
        if sig_ratio < 0.05:
            pattern_type = 'MCAR'
            little_p = 0.5 + 0.5 * (1 - sig_ratio / 0.05)
        elif sig_ratio < 0.20:
            pattern_type = 'MAR'
            little_p = 0.05 * (1 - (sig_ratio - 0.05) / 0.15)
        else:
            pattern_type = 'MNAR'
            little_p = 0.001

    return {
        'missing_pct': missing_pct,
        'missing_matrix': missing_matrix,
        'pattern_type': pattern_type,
        'little_test_p': round(little_p, 4),
    }


# ---------------------------------------------------------------------------
# 8. Full EDA report
# ---------------------------------------------------------------------------

def generate_eda_report(df: pd.DataFrame, output_dir: str,
                        target: str = 'recovery') -> str:
    """Run the full EDA pipeline, save figures and tables, compile a markdown
    report.  Returns the path to ``eda_report.md``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)

    lines: list[str] = ['# Automated EDA Report\n']
    lines.append(f'**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')
    lines.append(f'**Rows:** {len(df)} | **Columns:** {len(df.columns)}\n')
    lines.append(f'**Target variable:** `{target}`\n\n')

    # --- 1. Univariate profiles ----------------------------------------
    lines.append('## 1. Univariate Profiles\n')
    profile = univariate_profile(df)
    profile.to_csv(out / 'univariate_profile.csv')
    lines.append(profile.to_markdown() + '\n\n')

    # --- 2. Correlation matrix with lags --------------------------------
    lines.append('## 2. Correlations with Target (by lag)\n')
    if target in df.columns:
        lag_corrs = correlation_matrix_with_lags(df, target=target, max_lag=3)
        for lag, corr_df in sorted(lag_corrs.items()):
            lines.append(f'### Lag {lag}\n')
            sig = corr_df[corr_df['significant']] if 'significant' in corr_df.columns else corr_df
            if len(sig) > 0:
                lines.append(sig.to_markdown(index=False) + '\n\n')
            else:
                lines.append('No variables with |r| > 0.3 and p < 0.05.\n\n')

            # Heatmap for lag 0
            if lag == 0 and len(corr_df) > 0:
                try:
                    num = _numeric_cols(df)
                    corr_full = df[num].corr(method='spearman')
                    fig, ax = plt.subplots(figsize=(max(8, len(num) * 0.6),
                                                    max(6, len(num) * 0.5)))
                    im = ax.imshow(corr_full.values, cmap='RdBu_r', vmin=-1, vmax=1,
                                   aspect='auto')
                    ax.set_xticks(range(len(num)))
                    ax.set_yticks(range(len(num)))
                    ax.set_xticklabels(num, rotation=90, fontsize=7)
                    ax.set_yticklabels(num, fontsize=7)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title('Spearman Correlation Matrix (lag 0)')
                    fig.tight_layout()
                    fig.savefig(fig_dir / 'correlation_heatmap_lag0.png', dpi=150)
                    plt.close(fig)
                    lines.append('![Correlation heatmap](figures/correlation_heatmap_lag0.png)\n\n')
                except Exception as exc:
                    lines.append(f'*Heatmap generation failed: {exc}*\n\n')
    else:
        lines.append(f'*Target `{target}` not found in data.*\n\n')

    # --- 3. SHAP feature importance ------------------------------------
    lines.append('## 3. Feature Importance (Random Forest + SHAP)\n')
    if target in df.columns:
        shap_res = shap_feature_importance(df, target=target)
        perf = shap_res.get('model_performance', {})
        if perf:
            lines.append(f"**Model CV R2:** {perf.get('r2_cv_mean', 'N/A')} "
                         f"(+/- {perf.get('r2_cv_std', 'N/A')}) | "
                         f"**MAE:** {perf.get('mae_cv_mean', 'N/A')}\n\n")
        imp = shap_res.get('importance', pd.DataFrame())
        if len(imp) > 0:
            lines.append(imp.head(20).to_markdown(index=False) + '\n\n')
            # Save bar chart
            try:
                fig, ax = plt.subplots(figsize=(8, max(4, len(imp.head(20)) * 0.35)))
                imp_plot = imp.head(20).iloc[::-1]
                val_col = imp_plot.columns[1]  # mean_abs_shap or rf_importance
                ax.barh(imp_plot.iloc[:, 0], imp_plot[val_col], color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Top-20 Feature Importance')
                fig.tight_layout()
                fig.savefig(fig_dir / 'feature_importance.png', dpi=150)
                plt.close(fig)
                lines.append('![Feature importance](figures/feature_importance.png)\n\n')
            except Exception as exc:
                lines.append(f'*Feature importance plot failed: {exc}*\n\n')

        interactions = shap_res.get('interactions', [])
        if interactions:
            lines.append('### Top Interactions\n')
            lines.append('| Feature 1 | Feature 2 | Mean |SHAP interaction| |\n')
            lines.append('|---|---|---|\n')
            for f1, f2, val in interactions[:10]:
                lines.append(f'| {f1} | {f2} | {val:.4f} |\n')
            lines.append('\n')

    # --- 4. Time series decomposition ----------------------------------
    lines.append('## 4. Time Series Decomposition\n')
    key_vars = [target, 'hrv', 'rhr', 'sleep_hours', 'strain']
    key_vars = [v for v in key_vars if v in df.columns]
    df_ts = _ensure_date_index(df)
    for var in key_vars:
        if var not in df_ts.columns:
            continue
        s = df_ts[var].dropna()
        if len(s) < 15:
            continue
        decomp = time_series_decomposition(s, period=7)
        lines.append(f'### {var}\n')
        lines.append(f'- **Trend slope:** {decomp["trend_slope"]} per day '
                     f'(p = {decomp["trend_p"]})\n')
        lines.append(f'- **Seasonal strength:** {decomp["seasonal_strength"]}\n\n')
        # Plot
        try:
            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(s.index, s.values, linewidth=0.8)
            axes[0].set_ylabel('Observed')
            axes[0].set_title(f'STL Decomposition: {var}')
            if len(decomp['trend']) > 0:
                axes[1].plot(decomp['trend'].index, decomp['trend'].values,
                             color='C1', linewidth=1.2)
            axes[1].set_ylabel('Trend')
            if len(decomp['seasonal']) > 0:
                axes[2].plot(decomp['seasonal'].index, decomp['seasonal'].values,
                             color='C2', linewidth=0.8)
            axes[2].set_ylabel('Seasonal')
            if len(decomp['residual']) > 0:
                axes[3].plot(decomp['residual'].index, decomp['residual'].values,
                             color='C3', linewidth=0.5, alpha=0.7)
            axes[3].set_ylabel('Residual')
            fig.tight_layout()
            fname = f'stl_{var}.png'
            fig.savefig(fig_dir / fname, dpi=150)
            plt.close(fig)
            lines.append(f'![STL {var}](figures/{fname})\n\n')
        except Exception as exc:
            lines.append(f'*STL plot for {var} failed: {exc}*\n\n')

    # --- 5. Changepoint detection --------------------------------------
    lines.append('## 5. Changepoint Detection\n')
    for var in key_vars:
        if var not in df_ts.columns:
            continue
        s = df_ts[var].dropna()
        if len(s) < 30:
            continue
        cps = detect_changepoints(s, method='cusum', min_segment=14)
        lines.append(f'### {var}\n')
        if not cps:
            lines.append('No changepoints detected.\n\n')
        else:
            lines.append('| Date | Direction | Magnitude |\n|---|---|---|\n')
            for dt, direction, mag in cps:
                dt_str = str(dt)[:10] if hasattr(dt, 'strftime') else str(dt)
                lines.append(f'| {dt_str} | {direction} | {mag} |\n')
            lines.append('\n')

            # Changepoint plot
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(s.index, s.values, linewidth=0.8, alpha=0.7)
                for dt, direction, mag in cps:
                    color = 'green' if direction == 'increase' else 'red'
                    ax.axvline(dt, color=color, linestyle='--', alpha=0.7)
                ax.set_title(f'Changepoints: {var}')
                ax.set_ylabel(var)
                fig.tight_layout()
                fname = f'changepoints_{var}.png'
                fig.savefig(fig_dir / fname, dpi=150)
                plt.close(fig)
                lines.append(f'![Changepoints {var}](figures/{fname})\n\n')
            except Exception as exc:
                lines.append(f'*Changepoint plot for {var} failed: {exc}*\n\n')

    # --- 6. Missing data analysis --------------------------------------
    lines.append('## 6. Missing Data Analysis\n')
    miss = missing_data_analysis(df)
    missing_pct = miss['missing_pct']
    cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    lines.append(f'**Pattern type:** {miss["pattern_type"]} '
                 f'(approx. Little test p = {miss["little_test_p"]})\n\n')
    if len(cols_with_missing) > 0:
        lines.append('| Variable | Missing % |\n|---|---|\n')
        for var_name, pct in cols_with_missing.items():
            lines.append(f'| {var_name} | {pct:.1f}% |\n')
        lines.append('\n')

        # Missing pattern heatmap
        try:
            miss_cols = cols_with_missing.index.tolist()[:20]
            fig, ax = plt.subplots(figsize=(max(6, len(miss_cols) * 0.5),
                                            min(10, len(df) * 0.02 + 2)))
            mm = miss['missing_matrix'][miss_cols]
            ax.imshow(mm.values.astype(float), cmap='YlOrRd', aspect='auto',
                      interpolation='none')
            ax.set_xticks(range(len(miss_cols)))
            ax.set_xticklabels(miss_cols, rotation=90, fontsize=7)
            ax.set_ylabel('Row index')
            ax.set_title('Missing Data Pattern')
            fig.tight_layout()
            fig.savefig(fig_dir / 'missing_pattern.png', dpi=150)
            plt.close(fig)
            lines.append('![Missing pattern](figures/missing_pattern.png)\n\n')
        except Exception as exc:
            lines.append(f'*Missing pattern plot failed: {exc}*\n\n')
    else:
        lines.append('No missing data detected.\n\n')

    # --- Write report ---
    report_path = out / 'eda_report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return str(report_path)
