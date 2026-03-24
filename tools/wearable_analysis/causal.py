"""
Causal inference module for N=1 wearable data.
Granger causality, mediation analysis, DAG estimation.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def granger_causality_matrix(df: pd.DataFrame, variables: list,
                              max_lag: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    """Test Granger causality for all variable pairs.

    Returns DataFrame: rows=cause, cols=effect, values=min p-value across lags.
    Significant pairs (p < alpha) indicate potential causal direction.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    n = len(variables)
    results = pd.DataFrame(np.ones((n, n)), index=variables, columns=variables)

    for i, cause in enumerate(variables):
        for j, effect in enumerate(variables):
            if i == j:
                results.iloc[i, j] = np.nan
                continue
            pair = df[[effect, cause]].dropna()
            if len(pair) < max_lag * 3 + 10:
                results.iloc[i, j] = np.nan
                continue
            try:
                test = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
                min_p = min(test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
                results.iloc[i, j] = min_p
            except Exception:
                results.iloc[i, j] = np.nan

    return results


def granger_test_pair(df: pd.DataFrame, cause: str, effect: str,
                       max_lag: int = 5) -> dict:
    """Detailed Granger causality test for a single pair.

    Returns dict with lag-by-lag results and best lag.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    except ImportError:
        raise ImportError("statsmodels required")

    pair = df[[effect, cause]].dropna()
    n = len(pair)

    if n < max_lag * 3 + 10:
        return {'error': f'Insufficient data: {n} rows', 'is_causal': False}

    # Stationarity check (ADF test)
    stationarity = {}
    for col in [cause, effect]:
        adf_stat, adf_p, _, _, _, _ = adfuller(pair[col].values, maxlag=max_lag)
        stationarity[col] = {'adf_stat': adf_stat, 'adf_p': adf_p,
                              'is_stationary': adf_p < 0.05}

    # Differencing if non-stationary
    if not stationarity[cause]['is_stationary'] or not stationarity[effect]['is_stationary']:
        pair_diff = pair.diff().dropna()
        use_data = pair_diff
        differenced = True
    else:
        use_data = pair
        differenced = False

    try:
        test = grangercausalitytests(use_data, maxlag=max_lag, verbose=False)
    except Exception as e:
        return {'error': str(e), 'is_causal': False}

    lag_results = []
    for lag in range(1, max_lag + 1):
        f_stat = test[lag][0]['ssr_ftest'][0]
        p_val = test[lag][0]['ssr_ftest'][1]
        lag_results.append({'lag': lag, 'f_stat': f_stat, 'p_value': p_val})

    best = min(lag_results, key=lambda x: x['p_value'])

    # Also test reverse direction
    try:
        rev_test = grangercausalitytests(use_data[[cause, effect]], maxlag=max_lag, verbose=False)
        rev_best_p = min(rev_test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
    except Exception:
        rev_best_p = 1.0

    return {
        'cause': cause,
        'effect': effect,
        'n': n,
        'differenced': differenced,
        'stationarity': stationarity,
        'lag_results': lag_results,
        'best_lag': best['lag'],
        'best_p': best['p_value'],
        'best_f': best['f_stat'],
        'is_causal': best['p_value'] < 0.05,
        'reverse_p': rev_best_p,
        'direction': 'forward' if best['p_value'] < rev_best_p else 'bidirectional' if rev_best_p < 0.05 else 'reverse',
    }


def mediation_analysis(df: pd.DataFrame, x: str, mediator: str, y: str) -> dict:
    """Baron-Kenny mediation analysis.

    Tests the causal chain: X → M → Y
    Steps:
        1. X → Y (total effect, path c)
        2. X → M (path a)
        3. M → Y controlling for X (path b)
        4. X → Y controlling for M (direct effect, path c')
        Indirect effect = a × b
        Proportion mediated = indirect / total

    Returns full mediation results including Sobel test.
    """
    from sklearn.linear_model import LinearRegression

    data = df[[x, mediator, y]].dropna()
    X_val = data[x].values.reshape(-1, 1)
    M_val = data[mediator].values.reshape(-1, 1)
    Y_val = data[y].values.reshape(-1, 1)
    n = len(data)

    if n < 30:
        return {'error': f'Insufficient data: {n} rows (need >=30)', 'valid': False}

    # Step 1: Total effect (c path): Y = c*X + e
    reg_total = LinearRegression().fit(X_val, Y_val)
    c = reg_total.coef_[0][0]
    y_pred = reg_total.predict(X_val)
    ss_res = np.sum((Y_val - y_pred) ** 2)
    se_c = np.sqrt(ss_res / (n - 2) / np.sum((X_val - X_val.mean()) ** 2))

    # Step 2: a path: M = a*X + e
    reg_a = LinearRegression().fit(X_val, M_val)
    a = reg_a.coef_[0][0]
    m_pred = reg_a.predict(X_val)
    ss_res_a = np.sum((M_val - m_pred) ** 2)
    se_a = np.sqrt(ss_res_a / (n - 2) / np.sum((X_val - X_val.mean()) ** 2))

    # Step 3 & 4: Y = c'*X + b*M + e
    XM = np.hstack([X_val, M_val])
    reg_direct = LinearRegression().fit(XM, Y_val)
    c_prime = reg_direct.coef_[0][0]  # direct effect
    b = reg_direct.coef_[0][1]  # mediator effect
    y_pred_full = reg_direct.predict(XM)
    ss_res_full = np.sum((Y_val - y_pred_full) ** 2)
    mse_full = ss_res_full / (n - 3)
    xtx_inv = np.linalg.inv(XM.T @ XM)
    se_b = np.sqrt(mse_full * xtx_inv[1, 1])

    # Indirect effect
    indirect = a * b

    # Sobel test
    sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # Proportion mediated
    prop_mediated = indirect / c if abs(c) > 1e-10 else 0

    # Bootstrap CI for indirect effect
    boot_indirect = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        idx = rng.choice(n, n, replace=True)
        X_b, M_b, Y_b = X_val[idx], M_val[idx], Y_val[idx]
        try:
            a_b = LinearRegression().fit(X_b, M_b).coef_[0][0]
            XM_b = np.hstack([X_b, M_b])
            b_b = LinearRegression().fit(XM_b, Y_b).coef_[0][1]
            boot_indirect.append(a_b * b_b)
        except Exception:
            continue

    boot_ci = (np.percentile(boot_indirect, 2.5), np.percentile(boot_indirect, 97.5)) \
        if boot_indirect else (np.nan, np.nan)

    return {
        'x': x, 'mediator': mediator, 'y': y,
        'n': n,
        'total_effect_c': c,
        'path_a': a, 'se_a': se_a,
        'path_b': b, 'se_b': se_b,
        'direct_effect_c_prime': c_prime,
        'indirect_effect': indirect,
        'proportion_mediated': prop_mediated,
        'sobel_z': sobel_z,
        'sobel_p': sobel_p,
        'boot_ci_low': boot_ci[0],
        'boot_ci_high': boot_ci[1],
        'is_mediated': sobel_p < 0.05 and boot_ci[0] * boot_ci[1] > 0,
        'mediation_type': _classify_mediation(c, c_prime, indirect, sobel_p),
        'valid': True,
    }


def _classify_mediation(c, c_prime, indirect, sobel_p):
    """Classify mediation type."""
    if sobel_p >= 0.05:
        return 'no_mediation'
    if abs(c_prime) < 0.1 * abs(c):
        return 'full_mediation'
    if np.sign(c) == np.sign(c_prime):
        return 'partial_mediation'
    return 'suppression'


def build_causal_dag(df: pd.DataFrame, variables: list,
                      alpha: float = 0.05, max_lag: int = 3) -> dict:
    """Build causal DAG from Granger causality + correlation structure.

    Returns:
        'edges': list of (cause, effect, lag, strength)
        'nodes': list of variables with in/out degree
        'dot_string': GraphViz DOT format for visualization
    """
    gc_matrix = granger_causality_matrix(df, variables, max_lag=max_lag)

    edges = []
    for cause in variables:
        for effect in variables:
            if cause == effect:
                continue
            p = gc_matrix.loc[cause, effect]
            if pd.notna(p) and p < alpha:
                # Get correlation for strength
                pair = df[[cause, effect]].dropna()
                r = pair.corr().iloc[0, 1] if len(pair) > 10 else 0
                edges.append({
                    'cause': cause,
                    'effect': effect,
                    'p_value': p,
                    'correlation': r,
                    'strength': abs(r),
                })

    # Build node stats
    nodes = {}
    for v in variables:
        out_degree = sum(1 for e in edges if e['cause'] == v)
        in_degree = sum(1 for e in edges if e['effect'] == v)
        nodes[v] = {'out_degree': out_degree, 'in_degree': in_degree,
                     'centrality': out_degree + in_degree}

    # DOT string for GraphViz
    dot_lines = ['digraph CausalDAG {', '  rankdir=LR;', '  node [shape=box];']
    for e in sorted(edges, key=lambda x: -x['strength']):
        width = max(0.5, min(3.0, e['strength'] * 4))
        color = 'red' if e['correlation'] < 0 else 'blue'
        label = f"r={e['correlation']:.2f}"
        dot_lines.append(f'  "{e["cause"]}" -> "{e["effect"]}" '
                         f'[penwidth={width:.1f}, color={color}, label="{label}"];')
    dot_lines.append('}')

    return {
        'edges': edges,
        'nodes': nodes,
        'n_edges': len(edges),
        'dot_string': '\n'.join(dot_lines),
    }


def impulse_response(df: pd.DataFrame, variables: list,
                      impulse: str, response: str,
                      periods: int = 7) -> dict:
    """Estimate impulse response function using VAR model.

    How does a 1-SD shock to 'impulse' variable affect 'response' over time?
    """
    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        return {'error': 'statsmodels VAR not available'}

    data = df[variables].dropna()
    if len(data) < 50:
        return {'error': f'Insufficient data: {len(data)} rows'}

    # Standardize
    data_std = (data - data.mean()) / data.std()

    try:
        model = VAR(data_std)
        fitted = model.fit(maxlags=5, ic='aic')
        irf = fitted.irf(periods=periods)

        impulse_idx = variables.index(impulse)
        response_idx = variables.index(response)

        irf_values = irf.irfs[:, response_idx, impulse_idx]
        irf_lower = irf.ci[:, response_idx, impulse_idx, 0] if hasattr(irf, 'ci') else None
        irf_upper = irf.ci[:, response_idx, impulse_idx, 1] if hasattr(irf, 'ci') else None

        return {
            'impulse': impulse,
            'response': response,
            'periods': list(range(periods + 1)),
            'irf_values': irf_values.tolist(),
            'irf_lower': irf_lower.tolist() if irf_lower is not None else None,
            'irf_upper': irf_upper.tolist() if irf_upper is not None else None,
            'var_order': fitted.k_ar,
            'aic': fitted.aic,
        }
    except Exception as e:
        return {'error': str(e)}


def find_causal_paths(dag: dict, source: str, target: str, max_depth: int = 4) -> list:
    """Find all causal paths from source to target in DAG.

    Returns list of paths, each path is list of variable names.
    """
    edges_by_cause = {}
    for e in dag['edges']:
        edges_by_cause.setdefault(e['cause'], []).append(e['effect'])

    paths = []

    def dfs(current, path, visited):
        if current == target:
            paths.append(list(path))
            return
        if len(path) >= max_depth:
            return
        for neighbor in edges_by_cause.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    dfs(source, [source], {source})
    return paths
