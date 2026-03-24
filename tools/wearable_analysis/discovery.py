"""
Data-first discovery: all-pairs correlation scanning + hypothesis coverage mapping.

Dual pipeline:
1. DISCOVERY: data -> all correlations -> filter significant -> new findings
2. COVERAGE: significant correlations -> map to hypothesis database -> gap report
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import logging
import os
import yaml

logger = logging.getLogger(__name__)

# Human-readable labels for WHOOP metrics (Russian)
METRIC_LABELS_RU = {
    'recovery': 'Восстановление %',
    'hrv': 'Вариабельность ЧСС (мс)',
    'rhr': 'Пульс покоя (уд/мин)',
    'resp_rate': 'Частота дыхания',
    'spo2': 'SpO₂',
    'skin_temp': 'Температура кожи',
    'sleep_hours': 'Длительность сна (ч)',
    'sleep_efficiency': 'Эффективность сна %',
    'sleep_consistency': 'Стабильность сна %',
    'deep_pct': 'Глубокий сон %',
    'rem_pct': 'REM сон %',
    'light_pct': 'Лёгкий сон %',
    'deep_hrs': 'Глубокий сон (ч)',
    'rem_hrs': 'REM сон (ч)',
    'wake_events': 'Пробуждения',
    'sleep_debt_hrs': 'Дефицит сна (ч)',
    'bed_time_hour': 'Время отбоя',
    'wake_time_hour': 'Время подъёма',
    'sleep_hr_avg': 'ЧСС во сне (средн)',
    'sleep_perf': 'Качество сна %',
    'strain': 'Нагрузка (strain)',
    'steps': 'Шаги',
    'calories': 'Калории',
    'vo2max': 'VO₂max',
    'stress_high_pct': 'Высокий стресс %',
    'stress_high_min': 'Высокий стресс (мин)',
    'stress_med_min': 'Средний стресс (мин)',
    'stress_low_min': 'Низкий стресс (мин)',
    'healthspan_age': 'Биологический возраст',
    'pace_of_aging': 'Темп старения',
    'years_older_younger': 'Моложе/старше (лет)',
    'weight_kg': 'Вес (кг)',
    'body_fat_pct': 'Жир %',
    'lean_mass_pct': 'Мышечная масса %',
    'hrv_rhr_ratio': 'HRV/RHR соотношение',
    'sleep_stress_pct': 'Стресс во сне %',
    'awake_pct': 'Бодрствование %',
    'awake_time_hrs': 'Бодрствование (ч)',
    'restorative_pct': 'Восстанавливающий сон %',
    'hr_zones_13_hrs': 'Зоны ЧСС 1-3 (ч)',
    'hr_zones_45_hrs': 'Зоны ЧСС 4-5 (ч)',
    'strength_time_hrs': 'Силовые (ч)',
    'nap_hours': 'Дневной сон (ч)',
    'dow': 'День недели',
    'month': 'Месяц',
    'n_activities': 'Кол-во тренировок',
    'is_game_sport_day': 'Игровой день',
    'stress_3d': 'Стресс 3-дн среднее',
}

METRIC_LABELS_EN = {
    'recovery': 'Recovery %',
    'hrv': 'HRV (ms)',
    'rhr': 'Resting HR (bpm)',
    'resp_rate': 'Respiratory Rate',
    'spo2': 'SpO₂',
    'skin_temp': 'Skin Temperature',
    'sleep_hours': 'Sleep Duration (h)',
    'sleep_efficiency': 'Sleep Efficiency %',
    'sleep_consistency': 'Sleep Consistency %',
    'deep_pct': 'Deep Sleep %',
    'rem_pct': 'REM Sleep %',
    'light_pct': 'Light Sleep %',
    'wake_events': 'Wake Events',
    'sleep_debt_hrs': 'Sleep Debt (h)',
    'bed_time_hour': 'Bedtime',
    'wake_time_hour': 'Wake Time',
    'strain': 'Strain',
    'steps': 'Steps',
    'calories': 'Calories',
    'vo2max': 'VO₂max',
    'stress_high_pct': 'High Stress %',
    'healthspan_age': 'Biological Age',
    'pace_of_aging': 'Pace of Aging',
    'hrv_rhr_ratio': 'HRV/RHR Ratio',
    'sleep_stress_pct': 'Sleep Stress %',
    'dow': 'Day of Week',
}


def humanize_metric(name: str, lang: str = 'ru') -> str:
    """Convert internal metric name to human-readable label."""
    labels = METRIC_LABELS_RU if lang == 'ru' else METRIC_LABELS_EN
    if name in labels:
        return labels[name]
    # Fallback: clean up underscores
    return name.replace('_', ' ').replace('hs ', 'Healthspan ').title()


def all_pairs_correlation(df: pd.DataFrame,
                          method: str = 'spearman',
                          min_obs: int = 30,
                          fdr_alpha: float = 0.05) -> pd.DataFrame:
    """Compute correlations between ALL numeric column pairs.

    Returns DataFrame with columns:
        var1, var2, r, p_raw, p_fdr, significant, abs_r, n_obs
    Sorted by abs_r descending.
    """
    # Guard: empty or near-empty DataFrame
    if df.empty:
        logger.warning("all_pairs_correlation: empty DataFrame")
        return pd.DataFrame(columns=['var1', 'var2', 'r', 'p_raw', 'p_fdr',
                                      'significant', 'abs_r', 'n_obs'])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out columns with too few non-null values
    valid_cols = [c for c in numeric_cols if df[c].notna().sum() >= min_obs]

    # Guard: need at least 2 valid columns for pairwise correlation
    if len(valid_cols) < 2:
        logger.warning(f"all_pairs_correlation: only {len(valid_cols)} valid columns, need >=2")
        return pd.DataFrame(columns=['var1', 'var2', 'r', 'p_raw', 'p_fdr',
                                      'significant', 'abs_r', 'n_obs'])

    # Remove rolling/lag/derived duplicates to reduce trivial correlations
    skip_patterns = ['_14d', '_30d', 'prev_2d', 'prev_3d', '_7d',
                     '_raw', '_hm', 'prev_1d_']
    filtered_cols = [c for c in valid_cols
                     if not any(p in c for p in skip_patterns)]

    # Guard: after filtering still need >=2
    if len(filtered_cols) < 2:
        filtered_cols = valid_cols[:max(2, len(valid_cols))]

    # Static trivial pairs
    trivial_pairs = {
        frozenset({'hrv', 'hrv_rhr_ratio'}),
        frozenset({'stress_high_min', 'stress_high_pct'}),
        frozenset({'stress_med_min', 'stress_low_min'}),
        frozenset({'sleep_perf', 'sleep_perf_recovery'}),
        frozenset({'deep_pct', 'light_pct'}),
        frozenset({'rem_pct', 'light_pct'}),
        frozenset({'deep_hrs', 'deep_pct'}),
        frozenset({'rem_hrs', 'rem_pct'}),
        frozenset({'light_hrs', 'light_pct'}),
        frozenset({'sleep_hours', 'in_bed_hrs'}),
        frozenset({'healthspan_age', 'years_older_younger'}),
        frozenset({'stress_3d', 'stress_high_pct'}),
        frozenset({'stress_high_pct_3d', 'stress_3d'}),
        frozenset({'sleep_debt_hrs', 'sleep_debt_added'}),
        frozenset({'restorative_hrs', 'restorative_pct'}),
        frozenset({'awake_time_hrs', 'awake_pct'}),
        frozenset({'sleep_stress_pct', 'sleep_stress_high_hrs'}),
        frozenset({'sleep_stress_pct', 'sleep_stress_med_hrs'}),
        frozenset({'hr_zones_13_hrs', 'hr_zones_45_hrs'}),
    }

    def _trivially_related(a: str, b: str) -> bool:
        """Check if two columns are trivial derivations of each other."""
        # Same base with different suffix
        suffixes = ['_pct', '_min', '_hrs', '_impact', '_3d',
                    '_mean', '_30d', '_6mo', '_weekly']
        for s in suffixes:
            if a.replace(s, '') == b.replace(s, '') and a != b:
                return True
        # All healthspan components are algorithmically linked
        if a.startswith('hs_') and b.startswith('hs_'):
            return True
        hs_roots = {'healthspan_age', 'years_older_younger', 'pace_of_aging'}
        if (a.startswith('hs_') and b in hs_roots) or (b.startswith('hs_') and a in hs_roots):
            return True
        # Sleep stage percentages sum to ~100%
        sleep_stages = {'deep_pct', 'rem_pct', 'light_pct', 'awake_pct'}
        if a in sleep_stages and b in sleep_stages:
            return True
        return False

    results = []
    n_cols = len(filtered_cols)
    total_pairs = n_cols * (n_cols - 1) // 2
    logger.info(f"Scanning {total_pairs:,} pairs from {n_cols} columns...")

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = filtered_cols[i], filtered_cols[j]

            # Skip trivial pairs (static + dynamic)
            if frozenset({col1, col2}) in trivial_pairs:
                continue
            if col1 in col2 or col2 in col1:
                continue
            if _trivially_related(col1, col2):
                continue

            mask = df[col1].notna() & df[col2].notna()
            n = mask.sum()
            if n < min_obs:
                continue

            if method == 'spearman':
                r, p = stats.spearmanr(df.loc[mask, col1], df.loc[mask, col2])
            else:
                r, p = stats.pearsonr(df.loc[mask, col1], df.loc[mask, col2])

            if np.isnan(r):
                continue

            results.append({
                'var1': col1, 'var2': col2,
                'r': round(r, 4), 'p_raw': p,
                'abs_r': round(abs(r), 4), 'n_obs': n
            })

    if not results:
        logger.warning("No valid correlation pairs found.")
        return pd.DataFrame(columns=['var1', 'var2', 'r', 'p_raw', 'p_fdr',
                                      'significant', 'abs_r', 'n_obs'])

    result_df = pd.DataFrame(results)

    # FDR correction
    reject, p_fdr, _, _ = multipletests(result_df['p_raw'], alpha=fdr_alpha,
                                         method='fdr_bh')
    result_df['p_fdr'] = p_fdr
    result_df['significant'] = reject

    return result_df.sort_values('abs_r', ascending=False).reset_index(drop=True)


def map_correlations_to_hypotheses(correlations: pd.DataFrame,
                                    hypotheses_dir: str) -> pd.DataFrame:
    """Map each significant correlation to hypothesis database.

    For each correlation pair, check if ANY hypothesis tests those variables.

    Returns correlations DataFrame with added columns:
        covered, hypothesis_ids, coverage_type (exact/partial/none)
    """
    if correlations.empty:
        correlations = correlations.copy()
        correlations['covered'] = pd.Series(dtype=bool)
        correlations['hypothesis_ids'] = pd.Series(dtype=str)
        correlations['coverage_type'] = pd.Series(dtype=str)
        return correlations

    # Load all hypothesis files
    all_hypotheses = []
    if os.path.isdir(hypotheses_dir):
        for fname in os.listdir(hypotheses_dir):
            if not fname.endswith('.yaml'):
                continue
            fpath = os.path.join(hypotheses_dir, fname)
            try:
                with open(fpath) as f:
                    content = yaml.safe_load(f)
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to load {fpath}: {e}")
                continue

            # Handle both formats (list of dicts, or dict with 'hypotheses' key)
            if isinstance(content, list):
                hyps = content
            elif isinstance(content, dict) and 'hypotheses' in content:
                hyps = content['hypotheses']
            else:
                continue

            for h in hyps:
                if not isinstance(h, dict):
                    continue
                variables = h.get('variables', [])
                if isinstance(variables, str):
                    variables = [v.strip() for v in variables.split(',')]
                all_hypotheses.append({
                    'id': h.get('id', ''),
                    'variables': set(v.lower().replace(' ', '_') for v in variables),
                    'domain': h.get('domain', ''),
                    'hypothesis': h.get('hypothesis', h.get('description', '')),
                })
    else:
        logger.warning(f"Hypotheses directory not found: {hypotheses_dir}")

    # Build variable -> hypothesis index
    var_to_hyp = {}
    for h in all_hypotheses:
        for v in h['variables']:
            var_to_hyp.setdefault(v, []).append(h['id'])

    # Map each correlation
    covered = []
    hyp_ids = []
    coverage_types = []

    for _, row in correlations.iterrows():
        v1 = row['var1'].lower()
        v2 = row['var2'].lower()

        # Find hypotheses that mention BOTH variables
        h1 = set(var_to_hyp.get(v1, []))
        h2 = set(var_to_hyp.get(v2, []))
        exact_match = h1 & h2

        # Partial: hypothesis mentions one variable
        partial = (h1 | h2) - exact_match

        if exact_match:
            covered.append(True)
            hyp_ids.append(', '.join(sorted(exact_match)))
            coverage_types.append('exact')
        elif partial:
            covered.append(True)
            hyp_ids.append(', '.join(sorted(partial)[:3]))  # top 3
            coverage_types.append('partial')
        else:
            covered.append(False)
            hyp_ids.append('')
            coverage_types.append('none')

    correlations = correlations.copy()
    correlations['covered'] = covered
    correlations['hypothesis_ids'] = hyp_ids
    correlations['coverage_type'] = coverage_types

    return correlations


def generate_discovery_hypotheses(uncovered: pd.DataFrame,
                                   max_hypotheses: int = 10) -> list:
    """Generate new hypothesis proposals from uncovered significant correlations.

    Returns list of dicts with:
        id, var1, var2, r, p_fdr, n_obs, direction, priority,
        proposed_mechanism, test_suggestion
    """
    if uncovered.empty:
        return []

    new_hyps = []
    for i, row in uncovered.head(max_hypotheses).iterrows():
        direction = 'positive' if row['r'] > 0 else 'negative'
        r_abs = row['abs_r']
        priority = 'HIGH' if r_abs > 0.3 else 'MEDIUM' if r_abs > 0.2 else 'LOW'

        new_hyps.append({
            'id': f'NH_{i + 1:03d}',
            'var1': row['var1'],
            'var2': row['var2'],
            'r': row['r'],
            'p_fdr': row['p_fdr'],
            'n_obs': row['n_obs'],
            'direction': direction,
            'priority': priority,
            'proposed_mechanism': (
                f"Significant {direction} correlation (r={row['r']:.3f}) "
                f"between {row['var1']} and {row['var2']} "
                f"-- mechanism unknown, requires investigation"
            ),
            'test_suggestion': (
                f"Test with lagged correlation, controlling for confounders "
                f"(sleep_hours, strain, day_of_week)"
            ),
        })

    return new_hyps


BASE_METRICS = [
    'recovery', 'hrv', 'rhr', 'resp_rate', 'spo2', 'skin_temp',
    'sleep_hours', 'sleep_efficiency', 'sleep_consistency', 'deep_pct', 'rem_pct',
    'wake_events', 'sleep_debt_hrs', 'bed_time_hour', 'sleep_perf',
    'strain', 'steps', 'calories', 'vo2max',
    'stress_high_pct', 'stress_high_min',
    'healthspan_age', 'pace_of_aging',
    'weight_kg', 'body_fat_pct',
    'n_activities', 'is_game_sport_day',
    'dow', 'month',
]


def run_discovery(df: pd.DataFrame, hypotheses_dir: str, output_dir: str,
                  base_metrics_only: bool = False) -> dict:
    """Run full discovery pipeline.

    Args:
        df: enriched master DataFrame
        hypotheses_dir: path to hypothesis YAML files
        output_dir: path to output directory
        base_metrics_only: if True, restrict correlation scan to BASE_METRICS
            only (no derived features like ratios, interactions, rolling
            averages). This gives an "honest" coverage number — what %
            of REAL metric relationships are covered by hypotheses.

    Returns dict with:
        all_correlations, significant, mapped, covered, uncovered,
        coverage_pct, new_hypotheses, summary
    """
    # Guard: empty DataFrame
    if df.empty:
        logger.warning("run_discovery: empty DataFrame")
        return {
            'all_correlations': pd.DataFrame(),
            'significant': pd.DataFrame(),
            'mapped': pd.DataFrame(),
            'covered': pd.DataFrame(),
            'uncovered': pd.DataFrame(),
            'coverage_pct': 0,
            'new_hypotheses': [],
            'summary': {
                'total_pairs_tested': 0, 'significant_after_fdr': 0,
                'covered_by_hypotheses': 0, 'uncovered': 0, 'coverage_pct': 0,
                'base_metrics_only': base_metrics_only,
                'n_metrics_scanned': 0, 'new_hypotheses_generated': 0,
                'top_10_correlations': [], 'top_uncovered': [],
            },
        }

    if base_metrics_only:
        # Keep only base metrics + date for honest coverage
        keep_cols = [c for c in BASE_METRICS if c in df.columns] + ['date']
        df_scan = df[keep_cols].copy()
        logger.info(f"Base-metrics-only mode: {len(keep_cols) - 1} metrics "
                    f"(filtered from {len(df.columns)})")
    else:
        df_scan = df

    logger.info("Running all-pairs correlation scan...")
    all_corr = all_pairs_correlation(df_scan)
    significant = all_corr[all_corr['significant']].copy()

    logger.info(f"Found {len(significant)} significant correlations "
                f"out of {len(all_corr)} pairs")

    # Map to hypotheses
    logger.info("Mapping to hypothesis database...")
    mapped = map_correlations_to_hypotheses(significant, hypotheses_dir)

    covered = mapped[mapped['covered']].copy() if not mapped.empty else mapped
    uncovered = mapped[~mapped['covered']].copy() if not mapped.empty else mapped

    coverage_pct = (len(covered) / len(mapped) * 100) if len(mapped) > 0 else 0

    # Generate new hypotheses from uncovered
    new_hyps = generate_discovery_hypotheses(uncovered)

    # Save outputs
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    all_corr.to_csv(os.path.join(data_dir, 'all_correlations.csv'), index=False)
    significant.to_csv(os.path.join(data_dir, 'significant_correlations.csv'),
                       index=False)
    mapped.to_csv(os.path.join(data_dir, 'correlation_coverage.csv'), index=False)

    if new_hyps:
        pd.DataFrame(new_hyps).to_csv(
            os.path.join(data_dir, 'new_hypotheses.csv'), index=False)

    summary = {
        'total_pairs_tested': len(all_corr),
        'significant_after_fdr': len(significant),
        'covered_by_hypotheses': len(covered),
        'uncovered': len(uncovered),
        'coverage_pct': round(coverage_pct, 1),
        'base_metrics_only': base_metrics_only,
        'n_metrics_scanned': len(df_scan.select_dtypes(include=[np.number]).columns),
        'new_hypotheses_generated': len(new_hyps),
        'top_10_correlations': (
            significant.head(10).to_dict('records') if not significant.empty else []
        ),
        'top_uncovered': (
            uncovered.head(5).to_dict('records') if not uncovered.empty else []
        ),
    }

    logger.info(f"Discovery complete: {coverage_pct:.0f}% coverage, "
                f"{len(new_hyps)} new hypotheses")

    return {
        'all_correlations': all_corr,
        'significant': significant,
        'mapped': mapped,
        'covered': covered,
        'uncovered': uncovered,
        'coverage_pct': coverage_pct,
        'new_hypotheses': new_hyps,
        'summary': summary,
    }
