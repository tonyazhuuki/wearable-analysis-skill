"""
Personalization module: population comparison, anomaly detection, actionability scoring.
"""

import pandas as pd
import numpy as np
from .config import POPULATION_NORMS, get_percentile


def population_comparison(df: pd.DataFrame, sex: str, age: int) -> pd.DataFrame:
    """Compare user's metrics to population norms.

    Returns DataFrame with columns:
        variable, user_mean, user_median, percentile, category, norm_source
    """
    results = []

    if df.empty:
        return pd.DataFrame(results)

    # VO2max
    if 'vo2max' in df.columns:
        vo2 = df['vo2max'].dropna()
        if len(vo2) > 0:
            user_mean = vo2.mean()
            user_latest = vo2.iloc[-30:].mean() if len(vo2) >= 30 else vo2.mean()
            pct = get_percentile(user_latest, POPULATION_NORMS['vo2max'], sex, age)
            results.append({
                'variable': 'VO2max',
                'user_mean': user_mean,
                'user_latest_30d': user_latest,
                'percentile': pct,
                'category': _pct_category(pct),
                'unit': 'ml/kg/min',
                'source': POPULATION_NORMS['vo2max']['source'],
            })

    # HRV
    if 'hrv' in df.columns:
        hrv = df['hrv'].dropna()
        if len(hrv) > 0:
            user_mean = hrv.mean()
            pct = get_percentile(user_mean, POPULATION_NORMS['hrv'], sex, age)
            results.append({
                'variable': 'HRV (rMSSD)',
                'user_mean': user_mean,
                'user_latest_30d': hrv.iloc[-30:].mean() if len(hrv) >= 30 else user_mean,
                'percentile': pct,
                'category': _pct_category(pct),
                'unit': 'ms',
                'source': POPULATION_NORMS['hrv']['source'],
            })

    # RHR
    if 'rhr' in df.columns:
        rhr = df['rhr'].dropna()
        if len(rhr) > 0:
            user_mean = rhr.mean()
            # RHR is inverted — lower is better
            rhr_cat = 'excellent'
            for cat, (lo, hi) in POPULATION_NORMS['rhr']['general'].items():
                if lo <= user_mean < hi:
                    rhr_cat = cat
                    break
            # Estimate RHR percentile (inverted: lower = better)
            # Population: <45=p99, 45-50=p95, 50-55=p85, 55-60=p70, 60-65=p50, 65-72=p30, >72=p10
            rhr_pct_map = [(45, 99), (50, 95), (55, 85), (60, 70), (65, 50), (72, 30), (80, 10)]
            rhr_pct = 5  # default
            for threshold, pct in rhr_pct_map:
                if user_mean < threshold:
                    rhr_pct = pct
                    break
            results.append({
                'variable': 'RHR',
                'user_mean': user_mean,
                'user_latest_30d': rhr.iloc[-30:].mean() if len(rhr) >= 30 else user_mean,
                'percentile': rhr_pct,
                'category': rhr_cat,
                'unit': 'bpm',
                'source': POPULATION_NORMS['rhr']['source'],
            })

    # Sleep hours
    if 'sleep_hours' in df.columns:
        sleep = df['sleep_hours'].dropna()
        if len(sleep) > 0:
            user_mean = sleep.mean()
            norms = POPULATION_NORMS['sleep_hours']['adult_26-64']
            if norms['recommended'][0] <= user_mean <= norms['recommended'][1]:
                cat = 'recommended'
            elif norms['may_be_appropriate'][0] <= user_mean <= norms['may_be_appropriate'][1]:
                cat = 'may_be_appropriate'
            else:
                cat = 'not_recommended'
            # Estimate sleep percentile: 7-8h=p60, 8-9h=p75, 6-7h=p40, <6h=p15, >9h=p50
            sleep_pct = 50
            if 7.0 <= user_mean <= 8.5:
                sleep_pct = 65
            elif 8.5 < user_mean <= 9.0:
                sleep_pct = 55
            elif 6.0 <= user_mean < 7.0:
                sleep_pct = 35
            elif user_mean < 6.0:
                sleep_pct = 10
            results.append({
                'variable': 'Sleep Duration',
                'user_mean': user_mean,
                'user_latest_30d': sleep.iloc[-30:].mean() if len(sleep) >= 30 else user_mean,
                'percentile': sleep_pct,
                'category': cat,
                'unit': 'hours',
                'source': POPULATION_NORMS['sleep_hours']['source'],
            })

    # Sleep efficiency
    if 'sleep_efficiency' in df.columns:
        eff = df['sleep_efficiency'].dropna()
        if len(eff) > 0:
            user_mean = eff.mean()
            norms = POPULATION_NORMS['sleep_efficiency']['adult']
            cat = 'poor'
            for c, (lo, hi) in norms.items():
                if lo <= user_mean <= hi:
                    cat = c
                    break
            # Efficiency percentile: >90%=p80, 85-90%=p60, 75-85%=p30, <75%=p10
            eff_pct = 50
            if user_mean >= 92:
                eff_pct = 90
            elif user_mean >= 88:
                eff_pct = 75
            elif user_mean >= 85:
                eff_pct = 60
            elif user_mean >= 75:
                eff_pct = 30
            else:
                eff_pct = 10
            results.append({
                'variable': 'Sleep Efficiency',
                'user_mean': user_mean,
                'user_latest_30d': eff.iloc[-30:].mean() if len(eff) >= 30 else user_mean,
                'percentile': eff_pct,
                'category': cat,
                'unit': '%',
                'source': POPULATION_NORMS['sleep_efficiency']['source'],
            })

    # Steps
    if 'steps' in df.columns:
        steps = df['steps'].dropna()
        if len(steps) > 0:
            user_mean = steps.mean()
            norms = POPULATION_NORMS['steps']['adult']
            cat = 'sedentary'
            for c, (lo, hi) in norms.items():
                if lo <= user_mean < hi:
                    cat = c
                    break
            results.append({
                'variable': 'Daily Steps',
                'user_mean': user_mean,
                'user_latest_30d': steps.iloc[-30:].mean() if len(steps) >= 30 else user_mean,
                'percentile': None,
                'category': cat,
                'unit': 'steps/day',
                'source': POPULATION_NORMS['steps']['source'],
            })

    return pd.DataFrame(results)


def _pct_category(pct):
    """Convert percentile to descriptive category."""
    if pct is None:
        return 'unknown'
    if pct >= 95:
        return 'elite'
    if pct >= 90:
        return 'excellent'
    if pct >= 75:
        return 'above_average'
    if pct >= 50:
        return 'average'
    if pct >= 25:
        return 'below_average'
    return 'low'


def detect_anomalies(df: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Identify where user significantly differs from population norms.

    Returns DataFrame with:
        variable, direction (positive/negative), magnitude, interpretation
    """
    anomalies = []
    for _, row in population.iterrows():
        pct = row.get('percentile')
        cat = row['category']
        var = row['variable']

        if pct is not None:
            if pct >= 90:
                anomalies.append({
                    'variable': var,
                    'direction': 'positive',
                    'magnitude': f'{pct:.0f}th percentile',
                    'interpretation': f'{var} is exceptional — well above average for age/sex',
                })
            elif pct <= 10:
                anomalies.append({
                    'variable': var,
                    'direction': 'negative',
                    'magnitude': f'{pct:.0f}th percentile',
                    'interpretation': f'{var} is below average — may indicate room for improvement or underlying condition',
                })
        else:
            if cat in ('excellent', 'elite', 'recommended', 'good', 'optimal'):
                anomalies.append({
                    'variable': var,
                    'direction': 'positive',
                    'magnitude': cat,
                    'interpretation': f'{var} is in the optimal range',
                })
            elif cat in ('poor', 'low', 'not_recommended', 'below_average', 'sedentary'):
                anomalies.append({
                    'variable': var,
                    'direction': 'negative',
                    'magnitude': cat,
                    'interpretation': f'{var} is below optimal — prioritize for improvement',
                })

    return pd.DataFrame(anomalies)


def actionability_scoring(hypothesis_results: pd.DataFrame,
                          current_values: dict = None) -> pd.DataFrame:
    """Score each finding by actionability.

    actionability = effect_size × modifiability × confidence × (1 - current_compliance)

    Args:
        hypothesis_results: DataFrame from hypothesis testing with columns:
            id, hypothesis, effect_size, p_value, concordance, actionable, action_if_confirmed
        current_values: dict of current metric values (to assess compliance)

    Returns DataFrame sorted by actionability_score descending.
    """
    # Modifiability map (how easy is it to change this?)
    MODIFIABILITY = {
        'bed_time_hour': 0.9,      # very modifiable
        'sleep_hours': 0.8,        # modifiable with bedtime change
        'sleep_consistency': 0.7,   # modifiable with routine
        'strain': 0.7,             # modifiable (training planning)
        'zone2_min': 0.8,          # modifiable (add Z2 sessions)
        'zone3_min': 0.3,          # low — comes from game sports
        'steps': 0.8,              # modifiable (add walking)
        'stress_high_pct': 0.4,    # partially modifiable
        'hrv': 0.2,                # low direct control (indirect via sleep/stress)
        'rhr': 0.3,                # indirect (via iron, fitness)
        'vo2max': 0.5,             # modifiable long-term
        'deep_pct': 0.3,           # limited direct control
        'wake_events': 0.4,        # some control (environment, caffeine)
        'sleep_debt_hrs': 0.7,     # modifiable via sleep schedule
        'resp_rate': 0.1,          # not directly controllable
        'skin_temp_deviation': 0.1, # not controllable
    }

    results = hypothesis_results.copy()

    scores = []
    for _, row in results.iterrows():
        # Effect size — try numeric columns first, then label
        raw_es = None
        for es_col in ['r', 'cohen_d', 'interaction', 'linear_r', 'slope_per_month',
                        'indirect_effect', 'total_effect']:
            val = row.get(es_col)
            if val is not None and pd.notna(val) and isinstance(val, (int, float)):
                raw_es = abs(float(val))
                break
        if raw_es is None:
            raw_es_label = row.get('effect_size', '')
            if isinstance(raw_es_label, str):
                raw_es = {'negligible': 0.05, 'small': 0.15, 'medium': 0.35, 'large': 0.6,
                          'insufficient_data': 0.1}.get(raw_es_label, 0.1)
            elif pd.notna(raw_es_label):
                raw_es = min(abs(float(raw_es_label)), 1.0)
            else:
                raw_es = 0.1
        es = min(raw_es, 1.0)

        # Modifiability
        variables = row.get('variables_requested', row.get('variables', []))
        if isinstance(variables, str):
            variables = [v.strip() for v in variables.split(',')]
        if not isinstance(variables, list):
            variables = []
        mod = max((MODIFIABILITY.get(v, 0.5) for v in variables), default=0.5)

        # Confidence — use FDR-corrected p if available
        p_val = None
        for pc in ['p_fdr', 'p_unified', 'p', 'p_value', 'linear_p', 'interaction_p']:
            val = row.get(pc)
            if val is not None and pd.notna(val) and isinstance(val, (int, float)):
                p_val = float(val)
                break
        conf = 1 - p_val if p_val is not None else 0.5
        conf = max(0.1, min(0.99, conf))

        # Current compliance (lower = more room for improvement)
        compliance = 0.3  # default: assume 30% compliance (more room for improvement)
        if current_values and variables:
            # TODO: implement compliance checking based on optimal ranges
            pass

        score = es * mod * conf * (1 - compliance)
        scores.append({
            'id': row.get('hypothesis_id', row.get('id', '')),
            'hypothesis': row.get('hypothesis_text', row.get('hypothesis', '')),
            'effect_size': es,
            'modifiability': mod,
            'confidence': conf,
            'compliance': compliance,
            'actionability_score': score,
            'action': row.get('action_if_confirmed', ''),
            'priority': 'high' if score > 0.15 else 'medium' if score > 0.08 else 'low',
        })

    return pd.DataFrame(scores).sort_values('actionability_score', ascending=False)


def generate_personalization_report(df: pd.DataFrame, sex: str, age: int,
                                     hypothesis_results: pd.DataFrame,
                                     output_dir: str) -> str:
    """Generate full personalization report.

    Returns path to personalization_report.md.
    """
    import os

    pop = population_comparison(df, sex, age)
    anomalies = detect_anomalies(df, pop)
    actions = actionability_scoring(hypothesis_results)

    report_lines = [
        '# Personalization Report\n',
        '## Population Comparison\n',
    ]

    # Population table
    report_lines.append('| Metric | Your Value | Latest 30d | Percentile/Category | Source |')
    report_lines.append('|--------|-----------|-----------|-------------------|--------|')
    for _, row in pop.iterrows():
        pct_str = f"p{row['percentile']:.0f}" if pd.notna(row.get('percentile')) else row['category']
        report_lines.append(
            f"| {row['variable']} | {row['user_mean']:.1f} {row['unit']} | "
            f"{row['user_latest_30d']:.1f} | **{pct_str}** | {row['source']} |"
        )

    report_lines.append('\n## Anomalies (Where You Differ)\n')

    positive = anomalies[anomalies['direction'] == 'positive']
    negative = anomalies[anomalies['direction'] == 'negative']

    if len(positive) > 0:
        report_lines.append('### Strengths\n')
        for _, row in positive.iterrows():
            report_lines.append(f"- **{row['variable']}** ({row['magnitude']}): {row['interpretation']}")

    if len(negative) > 0:
        report_lines.append('\n### Areas for Improvement\n')
        for _, row in negative.iterrows():
            report_lines.append(f"- **{row['variable']}** ({row['magnitude']}): {row['interpretation']}")

    report_lines.append('\n## Actionability Ranking\n')
    report_lines.append('| Priority | Action | Score | Effect Size | Modifiability | Confidence |')
    report_lines.append('|----------|--------|-------|-------------|---------------|------------|')
    for _, row in actions.head(15).iterrows():
        emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['priority'], '')
        report_lines.append(
            f"| {emoji} {row['priority']} | {row['action'][:60]} | "
            f"{row['actionability_score']:.3f} | {row['effect_size']:.2f} | "
            f"{row['modifiability']:.1f} | {row['confidence']:.2f} |"
        )

    report_path = os.path.join(output_dir, 'reports', 'personalization_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Save data
    pop.to_csv(os.path.join(output_dir, 'data', 'population_comparison.csv'), index=False)
    anomalies.to_csv(os.path.join(output_dir, 'data', 'anomalies.csv'), index=False)
    actions.to_csv(os.path.join(output_dir, 'data', 'actionability_scores.csv'), index=False)

    return report_path
