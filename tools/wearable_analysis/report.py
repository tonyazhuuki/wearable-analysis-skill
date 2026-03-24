"""
Health Portrait Report Generator.

Produces a standalone HTML report combining:
1. Key Metrics Summary (6 cards)
2. Cardio Deep Dive (VO2max, RHR, HRV, Z2 gap)
3. Sleep Deep Dive (duration, efficiency, stages, consistency, bedtime)
4. Recovery Deep Dive (distribution, sleep-bucket breakdown, cascade)
5. Training Deep Dive (strain, steps, game sports, Z2/RT gaps)
6. Stress Deep Dive (decomposition, interventions)
7. Correlation Discovery (top-10 with scientific context)
8. Recommendations (top-5 ranked)
9. Discovery Zone (new findings, coverage)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional

from .config import (POPULATION_NORMS, get_percentile, get_user_profile,
                     get_age_bracket, load_user_config)
from .personalize import population_comparison, detect_anomalies
from .discovery import humanize_metric


# ════════════════════════════════════════════════════════════════
# Grading System
# ════════════════════════════════════════════════════════════════

def _grade(value, thresholds):
    """Convert a numeric value to letter grade based on thresholds.

    Args:
        value: numeric score (0-100)
        thresholds: list of (min_value, grade) tuples, sorted ascending

    Returns:
        Letter grade string
    """
    result = thresholds[0][1]  # default to lowest grade
    for threshold, grade in thresholds:
        if value >= threshold:
            result = grade
    return result


GRADE_THRESHOLDS = [
    (0, 'F'), (30, 'D'), (50, 'C'), (60, 'C+'),
    (65, 'B-'), (75, 'B'), (85, 'B+'),
    (90, 'A-'), (95, 'A'), (98, 'A+'),
]

GRADE_COLORS = {
    'A+': '#00e676', 'A': '#00c853', 'A-': '#69f0ae',
    'B+': '#64dd17', 'B': '#aeea00', 'B-': '#c6ff00',
    'C+': '#ffd600', 'C': '#ffab00', 'C-': '#ff9100',
    'D': '#ff6d00', 'F': '#dd2c00',
}

DOMAIN_NAMES = {
    'cardio': 'Cardio Fitness',
    'sleep': 'Sleep Quality',
    'recovery': 'Recovery',
    'training': 'Training Load',
    'stress': 'Stress Management',
}

DOMAIN_NAMES_RU = {
    'cardio': 'Кардио',
    'sleep': 'Сон',
    'recovery': 'Восстановление',
    'training': 'Тренировки',
    'stress': 'Стресс',
}

TREND_LABELS_RU = {
    'improving': 'улучшение',
    'declining': 'снижение',
    'stable': 'стабильно',
    'insufficient_data': 'мало данных',
    'unknown': 'нет данных',
}

GRADE_DESCRIPTIONS_RU = {
    'A+': 'Элитный уровень',
    'A': 'Отличный результат',
    'A-': 'Уровень тренированного атлета',
    'B+': 'Выше среднего, хорошая форма',
    'B': 'Хороший уровень',
    'B-': 'Чуть выше среднего',
    'C+': 'Средний уровень, есть потенциал',
    'C': 'Средний, требует внимания',
    'C-': 'Ниже среднего',
    'D': 'Слабый, требует работы',
    'F': 'Критический уровень',
}

GRADE_DESCRIPTIONS_EN = {
    'A+': 'Elite level',
    'A': 'Excellent',
    'A-': 'Well-trained athlete',
    'B+': 'Above average, good shape',
    'B': 'Good level',
    'B-': 'Slightly above average',
    'C+': 'Average, room for improvement',
    'C': 'Average, needs attention',
    'C-': 'Below average',
    'D': 'Poor, needs work',
    'F': 'Critical',
}


# ════════════════════════════════════════════════════════════════
# Core Helpers
# ════════════════════════════════════════════════════════════════

def _trend(series, window=90):
    """Calculate trend direction from last `window` days vs prior.

    Returns one of: 'improving', 'declining', 'stable', 'insufficient_data'
    """
    if len(series) < window * 2:
        return 'insufficient_data'
    recent = series.iloc[-window:].mean()
    prior = series.iloc[-window * 2:-window].mean()
    if prior == 0:
        return 'stable'
    pct_change = (recent - prior) / abs(prior) * 100
    if pct_change > 5:
        return 'improving'
    elif pct_change < -5:
        return 'declining'
    return 'stable'


def _trend_pct(series, window=90):
    """Calculate trend percentage change (last window vs prior window)."""
    if len(series) < window * 2:
        return None
    recent = series.iloc[-window:].mean()
    prior = series.iloc[-window * 2:-window].mean()
    if prior == 0:
        return 0.0
    return (recent - prior) / abs(prior) * 100


def _trend_arrow(trend_str):
    """Convert trend string to arrow character."""
    arrows = {
        'improving': '&#x2197;',   # ↗
        'declining': '&#x2198;',   # ↘
        'stable': '&#x2192;',     # →
    }
    return arrows.get(trend_str, '?')


def _safe_mean(series, last_n=None):
    """Safely compute mean, optionally for last N values."""
    s = series.dropna()
    if len(s) == 0:
        return None
    if last_n and len(s) >= last_n:
        return s.iloc[-last_n:].mean()
    return s.mean()


def _safe_median(series):
    """Safely compute median."""
    s = series.dropna()
    return s.median() if len(s) > 0 else None


def _safe_std(series):
    """Safely compute std."""
    s = series.dropna()
    return s.std() if len(s) > 1 else None


def _safe_min(series):
    s = series.dropna()
    return s.min() if len(s) > 0 else None


def _safe_max(series):
    s = series.dropna()
    return s.max() if len(s) > 0 else None


def _first_90d_mean(series):
    """Mean of first 90 days."""
    s = series.dropna()
    if len(s) < 90:
        return s.mean() if len(s) > 0 else None
    return s.iloc[:90].mean()


def _last_90d_mean(series):
    """Mean of last 90 days."""
    s = series.dropna()
    if len(s) < 90:
        return s.mean() if len(s) > 0 else None
    return s.iloc[-90:].mean()


def _last_30d_mean(series):
    """Mean of last 30 days."""
    s = series.dropna()
    if len(s) < 30:
        return s.mean() if len(s) > 0 else None
    return s.iloc[-30:].mean()


def _status_icon(value, thresholds_good, thresholds_warn):
    """Return status icon based on value and thresholds.

    thresholds_good: (min, max) for green check
    thresholds_warn: (min, max) for yellow warning
    Outside both = red cross
    """
    if thresholds_good[0] <= value <= thresholds_good[1]:
        return '✅'
    if thresholds_warn[0] <= value <= thresholds_warn[1]:
        return '⚠️'
    return '❌'


def _badge_class(value, thresholds_good, thresholds_warn):
    """Return badge CSS class."""
    if thresholds_good[0] <= value <= thresholds_good[1]:
        return 'badge-green'
    if thresholds_warn[0] <= value <= thresholds_warn[1]:
        return 'badge-yellow'
    return 'badge-red'


def _trend_css(pct_change, inverted=False):
    """Return trend CSS class and arrow for a percentage change.

    inverted=True means negative change is good (e.g., RHR going down).
    """
    if pct_change is None:
        return 'trend-stable', '→', '?'
    is_positive = pct_change > 0
    if inverted:
        is_positive = not is_positive
    if abs(pct_change) < 2:
        return 'trend-stable', '→', f'{pct_change:+.1f}%'
    if is_positive:
        css = 'trend-up' if not inverted else 'trend-down-good'
    else:
        css = 'trend-down-bad' if not inverted else 'trend-down-bad'
    # Correct: for inverted metrics, decreasing value is good
    if inverted:
        if pct_change < -2:
            css = 'trend-down-good'
        elif pct_change > 2:
            css = 'trend-down-bad'
    else:
        if pct_change > 2:
            css = 'trend-up'
        elif pct_change < -2:
            css = 'trend-down-bad'
    arrow = '↗' if pct_change > 0 else ('↘' if pct_change < 0 else '→')
    return css, arrow, f'{pct_change:+.1f}%'


def _format_number(val, decimals=1):
    """Format number with space-separated thousands."""
    if val is None:
        return '—'
    if abs(val) >= 1000:
        return f'{val:,.0f}'.replace(',', ' ')
    return f'{val:.{decimals}f}'


def _compute_correlation(df, col1, col2):
    """Compute Pearson correlation between two columns, handling NaN."""
    if col1 not in df.columns or col2 not in df.columns:
        return None, 0
    mask = df[[col1, col2]].dropna()
    if len(mask) < 30:
        return None, len(mask)
    r = mask[col1].corr(mask[col2])
    return round(r, 2) if not np.isnan(r) else None, len(mask)


# ════════════════════════════════════════════════════════════════
# Domain Grade Computation
# ════════════════════════════════════════════════════════════════

def compute_domain_grades(df: pd.DataFrame, sex: str, age: int) -> dict:
    """Compute letter grades for each health domain.

    Args:
        df: master DataFrame with DatetimeIndex
        sex: 'male' or 'female'
        age: user's age in years

    Returns:
        dict mapping domain key -> {grade, score, trend}
    """
    grades = {}

    if df.empty:
        return grades

    # --- CARDIO FITNESS ---
    cardio_scores = []
    if 'vo2max' in df.columns:
        vo2 = df['vo2max'].dropna()
        if len(vo2) > 0:
            vo2_val = vo2.iloc[-30:].mean() if len(vo2) >= 30 else vo2.mean()
            pct = get_percentile(vo2_val, POPULATION_NORMS.get('vo2max', {}),
                                 sex, age)
            if pct is not None:
                cardio_scores.append(pct)
    if 'rhr' in df.columns:
        rhr = df['rhr'].dropna()
        if len(rhr) > 0:
            rhr_mean = rhr.iloc[-30:].mean() if len(rhr) >= 30 else rhr.mean()
            # RHR percentile-based scoring (inverted: lower is better)
            if rhr_mean < 45:
                rhr_score = 99
            elif rhr_mean < 50:
                rhr_score = 90 + (50 - rhr_mean) * 1.8
            elif rhr_mean < 55:
                rhr_score = 80 + (55 - rhr_mean) * 2.0
            elif rhr_mean < 60:
                rhr_score = 70 + (60 - rhr_mean) * 2.0
            elif rhr_mean < 70:
                rhr_score = 40 + (70 - rhr_mean) * 3.0
            else:
                rhr_score = max(0, 40 - (rhr_mean - 70) * 2)
            cardio_scores.append(rhr_score)

    if cardio_scores:
        avg = np.mean(cardio_scores)
        grades['cardio'] = {
            'grade': _grade(avg, GRADE_THRESHOLDS),
            'score': round(avg, 1),
            'trend': (_trend(df['vo2max'].dropna())
                      if 'vo2max' in df.columns else 'unknown'),
        }

    # --- SLEEP QUALITY ---
    sleep_scores = []
    if 'sleep_hours' in df.columns:
        sh = df['sleep_hours'].dropna()
        if len(sh) > 0:
            mean_sh = sh.mean()
            sleep_dur_score = max(0, 100 - abs(mean_sh - 7.75) * 30)
            sleep_scores.append(sleep_dur_score)
    if 'sleep_efficiency' in df.columns:
        eff = df['sleep_efficiency'].dropna()
        if len(eff) > 0:
            sleep_scores.append(min(100, eff.mean()))
    if 'bed_time_hour' in df.columns:
        bt = df['bed_time_hour'].dropna()
        if len(bt) > 0:
            bt_sd = bt.rolling(7, min_periods=3).std().mean()
            if not np.isnan(bt_sd):
                consistency_score = max(0, 100 - bt_sd * 50)
                sleep_scores.append(consistency_score)

    if sleep_scores:
        avg = np.mean(sleep_scores)
        grades['sleep'] = {
            'grade': _grade(avg, GRADE_THRESHOLDS),
            'score': round(avg, 1),
            'trend': (_trend(df['sleep_hours'].dropna())
                      if 'sleep_hours' in df.columns else 'unknown'),
        }

    # --- RECOVERY ---
    recovery_scores = []
    if 'recovery' in df.columns:
        rec = df['recovery'].dropna()
        if len(rec) > 0:
            recovery_scores.append(rec.mean())
    if 'hrv' in df.columns:
        hrv = df['hrv'].dropna()
        if len(hrv) > 0:
            pct = get_percentile(hrv.mean(),
                                 POPULATION_NORMS.get('hrv', {}), sex, age)
            if pct is not None:
                recovery_scores.append(pct)

    if recovery_scores:
        avg = np.mean(recovery_scores)
        grades['recovery'] = {
            'grade': _grade(avg, GRADE_THRESHOLDS),
            'score': round(avg, 1),
            'trend': (_trend(df['recovery'].dropna())
                      if 'recovery' in df.columns else 'unknown'),
        }

    # --- TRAINING LOAD ---
    training_scores = []
    if 'strain' in df.columns:
        s = df['strain'].dropna()
        if len(s) > 0:
            mean_s = s.mean()
            strain_score = max(0, 100 - abs(mean_s - 11) * 10)
            training_scores.append(strain_score)
    if 'steps' in df.columns:
        st = df['steps'].dropna()
        if len(st) > 0:
            steps_mean = st.mean()
            steps_score = min(100, steps_mean / 100)
            training_scores.append(steps_score)

    if training_scores:
        avg = np.mean(training_scores)
        grades['training'] = {
            'grade': _grade(avg, GRADE_THRESHOLDS),
            'score': round(avg, 1),
            'trend': (_trend(df['strain'].dropna())
                      if 'strain' in df.columns else 'unknown'),
        }

    # --- STRESS ---
    if 'stress_high_pct' in df.columns:
        stress = df['stress_high_pct'].dropna()
        if len(stress) > 0:
            stress_score = max(0, 100 - stress.mean() * 2)
            neg_stress = -df['stress_high_pct'].dropna()
            grades['stress'] = {
                'grade': _grade(stress_score, GRADE_THRESHOLDS),
                'score': round(stress_score, 1),
                'trend': _trend(neg_stress),
            }

    return grades


# ════════════════════════════════════════════════════════════════
# Existing builder helpers (preserved for backward compat)
# ════════════════════════════════════════════════════════════════

def _build_grade_cards_html(grades: dict, lang: str = 'en') -> str:
    """Build HTML for the grade cards grid with human-readable descriptions."""
    names = DOMAIN_NAMES_RU if lang == 'ru' else DOMAIN_NAMES
    descriptions = GRADE_DESCRIPTIONS_RU if lang == 'ru' else GRADE_DESCRIPTIONS_EN
    cards = ''
    for domain_key, domain_name in names.items():
        if domain_key not in grades:
            continue
        g = grades[domain_key]
        color = GRADE_COLORS.get(g['grade'], '#ffffff')
        arrow = _trend_arrow(g.get('trend', ''))
        trend_raw = g.get('trend', 'unknown')
        trend_text = TREND_LABELS_RU.get(trend_raw, trend_raw) if lang == 'ru' else trend_raw.replace('_', ' ')
        desc = descriptions.get(g['grade'], '')
        cards += f'''
            <div class="grade-card">
                <div class="grade-letter" style="color: {color}">{g['grade']}</div>
                <div class="grade-domain">{domain_name}</div>
                <div class="grade-desc">{desc}</div>
                <div class="grade-trend">{arrow} {trend_text}</div>
            </div>'''
    return cards


def _build_top_correlations_html(summary: dict, lang: str = 'en') -> str:
    """Build HTML rows for top correlations table with human-readable names."""
    covered_label = 'ПОКРЫТО' if lang == 'ru' else 'COVERED'
    new_label = 'НОВОЕ' if lang == 'ru' else 'NEW'
    rows = ''
    for item in summary.get('top_10_correlations', [])[:10]:
        is_covered = item.get('covered', False)
        if is_covered:
            badge = f'<span class="badge covered">{covered_label}</span>'
        else:
            badge = f'<span class="badge uncovered">{new_label}</span>'
        hyp = item.get('hypothesis_ids', '')
        r_val = item.get('r', 0)
        css_class = 'positive' if r_val > 0 else 'negative'
        v1 = humanize_metric(item.get('var1', ''), lang)
        v2 = humanize_metric(item.get('var2', ''), lang)
        rows += f'''
        <tr>
            <td>{v1}</td>
            <td>{v2}</td>
            <td class="{css_class}">{r_val:+.3f}</td>
            <td>{item.get('n_obs', '')}</td>
            <td>{badge}</td>
            <td class="hyp-ids">{hyp}</td>
        </tr>'''
    return rows


def _build_population_html(population: pd.DataFrame) -> str:
    """Build HTML rows for population comparison table."""
    rows = ''
    for _, row in population.iterrows():
        pct = row.get('percentile')
        if pd.notna(pct):
            pct_str = f"p{pct:.0f}"
        else:
            pct_str = str(row.get('category', ''))
        unit = row.get('unit', '')
        latest = row.get('user_latest_30d', row['user_mean'])
        rows += f'''
        <tr>
            <td>{row['variable']}</td>
            <td>{row['user_mean']:.1f} {unit}</td>
            <td>{latest:.1f}</td>
            <td><strong>{pct_str}</strong></td>
        </tr>'''
    return rows


def _build_new_hypotheses_html(new_hypotheses: list, lang: str = 'en') -> str:
    """Build HTML for new hypothesis cards with human-readable names."""
    if not new_hypotheses:
        msg = ('Все значимые корреляции покрыты существующими гипотезами.' if lang == 'ru'
               else 'All significant correlations are covered by existing hypotheses.')
        return f'<p style="color:#666">{msg}</p>'

    html = ''
    for h in new_hypotheses[:7]:
        priority_class = h['priority'].lower()
        v1 = humanize_metric(h['var1'], lang)
        v2 = humanize_metric(h['var2'], lang)
        html += f'''
        <div class="hypothesis-card {priority_class}">
            <div class="hyp-header">
                <span class="hyp-id">{h['id']}</span>
                <span class="badge {priority_class}">{h['priority']}</span>
            </div>
            <div class="hyp-vars">{v1} &#x2194; {v2} (r={h['r']:+.3f})</div>
            <div class="hyp-mechanism">{h['proposed_mechanism']}</div>
        </div>'''
    return html


def _build_action_rows_html(actions: Optional[pd.DataFrame]) -> str:
    """Build HTML rows for action plan table."""
    if actions is None or len(actions) == 0:
        return ('<tr><td colspan="3" style="color:#666">'
                'Run hypothesis testing for actionability ranking</td></tr>')

    rows = ''
    priority_dots = {'high': '&#x1F534;', 'medium': '&#x1F7E1;',
                     'low': '&#x1F7E2;'}
    for _, row in actions.head(10).iterrows():
        priority = row.get('priority', 'low')
        dot = priority_dots.get(priority, '')
        action_text = str(row.get('action', ''))[:80]
        score = row.get('actionability_score', 0)
        rows += f'''
            <tr>
                <td>{dot} {priority.upper()}</td>
                <td>{action_text}</td>
                <td>{score:.3f}</td>
            </tr>'''
    return rows


# ════════════════════════════════════════════════════════════════
# Grade CSS helper
# ════════════════════════════════════════════════════════════════

def _grade_css_class(grade: str) -> str:
    """Map grade letter to CSS class for the section-grade span."""
    g = grade.upper().replace('+', 'p').replace('-', 'm')
    first = grade[0].lower()
    return f'grade-{g.lower()}'


def _grade_strip_class(grade: str) -> str:
    """Map grade to the domain-card left strip CSS class."""
    first = grade[0].upper()
    if first == 'A':
        return 'grade-strip-a'
    elif first == 'B':
        return 'grade-strip-b'
    elif first == 'C':
        return 'grade-strip-c'
    return 'grade-strip-d'


def _section_grade_span(grade: str) -> str:
    """Build the inline grade badge for section headers."""
    # Map grades to CSS classes matching the portrait
    g = grade.upper()
    css_map = {
        'A+': 'grade-a', 'A': 'grade-a', 'A-': 'grade-am',
        'B+': 'grade-bp', 'B': 'grade-b', 'B-': 'grade-bm',
        'C+': 'grade-c', 'C': 'grade-c', 'C-': 'grade-c',
        'D': 'grade-d', 'F': 'grade-d',
    }
    css = css_map.get(g, 'grade-b')
    return f'<span class="section-grade {css}">{g}</span>'


# ════════════════════════════════════════════════════════════════
# Scientific Reference Constants (hardcoded — science doesn't change)
# ════════════════════════════════════════════════════════════════

_SCIENCE = {
    'ru': {
        # Cardio
        'vo2max_elite': (
            'По классификации ACSM (11-е издание), для женщин 30 лет значение выше '
            '44 мл/кг/мин — это верхний 1−2%.'
        ),
        'vo2max_mandsager': (
            'В масштабном исследовании <strong>Mandsager 2018</strong> (n=122 007, '
            'Cleveland Clinic) каждый прирост на 1 MET (~3.5 мл/кг/мин) снижал '
            'смертность от всех причин на 13%.'
        ),
        'vo2max_ref': 'Mandsager et al., JAMA Network Open, 2018 &middot; ACSM Guidelines 11th ed., 2021',
        'rhr_quer': (
            'В исследовании <strong>Quer 2020</strong> (n=92 457, данные Fitbit) '
            'низкий пульс покоя — признак хорошей аэробной адаптации.'
        ),
        'rhr_ref': 'Quer et al., Lancet Digital Health, 2020 &middot; Recovery consensus (16 агентов)',
        'hrv_shaffer': (
            'Вариабельность сердечного ритма — показатель того, насколько хорошо '
            'вегетативная нервная система адаптируется к нагрузкам. По данным '
            '<strong>Shaffer & Ginsberg 2017</strong>, высокая HRV = '
            '<strong>сильный парасимпатический тонус</strong>.'
        ),
        'hrv_ref': 'Shaffer & Ginsberg, Front. Public Health, 2017',
        'z2_gap': (
            'По данным Training Polarization consensus (16 агентов), зона 2 = '
            '6 ключевых адаптаций: митохондриальный биогенез, капилляризация, '
            'жировое окисление, утилизация лактата, увеличение ударного объёма, '
            'мышечная выносливость. Игровые виды спорта покрывают ~60% этих адаптаций, но не все.'
        ),
        # Sleep
        'sleep_nsf': (
            'Рекомендация <strong>National Sleep Foundation</strong> (Hirshkowitz 2015): '
            '7−9 часов для взрослых 26−64 лет.'
        ),
        'sleep_nsf_ref': 'Hirshkowitz et al., Sleep Health, 2015',
        'sleep_efficiency_ref': 'Buysse et al., Psychiatry Research, 1989',
        'deep_sleep_ref': 'Van Cauter et al., JAMA, 2000',
        'deep_sleep_gh': (
            'По данным <strong>Van Cauter 2000</strong>, 70−80% суточного гормона '
            'роста выделяется именно во время глубокого сна.'
        ),
        'rem_ref': 'Walker, Why We Sleep, 2017',
        'consistency_phillips': (
            'В исследовании <strong>Phillips 2017</strong> (n=1 978, студенты '
            'Гарварда) нерегулярный сон был связан с <strong>+27% кардиоваскулярного '
            'риска</strong> независимо от длительности сна.'
        ),
        'consistency_ref': 'Phillips et al., Scientific Reports, 2017 &middot; Sleep consensus',
        'bedtime_wittmann': (
            'По <strong>Wittmann 2006</strong>, разница >2 часа между рабочими и '
            'выходными днями снижает качество глубокого сна на 10−15%.'
        ),
        'bedtime_ref': 'Wittmann et al., Chronobiology International, 2006',
        'fragmentation_ref': 'Bonnet & Arand, Sleep Med. Reviews, 2003',
        'fragmentation_text': (
            'По <strong>Bonnet & Arand 2003</strong>, фрагментация сна может быть '
            'функционально эквивалентна потере 2 часов сна даже при сохранении общей длительности.'
        ),
        # Recovery
        'recovery_grade_a': (
            'По данным Recovery consensus (16 агентов, 27 файлов), только '
            '<strong>3 интервенции имеют Grade A</strong> доказательности для '
            'улучшения восстановления: сон, железо и креатин.'
        ),
        'recovery_cascade': (
            '<strong>Модель iron → RHR → HRV → Recovery:</strong> пульс покоя — '
            'главный bottleneck.'
        ),
        'recovery_ref': 'Recovery & HRV consensus (16 агентов)',
        # Training
        'steps_paluch': (
            'Мета-анализ <strong>Paluch 2022</strong> (n=47 471, 15 исследований) '
            'показал, что <strong>7 000−10 000 шагов = максимальная польза для '
            'долголетия</strong>. Дальше кривая выходит на плато.'
        ),
        'steps_ref': 'Paluch et al., Lancet Public Health, 2022',
        'rt_momma': (
            'По мета-анализу <strong>Momma 2022</strong> (British Journal of Sports '
            'Medicine), 30−60 мин силовых в неделю снижают смертность от всех причин '
            'на <strong>10−17%</strong>.'
        ),
        'rt_ref': 'Momma et al., British J Sports Med, 2022',
        'game_sports_text': (
            'По Training Polarization consensus, игровое кардио покрывает ~60% '
            'адаптаций зоны 2. По Stress Management consensus, социальные активности > '
            'CBT > дыхательные упражнения > медитация для управления стрессом.'
        ),
        # Stress
        'stress_consensus': (
            'По Stress Management consensus (16 агентов), типичный стресс WHOOP '
            'раскладывается примерно так: ~50−65% — психологический стресс, остальное — '
            'спортивная нагрузка, дефицит железа, недосып. Эффективность интервенций: '
            'социальная поддержка > когнитивно-поведенческая терапия > дыхательные '
            'практики > прогрессивная мышечная релаксация > медитация.'
        ),
        'stress_ref': 'Stress Management consensus (16 агентов, 28 файлов)',
    },
    'en': {
        'vo2max_elite': (
            'Per ACSM Guidelines (11th ed.), for females aged 30 a value above '
            '44 ml/kg/min places you in the top 1-2%.'
        ),
        'vo2max_mandsager': (
            'In <strong>Mandsager 2018</strong> (n=122,007, Cleveland Clinic), '
            'each +1 MET (~3.5 ml/kg/min) reduced all-cause mortality by 13%.'
        ),
        'vo2max_ref': 'Mandsager et al., JAMA Network Open, 2018 &middot; ACSM Guidelines 11th ed., 2021',
        'rhr_quer': (
            '<strong>Quer 2020</strong> (n=92,457, Fitbit data): low resting HR '
            'indicates strong aerobic adaptation.'
        ),
        'rhr_ref': 'Quer et al., Lancet Digital Health, 2020 &middot; Recovery consensus',
        'hrv_shaffer': (
            'Heart rate variability measures autonomic nervous system adaptability. '
            'Per <strong>Shaffer & Ginsberg 2017</strong>, high HRV = '
            '<strong>strong parasympathetic tone</strong>.'
        ),
        'hrv_ref': 'Shaffer & Ginsberg, Front. Public Health, 2017',
        'z2_gap': (
            'Per Training Polarization consensus: Zone 2 = 6 key adaptations: '
            'mitochondrial biogenesis, capillarization, fat oxidation, lactate '
            'clearance, stroke volume, muscular endurance. Game sports cover ~60%.'
        ),
        'sleep_nsf': (
            '<strong>National Sleep Foundation</strong> (Hirshkowitz 2015): '
            '7-9 hours recommended for adults 26-64.'
        ),
        'sleep_nsf_ref': 'Hirshkowitz et al., Sleep Health, 2015',
        'sleep_efficiency_ref': 'Buysse et al., Psychiatry Research, 1989',
        'deep_sleep_ref': 'Van Cauter et al., JAMA, 2000',
        'deep_sleep_gh': (
            'Per <strong>Van Cauter 2000</strong>, 70-80% of daily growth hormone '
            'is released during deep sleep.'
        ),
        'rem_ref': 'Walker, Why We Sleep, 2017',
        'consistency_phillips': (
            '<strong>Phillips 2017</strong> (n=1,978, Harvard): irregular sleep '
            'was associated with <strong>+27% cardiovascular risk</strong> '
            'independent of sleep duration.'
        ),
        'consistency_ref': 'Phillips et al., Scientific Reports, 2017 &middot; Sleep consensus',
        'bedtime_wittmann': (
            'Per <strong>Wittmann 2006</strong>, >2h difference between work and '
            'weekend bedtimes reduces deep sleep quality by 10-15%.'
        ),
        'bedtime_ref': 'Wittmann et al., Chronobiology International, 2006',
        'fragmentation_ref': 'Bonnet & Arand, Sleep Med. Reviews, 2003',
        'fragmentation_text': (
            'Per <strong>Bonnet & Arand 2003</strong>, sleep fragmentation can be '
            'functionally equivalent to losing 2 hours of sleep.'
        ),
        'recovery_grade_a': (
            'Per Recovery consensus: only <strong>3 Grade A interventions</strong> '
            'for recovery: sleep, iron, creatine.'
        ),
        'recovery_cascade': (
            '<strong>Iron → RHR → HRV → Recovery model:</strong> resting HR is '
            'the main bottleneck.'
        ),
        'recovery_ref': 'Recovery & HRV consensus',
        'steps_paluch': (
            '<strong>Paluch 2022</strong> meta-analysis (n=47,471): '
            '<strong>7,000-10,000 steps = maximum longevity benefit</strong>.'
        ),
        'steps_ref': 'Paluch et al., Lancet Public Health, 2022',
        'rt_momma': (
            '<strong>Momma 2022</strong> (British J Sports Med): 30-60 min/wk '
            'resistance training reduces all-cause mortality by <strong>10-17%</strong>.'
        ),
        'rt_ref': 'Momma et al., British J Sports Med, 2022',
        'game_sports_text': (
            'Per Training Polarization consensus, game sports cover ~60% of Z2 '
            'adaptations. Per Stress Management consensus, social activities > '
            'CBT > breathwork > meditation for stress management.'
        ),
        'stress_consensus': (
            'Per Stress Management consensus: typical WHOOP stress decomposes to '
            '~50-65% psychological, rest = sport + iron deficiency + sleep debt. '
            'Effectiveness: social support > CBT > breathwork > PMR > meditation.'
        ),
        'stress_ref': 'Stress Management consensus',
    },
}


# ════════════════════════════════════════════════════════════════
# Deep Dive Section Generators
# ════════════════════════════════════════════════════════════════

def _metric_row(name_html, badge_text, badge_css, desc_html, ref_html=''):
    """Build a single metric-row card."""
    ref = f'<div class="metric-ref">{ref_html}</div>' if ref_html else ''
    return f'''
    <div class="metric-row">
        <div class="metric-header">
            <span class="metric-name">{name_html}</span>
            <span class="metric-value-badge {badge_css}">{badge_text}</span>
        </div>
        <div class="metric-desc">{desc_html}</div>
        {ref}
    </div>'''


def _gap_box(html_content):
    """Build a gap/warning box."""
    return f'<div class="gap-box">{html_content}</div>'


def _summary_card(value_html, label, context='', trend_css='', trend_text=''):
    """Build a single summary card."""
    ctx = f'<div class="summary-context">{context}</div>' if context else ''
    trend = ''
    if trend_text:
        trend = f'<div class="summary-trend {trend_css}">{trend_text}</div>'
    return f'''
    <div class="summary-card">
        <div class="summary-value">{value_html}</div>
        <div class="summary-label">{label}</div>
        {ctx}
        {trend}
    </div>'''


def _build_key_metrics(df, grades, sex, age, lang='ru'):
    """Build section 1: Key Metrics Summary (6 cards)."""
    sci = _SCIENCE[lang]
    cards = []

    # VO2max
    if 'vo2max' in df.columns:
        vo2 = df['vo2max'].dropna()
        if len(vo2) > 0:
            val = _last_30d_mean(vo2) or vo2.mean()
            pct = get_percentile(val, POPULATION_NORMS.get('vo2max', {}), sex, age)
            pct_str = f'{pct:.0f}' if pct else '?'
            tp = _trend_pct(vo2)
            css, arrow, ttext = _trend_css(tp)
            unit = 'мл/кг/мин' if lang == 'ru' else 'ml/kg/min'
            pct_label = f'{pct_str}-й перцентиль' if lang == 'ru' else f'{pct_str}th percentile'
            cards.append(_summary_card(
                f'{val:.1f} <span class="summary-unit">{unit}</span>',
                'VO₂max', pct_label, css, f'{arrow} {ttext}'))

    # RHR
    if 'rhr' in df.columns:
        rhr = df['rhr'].dropna()
        if len(rhr) > 0:
            val = _last_30d_mean(rhr) or rhr.mean()
            tp = _trend_pct(rhr)
            css, arrow, ttext = _trend_css(tp, inverted=True)
            unit = 'уд/мин' if lang == 'ru' else 'bpm'
            ctx = ('Нижние 5% популяции' if val < 55 else 'Норма') if lang == 'ru' else \
                  ('Bottom 5% of population' if val < 55 else 'Normal')
            lbl = 'Пульс покоя' if lang == 'ru' else 'Resting HR'
            cards.append(_summary_card(
                f'{val:.0f} <span class="summary-unit">{unit}</span>',
                lbl, ctx, css, f'{arrow} {ttext}'))

    # HRV
    if 'hrv' in df.columns:
        hrv = df['hrv'].dropna()
        if len(hrv) > 0:
            val = _last_30d_mean(hrv) or hrv.mean()
            pct = get_percentile(val, POPULATION_NORMS.get('hrv', {}), sex, age)
            pct_str = f'{pct:.0f}' if pct else '?'
            tp = _trend_pct(hrv)
            css, arrow, ttext = _trend_css(tp)
            unit = 'мс' if lang == 'ru' else 'ms'
            pct_label = f'{pct_str}-й перцентиль' if lang == 'ru' else f'{pct_str}th percentile'
            cards.append(_summary_card(
                f'{val:.0f} <span class="summary-unit">{unit}</span>',
                'HRV', pct_label, css, f'{arrow} {ttext}'))

    # Sleep
    if 'sleep_hours' in df.columns:
        sh = df['sleep_hours'].dropna()
        if len(sh) > 0:
            val = _last_30d_mean(sh) or sh.mean()
            tp = _trend_pct(sh)
            css, arrow, ttext = _trend_css(tp)
            unit = 'ч' if lang == 'ru' else 'h'
            norm = 'Норма 7−9 ч' if lang == 'ru' else 'Norm 7-9 h'
            lbl = 'Сон' if lang == 'ru' else 'Sleep'
            cards.append(_summary_card(
                f'{val:.1f} <span class="summary-unit">{unit}</span>',
                lbl, norm, css, f'{arrow} {ttext}'))

    # Recovery
    if 'recovery' in df.columns:
        rec = df['recovery'].dropna()
        if len(rec) > 0:
            val = _last_30d_mean(rec) or rec.mean()
            green_pct = (rec > 66).mean() * 100
            tp = _trend_pct(rec)
            css, arrow, ttext = _trend_css(tp)
            unit = '%'
            green_lbl = f'{green_pct:.0f}% зелёных дней' if lang == 'ru' else f'{green_pct:.0f}% green days'
            lbl = 'Восстановление' if lang == 'ru' else 'Recovery'
            cards.append(_summary_card(
                f'{val:.0f}<span class="summary-unit">{unit}</span>',
                lbl, green_lbl, css, f'{arrow} {ttext}'))

    # Steps
    if 'steps' in df.columns:
        st = df['steps'].dropna()
        if len(st) > 0:
            val = _last_30d_mean(st) or st.mean()
            tp = _trend_pct(st)
            css, arrow, ttext = _trend_css(tp)
            zone = 'Оптимальная зона' if 7000 <= val <= 12000 else ('Ниже оптимума' if val < 7000 else 'Высокая активность')
            if lang == 'en':
                zone = 'Optimal zone' if 7000 <= val <= 12000 else ('Below optimal' if val < 7000 else 'Highly active')
            lbl = 'Шаги/день' if lang == 'ru' else 'Steps/day'
            cards.append(_summary_card(
                _format_number(val, 0),
                lbl, zone, css, f'{arrow} {ttext}'))

    cards_html = '\n'.join(cards)
    title = '1. Ключевые показатели' if lang == 'ru' else '1. Key Metrics'
    return f'''
    <h2>{title}</h2>
    <div class="summary-grid">
        {cards_html}
    </div>'''


def _cardio_deep_dive(df, grades, sex, age, lang='ru'):
    """Build section 2: Cardio deep dive."""
    sci = _SCIENCE[lang]
    html = ''
    g = grades.get('cardio', {})
    grade = g.get('grade', 'B')
    n_days = len(df)

    title = 'Сердечно-сосудистая система' if lang == 'ru' else 'Cardiovascular Fitness'
    html += f'<h2>2. {title} {_section_grade_span(grade)}</h2>'

    # --- VO2max ---
    if 'vo2max' in df.columns:
        vo2 = df['vo2max'].dropna()
        if len(vo2) > 0:
            val_30d = _last_30d_mean(vo2) or vo2.mean()
            pct = get_percentile(val_30d, POPULATION_NORMS.get('vo2max', {}), sex, age)
            pct_str = f'{pct:.0f}' if pct else '?'
            first90 = _first_90d_mean(vo2)
            last90 = _last_90d_mean(vo2)
            mets = val_30d / 3.5

            if pct and pct >= 95:
                icon, badge_text = '✅', ('Элитный уровень' if lang == 'ru' else 'Elite')
            elif pct and pct >= 75:
                icon, badge_text = '✅', ('Выше среднего' if lang == 'ru' else 'Above average')
            else:
                icon, badge_text = '⚠️', ('Средний' if lang == 'ru' else 'Average')

            badge_css = 'badge-green' if pct and pct >= 75 else 'badge-yellow'
            pct_label = f'{pct_str}-й перцентиль' if lang == 'ru' else f'{pct_str}th percentile'

            growth_text = ''
            if first90 and last90 and first90 > 0:
                growth_pct = (last90 - first90) / first90 * 100
                if lang == 'ru':
                    growth_text = (
                        f'<strong>Рост за {n_days // 30} месяцев:</strong> с {first90:.1f} '
                        f'до {val_30d:.1f} ({growth_pct:+.0f}%). Первые 90 дней — {first90:.1f}, '
                        f'последние 90 дней — {last90:.1f}.'
                    )
                else:
                    growth_text = (
                        f'<strong>Growth over {n_days // 30} months:</strong> from {first90:.1f} '
                        f'to {val_30d:.1f} ({growth_pct:+.0f}%). First 90 days — {first90:.1f}, '
                        f'last 90 days — {last90:.1f}.'
                    )

            if lang == 'ru':
                desc = (
                    f'Это <strong>элитный уровень кардиоприспособленности</strong>. '
                    f'{sci["vo2max_elite"]}<br><br>'
                    f'{sci["vo2max_mandsager"]} '
                    f'Ваш уровень ~{mets:.1f} MET ставит вас в категорию высокой выносливости, '
                    f'которая показала <strong>на 80% ниже риск смерти</strong> по сравнению с нижним квинтилем.'
                    f'<br><br>{growth_text}'
                ) if pct and pct >= 90 else (
                    f'{sci["vo2max_elite"]}<br><br>'
                    f'{sci["vo2max_mandsager"]}<br><br>{growth_text}'
                )
            else:
                desc = (
                    f'This is <strong>elite cardiorespiratory fitness</strong>. '
                    f'{sci["vo2max_elite"]}<br><br>'
                    f'{sci["vo2max_mandsager"]} '
                    f'Your level of ~{mets:.1f} MET puts you in the high endurance category, '
                    f'with <strong>80% lower mortality risk</strong> vs bottom quintile.'
                    f'<br><br>{growth_text}'
                ) if pct and pct >= 90 else (
                    f'{sci["vo2max_elite"]}<br><br>'
                    f'{sci["vo2max_mandsager"]}<br><br>{growth_text}'
                )

            unit = 'мл/кг/мин' if lang == 'ru' else 'ml/kg/min'
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> VO₂max — {val_30d:.1f} {unit}',
                pct_label, badge_css, desc, sci['vo2max_ref'])

    # --- RHR ---
    if 'rhr' in df.columns:
        rhr = df['rhr'].dropna()
        if len(rhr) > 0:
            val_30d = _last_30d_mean(rhr) or rhr.mean()
            val_all = rhr.mean()
            val_min = rhr.min()
            val_max = rhr.max()
            val_med = rhr.median()
            first90 = _first_90d_mean(rhr)
            last90 = _last_90d_mean(rhr)

            if val_30d < 50:
                icon, badge_text = '✅', ('Отлично' if lang == 'ru' else 'Excellent')
                badge_css = 'badge-green'
            elif val_30d < 60:
                icon, badge_text = '✅', ('Хорошо' if lang == 'ru' else 'Good')
                badge_css = 'badge-green'
            elif val_30d < 70:
                icon, badge_text = '⚠️', ('Средний' if lang == 'ru' else 'Average')
                badge_css = 'badge-yellow'
            else:
                icon, badge_text = '❌', ('Повышен' if lang == 'ru' else 'Elevated')
                badge_css = 'badge-red'

            # Compute correlation with recovery
            r_recovery, n_corr = _compute_correlation(df, 'rhr', 'recovery')

            trend_text = ''
            if first90 and last90:
                direction = ('снижение' if last90 < first90 else 'рост') if lang == 'ru' else \
                            ('decreasing' if last90 < first90 else 'increasing')
                good_bad = ('. Это хорошо.' if last90 < first90 else '.') if lang == 'ru' else \
                           ('. This is good.' if last90 < first90 else '.')
                trend_text = (
                    f'<strong>{"Тренд" if lang == "ru" else "Trend"}: {direction}</strong> — '
                    f'{"с" if lang == "ru" else "from"} {first90:.1f} '
                    f'{"до" if lang == "ru" else "to"} {last90:.1f} '
                    f'{"за последние 180 дней" if lang == "ru" else "over last 180 days"}{good_bad}'
                )

            unit = 'уд/мин' if lang == 'ru' else 'bpm'

            if lang == 'ru':
                desc = (
                    f'По классификации AHA, пульс покоя ниже 60 — «отлично». '
                    f'{sci["rhr_quer"]}<br><br>'
                    f'{trend_text}<br>'
                    f'Всё время наблюдения: от {val_max:.0f} (макс.) до {val_min:.0f} (мин.), '
                    f'медиана {val_med:.0f}.'
                )
                if r_recovery is not None:
                    desc += (
                        f'<br><br>Из наших данных: <strong>пульс покоя — '
                        f'{"главный" if abs(r_recovery) > 0.5 else "значимый"} предиктор '
                        f'восстановления</strong> (корреляция {r_recovery:+.2f}).'
                    )
            else:
                desc = (
                    f'Per AHA, resting HR below 60 is "excellent". '
                    f'{sci["rhr_quer"]}<br><br>'
                    f'{trend_text}<br>'
                    f'All-time range: {val_min:.0f} (min) to {val_max:.0f} (max), '
                    f'median {val_med:.0f}.'
                )
                if r_recovery is not None:
                    desc += (
                        f'<br><br>From your data: <strong>resting HR is the '
                        f'{"primary" if abs(r_recovery) > 0.5 else "significant"} predictor '
                        f'of recovery</strong> (r={r_recovery:+.2f}).'
                    )

            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Пульс покоя" if lang == "ru" else "Resting HR"} — {val_30d:.0f} {unit}',
                badge_text, badge_css, desc, sci['rhr_ref'])

    # --- HRV ---
    if 'hrv' in df.columns:
        hrv = df['hrv'].dropna()
        if len(hrv) > 0:
            val_30d = _last_30d_mean(hrv) or hrv.mean()
            pct = get_percentile(val_30d, POPULATION_NORMS.get('hrv', {}), sex, age)
            pct_str = f'{pct:.0f}' if pct else '?'
            first90 = _first_90d_mean(hrv)
            last90 = _last_90d_mean(hrv)
            val_min = hrv.min()
            val_max = hrv.max()
            val_med = hrv.median()

            if pct and pct >= 90:
                icon, badge_css = '✅', 'badge-green'
            elif pct and pct >= 50:
                icon, badge_css = '✅', 'badge-green'
            else:
                icon, badge_css = '⚠️', 'badge-yellow'

            pct_label = f'{pct_str}-й перцентиль' if lang == 'ru' else f'{pct_str}th percentile'

            r_recovery, _ = _compute_correlation(df, 'hrv', 'recovery')
            trend_text = ''
            if first90 and last90 and first90 > 0:
                tp = (last90 - first90) / first90 * 100
                if lang == 'ru':
                    trend_text = (
                        f'<strong>Тренд: {"рост" if tp > 0 else "снижение"}</strong> — '
                        f'с {first90:.1f} до {last90:.1f} ({tp:+.1f}%) за последние 180 дней.'
                    )
                else:
                    trend_text = (
                        f'<strong>Trend: {"increasing" if tp > 0 else "decreasing"}</strong> — '
                        f'from {first90:.1f} to {last90:.1f} ({tp:+.1f}%) over last 180 days.'
                    )

            unit = 'мс' if lang == 'ru' else 'ms'
            if lang == 'ru':
                desc = (
                    f'{sci["hrv_shaffer"]}<br><br>'
                    f'Высокая HRV = организм хорошо переключается между нагрузкой и отдыхом. '
                    f'Это одна из сильнейших сторон вашего профиля.<br><br>'
                    f'{trend_text} Диапазон за всё время: {val_min:.0f}−{val_max:.0f}, '
                    f'медиана {val_med:.0f}.'
                )
                if r_recovery is not None:
                    desc += (
                        f'<br><br>В ваших данных HRV и восстановление коррелируют на '
                        f'{r_recovery:+.2f} — <strong>{"самая сильная связь" if abs(r_recovery) > 0.7 else "сильная связь"}</strong> '
                        f'из всех метрик.'
                    )
            else:
                desc = (
                    f'{sci["hrv_shaffer"]}<br><br>'
                    f'High HRV = your body switches effectively between load and rest. '
                    f'This is one of the strongest aspects of your profile.<br><br>'
                    f'{trend_text} All-time range: {val_min:.0f}−{val_max:.0f}, '
                    f'median {val_med:.0f}.'
                )
                if r_recovery is not None:
                    desc += (
                        f'<br><br>In your data, HRV and recovery correlate at '
                        f'{r_recovery:+.2f} — <strong>{"strongest" if abs(r_recovery) > 0.7 else "strong"} relationship</strong>.'
                    )

            html += _metric_row(
                f'<span class="status-icon">{icon}</span> HRV — {val_30d:.0f} {unit} (rMSSD)',
                pct_label, badge_css, desc,
                f'{sci["hrv_ref"]} &middot; '
                f'{"Собственные данные" if lang == "ru" else "Own data"} {n_days} '
                f'{"день" if lang == "ru" else "days"}')

    # --- Z2 gap ---
    z2_col = None
    for col in ['zone2_min', 'z2_min', 'hr_zone2_min']:
        if col in df.columns:
            z2_col = col
            break
    if z2_col:
        z2 = df[z2_col].dropna()
        if len(z2) > 0:
            z2_weekly = z2.mean() * 7
            z2_label = f'{z2_weekly:.0f}'
            gap_title = '⚠️ Разрыв:' if lang == 'ru' else '⚠️ Gap:'
            if z2_weekly < 120:
                if lang == 'ru':
                    html += _gap_box(
                        f'<strong>{gap_title}</strong> Мало целенаправленных тренировок в зоне 2 '
                        f'(лёгкое кардио). Текущий объём ~{z2_label} мин/нед при рекомендованных 120+. '
                        f'{sci["z2_gap"]}'
                    )
                else:
                    html += _gap_box(
                        f'<strong>{gap_title}</strong> Low dedicated Zone 2 training '
                        f'(easy cardio). Current volume ~{z2_label} min/wk vs recommended 120+. '
                        f'{sci["z2_gap"]}'
                    )

    return html


def _sleep_deep_dive(df, grades, lang='ru'):
    """Build section 3: Sleep deep dive."""
    sci = _SCIENCE[lang]
    html = ''
    g = grades.get('sleep', {})
    grade = g.get('grade', 'C')
    n_days = len(df)

    title = 'Сон' if lang == 'ru' else 'Sleep'
    html += f'<h2>3. {title} {_section_grade_span(grade)}</h2>'

    # --- Duration ---
    if 'sleep_hours' in df.columns:
        sh = df['sleep_hours'].dropna()
        if len(sh) > 0:
            val_30d = _last_30d_mean(sh) or sh.mean()
            val_all = sh.mean()
            icon = '✅' if 7 <= val_30d <= 9 else '⚠️'
            badge_css = 'badge-green' if 7 <= val_30d <= 9 else 'badge-yellow'
            badge_text = 'Норма' if lang == 'ru' else 'Normal'
            if val_30d < 6:
                badge_text = 'Мало' if lang == 'ru' else 'Low'
                badge_css = 'badge-red'
                icon = '❌'
            elif val_30d < 7:
                badge_text = 'Ниже нормы' if lang == 'ru' else 'Below norm'
                badge_css = 'badge-yellow'

            if lang == 'ru':
                desc = (
                    f'{sci["sleep_nsf"]} Вы попадаете в '
                    f'{"нижнюю границу рекомендованного диапазона" if 7 <= val_30d < 7.5 else "рекомендованный диапазон" if val_30d >= 7 else "ниже рекомендации"}. '
                    f'Среднее за всё время — {val_all:.2f} ч, '
                    f'{"последний месяц немного лучше" if val_30d > val_all else "последний месяц немного хуже"}.'
                )
            else:
                desc = (
                    f'{sci["sleep_nsf"]} You are '
                    f'{"at the lower boundary" if 7 <= val_30d < 7.5 else "within" if val_30d >= 7 else "below"} '
                    f'the recommended range. All-time mean: {val_all:.2f} h.'
                )
            unit = 'ч' if lang == 'ru' else 'h'
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Длительность" if lang == "ru" else "Duration"} — {val_30d:.1f} {unit}',
                badge_text, badge_css, desc, sci['sleep_nsf_ref'])

    # --- Efficiency ---
    if 'sleep_efficiency' in df.columns:
        eff = df['sleep_efficiency'].dropna()
        if len(eff) > 0:
            val = _last_30d_mean(eff) or eff.mean()
            icon = '✅' if val >= 85 else '⚠️'
            badge_css = 'badge-green' if val >= 85 else 'badge-yellow'
            badge_text = ('Отлично' if val >= 90 else 'Хорошо' if val >= 85 else 'Ниже идеала') if lang == 'ru' else \
                         ('Excellent' if val >= 90 else 'Good' if val >= 85 else 'Below ideal')
            n_obs = len(eff)
            if lang == 'ru':
                desc = (
                    f'Эффективность сна — это доля времени в кровати, проведённая собственно во сне. '
                    f'По <strong>PSQI</strong> (Buysse 1989), порог «хорошо» — 85%. '
                    f'Ваши {val:.1f}% — это <strong>{"отличный результат" if val >= 90 else "хороший результат"}</strong>, '
                    f'стабильный на протяжении всех {n_obs} ночей.'
                )
            else:
                desc = (
                    f'Sleep efficiency = fraction of time in bed spent sleeping. '
                    f'Per <strong>PSQI</strong> (Buysse 1989), threshold: 85%. '
                    f'Your {val:.1f}% is <strong>{"excellent" if val >= 90 else "good"}</strong> '
                    f'across {n_obs} nights.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Эффективность" if lang == "ru" else "Efficiency"} — {val:.1f}%',
                badge_text, badge_css, desc, sci['sleep_efficiency_ref'])

    # --- Deep sleep ---
    if 'deep_pct' in df.columns:
        dp = df['deep_pct'].dropna()
        if len(dp) > 0:
            val_30d = _last_30d_mean(dp) or dp.mean()
            val_all = dp.mean()
            deep_hrs = None
            if 'sleep_hours' in df.columns:
                sh_30d = _last_30d_mean(df['sleep_hours'].dropna())
                if sh_30d:
                    deep_hrs = sh_30d * val_30d / 100

            icon = '✅' if 15 <= val_30d <= 25 else '⚠️'
            badge_css = 'badge-green' if 15 <= val_30d <= 25 else 'badge-yellow'
            badge_text = ('Оптимально' if lang == 'ru' else 'Optimal') if 15 <= val_30d <= 25 else \
                         ('Ниже нормы' if lang == 'ru' else 'Below normal')

            hrs_str = f' ({deep_hrs:.1f} {"ч" if lang == "ru" else "h"})' if deep_hrs else ''
            if lang == 'ru':
                desc = (
                    f'Норма глубокого сна: 15−25% от общего времени. '
                    f'Вы {"в верхней части диапазона" if val_30d >= 20 else "в пределах нормы" if val_30d >= 15 else "ниже нормы"}. '
                    f'{sci["deep_sleep_gh"]}<br><br>'
                    f'<strong>Тренд:</strong> последние 30 дней — {val_30d:.1f}%, '
                    f'среднее за всё время — {val_all:.1f}%.'
                )
            else:
                desc = (
                    f'Normal deep sleep: 15-25% of total. '
                    f'You are {"in the upper range" if val_30d >= 20 else "within normal" if val_30d >= 15 else "below normal"}. '
                    f'{sci["deep_sleep_gh"]}<br><br>'
                    f'<strong>Trend:</strong> last 30d — {val_30d:.1f}%, all-time — {val_all:.1f}%.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Глубокий сон" if lang == "ru" else "Deep sleep"} — {val_30d:.0f}%{hrs_str}',
                badge_text, badge_css, desc, sci['deep_sleep_ref'])

    # --- REM ---
    if 'rem_pct' in df.columns:
        rp = df['rem_pct'].dropna()
        if len(rp) > 0:
            val_30d = _last_30d_mean(rp) or rp.mean()
            rem_hrs = None
            if 'sleep_hours' in df.columns:
                sh_30d = _last_30d_mean(df['sleep_hours'].dropna())
                if sh_30d:
                    rem_hrs = sh_30d * val_30d / 100

            icon = '✅' if val_30d >= 20 else '⚠️'
            badge_css = 'badge-green' if val_30d >= 20 else 'badge-yellow'
            above = val_30d > 25
            badge_text = ('Выше нормы' if above else 'Норма') if lang == 'ru' else ('Above normal' if above else 'Normal')

            hrs_str = f' ({rem_hrs:.1f} {"ч" if lang == "ru" else "h"})' if rem_hrs else ''
            if lang == 'ru':
                desc = (
                    f'Типичная доля REM: 20−25%. '
                    f'Вы {"чуть выше нормы — это хорошо" if above else "в пределах нормы"}. '
                    f'REM-фаза отвечает за <strong>эмоциональную регуляцию</strong> и '
                    f'<strong>консолидацию процедурной памяти</strong>.'
                )
            else:
                desc = (
                    f'Typical REM: 20-25%. '
                    f'You are {"slightly above normal — this is good" if above else "within normal range"}. '
                    f'REM is responsible for <strong>emotional regulation</strong> and '
                    f'<strong>procedural memory consolidation</strong>.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'REM-{"сон" if lang == "ru" else "sleep"} — {val_30d:.0f}%{hrs_str}',
                badge_text, badge_css, desc, sci['rem_ref'])

    # --- Consistency ---
    if 'sleep_consistency' in df.columns:
        sc = df['sleep_consistency'].dropna()
        if len(sc) > 0:
            val = _last_30d_mean(sc) or sc.mean()
            icon = '✅' if val >= 85 else '⚠️'
            badge_css = 'badge-green' if val >= 85 else 'badge-yellow'
            badge_text = ('Хорошо' if val >= 85 else 'Ниже идеала') if lang == 'ru' else \
                         ('Good' if val >= 85 else 'Below ideal')

            if lang == 'ru':
                desc = (
                    f'Стабильность показывает, насколько одинаково вы ложитесь и встаёте каждый день. '
                    f'Идеал: >85%. {sci["consistency_phillips"]}<br><br>'
                    f'По нашему Sleep consensus (16 агентов): <strong>стабильность > длительность</strong> (Grade A).'
                )
            else:
                desc = (
                    f'Consistency measures how regular your sleep schedule is. '
                    f'Ideal: >85%. {sci["consistency_phillips"]}<br><br>'
                    f'Per Sleep consensus: <strong>consistency > duration</strong> (Grade A).'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Стабильность сна" if lang == "ru" else "Sleep consistency"} — {val:.0f}%',
                badge_text, badge_css, desc, sci['consistency_ref'])

    # --- Bedtime ---
    if 'bed_time_hour' in df.columns:
        bt = df['bed_time_hour'].dropna()
        if len(bt) > 0:
            # bed_time_hour is decimal hours from midnight (can be >24 for after midnight)
            val_mean = bt.mean()
            val_median = bt.median()
            val_std = bt.std()
            std_min = val_std * 60  # convert to minutes

            # Convert to time string
            def _hours_to_time(h):
                h = h % 24
                hh = int(h)
                mm = int((h - hh) * 60)
                return f'{hh:02d}:{mm:02d}'

            mean_time = _hours_to_time(val_mean)
            median_time = _hours_to_time(val_median)

            late = val_mean > 24  # after midnight
            icon = '❌' if std_min > 60 or late else ('⚠️' if std_min > 45 else '✅')
            badge_css = 'badge-red' if std_min > 60 or late else ('badge-yellow' if std_min > 45 else 'badge-green')
            badge_text = ('Проблема' if std_min > 60 else 'Ниже идеала') if lang == 'ru' else \
                         ('Problem' if std_min > 60 else 'Below ideal')

            # Bedtime distribution
            before_midnight = (bt < 24).mean() * 100
            after_1am = (bt > 25).mean() * 100
            after_2am = (bt > 26).mean() * 100
            after_3am = (bt > 27).mean() * 100

            if lang == 'ru':
                desc = (
                    f'Медианный отбой — {median_time}, среднее — {mean_time}. '
                    f'Разброс: ±{std_min:.0f} мин. '
                    f'{before_midnight:.0f}% ночей ложитесь до полуночи, '
                    f'{after_1am:.0f}% — после 01:00'
                )
                if after_2am > 5:
                    desc += f', {after_2am:.0f}% — после 02:00'
                if after_3am > 3:
                    desc += f', {after_3am:.0f}% — после 03:00'
                desc += f'.<br><br>{sci["bedtime_wittmann"]}'

                # Recovery by bedtime if available
                if 'recovery' in df.columns:
                    mask_early = (bt < 24) & df['recovery'].notna()
                    mask_late = (bt > 25) & df['recovery'].notna()
                    if mask_early.sum() > 10 and mask_late.sum() > 10:
                        rec_early = df.loc[mask_early, 'recovery'].mean()
                        rec_late = df.loc[mask_late, 'recovery'].mean()
                        desc += (
                            f'<br><br><strong>По данным WHOOP:</strong> при отбое до полуночи '
                            f'восстановление {rec_early:.0f}%, после 01:00 — {rec_late:.0f}%.'
                        )
            else:
                desc = (
                    f'Median bedtime: {median_time}, mean: {mean_time}. '
                    f'Spread: ±{std_min:.0f} min. '
                    f'{before_midnight:.0f}% of nights before midnight, '
                    f'{after_1am:.0f}% after 01:00.'
                    f'<br><br>{sci["bedtime_wittmann"]}'
                )

            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Время отбоя" if lang == "ru" else "Bedtime"} — {mean_time}, '
                f'{"разброс" if lang == "ru" else "spread"} ±{std_min:.0f} '
                f'{"мин" if lang == "ru" else "min"}',
                badge_text, badge_css, desc, sci['bedtime_ref'])

    # --- Wake events ---
    if 'wake_events' in df.columns:
        we = df['wake_events'].dropna()
        if len(we) > 0:
            val = _last_30d_mean(we) or we.mean()
            icon = '✅' if val <= 5 else '⚠️'
            badge_css = 'badge-green' if val <= 5 else 'badge-yellow'
            badge_text = ('Норма' if val <= 5 else 'Умеренно') if lang == 'ru' else \
                         ('Normal' if val <= 5 else 'Moderate')

            awake_time = ''
            if 'awake_time_hrs' in df.columns:
                at = df['awake_time_hrs'].dropna()
                if len(at) > 0:
                    at_val = _last_30d_mean(at) or at.mean()
                    awake_time = (
                        f'{"Время бодрствования за ночь" if lang == "ru" else "Awake time per night"}: '
                        f'{at_val * 60:.0f} {"мин" if lang == "ru" else "min"} '
                        f'({at_val:.2f} {"ч" if lang == "ru" else "h"}).'
                    )

            if lang == 'ru':
                desc = (
                    f'Для молодых взрослых норма — 2−5 пробуждений. '
                    f'{sci["fragmentation_text"]}<br><br>{awake_time}'
                )
            else:
                desc = (
                    f'Normal for young adults: 2-5 awakenings. '
                    f'{sci["fragmentation_text"]}<br><br>{awake_time}'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Пробуждения" if lang == "ru" else "Wake events"} — {val:.0f} '
                f'{"за ночь" if lang == "ru" else "per night"}',
                badge_text, badge_css, desc, sci['fragmentation_ref'])

    # --- Sleep debt ---
    if 'sleep_debt_hrs' in df.columns:
        sd = df['sleep_debt_hrs'].dropna()
        if len(sd) > 0:
            val_30d = _last_30d_mean(sd) or sd.mean()
            val_all = sd.mean()
            icon = '✅' if val_30d < 1 else '⚠️'
            badge_css = 'badge-green' if val_30d < 1 else 'badge-yellow'
            trend_dir = 'улучшение' if val_30d < val_all else 'рост'
            if lang == 'en':
                trend_dir = 'improving' if val_30d < val_all else 'increasing'
            badge_text = ('Лучше среднего' if val_30d < val_all else 'Средний') if lang == 'ru' else \
                         ('Better than average' if val_30d < val_all else 'Average')

            unit = 'ч' if lang == 'ru' else 'h'
            if lang == 'ru':
                desc = (
                    f'WHOOP рассчитывает накопленный дефицит относительно индивидуальной потребности. '
                    f'Среднее за всё время — {val_all:.1f} ч, последний месяц — {val_30d:.1f} ч. '
                    f'<strong>Тренд: {trend_dir}</strong>. Идеал: <1 ч.'
                )
            else:
                desc = (
                    f'WHOOP calculates accumulated debt vs individual need. '
                    f'All-time mean: {val_all:.1f} h, last month: {val_30d:.1f} h. '
                    f'<strong>Trend: {trend_dir}</strong>. Ideal: <1 h.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Дефицит сна" if lang == "ru" else "Sleep debt"} — {val_30d:.1f} {unit}',
                badge_text, badge_css, desc)

    # --- Sleep stress ---
    if 'sleep_stress_pct' in df.columns:
        ss = df['sleep_stress_pct'].dropna()
        if len(ss) > 0:
            val_30d = _last_30d_mean(ss) or ss.mean()
            val_all = ss.mean()
            icon = '✅' if val_30d < 10 else '⚠️'
            badge_css = 'badge-green' if val_30d < 10 else 'badge-yellow'
            badge_text = ('Повышен' if val_30d > val_all * 1.2 else 'Норма') if lang == 'ru' else \
                         ('Elevated' if val_30d > val_all * 1.2 else 'Normal')

            if lang == 'ru':
                desc = (
                    f'Доля времени сна, проведённого в состоянии физиологического стресса. '
                    f'Среднее за всё время — {val_all:.1f}%, последний месяц — {val_30d:.1f}%.'
                )
            else:
                desc = (
                    f'Fraction of sleep time in physiological stress. '
                    f'All-time: {val_all:.1f}%, last month: {val_30d:.1f}%.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Стресс во сне" if lang == "ru" else "Sleep stress"} — {val_30d:.1f}%',
                badge_text, badge_css, desc)

    return html


def _recovery_deep_dive(df, grades, discovery, lang='ru'):
    """Build section 4: Recovery deep dive."""
    sci = _SCIENCE[lang]
    html = ''
    g = grades.get('recovery', {})
    grade = g.get('grade', 'B')

    title = 'Восстановление' if lang == 'ru' else 'Recovery'
    html += f'<h2>4. {title} {_section_grade_span(grade)}</h2>'

    if 'recovery' not in df.columns:
        return html

    rec = df['recovery'].dropna()
    if len(rec) == 0:
        return html

    val_30d = _last_30d_mean(rec) or rec.mean()
    n_days = len(rec)

    badge_text = ('Среднее' if val_30d < 66 else 'Хорошо') if lang == 'ru' else \
                 ('Average' if val_30d < 66 else 'Good')
    badge_css = 'badge-blue' if val_30d < 66 else 'badge-green'

    # Distribution
    green_pct = (rec > 66).mean() * 100
    yellow_pct = ((rec >= 34) & (rec <= 66)).mean() * 100
    red_pct = (rec < 34).mean() * 100

    dist_label = (
        f'<strong>{"Распределение за" if lang == "ru" else "Distribution over"} {n_days} '
        f'{"дней" if lang == "ru" else "days"}:</strong>'
    )
    html += _metric_row(
        f'{"Общий показатель" if lang == "ru" else "Overall"} — {val_30d:.0f}%',
        badge_text, badge_css, dist_label)

    # Recovery zone buckets
    green_label = 'Зелёные' if lang == 'ru' else 'Green'
    yellow_label = 'Жёлтые' if lang == 'ru' else 'Yellow'
    red_label = 'Красные' if lang == 'ru' else 'Red'
    html += f'''
    <div class="bucket-grid">
        <div class="bucket-item">
            <div class="bucket-label" style="color:#4ade80">{green_label}</div>
            <div class="bucket-val" style="color:#4ade80">{green_pct:.0f}%</div>
            <div class="bucket-n">&gt;66%</div>
        </div>
        <div class="bucket-item">
            <div class="bucket-label" style="color:#fbbf24">{yellow_label}</div>
            <div class="bucket-val" style="color:#fbbf24">{yellow_pct:.0f}%</div>
            <div class="bucket-n">34−66%</div>
        </div>
        <div class="bucket-item">
            <div class="bucket-label" style="color:#f87171">{red_label}</div>
            <div class="bucket-val" style="color:#f87171">{red_pct:.0f}%</div>
            <div class="bucket-n">&lt;34%</div>
        </div>
    </div>'''

    # Recovery by sleep bucket
    if 'sleep_hours' in df.columns:
        sh = df['sleep_hours']
        merged = pd.DataFrame({'sleep': sh, 'recovery': df['recovery']}).dropna()
        if len(merged) > 30:
            buckets = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
            bucket_html = ''
            for lo, hi in buckets:
                mask = (merged['sleep'] >= lo) & (merged['sleep'] < hi)
                n = mask.sum()
                if n >= 5:
                    mean_rec = merged.loc[mask, 'recovery'].mean()
                    color = '#4ade80' if mean_rec >= 66 else '#fbbf24' if mean_rec >= 50 else '#f87171'
                    bucket_html += f'''
                    <div class="bucket-item">
                        <div class="bucket-label">{lo}−{hi} {"ч" if lang == "ru" else "h"}</div>
                        <div class="bucket-val" style="color:{color}">{mean_rec:.0f}%</div>
                        <div class="bucket-n">n={n}</div>
                    </div>'''

            if bucket_html:
                header = 'Восстановление по длительности сна' if lang == 'ru' else 'Recovery by sleep duration'
                each_hour = (
                    'Каждый час сна даёт ощутимый прирост.' if lang == 'ru' else
                    'Each hour of sleep provides measurable improvement.'
                )
                html += f'''
                <h3>{header}</h3>
                <div class="metric-desc" style="margin-bottom: 8px;">{each_hour}</div>
                <div class="bucket-grid">{bucket_html}</div>'''

    # Science context
    r_hrv, _ = _compute_correlation(df, 'hrv', 'recovery')
    r_rhr, _ = _compute_correlation(df, 'rhr', 'recovery')
    r_sleep, _ = _compute_correlation(df, 'sleep_hours', 'recovery')
    r_strain, _ = _compute_correlation(df, 'strain', 'recovery')

    first90 = _first_90d_mean(rec)
    last90 = _last_90d_mean(rec)
    trend_text = ''
    if first90 and last90 and first90 > 0:
        tp = (last90 - first90) / first90 * 100
        direction = ('улучшение' if tp > 0 else 'снижение') if lang == 'ru' else ('improving' if tp > 0 else 'declining')
        trend_text = (
            f'<strong>{"Тренд" if lang == "ru" else "Trend"}: {direction}</strong> — '
            f'{"с" if lang == "ru" else "from"} {first90:.1f} '
            f'{"до" if lang == "ru" else "to"} {last90:.1f} ({tp:+.1f}%) '
            f'{"за последние 180 дней" if lang == "ru" else "over last 180 days"}.'
        )

    correlations_text = ''
    parts = []
    if r_hrv is not None:
        parts.append(f'HRV {"объясняет" if lang == "ru" else "explains"} {abs(r_hrv)*100:.0f}% '
                     f'{"вариации" if lang == "ru" else "of variation"} ({r_hrv:+.2f})')
    if r_rhr is not None:
        parts.append(f'{"пульс покоя" if lang == "ru" else "resting HR"} — {abs(r_rhr)*100:.0f}% ({r_rhr:+.2f})')
    if r_sleep is not None:
        parts.append(f'{"сон" if lang == "ru" else "sleep"} ({r_sleep:+.2f})')
    if r_strain is not None:
        parts.append(f'{"нагрузка" if lang == "ru" else "strain"} ({r_strain:+.2f})')
    if parts:
        correlations_text = '. '.join(parts) + '.'

    if lang == 'ru':
        desc = (
            f'<strong>Что говорит наука:</strong> {sci["recovery_grade_a"]}<br><br>'
            f'{sci["recovery_cascade"]} В ваших данных: {correlations_text}<br><br>'
            f'{trend_text}'
        )
    else:
        desc = (
            f'<strong>Scientific context:</strong> {sci["recovery_grade_a"]}<br><br>'
            f'{sci["recovery_cascade"]} In your data: {correlations_text}<br><br>'
            f'{trend_text}'
        )

    html += f'''
    <div class="metric-row" style="margin-top: 12px;">
        <div class="metric-desc">{desc}</div>
        <div class="metric-ref">{sci["recovery_ref"]} &middot; '
        {"Собственные данные" if lang == "ru" else "Own data"} n={n_days}</div>
    </div>'''

    return html


def _training_deep_dive(df, grades, lang='ru'):
    """Build section 5: Training deep dive."""
    sci = _SCIENCE[lang]
    html = ''
    g = grades.get('training', {})
    grade = g.get('grade', 'B')

    title = 'Физическая активность' if lang == 'ru' else 'Physical Activity'
    html += f'<h2>5. {title} {_section_grade_span(grade)}</h2>'

    # --- Strain ---
    if 'strain' in df.columns:
        s = df['strain'].dropna()
        if len(s) > 0:
            val = _last_30d_mean(s) or s.mean()
            icon = '✅' if 8 <= val <= 14 else '⚠️'
            badge_css = 'badge-green' if 8 <= val <= 14 else 'badge-yellow'
            badge_text = ('Оптимум' if 8 <= val <= 14 else 'Высокая' if val > 14 else 'Низкая') if lang == 'ru' else \
                         ('Optimal' if 8 <= val <= 14 else 'High' if val > 14 else 'Low')

            if lang == 'ru':
                desc = (
                    f'WHOOP Strain от 0 до 21, оптимальный диапазон — 8−14. '
                    f'Ваши {val:.1f} — {"в зоне оптимума" if 8 <= val <= 14 else "выше оптимума" if val > 14 else "ниже оптимума"}.'
                )
            else:
                desc = (
                    f'WHOOP Strain 0-21, optimal: 8-14. '
                    f'Your {val:.1f} is {"in the optimal zone" if 8 <= val <= 14 else "above optimal" if val > 14 else "below optimal"}.'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Нагрузка (Strain)" if lang == "ru" else "Strain"} — {val:.1f}',
                badge_text, badge_css, desc)

    # --- Steps ---
    if 'steps' in df.columns:
        st = df['steps'].dropna()
        if len(st) > 0:
            val_30d = _last_30d_mean(st) or st.mean()
            val_all = st.mean()
            tp = _trend_pct(st)
            icon = '✅' if val_30d >= 7000 else '⚠️'
            badge_css = 'badge-green' if val_30d >= 7000 else 'badge-yellow'
            badge_text = ('Оптимально' if val_30d >= 7000 else 'Ниже оптимума') if lang == 'ru' else \
                         ('Optimal' if val_30d >= 7000 else 'Below optimal')

            trend_text = ''
            if tp is not None:
                direction = ('снижение' if tp < -5 else 'рост' if tp > 5 else 'стабильно') if lang == 'ru' else \
                            ('declining' if tp < -5 else 'increasing' if tp > 5 else 'stable')
                trend_text = (
                    f'<strong>{"Тренд" if lang == "ru" else "Trend"}: {direction}</strong> ({tp:+.1f}%).'
                )

            if lang == 'ru':
                desc = (
                    f'{sci["steps_paluch"]}<br><br>'
                    f'Среднее за всё время — {_format_number(val_all, 0)}, '
                    f'последний месяц — {_format_number(val_30d, 0)}. {trend_text}'
                )
            else:
                desc = (
                    f'{sci["steps_paluch"]}<br><br>'
                    f'All-time mean: {_format_number(val_all, 0)}, '
                    f'last month: {_format_number(val_30d, 0)}. {trend_text}'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Шаги" if lang == "ru" else "Steps"} — {_format_number(val_30d, 0)} '
                f'{"в день" if lang == "ru" else "per day"}',
                badge_text, badge_css, desc, sci['steps_ref'])

    # --- Game sports ---
    if 'is_game_sport_day' in df.columns:
        gsd = df['is_game_sport_day'].dropna()
        if gsd.sum() > 0:
            icon = '✅'
            badge_text = ('Сильная сторона' if lang == 'ru' else 'Strength')
            badge_css = 'badge-green'
            if lang == 'ru':
                desc = (
                    f'Падел, футбол, теннис — это <strong>двойная функция</strong>: '
                    f'физическая нагрузка + социальное взаимодействие. {sci["game_sports_text"]}'
                )
            else:
                desc = (
                    f'Padel, football, tennis — <strong>dual function</strong>: '
                    f'physical load + social interaction. {sci["game_sports_text"]}'
                )
            html += _metric_row(
                f'<span class="status-icon">{icon}</span> '
                f'{"Игровые виды спорта" if lang == "ru" else "Game sports"}',
                badge_text, badge_css, desc)

    # --- Z2 gap ---
    z2_col = None
    for col in ['zone2_min', 'z2_min', 'hr_zone2_min']:
        if col in df.columns:
            z2_col = col
            break
    if z2_col:
        z2 = df[z2_col].dropna()
        if len(z2) > 0:
            z2_weekly = z2.mean() * 7
            if z2_weekly < 120:
                gap_title = '⚠️ Разрыв 1: Зона 2' if lang == 'ru' else '⚠️ Gap 1: Zone 2'
                rec_text = '120+' if lang == 'ru' else '120+'
                if lang == 'ru':
                    html += _gap_box(
                        f'<strong>{gap_title}</strong> — {z2_weekly:.0f} мин/нед '
                        f'(рекомендация: {rec_text}). Зона 2 — это интенсивность, при которой можно '
                        f'поддерживать разговор. Именно здесь происходит основной рост митохондрий.'
                    )
                else:
                    html += _gap_box(
                        f'<strong>{gap_title}</strong> — {z2_weekly:.0f} min/wk '
                        f'(recommended: {rec_text}). Zone 2 is conversational intensity. '
                        f'This is where most mitochondrial growth occurs.'
                    )

    # --- RT gap ---
    rt_col = None
    for col in ['strength_time_hrs', 'strength_min']:
        if col in df.columns:
            rt_col = col
            break
    rt_weekly = 0
    if rt_col:
        rt = df[rt_col].dropna()
        if len(rt) > 0:
            rt_weekly = rt.mean() * 7
            if 'hrs' in rt_col:
                rt_weekly *= 60  # convert to min

    if rt_weekly < 30:
        gap_title = '⚠️ Разрыв 2: Силовые тренировки' if lang == 'ru' else '⚠️ Gap 2: Resistance Training'
        if lang == 'ru':
            html += _gap_box(
                f'<strong>{gap_title}</strong> — фактически {rt_weekly:.0f} мин/нед. '
                f'{sci["rt_momma"]} Кроме того, силовые — <strong>самый дружественный к железу</strong> '
                f'вид тренировки: гепцидин после силовых на 40−60% ниже, чем после бега.'
            )
        else:
            html += _gap_box(
                f'<strong>{gap_title}</strong> — currently {rt_weekly:.0f} min/wk. '
                f'{sci["rt_momma"]} Additionally, RT is the most <strong>iron-friendly</strong> '
                f'training: post-RT hepcidin is 40-60% lower than after running.'
            )

    return html


def _stress_deep_dive(df, grades, lang='ru'):
    """Build section 6: Stress deep dive."""
    sci = _SCIENCE[lang]
    html = ''
    g = grades.get('stress', {})
    grade = g.get('grade', 'B')

    title = 'Стресс' if lang == 'ru' else 'Stress'
    html += f'<h2>6. {title} {_section_grade_span(grade)}</h2>'

    if 'stress_high_pct' in df.columns:
        sp = df['stress_high_pct'].dropna()
        if len(sp) > 0:
            val_30d = _last_30d_mean(sp) or sp.mean()
            first90 = _first_90d_mean(sp)
            last90 = _last_90d_mean(sp)

            badge_text = ('Умеренно' if lang == 'ru' else 'Moderate')
            badge_css = 'badge-blue'

            trend_text = ''
            if first90 and last90 and first90 > 0:
                tp = (last90 - first90) / first90 * 100
                direction = ('снижение' if tp < 0 else 'рост') if lang == 'ru' else ('decreasing' if tp < 0 else 'increasing')
                good_bad = ('. Это улучшение.' if tp < 0 else '.') if lang == 'ru' else \
                           ('. This is improvement.' if tp < 0 else '.')
                trend_text = (
                    f'<strong>{"Тренд" if lang == "ru" else "Trend"}: {direction}</strong> — '
                    f'{tp:+.1f}%{good_bad}'
                )

            # Stress minutes breakdown
            stress_detail = ''
            for col, label_ru, label_en in [
                ('stress_high_min', 'высокий стресс', 'high stress'),
                ('stress_med_min', 'средний', 'medium'),
                ('stress_low_min', 'низкий', 'low'),
            ]:
                if col in df.columns:
                    v = _last_30d_mean(df[col].dropna())
                    if v is not None:
                        label = label_ru if lang == 'ru' else label_en
                        stress_detail += f'{label} {v:.1f} мин/день, ' if lang == 'ru' else f'{label} {v:.1f} min/day, '
            if stress_detail:
                stress_detail = stress_detail.rstrip(', ')
                prefix = 'За последний месяц' if lang == 'ru' else 'Last month'
                stress_detail = f'{prefix}: {stress_detail}.'

            if lang == 'ru':
                desc = (
                    f'{stress_detail}<br><br>{trend_text}<br><br>'
                    f'<strong>Что говорит наука:</strong> {sci["stress_consensus"]}'
                )
            else:
                desc = (
                    f'{stress_detail}<br><br>{trend_text}<br><br>'
                    f'<strong>Scientific context:</strong> {sci["stress_consensus"]}'
                )

            html += _metric_row(
                f'{"Высокий стресс" if lang == "ru" else "High stress"} — {val_30d:.1f}% '
                f'{"дня" if lang == "ru" else "of day"}',
                badge_text, badge_css, desc, sci['stress_ref'])

    # Stress vs recovery
    r_stress_rec, _ = _compute_correlation(df, 'stress_high_pct', 'recovery')
    r_rhr_rec, _ = _compute_correlation(df, 'rhr', 'recovery')
    r_hrv_rec, _ = _compute_correlation(df, 'hrv', 'recovery')

    if r_stress_rec is not None:
        if lang == 'ru':
            desc = (
                f'<strong>В ваших данных:</strong> стресс влияет на восстановление '
                f'(корреляция {r_stress_rec:+.2f}), но '
            )
            if r_rhr_rec is not None:
                desc += f'<strong>слабее, чем пульс покоя</strong> ({r_rhr_rec:+.2f}) '
            if r_hrv_rec is not None:
                desc += f'и <strong>HRV</strong> ({r_hrv_rec:+.2f}). '
            desc += (
                'Это означает, что для улучшения восстановления эффективнее работать '
                'с пульсом покоя (через железо и сон), чем со стрессом напрямую.'
            )
        else:
            desc = (
                f'<strong>In your data:</strong> stress affects recovery '
                f'(r={r_stress_rec:+.2f}), but '
            )
            if r_rhr_rec is not None:
                desc += f'<strong>weaker than resting HR</strong> ({r_rhr_rec:+.2f}) '
            if r_hrv_rec is not None:
                desc += f'and <strong>HRV</strong> ({r_hrv_rec:+.2f}). '
            desc += (
                'For recovery improvement, working on resting HR (via iron and sleep) '
                'is more effective than targeting stress directly.'
            )

        html += f'''
        <div class="metric-row">
            <div class="metric-desc">{desc}</div>
        </div>'''

    return html


def _correlations_deep_dive(df, discovery, lang='ru'):
    """Build section 7: Correlations with scientific context."""
    html = ''
    title = 'Что влияет на что' if lang == 'ru' else 'What Drives What'
    subtitle = (
        f'Топ-10 взаимосвязей из ваших {len(df)} дней данных, подтверждённые научной литературой.'
        if lang == 'ru' else
        f'Top 10 relationships from your {len(df)} days of data, confirmed by scientific literature.'
    )
    html += f'''
    <h2>7. {title}</h2>
    <p style="color: #71717a; font-size: 13px; margin-bottom: 12px;">{subtitle}</p>'''

    summary = discovery.get('summary', {})
    top_corrs = summary.get('top_10_correlations', [])

    for i, item in enumerate(top_corrs[:10], 1):
        v1 = humanize_metric(item.get('var1', ''), lang)
        v2 = humanize_metric(item.get('var2', ''), lang)
        r_val = item.get('r', 0)
        n_obs = item.get('n_obs', 0)
        is_covered = item.get('covered', False)

        badges = ''
        if is_covered:
            sci_label = 'Наука ✓' if lang == 'ru' else 'Evidence ✓'
            badges += f'<span class="corr-badge corr-evidence">{sci_label}</span>'
        data_label = f'{"Ваши данные" if lang == "ru" else "Your data"}: {r_val:+.2f}'
        badges += f'<span class="corr-badge corr-your-data">{data_label}</span>'

        # Generate description from hypothesis if available
        mechanism = item.get('mechanism', '')
        if not mechanism:
            mechanism = item.get('hypothesis_mechanism', '')

        html += f'''
        <div class="corr-card">
            <div class="corr-title">{i}. {v1} {"→" if abs(r_val) > 0.5 else "↔"} {v2}</div>
            <div>{badges}</div>
            <div class="corr-desc">{mechanism}</div>
        </div>'''

    return html


def _recommendations_section(df, grades, discovery, actions, lang='ru'):
    """Build section 8: Recommendations."""
    html = ''
    title = 'Рекомендации' if lang == 'ru' else 'Recommendations'
    subtitle = (
        '5 действий с наибольшим ожидаемым эффектом, ранжированные по силе доказательств.'
        if lang == 'ru' else
        '5 actions with highest expected impact, ranked by evidence strength.'
    )
    html += f'''
    <h2>8. {title}</h2>
    <p style="color: #71717a; font-size: 13px; margin-bottom: 12px;">{subtitle}</p>'''

    # If we have computed actions, use them
    if actions is not None and len(actions) > 0:
        for i, (_, row) in enumerate(actions.head(5).iterrows(), 1):
            priority = row.get('priority', 'medium')
            css_class = 'urgent' if priority == 'high' else ('important' if priority == 'medium' else 'good')
            action_text = str(row.get('action', ''))
            effect_text = str(row.get('expected_effect', ''))
            desc = str(row.get('description', ''))
            html += f'''
            <div class="action-card {css_class}">
                <div class="action-num">{"Приоритет" if lang == "ru" else "Priority"} №{i}</div>
                <div class="action-title">{action_text}</div>
                <div class="action-desc">{desc}</div>
                {"<div class='action-effect'>" + effect_text + "</div>" if effect_text else ""}
            </div>'''
    else:
        # Generate data-driven recommendations
        recs = _generate_recommendations(df, grades, lang)
        for i, rec in enumerate(recs[:5], 1):
            html += f'''
            <div class="action-card {rec["css"]}">
                <div class="action-num">{"Приоритет" if lang == "ru" else "Priority"} №{i}</div>
                <div class="action-title">{rec["title"]}</div>
                <div class="action-desc">{rec["desc"]}</div>
                <div class="action-effect">{rec["effect"]}</div>
            </div>'''

    return html


def _generate_recommendations(df, grades, lang='ru'):
    """Generate data-driven recommendations based on grades and metrics."""
    sci = _SCIENCE[lang]
    recs = []

    # Check bedtime consistency
    if 'bed_time_hour' in df.columns:
        bt = df['bed_time_hour'].dropna()
        if len(bt) > 0:
            bt_std = bt.std()
            if bt_std > 1.0:  # > 60 min spread
                std_min = bt_std * 60
                if lang == 'ru':
                    recs.append({
                        'css': 'urgent', 'priority': 1,
                        'title': 'Стабилизировать время отбоя',
                        'desc': (
                            f'Текущий разброс: ±{std_min:.0f} минут. Цель: ±30 минут. '
                            f'Фиксированное время подъёма (якорь) + утренний свет 10−15 мин. '
                            f'{sci["consistency_phillips"]}'
                        ),
                        'effect': 'Ожидаемый эффект: +5−10 пунктов восстановления, +10−15% глубокого сна',
                    })
                else:
                    recs.append({
                        'css': 'urgent', 'priority': 1,
                        'title': 'Stabilize bedtime',
                        'desc': (
                            f'Current spread: ±{std_min:.0f} min. Target: ±30 min. '
                            f'Fixed wake time (anchor) + morning light 10-15 min. '
                            f'{sci["consistency_phillips"]}'
                        ),
                        'effect': 'Expected: +5-10 recovery points, +10-15% deep sleep',
                    })

    # Check sleep grade
    sleep_g = grades.get('sleep', {}).get('score', 100)
    if sleep_g < 75:
        if lang == 'ru':
            recs.append({
                'css': 'urgent', 'priority': 2,
                'title': 'Улучшить качество сна',
                'desc': (
                    f'Оценка сна: {grades.get("sleep", {}).get("grade", "?")}. '
                    f'По Sleep consensus: стабильность > длительность (Grade A). '
                    f'Температура 18-19°C, темнота, экраны за 30 мин до сна.'
                ),
                'effect': 'Ожидаемый эффект: +5−15 пунктов восстановления',
            })
        else:
            recs.append({
                'css': 'urgent', 'priority': 2,
                'title': 'Improve sleep quality',
                'desc': (
                    f'Sleep grade: {grades.get("sleep", {}).get("grade", "?")}. '
                    f'Per Sleep consensus: consistency > duration (Grade A). '
                    f'Temperature 18-19°C, darkness, screens off 30 min before bed.'
                ),
                'effect': 'Expected: +5-15 recovery points',
            })

    # Check Z2
    z2_col = None
    for col in ['zone2_min', 'z2_min', 'hr_zone2_min']:
        if col in df.columns:
            z2_col = col
            break
    if z2_col:
        z2_weekly = df[z2_col].dropna().mean() * 7
        if z2_weekly < 90:
            if lang == 'ru':
                recs.append({
                    'css': 'important', 'priority': 4,
                    'title': f'Увеличить объём зоны 2 до 90+ мин/нед',
                    'desc': (
                        f'Сейчас: ~{z2_weekly:.0f} мин/нед. Рекомендация: 120+ мин/нед. '
                        f'Зона 2 = митохондриальный биогенез, капилляризация, жировое окисление.'
                    ),
                    'effect': 'Ожидаемый эффект: VO₂max +2−3 мл/кг/мин',
                })
            else:
                recs.append({
                    'css': 'important', 'priority': 4,
                    'title': f'Increase Zone 2 to 90+ min/wk',
                    'desc': (
                        f'Current: ~{z2_weekly:.0f} min/wk. Target: 120+ min/wk. '
                        f'Zone 2 = mitochondrial biogenesis, capillarization, fat oxidation.'
                    ),
                    'effect': 'Expected: VO₂max +2-3 ml/kg/min',
                })

    # Check RT
    rt_col = None
    for col in ['strength_time_hrs', 'strength_min']:
        if col in df.columns:
            rt_col = col
            break
    rt_weekly = 0
    if rt_col:
        rt = df[rt_col].dropna()
        if len(rt) > 0:
            rt_weekly = rt.mean() * 7
            if 'hrs' in rt_col:
                rt_weekly *= 60
    if rt_weekly < 30:
        if lang == 'ru':
            recs.append({
                'css': 'important', 'priority': 3,
                'title': 'Добавить 2 силовые тренировки в неделю',
                'desc': (
                    f'Сейчас: {rt_weekly:.0f} мин/нед. Минимальная доза: 30−60 мин/нед. '
                    f'{sci["rt_momma"]}'
                ),
                'effect': 'Ожидаемый эффект: −10−17% ACM, сохранение мышечной массы',
            })
        else:
            recs.append({
                'css': 'important', 'priority': 3,
                'title': 'Add 2 resistance training sessions per week',
                'desc': (
                    f'Current: {rt_weekly:.0f} min/wk. Minimum effective dose: 30-60 min/wk. '
                    f'{sci["rt_momma"]}'
                ),
                'effect': 'Expected: -10-17% ACM, muscle mass preservation',
            })

    # Check wake events
    if 'wake_events' in df.columns:
        we = df['wake_events'].dropna()
        if len(we) > 0:
            val = _last_30d_mean(we) or we.mean()
            if val > 5:
                if lang == 'ru':
                    recs.append({
                        'css': 'good', 'priority': 5,
                        'title': f'Снизить пробуждения до 4−5 за ночь',
                        'desc': (
                            f'Сейчас: {val:.1f} за ночь. Причины: нестабильный отбой, шум, '
                            f'температура. Sleep consensus: 18−19°C + темнота.'
                        ),
                        'effect': 'Ожидаемый эффект: эффективность сна +2−3%',
                    })
                else:
                    recs.append({
                        'css': 'good', 'priority': 5,
                        'title': f'Reduce awakenings to 4-5 per night',
                        'desc': (
                            f'Current: {val:.1f} per night. Causes: irregular bedtime, noise, '
                            f'temperature. Sleep consensus: 18-19°C + darkness.'
                        ),
                        'effect': 'Expected: sleep efficiency +2-3%',
                    })

    # Sort by priority
    recs.sort(key=lambda x: x['priority'])
    return recs


def _discovery_zone(df, discovery, lang='ru'):
    """Build section 9: Discovery Zone with new findings + coverage."""
    html = ''
    title = 'Зона открытий' if lang == 'ru' else 'Discovery Zone'
    subtitle = (
        'Значимые корреляции, не покрытые заранее сформулированными гипотезами.'
        if lang == 'ru' else
        'Significant correlations not covered by pre-registered hypotheses.'
    )
    html += f'''
    <h2>9. {title}</h2>
    <p style="color: #71717a; font-size: 13px; margin-bottom: 12px;">{subtitle}</p>'''

    summary = discovery.get('summary', {})
    total_pairs = summary.get('total_pairs_tested', 0)
    significant = summary.get('significant_after_fdr', 0)
    coverage_pct = summary.get('coverage_pct', 0)
    new_count = summary.get('new_hypotheses_generated', 0)

    pairs_label = 'Пар проверено' if lang == 'ru' else 'Pairs tested'
    sig_label = 'Значимых (FDR)' if lang == 'ru' else 'Significant (FDR)'
    cov_label = 'Покрыто гипотезами' if lang == 'ru' else 'Covered'
    new_label = 'Новых находок' if lang == 'ru' else 'New findings'

    html += f'''
    <div class="stats-row">
        <div class="stat-mini">
            <div class="stat-mini-val">{_format_number(total_pairs, 0)}</div>
            <div class="stat-mini-label">{pairs_label}</div>
        </div>
        <div class="stat-mini">
            <div class="stat-mini-val">{_format_number(significant, 0)}</div>
            <div class="stat-mini-label">{sig_label}</div>
        </div>
        <div class="stat-mini">
            <div class="stat-mini-val">{coverage_pct:.0f}%</div>
            <div class="stat-mini-label">{cov_label}</div>
        </div>
        <div class="stat-mini">
            <div class="stat-mini-val">{new_count}</div>
            <div class="stat-mini-label">{new_label}</div>
        </div>
    </div>'''

    # Coverage bar
    cov_text = (
        f'Покрытие гипотезами: {coverage_pct:.0f}% из {significant} значимых корреляций'
        if lang == 'ru' else
        f'Hypothesis coverage: {coverage_pct:.0f}% of {significant} significant correlations'
    )
    html += f'''
    <div class="coverage-container">
        <div style="font-size: 12px; color: #71717a; margin-bottom: 4px;">{cov_text}</div>
        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {coverage_pct:.0f}%; background: linear-gradient(90deg, #1e3a5f, #818cf8);">
                {coverage_pct:.0f}%
            </div>
        </div>
    </div>'''

    # New hypotheses as discovery cards
    new_hypotheses = discovery.get('new_hypotheses', [])
    for h in new_hypotheses[:5]:
        v1 = humanize_metric(h.get('var1', ''), lang)
        v2 = humanize_metric(h.get('var2', ''), lang)
        mechanism = h.get('proposed_mechanism', '')
        html += f'''
        <div class="discovery-card">
            <div class="discovery-title">{v1} ↔ {v2} (r={h.get("r", 0):+.2f})</div>
            <div class="discovery-desc">{mechanism}</div>
        </div>'''

    return html


# ════════════════════════════════════════════════════════════════
# CSS Template (matching Health Portal portrait exactly)
# ════════════════════════════════════════════════════════════════

def _portrait_css():
    """Return the CSS matching the Health Portal dark theme."""
    return '''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #d4d4d8;
            line-height: 1.65;
            padding: 20px 16px;
            max-width: 900px;
            margin: 0 auto;
        }
        h1 { font-size: 26px; font-weight: 700; color: #fff; margin-bottom: 2px; }
        .subtitle { color: #71717a; font-size: 13px; margin-bottom: 28px; }
        h2 {
            font-size: 19px; font-weight: 600; color: #e4e4e7;
            margin: 36px 0 6px; padding-bottom: 8px;
            border-bottom: 1px solid #27272a;
        }
        .section-grade {
            display: inline-block;
            font-size: 14px; font-weight: 700;
            padding: 2px 10px; border-radius: 6px;
            margin-left: 10px; vertical-align: middle;
        }
        .grade-a  { background: rgba(74,222,128,0.15); color: #4ade80; }
        .grade-am { background: rgba(74,222,128,0.12); color: #86efac; }
        .grade-b  { background: rgba(129,140,248,0.15); color: #818cf8; }
        .grade-bp { background: rgba(129,140,248,0.18); color: #a5b4fc; }
        .grade-bm { background: rgba(129,140,248,0.10); color: #a5b4fc; }
        .grade-c  { background: rgba(251,191,36,0.15); color: #fbbf24; }
        .grade-d  { background: rgba(248,113,113,0.15); color: #f87171; }

        h3 { font-size: 15px; color: #a1a1aa; margin: 18px 0 8px; font-weight: 600; }

        /* Summary cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 16px 0 24px;
        }
        .summary-card {
            background: #141419;
            border: 1px solid #27272a;
            border-radius: 12px;
            padding: 14px 12px;
            text-align: center;
        }
        .summary-value {
            font-size: 28px; font-weight: 700; color: #fff;
            line-height: 1.1;
        }
        .summary-unit { font-size: 13px; color: #71717a; font-weight: 400; }
        .summary-label {
            font-size: 12px; color: #a1a1aa;
            margin-top: 4px;
        }
        .summary-context {
            font-size: 11px; color: #818cf8;
            margin-top: 3px; line-height: 1.3;
        }
        .summary-trend {
            font-size: 12px; margin-top: 4px; font-weight: 500;
        }
        .trend-up { color: #4ade80; }
        .trend-down-good { color: #4ade80; }
        .trend-down-bad { color: #f87171; }
        .trend-stable { color: #71717a; }

        /* Metric row */
        .metric-row {
            background: #141419;
            border: 1px solid #27272a;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
        }
        .metric-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 6px;
        }
        .metric-name { font-size: 14px; font-weight: 600; color: #e4e4e7; }
        .metric-value-badge {
            font-size: 14px; font-weight: 700;
            padding: 2px 10px; border-radius: 6px;
        }
        .badge-green  { background: rgba(74,222,128,0.15); color: #4ade80; }
        .badge-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; }
        .badge-red    { background: rgba(248,113,113,0.15); color: #f87171; }
        .badge-blue   { background: rgba(129,140,248,0.12); color: #a5b4fc; }
        .metric-desc {
            font-size: 13px; color: #a1a1aa; line-height: 1.55;
        }
        .metric-desc strong { color: #d4d4d8; font-weight: 600; }
        .metric-ref {
            font-size: 11px; color: #52525b; margin-top: 4px;
            font-style: italic;
        }
        .status-icon { margin-right: 4px; }

        /* Correlation cards */
        .corr-card {
            background: #141419;
            border: 1px solid #27272a;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
        }
        .corr-title {
            font-size: 14px; font-weight: 600; color: #e4e4e7;
            margin-bottom: 6px;
        }
        .corr-badge {
            display: inline-block;
            padding: 2px 8px; border-radius: 5px;
            font-size: 11px; font-weight: 600;
            margin-right: 6px;
        }
        .corr-evidence { background: rgba(74,222,128,0.12); color: #4ade80; }
        .corr-your-data { background: rgba(129,140,248,0.12); color: #a5b4fc; }
        .corr-new { background: rgba(251,191,36,0.12); color: #fbbf24; }
        .corr-desc {
            font-size: 13px; color: #a1a1aa; line-height: 1.5;
            margin-top: 4px;
        }

        /* Action cards */
        .action-card {
            background: #141419;
            border-left: 3px solid #818cf8;
            border-radius: 0 10px 10px 0;
            padding: 14px 16px;
            margin: 10px 0;
        }
        .action-card.urgent { border-left-color: #f87171; }
        .action-card.important { border-left-color: #fbbf24; }
        .action-card.good { border-left-color: #4ade80; }
        .action-num {
            font-size: 12px; font-weight: 700; color: #818cf8;
            margin-bottom: 4px;
        }
        .action-title {
            font-size: 14px; font-weight: 600; color: #e4e4e7;
            margin-bottom: 6px;
        }
        .action-desc {
            font-size: 13px; color: #a1a1aa; line-height: 1.5;
        }
        .action-effect {
            font-size: 12px; color: #4ade80; margin-top: 6px;
            font-weight: 500;
        }

        /* Recovery bucket grid */
        .bucket-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 6px;
            margin: 10px 0;
        }
        .bucket-item {
            background: #1e1e2a;
            border-radius: 6px;
            padding: 8px 6px;
            text-align: center;
        }
        .bucket-label { font-size: 11px; color: #71717a; }
        .bucket-val { font-size: 16px; font-weight: 700; color: #fff; }
        .bucket-n { font-size: 10px; color: #52525b; }

        /* Coverage bar */
        .coverage-container { margin: 12px 0; }
        .coverage-bar {
            background: #1e1e2a;
            border-radius: 6px;
            height: 28px;
            overflow: hidden;
        }
        .coverage-fill {
            height: 100%;
            border-radius: 6px;
            display: flex; align-items: center;
            padding: 0 10px;
            font-size: 12px; font-weight: 600; color: #fff;
        }

        /* Discovery */
        .discovery-card {
            background: #141419;
            border: 1px solid #27272a;
            border-left: 3px solid #fbbf24;
            border-radius: 0 10px 10px 0;
            padding: 12px 16px;
            margin: 8px 0;
        }
        .discovery-title { font-size: 13px; font-weight: 600; color: #e4e4e7; }
        .discovery-desc { font-size: 12px; color: #a1a1aa; margin-top: 3px; }

        /* Stats row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin: 12px 0;
        }
        .stat-mini {
            background: #141419;
            border-radius: 8px;
            padding: 10px 8px;
            text-align: center;
        }
        .stat-mini-val { font-size: 20px; font-weight: 700; color: #fff; }
        .stat-mini-label { font-size: 10px; color: #71717a; margin-top: 2px; }

        .gap-box {
            background: rgba(251,191,36,0.06);
            border: 1px solid rgba(251,191,36,0.15);
            border-radius: 8px;
            padding: 10px 14px;
            margin: 10px 0;
            font-size: 13px; color: #fbbf24;
        }
        .gap-box strong { color: #fde68a; }

        .footer {
            margin-top: 40px; padding-top: 14px;
            border-top: 1px solid #27272a;
            font-size: 11px; color: #3f3f46;
            text-align: center; line-height: 1.6;
        }

        @media (max-width: 600px) {
            body { padding: 12px 10px; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
            .summary-value { font-size: 24px; }
            .stats-row { grid-template-columns: repeat(2, 1fr); }
            .bucket-grid { grid-template-columns: repeat(3, 1fr); }
            h1 { font-size: 22px; }
            h2 { font-size: 17px; }
        }
    '''


# ════════════════════════════════════════════════════════════════
# Main Generator
# ════════════════════════════════════════════════════════════════

def generate_html_report(df: pd.DataFrame,
                          grades: dict,
                          discovery: dict,
                          hypothesis_results: Optional[pd.DataFrame],
                          population: pd.DataFrame,
                          actions: Optional[pd.DataFrame],
                          output_path: str,
                          user_config: dict = None,
                          lang: str = 'en') -> str:
    """Generate standalone HTML Health Portrait report.

    Produces a deep-dive report matching the Health Portal portrait format
    with 9 sections, scientific context, and computed metrics.

    Args:
        df: master DataFrame (DatetimeIndex)
        grades: domain grades from compute_domain_grades()
        discovery: output from discovery.run_discovery()
        hypothesis_results: from hypothesis testing (optional)
        population: from personalize.population_comparison()
        actions: from personalize.actionability_scoring() (optional)
        output_path: where to save the HTML
        user_config: user configuration dict
        lang: 'ru' or 'en'

    Returns:
        Path to generated HTML file
    """
    # Get user profile
    config = user_config or load_user_config()
    profile = config.get('profile', {})
    sex = profile.get('sex', 'female')
    age = profile.get('age', 30)

    n_days = len(df)
    n_cols = len(df.columns)
    now = datetime.now()

    # Date range
    if hasattr(df.index, 'min') and n_days > 0:
        try:
            date_range = (f"{df.index.min().strftime('%Y-%m-%d')} — "
                          f"{df.index.max().strftime('%Y-%m-%d')}")
        except (AttributeError, ValueError):
            date_range = "Unknown"
    else:
        date_range = "Unknown"

    # Build all sections — each wrapped in try/except to handle missing data gracefully
    no_data_msg = ('<p style="color:#888">Insufficient data for this section.</p>'
                   if lang == 'en' else
                   '<p style="color:#888">Недостаточно данных для этого раздела.</p>')

    try:
        key_metrics_html = _build_key_metrics(df, grades, sex, age, lang)
    except Exception:
        key_metrics_html = ''

    try:
        cardio_html = _cardio_deep_dive(df, grades, sex, age, lang)
    except Exception:
        cardio_html = f'<h2>2. Cardio</h2>{no_data_msg}'

    try:
        sleep_html = _sleep_deep_dive(df, grades, lang)
    except Exception:
        sleep_html = f'<h2>3. Sleep</h2>{no_data_msg}'

    try:
        recovery_html = _recovery_deep_dive(df, grades, discovery, lang)
    except Exception:
        recovery_html = f'<h2>4. Recovery</h2>{no_data_msg}'

    try:
        training_html = _training_deep_dive(df, grades, lang)
    except Exception:
        training_html = f'<h2>5. Training</h2>{no_data_msg}'

    try:
        stress_html = _stress_deep_dive(df, grades, lang)
    except Exception:
        stress_html = f'<h2>6. Stress</h2>{no_data_msg}'

    try:
        correlations_html = _correlations_deep_dive(df, discovery, lang)
    except Exception:
        correlations_html = ''

    try:
        recommendations_html = _recommendations_section(df, grades, discovery, actions, lang)
    except Exception:
        recommendations_html = ''

    try:
        discovery_html = _discovery_zone(df, discovery, lang)
    except Exception:
        discovery_html = ''

    # Header
    if lang == 'ru':
        title_text = 'Портрет здоровья N=1'
        date_str = now.strftime('%d %B %Y').lstrip('0')
        # Rough Russian month names
        months_ru = {
            'January': 'января', 'February': 'февраля', 'March': 'марта',
            'April': 'апреля', 'May': 'мая', 'June': 'июня',
            'July': 'июля', 'August': 'августа', 'September': 'сентября',
            'October': 'октября', 'November': 'ноября', 'December': 'декабря',
        }
        for en, ru in months_ru.items():
            date_str = date_str.replace(en, ru)
        subtitle = f'{n_days} день данных WHOOP &middot; {n_cols} метрик &middot; Обновлено {date_str}'
        footer_text = (
            f'Сгенерировано {date_str} &middot; {n_days} дней данных WHOOP &middot; '
            f'{n_cols} метрик &middot; 20+ consensus references<br>'
            f'Wearable Analysis v2.0 &middot; Literature-First N=1 Pipeline &middot; '
            f'Портал здоровья'
        )
    else:
        title_text = 'N=1 Health Portrait'
        date_str = now.strftime('%B %d, %Y')
        subtitle = f'{n_days} days of WHOOP data &middot; {n_cols} metrics &middot; Updated {date_str}'
        footer_text = (
            f'Generated {date_str} &middot; {n_days} days of WHOOP data &middot; '
            f'{n_cols} metrics &middot; 20+ consensus references<br>'
            f'Wearable Analysis v2.0 &middot; Literature-First N=1 Pipeline &middot; '
            f'Health Portal'
        )

    html_lang = 'ru' if lang == 'ru' else 'en'
    html = f'''<!DOCTYPE html>
<html lang="{html_lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_text}</title>
    <style>{_portrait_css()}</style>
</head>
<body>

<h1>{title_text}</h1>
<div class="subtitle">{subtitle}</div>

{key_metrics_html}

{cardio_html}

{sleep_html}

{recovery_html}

{training_html}

{stress_html}

{correlations_html}

{recommendations_html}

{discovery_html}

<div class="footer">
    {footer_text}
</div>

</body>
</html>'''

    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path
