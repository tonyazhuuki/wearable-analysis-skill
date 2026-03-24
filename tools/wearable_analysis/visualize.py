"""
Standardized visualization module for wearable data analysis.
All figures: 300 DPI, clean style, consistent color palette.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings('ignore')

# Consistent style
COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'accent': '#FF9800',
    'danger': '#F44336',
    'neutral': '#9E9E9E',
    'recovery_green': '#4CAF50',
    'recovery_yellow': '#FFC107',
    'recovery_red': '#F44336',
    'sleep_purple': '#9C27B0',
    'hrv_blue': '#2196F3',
    'rhr_red': '#F44336',
    'strain_orange': '#FF9800',
    'z1': '#E3F2FD',
    'z2': '#4CAF50',
    'z3': '#FF9800',
    'z4': '#F44336',
    'z5': '#B71C1C',
}

DPI = 300


def setup_style():
    """Apply consistent matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.figsize': (14, 8),
    })


def plot_timeline_dashboard(df: pd.DataFrame, output_path: str,
                             title: str = 'Wearable Data Dashboard') -> str:
    """9-panel dashboard of key metrics over time."""
    setup_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    panels = [
        ('recovery', 'Recovery %', COLORS['recovery_green'], '7d', None),
        ('hrv', 'HRV (ms)', COLORS['hrv_blue'], '7d', None),
        ('rhr', 'RHR (bpm)', COLORS['rhr_red'], '7d', None),
        ('vo2max', 'VO2max', '#9C27B0', '30d', None),
        ('steps', 'Steps/day', COLORS['strain_orange'], '7d', 10000),
        ('sleep_hours', 'Sleep (h)', COLORS['sleep_purple'], '7d', 7.5),
        ('stress_high_pct', 'High Stress %', COLORS['danger'], '7d', None),
        ('strain', 'Strain', COLORS['strain_orange'], '7d', None),
        ('deep_pct', 'Deep Sleep %', '#3F51B5', '7d', 15),
    ]

    for ax, (col, label, color, window, target) in zip(axes.flat, panels):
        if col not in df.columns or df[col].isna().all():
            ax.text(0.5, 0.5, f'{label}\n(no data)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(label, fontsize=9)
            continue

        data = df[['date', col]].dropna()
        if len(data) == 0:
            continue

        # Raw data (transparent)
        ax.plot(data['date'], data[col], color=color, alpha=0.2, linewidth=0.5)

        # Rolling average
        w = int(window.replace('d', ''))
        rolling = data.set_index('date')[col].rolling(f'{w}D').mean()
        ax.plot(rolling.index, rolling.values, color=color, linewidth=2)

        if target:
            ax.axhline(y=target, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(label, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str,
                              title: str = 'Correlation Matrix',
                              threshold: float = 0.3) -> str:
    """Heatmap with significant correlations highlighted."""
    setup_style()
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.abs(corr_matrix) < threshold
    sns.heatmap(corr_matrix, mask=None, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                annot_kws={'size': 7},
                cbar_kws={'label': 'Pearson r'})

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_dose_response(x_values: list, y_means: list, y_cis_low: list, y_cis_high: list,
                        x_label: str, y_label: str, title: str,
                        output_path: str, optimal_range: tuple = None,
                        literature_line: float = None) -> str:
    """Dose-response curve with confidence intervals."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.array(x_values)
    y = np.array(y_means)
    ci_lo = np.array(y_cis_low)
    ci_hi = np.array(y_cis_high)

    ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color=COLORS['primary'])
    ax.plot(x, y, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)

    if optimal_range:
        ax.axvspan(optimal_range[0], optimal_range[1], alpha=0.1, color=COLORS['secondary'],
                   label=f'Optimal: {optimal_range[0]}-{optimal_range[1]}')

    if literature_line is not None:
        ax.axhline(y=literature_line, color=COLORS['neutral'], linestyle='--',
                   label=f'Literature: {literature_line:.1f}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_shap_summary(shap_values, feature_names: list, output_path: str,
                       title: str = 'Feature Importance (SHAP)') -> str:
    """SHAP summary plot."""
    setup_style()
    try:
        import shap
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False,
                          max_display=20)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    except ImportError:
        # Fallback: bar chart of mean |SHAP|
        fig, ax = plt.subplots(figsize=(10, 8))
        importance = np.abs(shap_values).mean(axis=0)
        idx = np.argsort(importance)[-20:]
        ax.barh(range(len(idx)), importance[idx], color=COLORS['primary'])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    return output_path


def plot_population_comparison(comparison_df: pd.DataFrame, output_path: str,
                                user_name: str = 'You') -> str:
    """Population percentile chart."""
    setup_style()
    metrics_with_pct = comparison_df.dropna(subset=['percentile'])

    if len(metrics_with_pct) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(metrics_with_pct) * 0.8)))

    y_pos = range(len(metrics_with_pct))
    pcts = metrics_with_pct['percentile'].values
    labels = metrics_with_pct['variable'].values

    colors = [COLORS['secondary'] if p >= 75 else COLORS['accent'] if p >= 50
              else COLORS['danger'] for p in pcts]

    bars = ax.barh(y_pos, pcts, color=colors, alpha=0.7, height=0.6)

    # Reference lines
    for ref in [25, 50, 75, 90]:
        ax.axvline(x=ref, color='gray', linestyle=':', alpha=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{l}\n({metrics_with_pct.iloc[i]['user_latest_30d']:.1f} "
                        f"{metrics_with_pct.iloc[i]['unit']})"
                        for i, l in enumerate(labels)], fontsize=9)
    ax.set_xlabel('Population Percentile')
    ax.set_title(f'{user_name} vs Population Norms (age/sex adjusted)')
    ax.set_xlim(0, 100)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'p{pct:.0f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_causal_dag(dag_dict: dict, output_path: str) -> str:
    """Visualize causal DAG (matplotlib fallback if graphviz unavailable)."""
    setup_style()
    edges = dag_dict['edges']

    if not edges:
        return None

    # Get unique nodes
    nodes = list(set([e['cause'] for e in edges] + [e['effect'] for e in edges]))
    n = len(nodes)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a) * 3, np.sin(a) * 3) for node, a in zip(nodes, angles)}

    # Draw edges
    for e in edges:
        x1, y1 = pos[e['cause']]
        x2, y2 = pos[e['effect']]
        color = COLORS['danger'] if e['correlation'] < 0 else COLORS['primary']
        width = max(0.5, min(3.0, e['strength'] * 5))
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=width, alpha=0.7))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"r={e['correlation']:.2f}", fontsize=7, ha='center',
                color=color, alpha=0.8)

    # Draw nodes
    for node in nodes:
        x, y = pos[node]
        circle = plt.Circle((x, y), 0.4, color='white', ec='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node.replace('_', '\n'), ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=6)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Estimated Causal DAG (Granger causality)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_actionability_matrix(actions_df: pd.DataFrame, output_path: str) -> str:
    """Effect size × modifiability scatter plot."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    df = actions_df.copy()
    if len(df) == 0:
        return None

    colors = {'high': COLORS['danger'], 'medium': COLORS['accent'], 'low': COLORS['secondary']}
    for priority, group in df.groupby('priority'):
        ax.scatter(group['modifiability'], group['effect_size'],
                   s=group['actionability_score'] * 2000 + 50,
                   c=colors.get(priority, 'gray'), alpha=0.6,
                   label=f'{priority} priority', edgecolors='black', linewidth=0.5)

    # Label top items
    for _, row in df.head(8).iterrows():
        label = str(row.get('id', ''))[:12]
        ax.annotate(label, (row['modifiability'], row['effect_size']),
                    fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Modifiability (can you change this?)')
    ax.set_ylabel('Effect Size (how much does it matter?)')
    ax.set_title('Actionability Matrix — What to Focus On')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Quadrant labels
    ax.text(0.8, 0.9, 'HIGH PRIORITY\n(big effect, easy to change)',
            transform=ax.transAxes, ha='center', fontsize=8, color='green', alpha=0.5)
    ax.text(0.2, 0.1, 'LOW PRIORITY\n(small effect or hard to change)',
            transform=ax.transAxes, ha='center', fontsize=8, color='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path


def plot_monthly_scorecards(df: pd.DataFrame, output_path: str) -> str:
    """Monthly scorecards: grade each month by key metrics."""
    setup_style()
    monthly = df.set_index('date').resample('ME').agg({
        'recovery': 'mean', 'hrv': 'mean', 'rhr': 'mean',
        'sleep_hours': 'mean', 'strain': 'mean',
    }).dropna(how='all')

    if len(monthly) < 3:
        return None

    fig, axes = plt.subplots(len(monthly.columns), 1, figsize=(14, 3 * len(monthly.columns)),
                              sharex=True)

    for ax, col in zip(axes, monthly.columns):
        values = monthly[col].values
        colors_list = []
        for v in values:
            mean = monthly[col].mean()
            std = monthly[col].std()
            if col == 'rhr':  # lower is better
                colors_list.append(COLORS['secondary'] if v < mean - 0.5 * std
                                   else COLORS['danger'] if v > mean + 0.5 * std
                                   else COLORS['accent'])
            else:  # higher is better
                colors_list.append(COLORS['secondary'] if v > mean + 0.5 * std
                                   else COLORS['danger'] if v < mean - 0.5 * std
                                   else COLORS['accent'])

        ax.bar(monthly.index, values, color=colors_list, alpha=0.7, width=25)
        ax.axhline(y=monthly[col].mean(), color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.2)

    axes[0].set_title('Monthly Scorecards')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    return output_path
