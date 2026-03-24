"""
Configuration: schemas, population norms, constants for wearable analysis.

Central schema registry loaded from schema.yaml — single source of truth
for all WHOOP field names, internal names, types, and aliases.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

# ============================================================
# User Configuration — loaded from user_config.yaml
# ============================================================

_USER_CONFIG_CACHE = None


def load_user_config() -> dict:
    """Load user_config.yaml if present, else use defaults from template."""
    global _USER_CONFIG_CACHE
    if _USER_CONFIG_CACHE is not None:
        return _USER_CONFIG_CACHE

    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, 'user_config.yaml')
    template_path = os.path.join(config_dir, 'user_config.template.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            _USER_CONFIG_CACHE = yaml.safe_load(f) or {}
        logger.info(f"Loaded user config from {config_path}")
    elif os.path.exists(template_path):
        with open(template_path, 'r') as f:
            _USER_CONFIG_CACHE = yaml.safe_load(f) or {}
        logger.warning("No user_config.yaml found — using template defaults. "
                       "Copy user_config.template.yaml to user_config.yaml and customize.")
    else:
        _USER_CONFIG_CACHE = {}
        logger.warning("No user config found. Using built-in defaults.")

    return _USER_CONFIG_CACHE


def get_user_profile() -> dict:
    """Get user demographics from config."""
    config = load_user_config()
    profile = config.get('profile', {})
    return {
        'sex': profile.get('sex', 'female'),
        'age': profile.get('age', 30),
        'timezone': profile.get('timezone', 'UTC'),
    }


def get_data_dir(device: str = None) -> str:
    """Get data directory for specified device from user config."""
    config = load_user_config()
    device = device or config.get('device', {}).get('type', 'whoop')

    # First check user config
    user_dir = config.get('device', {}).get('data_dir', '')
    if user_dir:
        return user_dir

    # Fall back to DEVICE_CONFIGS defaults
    return DEVICE_CONFIGS.get(device, {}).get('data_dir', '')


def get_game_sports_from_config() -> set:
    """Get game sports list from user config, falling back to schema.yaml."""
    config = load_user_config()
    user_sports = config.get('game_sports', [])
    if user_sports:
        return set(user_sports)
    return get_game_sports()


def get_output_config() -> dict:
    """Get output configuration (base_dir, figure_dpi, figure_style)."""
    config = load_user_config()
    output = config.get('output', {})
    return {
        'base_dir': output.get('base_dir', 'wearable_analysis_output'),
        'figure_dpi': output.get('figure_dpi', ANALYSIS_PARAMS.get('figure_dpi', 300)),
        'figure_style': output.get('figure_style',
                                   ANALYSIS_PARAMS.get('figure_style', 'seaborn-v0_8-whitegrid')),
    }


def get_enabled_domains() -> dict:
    """Get which hypothesis domains are enabled."""
    config = load_user_config()
    defaults = {
        'recovery': True, 'sleep': True, 'training': True,
        'stress': True, 'interactions': True, 'cycle': False,
    }
    return {**defaults, **config.get('domains', {})}


def get_health_context() -> dict:
    """Get optional health context for deeper personalization."""
    config = load_user_config()
    return config.get('health_context', {
        'conditions': [], 'medications': [], 'supplements': [], 'genetics': {}
    })


# ============================================================
# Schema Registry — loaded from schema.yaml
# ============================================================

_SCHEMA_CACHE = None
_ALIAS_MAP_CACHE = None


def load_schema() -> dict:
    """Load schema.yaml — the single source of truth for all WHOOP fields."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.yaml')
    with open(schema_path, 'r') as f:
        _SCHEMA_CACHE = yaml.safe_load(f)
    return _SCHEMA_CACHE


def build_alias_map() -> dict:
    """Build a complete alias → internal_name mapping from schema.yaml.

    Used by ALL modules (hypothesis_test, eda, personalize) to resolve
    column names. Single source of truth — no more scattered aliases.

    Returns:
        dict mapping any alias/whoop_key → internal column name
    """
    global _ALIAS_MAP_CACHE
    if _ALIAS_MAP_CACHE is not None:
        return _ALIAS_MAP_CACHE

    schema = load_schema()
    alias_map = {}

    # Source fields
    for source_name, source_config in schema.get('sources', {}).items():
        for field in source_config.get('fields', []):
            internal = field['internal']
            # Self-map
            alias_map[internal] = internal
            # WHOOP keys
            for wk in field.get('whoop_keys', []):
                alias_map[wk] = internal
            # Aliases
            for a in field.get('aliases', []):
                alias_map[a] = internal

    # Derived fields
    for field in schema.get('derived', []):
        internal = field.get('internal', '')
        if field.get('pattern'):
            # Expand known pattern aliases (zone pcts, rolling, lags)
            if 'zone{n}' in internal:
                for n in range(1, 6):
                    expanded = internal.replace('{n}', str(n))
                    alias_map[expanded] = expanded
                    for a_pat in field.get('aliases_pattern', []):
                        alias_map[a_pat.replace('{n}', str(n))] = expanded
            continue
        alias_map[internal] = internal
        for a in field.get('aliases', []):
            alias_map[a] = internal

    # Game sports
    alias_map['_game_sports'] = set(schema.get('game_sports', []))

    _ALIAS_MAP_CACHE = alias_map
    return _ALIAS_MAP_CACHE


def resolve_column(name: str, df_columns: list) -> str:
    """Resolve a column name to an actual column in the DataFrame.

    Resolution order:
    1. Exact match in df_columns
    2. Alias map lookup → check if mapped name in df_columns
    3. Case-insensitive fuzzy match (strip underscores)
    4. None (with warning)

    Args:
        name: column name to resolve (from hypothesis YAML, user code, etc.)
        df_columns: list of actual column names in the DataFrame

    Returns:
        Resolved column name or None
    """
    cols_set = set(df_columns)

    # 1. Exact match
    if name in cols_set:
        return name

    # 2. Alias map
    alias_map = build_alias_map()
    mapped = alias_map.get(name)
    if mapped and mapped in cols_set:
        return mapped

    # 3. Rolling/pattern suffix stripping (e.g. rhr_30d_rolling → rhr_30d)
    for suffix in ('_rolling', '_avg', '_mean'):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if base in cols_set:
                return base

    # 4. Fuzzy match (strip underscores, case-insensitive)
    name_stripped = name.replace('_', '').lower()
    for col in df_columns:
        if col.replace('_', '').lower() == name_stripped:
            return col

    # 5. Not found
    logger.warning(f"Column not resolved: '{name}' (mapped to '{mapped}' but not in DataFrame)")
    return None


def get_whoop_keys(internal_name: str) -> list:
    """Get all WHOOP JSON key names for an internal column name."""
    schema = load_schema()
    for source_config in schema.get('sources', {}).values():
        for field in source_config.get('fields', []):
            if field['internal'] == internal_name:
                return field.get('whoop_keys', [])
    return []


def get_source_fields(source_file: str) -> list:
    """Get all field definitions for a WHOOP JSON source file."""
    schema = load_schema()
    source_config = schema.get('sources', {}).get(source_file, {})
    return source_config.get('fields', [])


def get_date_key(source_file: str) -> str:
    """Get the date field name for a WHOOP JSON source file."""
    schema = load_schema()
    source_config = schema.get('sources', {}).get(source_file, {})
    return source_config.get('date_key', 'Date')


def get_game_sports() -> set:
    """Get the set of game sport activity names."""
    schema = load_schema()
    return set(schema.get('game_sports', []))

# === Core data schema (universal across devices) ===
CORE_SCHEMA = {
    # Time
    'date': 'datetime',

    # Recovery
    'recovery': ('float', 'Recovery score 0-100%'),
    'hrv': ('float', 'Heart rate variability rMSSD (ms)'),
    'rhr': ('float', 'Resting heart rate (bpm)'),
    'resp_rate': ('float', 'Respiratory rate (brpm)'),
    'spo2': ('float', 'Blood oxygen saturation (%)'),

    # Sleep
    'sleep_hours': ('float', 'Total sleep duration (hours)'),
    'sleep_efficiency': ('float', 'Sleep efficiency (%)'),
    'deep_pct': ('float', 'Deep sleep percentage'),
    'rem_pct': ('float', 'REM sleep percentage'),
    'light_pct': ('float', 'Light sleep percentage'),
    'wake_events': ('int', 'Number of wake events'),
    'sleep_latency_min': ('float', 'Sleep onset latency (minutes)'),
    'bed_time_hour': ('float', 'Bedtime as decimal hours from midnight'),
    'wake_time_hour': ('float', 'Wake time as decimal hours from midnight'),
    'sleep_debt_hrs': ('float', 'Accumulated sleep debt (hours)'),
    'sleep_hr_avg': ('float', 'Average heart rate during sleep (bpm)'),
    'sleep_hr_min': ('float', 'Minimum heart rate during sleep (bpm)'),
    'sleep_stress_pct': ('float', 'Percentage of sleep in high stress'),
    'sleep_perf': ('float', 'Sleep performance score (%)'),
    'sleep_consistency': ('float', 'Sleep consistency score (%)'),

    # Strain / Training
    'strain': ('float', 'Daily strain score'),
    'steps': ('int', 'Daily step count'),
    'calories': ('float', 'Calories burned'),
    'zone1_min': ('float', 'Minutes in HR zone 1'),
    'zone2_min': ('float', 'Minutes in HR zone 2'),
    'zone3_min': ('float', 'Minutes in HR zone 3'),
    'zone4_min': ('float', 'Minutes in HR zone 4'),
    'zone5_min': ('float', 'Minutes in HR zone 5'),
    'workout_duration_min': ('float', 'Total workout duration (minutes)'),
    'n_workouts': ('int', 'Number of workouts'),

    # Cardio fitness
    'vo2max': ('float', 'Estimated VO2max (ml/kg/min)'),

    # Stress
    'stress_high_min': ('float', 'Minutes in high stress'),
    'stress_med_min': ('float', 'Minutes in medium stress'),
    'stress_low_min': ('float', 'Minutes in low stress'),
    'stress_high_pct': ('float', 'Percentage of day in high stress'),

    # Body
    'skin_temp': ('float', 'Skin temperature (°C)'),
    'skin_temp_deviation': ('float', 'Skin temp deviation from baseline (°C)'),
    'weight_kg': ('float', 'Body weight (kg)'),

    # Healthspan
    'healthspan_age': ('float', 'Estimated biological age'),
    'pace_of_aging': ('float', 'Pace of aging (1.0 = normal)'),
}


# === Derived features to compute automatically ===
DERIVED_FEATURES = {
    # Lag features (1-3 days)
    'lag_cols': ['recovery', 'hrv', 'rhr', 'strain', 'sleep_hours',
                 'stress_high_pct', 'sleep_debt_hrs', 'deep_pct'],
    'lag_days': [1, 2, 3],

    # Cumulative features
    'cumulative': {
        'sleep_deficit_3d': ('sleep_hours', 'sleep_needed', 3, 'sum_diff'),
        'strain_3d': ('strain', None, 3, 'mean'),
        'stress_3d': ('stress_high_pct', None, 3, 'mean'),
        'sleep_debt_3d': ('sleep_debt_hrs', None, 3, 'max'),
    },

    # Rolling averages
    'rolling_windows': [7, 14, 30],
    'rolling_cols': ['recovery', 'hrv', 'rhr', 'sleep_hours', 'strain',
                     'steps', 'vo2max', 'stress_high_pct'],

    # Ratios
    'ratios': {
        'hrv_rhr_ratio': ('hrv', 'rhr'),
        'deep_rem_ratio': ('deep_pct', 'rem_pct'),
        'sleep_vs_needed': ('sleep_hours', 'sleep_needed'),
    },

    # Interaction terms
    'interactions': {
        'stress_x_sleep_debt': ('stress_high_pct', 'sleep_debt_hrs'),
        'strain_x_sleep': ('strain', 'sleep_hours'),
        'bedtime_x_strain': ('bed_time_hour', 'strain'),
        'stress_x_sleep_hours': ('stress_high_pct', 'sleep_hours'),
        'deep_x_strain': ('deep_pct', 'strain'),
    },

    # Change features
    'deltas': ['hrv', 'rhr', 'recovery', 'sleep_hours'],
}


# === Population norms (for personalization) ===
POPULATION_NORMS = {
    'vo2max': {
        'source': 'ACSM Guidelines 11th ed, 2022',
        'unit': 'ml/kg/min',
        'female': {
            '20-29': {'p10': 28, 'p25': 32, 'p50': 36, 'p75': 41, 'p90': 45, 'p95': 49},
            '30-39': {'p10': 26, 'p25': 30, 'p50': 34, 'p75': 38, 'p90': 43, 'p95': 47},
            '40-49': {'p10': 24, 'p25': 28, 'p50': 32, 'p75': 36, 'p90': 40, 'p95': 44},
            '50-59': {'p10': 22, 'p25': 26, 'p50': 30, 'p75': 33, 'p90': 37, 'p95': 41},
        },
        'male': {
            '20-29': {'p10': 34, 'p25': 38, 'p50': 43, 'p75': 49, 'p90': 54, 'p95': 57},
            '30-39': {'p10': 32, 'p25': 36, 'p50': 41, 'p75': 46, 'p90': 51, 'p95': 55},
            '40-49': {'p10': 30, 'p25': 34, 'p50': 38, 'p75': 43, 'p90': 48, 'p95': 52},
            '50-59': {'p10': 27, 'p25': 31, 'p50': 35, 'p75': 40, 'p90': 44, 'p95': 48},
        },
    },
    'hrv': {
        'source': 'Shaffer & Ginsberg 2017 Front Public Health; Nunan 2010',
        'unit': 'ms (rMSSD)',
        'note': 'HRV varies enormously; these are rough population guides',
        'female': {
            '20-29': {'p10': 20, 'p25': 35, 'p50': 55, 'p75': 80, 'p90': 110},
            '30-39': {'p10': 18, 'p25': 30, 'p50': 48, 'p75': 70, 'p90': 100},
            '40-49': {'p10': 15, 'p25': 25, 'p50': 40, 'p75': 60, 'p90': 85},
            '50-59': {'p10': 12, 'p25': 20, 'p50': 33, 'p75': 50, 'p90': 72},
        },
        'male': {
            '20-29': {'p10': 22, 'p25': 38, 'p50': 60, 'p75': 90, 'p90': 125},
            '30-39': {'p10': 18, 'p25': 32, 'p50': 52, 'p75': 78, 'p90': 110},
            '40-49': {'p10': 14, 'p25': 26, 'p50': 42, 'p75': 64, 'p90': 92},
            '50-59': {'p10': 11, 'p25': 21, 'p50': 35, 'p75': 54, 'p90': 78},
        },
    },
    'rhr': {
        'source': 'AHA; Quer 2020 Lancet Digital Health (Fitbit N=92K)',
        'unit': 'bpm',
        'general': {
            'excellent': (0, 50),
            'good': (50, 60),
            'above_average': (60, 65),
            'average': (65, 72),
            'below_average': (72, 80),
            'poor': (80, 100),
        },
        'athlete_female': {
            'elite': (38, 48),
            'well_trained': (48, 55),
            'trained': (55, 62),
        },
    },
    'sleep_hours': {
        'source': 'Hirshkowitz 2015 NSF; Watson 2015 AASM',
        'unit': 'hours',
        'adult_26-64': {
            'recommended': (7.0, 9.0),
            'may_be_appropriate': (6.0, 10.0),
            'not_recommended_low': (0, 6.0),
            'not_recommended_high': (10.0, 24.0),
        },
    },
    'sleep_efficiency': {
        'source': 'Buysse 2014 PSQI; Ohayon 2017',
        'unit': '%',
        'adult': {
            'good': (85, 100),
            'borderline': (75, 85),
            'poor': (0, 75),
        },
    },
    'steps': {
        'source': 'Paluch 2022 Lancet Public Health meta-analysis',
        'unit': 'steps/day',
        'adult': {
            'sedentary': (0, 4000),
            'low_active': (4000, 7000),
            'optimal': (7000, 10000),
            'highly_active': (10000, 15000),
            'diminishing_returns': (15000, 50000),
        },
        'mortality_reduction': {
            4000: 0,
            7000: 0.50,
            10000: 0.65,
            12000: 0.70,
        },
    },
    'deep_sleep_pct': {
        'source': 'Ohayon 2004; Dijk 2010',
        'unit': '%',
        'adult_30-39': {
            'normal_range': (13, 23),
            'low': (0, 13),
            'optimal': (17, 23),
        },
    },
}


# === Device-specific adapter configs ===
DEVICE_CONFIGS = {
    'whoop': {
        'data_dir': '',  # Set in user_config.yaml
        'files': {
            'Recovery.json': 'recovery',
            'Sleep.json': 'sleep',
            'Strain.json': 'strain',
            'VO2_Max.json': 'vo2max',
            'Total_Day_Stress.json': 'stress',
            'Health_Monitor.json': 'body',
            'Healthspan.json': 'healthspan',
            'Stress_Monitor.json': 'stress_detail',
            'Strain_Activities.json': 'activities',
            'Recovery_Impact.json': 'recovery_impact',
            'Physiological_Cycles.json': 'cycles',
            'Heart_Rate_Zones.json': 'hr_zones',
        },
    },
    'oura': {
        'data_dir': '',  # Set in user_config.yaml
        'note': 'Oura exports JSON via API or CSV via app',
    },
    'garmin': {
        'data_dir': '',  # Set in user_config.yaml
        'note': 'Garmin Connect CSV export',
    },
    'apple': {
        'data_dir': '',  # Set in user_config.yaml
        'note': 'Apple Health XML export → needs xml parser',
    },
}


# === Analysis constants ===
ANALYSIS_PARAMS = {
    'min_days': 30,              # minimum dataset size for meaningful analysis
    'ideal_days': 180,           # ideal dataset size
    'correlation_threshold': 0.3, # |r| above this = "moderate"
    'p_threshold': 0.05,         # nominal significance
    'fdr_method': 'fdr_bh',     # Benjamini-Hochberg for multiple testing
    'shap_n_background': 100,    # SHAP background samples
    'granger_max_lag': 5,        # max lag for Granger causality
    'bootstrap_n': 1000,         # bootstrap iterations for CIs
    'figure_dpi': 300,
    'figure_style': 'seaborn-v0_8-whitegrid',
}


# === Bayesian N=1 defaults ===
BAYESIAN_PARAMS = {
    'prior_uncertainty_multiplier': 2.0,  # widen literature priors by 2x for N=1
    'min_credible_interval': 0.89,        # 89% CI (Bayesian convention)
}


def get_age_bracket(age: int) -> str:
    """Return age bracket string for population norm lookup."""
    if age < 20:
        return '10-19'
    decade = (age // 10) * 10
    return f'{decade}-{decade + 9}'


def get_percentile(value: float, norms: dict, sex: str, age: int) -> float:
    """Estimate percentile from population norms (linear interpolation)."""
    import numpy as np
    bracket = get_age_bracket(age)
    if sex not in norms or bracket not in norms[sex]:
        return None
    ref = norms[sex][bracket]
    percentiles = sorted(ref.keys())
    values = [ref[p] for p in percentiles]
    pcts = [int(p.replace('p', '')) for p in percentiles]
    if value <= values[0]:
        return pcts[0] * (value / values[0]) if values[0] > 0 else 0
    if value >= values[-1]:
        return min(99, pcts[-1] + (100 - pcts[-1]) * 0.5)
    return float(np.interp(value, values, pcts))
