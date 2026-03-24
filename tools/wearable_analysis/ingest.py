"""
Data ingestion adapters for wearable devices.
Reads raw data → standardized master.csv following CORE_SCHEMA.

Schema-driven: all field mappings come from schema.yaml via config.load_schema().
Audit report: every ingest produces a report of what was received/mapped/lost.
"""

import pandas as pd
import numpy as np
import json
import os
import re
import logging
import warnings
warnings.filterwarnings('ignore')

from .config import (CORE_SCHEMA, DERIVED_FEATURES, load_schema,
                     get_source_fields, get_date_key, get_game_sports)

logger = logging.getLogger(__name__)


# === Helpers (must be defined before ingest functions) ===

def _first(d: dict, *keys):
    """Return first non-None value from dict for given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None and v != '' and v != 'N/A':
            return v
    return None


def _parse_years_impact(val):
    """Parse Healthspan impact: '+1.2 years' → 1.2, '-0.8 years' → -0.8."""
    if pd.isna(val) or val is None or val == '' or val == 'N/A':
        return np.nan
    s = str(val).strip().lower().replace('years', '').replace('year', '').strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _to_float(val):
    """Safe float conversion. Handles thousand separators (9,190 → 9190)."""
    if pd.isna(val) or val == '' or val == 'N/A' or val is None:
        return np.nan
    s = str(val).strip()
    # Thousand separator: "9,190" → "9190" (comma followed by exactly 3 digits)
    if re.match(r'^[\d,]+$', s) and ',' in s:
        s = s.replace(',', '')
    # Decimal comma: "3,14" → "3.14" (single comma not matching thousand pattern)
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _to_int(val):
    """Safe int conversion."""
    f = _to_float(val)
    return int(f) if pd.notna(f) else np.nan


# === Parsers ===

def parse_duration(val):
    """Parse WHOOP duration strings: '2h 30m', '1:30:00', '45m', '7.5', etc."""
    if pd.isna(val) or val == '' or val == 'N/A':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # "2h 30m" or "2h30m"
    m = re.match(r'(\d+)\s*h\s*(\d+)\s*m', s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    # "2h" only
    m = re.match(r'(\d+)\s*h$', s)
    if m:
        return int(m.group(1))
    # "45m" only
    m = re.match(r'(\d+)\s*m$', s)
    if m:
        return int(m.group(1)) / 60
    # "1:30:00" or "1:30"
    m = re.match(r'(\d+):(\d+)(?::(\d+))?$', s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        sec = int(m.group(3)) if m.group(3) else 0
        return h + mi / 60 + sec / 3600
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def parse_percentage(val):
    """Parse percentage: '93%' → 93.0, '0.93' → 93.0, 93 → 93.0."""
    if pd.isna(val) or val == '' or val == 'N/A':
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if float(val) > 1 else float(val) * 100
    s = str(val).strip().rstrip('%')
    try:
        v = float(s)
        return v if v > 1 else v * 100
    except (ValueError, TypeError):
        return np.nan


def parse_time(val):
    """Parse time string to decimal hours from midnight.
    '01:30 AM' → 1.5, '11:30 PM' → 23.5, '2025-01-15T01:30' → 1.5
    """
    if pd.isna(val) or val == '' or val == 'N/A':
        return np.nan
    s = str(val).strip()

    # ISO format
    m = re.search(r'T(\d{2}):(\d{2})', s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60

    # "HH:MM AM/PM"
    m = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        ampm = m.group(3)
        if ampm and ampm.upper() == 'PM' and h != 12:
            h += 12
        elif ampm and ampm.upper() == 'AM' and h == 12:
            h = 0
        return h + mi / 60

    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


# === JSON loading ===

def load_whoop_json(filepath: str) -> dict:
    """Load WHOOP MCP JSON file, handling _metadata/data nesting."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        raw = json.load(f)

    # Handle _metadata/data wrapper
    if '_metadata' in raw and 'data' in raw:
        data = raw['data']
    elif isinstance(raw, dict):
        data = raw
    else:
        logger.warning(f"Unexpected format in {filepath}: type={type(raw)}")
        return {}

    return data


# === Schema-driven extraction ===

def _extract_row_from_schema(record: dict, source_file: str) -> dict:
    """Extract fields from a WHOOP record using schema.yaml definitions.

    This is the schema-driven extractor — reads field definitions from
    schema.yaml and applies the correct parser for each field type.

    Args:
        record: dict from WHOOP JSON (one day/record)
        source_file: name of the JSON file (e.g. 'Recovery.json')

    Returns:
        dict of {internal_name: parsed_value}
    """
    fields = get_source_fields(source_file)
    row = {}
    for field in fields:
        internal = field['internal']
        ftype = field.get('type', 'float')
        raw_val = _first(record, *field.get('whoop_keys', []))

        if raw_val is None or raw_val == '' or raw_val == 'N/A':
            row[internal] = np.nan if ftype != 'str' else ''
            continue

        # Parse by type
        if ftype == 'pct':
            row[internal] = parse_percentage(raw_val)
        elif ftype == 'duration':
            row[internal] = parse_duration(raw_val)
        elif ftype == 'float':
            row[internal] = _to_float(raw_val)
        elif ftype == 'int':
            row[internal] = _to_int(raw_val)
        elif ftype == 'str':
            row[internal] = str(raw_val)
        elif ftype == 'datetime':
            row[internal] = str(raw_val)
        elif ftype == 'years_impact':
            row[internal] = _parse_years_impact(raw_val)
        else:
            row[internal] = raw_val

    return row


# === Audit tracking ===

class IngestAudit:
    """Tracks what data was received, mapped, and lost during ingestion."""

    def __init__(self):
        self.sources = {}  # source_file → {received, mapped, lost, fill_rates}
        self.warnings = []
        self.errors = []

    def register_source(self, source_file: str, n_records: int, n_fields_available: int):
        self.sources[source_file] = {
            'records': n_records,
            'fields_available': n_fields_available,
            'fields_mapped': 0,
            'fields_lost': [],
            'fill_rates': {},
        }

    def register_field(self, source_file: str, field_name: str, fill_rate: float):
        if source_file in self.sources:
            self.sources[source_file]['fields_mapped'] += 1
            self.sources[source_file]['fill_rates'][field_name] = fill_rate

    def register_lost_field(self, source_file: str, field_name: str, reason: str):
        if source_file in self.sources:
            self.sources[source_file]['fields_lost'].append((field_name, reason))

    def register_missing_source(self, source_file: str):
        self.warnings.append(f"Source file not found: {source_file}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def error(self, msg: str):
        self.errors.append(msg)
        logger.error(msg)

    def generate_report(self, master_df: pd.DataFrame) -> str:
        """Generate human-readable audit report."""
        lines = ['=' * 60, 'INGEST AUDIT REPORT', '=' * 60, '']

        # Overall stats
        lines.append(f'Master dataset: {len(master_df)} days x {len(master_df.columns)} columns')
        non_null_cols = sum(1 for c in master_df.columns if master_df[c].notna().any())
        lines.append(f'Columns with data: {non_null_cols}/{len(master_df.columns)}')
        lines.append('')

        # Per-source
        total_mapped = 0
        total_lost = 0
        for source, info in self.sources.items():
            lines.append(f'--- {source} ({info["records"]} records) ---')
            lines.append(f'  Fields mapped: {info["fields_mapped"]}')
            total_mapped += info['fields_mapped']
            if info['fields_lost']:
                total_lost += len(info['fields_lost'])
                for fname, reason in info['fields_lost']:
                    lines.append(f'  LOST: {fname} ({reason})')
            # Low fill rates
            low_fill = [(f, r) for f, r in info['fill_rates'].items() if r < 0.5]
            if low_fill:
                for fname, rate in sorted(low_fill, key=lambda x: x[1]):
                    lines.append(f'  LOW FILL: {fname} = {rate:.1%}')
            lines.append('')

        # Schema coverage
        schema = load_schema()
        source_files = set(schema.get('sources', {}).keys())
        ingested_files = set(self.sources.keys())
        not_ingested = source_files - ingested_files
        if not_ingested:
            lines.append('SOURCES NOT INGESTED:')
            for f in sorted(not_ingested):
                lines.append(f'  - {f}')
            lines.append('')

        # Warnings
        if self.warnings:
            lines.append('WARNINGS:')
            for w in self.warnings:
                lines.append(f'  - {w}')
            lines.append('')

        # Errors
        if self.errors:
            lines.append('ERRORS:')
            for e in self.errors:
                lines.append(f'  !! {e}')
            lines.append('')

        # Required columns check
        required_missing = []
        for source_config in schema.get('sources', {}).values():
            for field in source_config.get('fields', []):
                if field.get('required') and field['internal'] not in master_df.columns:
                    required_missing.append(field['internal'])
                elif field.get('required') and field['internal'] in master_df.columns:
                    fill = master_df[field['internal']].notna().mean()
                    if fill < 0.5:
                        required_missing.append(f"{field['internal']} ({fill:.1%} fill)")

        if required_missing:
            lines.append('REQUIRED COLUMNS MISSING OR LOW:')
            for c in required_missing:
                lines.append(f'  !! {c}')
            lines.append('')

        # Summary
        lines.append(f'Total: {total_mapped} fields mapped, {total_lost} fields lost, '
                     f'{len(self.warnings)} warnings, {len(self.errors)} errors')
        lines.append('=' * 60)

        report = '\n'.join(lines)
        return report


# === Generic schema-driven source ingestion ===

def _ingest_daily_source(data_dir: str, source_file: str, audit: 'IngestAudit') -> pd.DataFrame:
    """Generic schema-driven ingestion for daily WHOOP JSON files.

    Reads schema.yaml for field definitions, extracts all fields, tracks audit.

    Args:
        data_dir: path to WHOOP MCP sync full/ directory
        source_file: JSON filename (e.g. 'Recovery.json')
        audit: IngestAudit instance for tracking

    Returns:
        DataFrame with 'date' + all schema-defined fields
    """
    filepath = os.path.join(data_dir, source_file)
    data = load_whoop_json(filepath)
    if not data:
        audit.register_missing_source(source_file)
        return pd.DataFrame()

    date_key = get_date_key(source_file)
    rows = []
    for date_str, record in data.items():
        if date_str.startswith('_'):
            continue
        if isinstance(record, dict):
            row = {'date': date_str}
            extracted = _extract_row_from_schema(record, source_file)
            row.update(extracted)
            rows.append(row)
        elif isinstance(record, list):
            # Multiple records per day (e.g. Naps, Recovery_Impact)
            for item in record:
                if isinstance(item, dict):
                    row = {'date': date_str}
                    extracted = _extract_row_from_schema(item, source_file)
                    row.update(extracted)
                    rows.append(row)
        else:
            audit.warn(f"{source_file}: record for {date_str} is {type(record).__name__}, not dict/list")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Track audit
        fields = get_source_fields(source_file)
        n_available = len(fields)
        audit.register_source(source_file, len(df), n_available)
        for field in fields:
            internal = field['internal']
            if internal in df.columns:
                fill = df[internal].notna().mean()
                audit.register_field(source_file, internal, fill)
                if fill == 0:
                    audit.warn(f"{source_file}: {internal} is 100% NaN (field exists but no data parsed)")
            else:
                audit.register_lost_field(source_file, internal, "column not created")

    return df


# === Main WHOOP ingestion ===

def ingest_whoop(data_dir: str, output_dir: str) -> tuple:
    """Ingest all WHOOP MCP JSON files → standardized master.csv.

    Schema-driven: all field mappings come from schema.yaml.
    Generates audit report showing what was received/mapped/lost.

    Args:
        data_dir: path to WHOOP MCP sync full/ directory
        output_dir: path to output data/ directory

    Returns:
        (master_df, activities_df, healthspan_df) — three DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)
    audit = IngestAudit()

    # === Recovery (schema-driven) ===
    df_recovery = _ingest_daily_source(data_dir, 'Recovery.json', audit)

    # === Sleep (schema-driven) ===
    df_sleep = _ingest_daily_source(data_dir, 'Sleep.json', audit)

    # === Strain (schema-driven) ===
    df_strain = _ingest_daily_source(data_dir, 'Strain.json', audit)

    # === VO2max (schema-driven) ===
    df_vo2 = _ingest_daily_source(data_dir, 'VO2_Max.json', audit)

    # === Stress (schema-driven) ===
    df_stress = _ingest_daily_source(data_dir, 'Total_Day_Stress.json', audit)
    # Compute stress_high_pct (derived from stress minutes)
    if len(df_stress) > 0 and 'stress_high_min' in df_stress.columns:
        total = (df_stress['stress_high_min'].fillna(0) +
                 df_stress['stress_med_min'].fillna(0) +
                 df_stress['stress_low_min'].fillna(0))
        df_stress['stress_high_pct'] = np.where(
            total > 0,
            df_stress['stress_high_min'].fillna(0) / total * 100,
            np.nan
        )

    # === Healthspan (schema-driven, weekly → forward-fill to daily) ===
    df_healthspan = _ingest_daily_source(data_dir, 'Healthspan.json', audit)

    # === Health Monitor (schema-driven, limited history) ===
    df_health_monitor = _ingest_daily_source(data_dir, 'Health_Monitor.json', audit)

    # === Naps (schema-driven, per-nap → aggregate to daily) ===
    df_naps_raw = _ingest_daily_source(data_dir, 'Naps.json', audit)
    df_naps = pd.DataFrame()
    if len(df_naps_raw) > 0:
        nap_daily = df_naps_raw.groupby('date').agg({
            'nap_hours': 'sum',
            'nap_efficiency': 'mean',
        }).reset_index()
        nap_daily['has_nap'] = 1
        nap_daily.rename(columns={'nap_hours': 'nap_total_hrs'}, inplace=True)
        df_naps = nap_daily

    # === Weight (schema-driven) ===
    df_weight = _ingest_daily_source(data_dir, 'Weight.json', audit)

    # === Recovery Impact (schema-driven, per-factor → pivot to daily) ===
    df_impact_raw = _ingest_daily_source(data_dir, 'Recovery_Impact.json', audit)
    df_impact = pd.DataFrame()
    if len(df_impact_raw) > 0 and 'impact_activity' in df_impact_raw.columns:
        # Pivot: each activity/behavior → column
        pivot = df_impact_raw.pivot_table(
            index='date', columns='impact_activity', values='impact_pct',
            aggfunc='first'
        ).reset_index()
        pivot.columns = ['date'] + [f'impact_{c.lower().replace(" ", "_").replace("/", "_")}'
                                     for c in pivot.columns[1:]]
        df_impact = pivot

    # === Strain Activities (schema-driven, per-activity) ===
    act_path = os.path.join(data_dir, 'Strain_Activities.json')
    df_activities = pd.DataFrame()
    if os.path.exists(act_path):
        act_data = load_whoop_json(act_path)
        rows = []
        for date_str, acts in act_data.items():
            if date_str.startswith('_'):
                continue
            if isinstance(acts, list):
                for a in acts:
                    row = {'date': date_str}
                    extracted = _extract_row_from_schema(a, 'Strain_Activities.json')
                    row.update(extracted)
                    # Zone fields need special handling (Zone 0-5 → zone0_min etc.)
                    for z in range(6):
                        zone_val = _first(a, f'Zone {z}', f'zone{z}_min')
                        row[f'zone{z}_min'] = parse_duration(zone_val) if zone_val else 0
                    row['duration_min'] = parse_duration(_first(a, 'Duration', 'duration'))
                    rows.append(row)
        df_activities = pd.DataFrame(rows)
        if len(df_activities) > 0:
            audit.register_source('Strain_Activities.json', len(df_activities),
                                  len(get_source_fields('Strain_Activities.json')))

        # Aggregate zones per day
        if len(df_activities) > 0:
            zone_daily = df_activities.groupby('date').agg({
                f'zone{z}_min': 'sum' for z in range(6)
            }).reset_index()
            if 'zone0_min' in zone_daily.columns:
                zone_daily['zone1_min'] = zone_daily.get('zone1_min', 0) + zone_daily.get('zone0_min', 0)
                zone_daily.drop(columns=['zone0_min'], inplace=True, errors='ignore')

            # Activity type aggregation
            GAME_SPORTS = get_game_sports()

            def _agg_activities(group):
                names = group.get('activity_name', pd.Series(dtype=str)).tolist()
                names = [n for n in names if isinstance(n, str) and n]
                is_game = any(n in GAME_SPORTS for n in names)
                dominant = max(set(names), key=names.count) if names else ''
                dur_col = 'duration_min' if 'duration_min' in group.columns else 'activity_duration_min'
                total_dur = group[dur_col].sum() if dur_col in group.columns else 0
                return pd.Series({
                    'n_activities': len(group),
                    'is_game_sport_day': int(is_game),
                    'dominant_activity': dominant,
                    'activity_types': ', '.join(sorted(set(names))),
                    'total_activity_min': total_dur,
                })

            act_agg = df_activities.groupby('date').apply(_agg_activities).reset_index()
            zone_daily = zone_daily.merge(act_agg, on='date', how='outer')
            df_strain = df_strain.merge(zone_daily, on='date', how='left')

    # === Health Monitor merge (fills gaps in Recovery) — schema-driven ===
    # Build merge_into map from schema: {source_col: target_col}
    schema = load_schema()
    merge_into_map = {}
    for source_name, source_config in schema.get('sources', {}).items():
        for field in source_config.get('fields', []):
            if 'merge_into' in field:
                merge_into_map[field['internal']] = field['merge_into']

    if len(df_health_monitor) > 0 and len(df_recovery) > 0:
        # Use outer merge so Health Monitor dates not in Recovery are preserved
        df_recovery = df_recovery.merge(df_health_monitor, on='date', how='outer',
                                        suffixes=('', '_merge_dup'))
        # Fill gaps using schema-driven merge_into map
        for source_col, target_col in merge_into_map.items():
            if source_col in df_recovery.columns:
                if target_col not in df_recovery.columns:
                    df_recovery[target_col] = np.nan
                df_recovery[target_col] = df_recovery[target_col].fillna(df_recovery[source_col])
        # Drop source columns that were merged
        drop_cols = [c for c in df_recovery.columns
                     if c in merge_into_map or c.endswith('_merge_dup')]
        df_recovery.drop(columns=drop_cols, inplace=True, errors='ignore')

    # === Parse date columns ===
    for df in [df_recovery, df_sleep, df_strain, df_vo2, df_stress,
               df_healthspan, df_naps, df_weight, df_impact]:
        if len(df) > 0 and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # === Sleep percentages (derive if raw available) ===
    if len(df_sleep) > 0:
        for pct_col, raw_col in [('deep_pct', 'deep_pct_raw'), ('rem_pct', 'rem_pct_raw'),
                                  ('light_pct', 'light_pct_raw')]:
            hrs_col = pct_col.replace('_pct', '_hrs')
            if raw_col in df_sleep.columns and df_sleep[raw_col].notna().any():
                df_sleep[pct_col] = df_sleep[raw_col]
            elif hrs_col in df_sleep.columns and 'sleep_hours' in df_sleep.columns:
                df_sleep[pct_col] = (df_sleep[hrs_col] / df_sleep['sleep_hours'] * 100).clip(0, 100)
            else:
                audit.warn(f"Cannot derive {pct_col}: missing {raw_col} and {hrs_col}")

    # === Bed/wake time parsing ===
    if len(df_sleep) > 0:
        for raw_col, hour_col in [('bed_time_raw', 'bed_time_hour'), ('wake_time_raw', 'wake_time_hour')]:
            if raw_col in df_sleep.columns:
                df_sleep[hour_col] = df_sleep[raw_col].apply(parse_time)

    # === Merge all into master (outer join on date) ===
    master = df_recovery.copy() if len(df_recovery) > 0 else pd.DataFrame(columns=['date'])
    for other_name, other_df in [
        ('Sleep', df_sleep), ('Strain', df_strain), ('VO2max', df_vo2),
        ('Stress', df_stress), ('Healthspan', df_healthspan),
        ('Naps', df_naps), ('Weight', df_weight), ('Recovery_Impact', df_impact),
    ]:
        if len(other_df) > 0:
            before_cols = set(master.columns)
            master = master.merge(other_df, on='date', how='outer', suffixes=('', '_dup'))
            dup_cols = [c for c in master.columns if c.endswith('_dup')]
            if dup_cols:
                audit.warn(f"Merge with {other_name}: {len(dup_cols)} duplicate columns dropped: {dup_cols}")
            master.drop(columns=dup_cols, inplace=True)

    # === Handle empty master gracefully ===
    if master.empty or 'date' not in master.columns:
        logger.warning("No data ingested — master DataFrame is empty.")
        audit_report = audit.generate_report(master)
        audit_path = os.path.join(output_dir, 'ingest_audit.txt')
        with open(audit_path, 'w') as f:
            f.write(audit_report)
        print(audit_report)
        print(f"\n⚠️ Master dataset: 0 days × {len(master.columns)} columns (no data found)")
        master_path = os.path.join(output_dir, 'master.csv')
        master.to_csv(master_path, index=False)
        return master, df_activities, df_healthspan

    # === Sort and fill nap gaps ===
    master.sort_values('date', inplace=True)
    master.reset_index(drop=True, inplace=True)
    if 'has_nap' in master.columns:
        master['has_nap'] = master['has_nap'].fillna(0).astype(int)

    # === Recovery zone ===
    if 'recovery' in master.columns:
        master['recovery_zone'] = pd.cut(
            master['recovery'], bins=[0, 34, 67, 100],
            labels=['red', 'yellow', 'green'], include_lowest=True
        )

    # === Time features ===
    if 'date' in master.columns and pd.api.types.is_datetime64_any_dtype(master['date']):
        try:
            master['dow'] = master['date'].dt.dayofweek
            master['month'] = master['date'].dt.month
            master['week'] = master['date'].dt.isocalendar().week.astype(int)
            master['year'] = master['date'].dt.year
        except Exception as e:
            logger.warning(f"Could not compute time features: {e}")
    elif 'date' in master.columns:
        logger.warning("Date column is not datetime — skipping time features")

    # === Generate audit report ===
    audit_report = audit.generate_report(master)
    audit_path = os.path.join(output_dir, 'ingest_audit.txt')
    with open(audit_path, 'w') as f:
        f.write(audit_report)
    print(audit_report)

    print(f"\n✅ Master dataset: {len(master)} days × {len(master.columns)} columns")
    print(f"   Audit report: {audit_path}")

    # Save master
    master_path = os.path.join(output_dir, 'master.csv')
    master.to_csv(master_path, index=False)

    return master, df_activities, df_healthspan


# === Derived features ===

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features, rolling averages, ratios, interactions.

    Gracefully handles empty DataFrames and missing columns — skips
    any derived feature whose source columns are absent.
    """
    if df.empty:
        logger.warning("add_derived_features: DataFrame is empty, returning as-is.")
        return df

    df = df.copy()

    # Sort by date if available
    if 'date' in df.columns:
        try:
            df = df.sort_values('date').reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Lag features
    for col in DERIVED_FEATURES.get('lag_cols', []):
        if col not in df.columns:
            continue
        for lag in DERIVED_FEATURES.get('lag_days', []):
            try:
                df[f'prev_{lag}d_{col}'] = df[col].shift(lag)
            except Exception as e:
                logger.warning(f"Lag feature prev_{lag}d_{col} failed: {e}")

    # Cumulative features
    for name, (col, ref_col, window, method) in DERIVED_FEATURES.get('cumulative', {}).items():
        if col not in df.columns:
            continue
        try:
            if method == 'sum_diff' and ref_col and ref_col in df.columns:
                diff = df[col] - df[ref_col]
                df[name] = diff.rolling(window, min_periods=1).sum()
            elif method == 'mean':
                df[name] = df[col].rolling(window, min_periods=1).mean()
            elif method == 'max':
                df[name] = df[col].rolling(window, min_periods=1).max()
        except Exception as e:
            logger.warning(f"Cumulative feature {name} failed: {e}")

    # Rolling averages
    for col in DERIVED_FEATURES.get('rolling_cols', []):
        if col not in df.columns:
            continue
        for w in DERIVED_FEATURES.get('rolling_windows', []):
            try:
                df[f'{col}_{w}d'] = df[col].rolling(w, min_periods=max(1, w // 3)).mean()
            except Exception as e:
                logger.warning(f"Rolling feature {col}_{w}d failed: {e}")

    # Ratios
    for name, (num, den) in DERIVED_FEATURES.get('ratios', {}).items():
        if num in df.columns and den in df.columns:
            try:
                df[name] = df[num] / df[den].replace(0, np.nan)
            except Exception as e:
                logger.warning(f"Ratio {name} failed: {e}")

    # Interaction terms
    for name, (col1, col2) in DERIVED_FEATURES.get('interactions', {}).items():
        if col1 not in df.columns or col2 not in df.columns:
            continue
        try:
            std1 = df[col1].std()
            std2 = df[col2].std()
            if std1 and std1 > 0 and std2 and std2 > 0:
                z1 = (df[col1] - df[col1].mean()) / std1
                z2 = (df[col2] - df[col2].mean()) / std2
                df[name] = z1 * z2
            else:
                logger.warning(f"Interaction {name}: zero variance in one or both columns")
        except Exception as e:
            logger.warning(f"Interaction {name} failed: {e}")

    # Delta features
    for col in DERIVED_FEATURES.get('deltas', []):
        if col not in df.columns:
            continue
        try:
            df[f'{col}_delta_1d'] = df[col].diff()
        except Exception as e:
            logger.warning(f"Delta feature {col}_delta_1d failed: {e}")

    # Z2 weekly
    if 'zone2_min' in df.columns:
        try:
            df['z2_weekly'] = df['zone2_min'].rolling(7, min_periods=1).sum()
        except Exception as e:
            logger.warning(f"z2_weekly failed: {e}")

    # Bed time adjusted (for correlation — shift so midnight=0, 1am=1, 11pm=-1)
    if 'bed_time_hour' in df.columns:
        try:
            df['bed_time_adj'] = df['bed_time_hour'].apply(
                lambda x: x - 24 if x > 12 else x if pd.notna(x) else np.nan
            )
        except Exception as e:
            logger.warning(f"bed_time_adj failed: {e}")

    # Zone percentages (from zone minutes)
    zone_cols = [c for c in df.columns if c.startswith('zone') and c.endswith('_min')]
    if zone_cols:
        try:
            zone_total = df[zone_cols].sum(axis=1).replace(0, np.nan)
            for zc in zone_cols:
                pct_name = zc.replace('_min', '_pct')
                df[pct_name] = (df[zc] / zone_total * 100)
        except Exception as e:
            logger.warning(f"Zone percentages failed: {e}")

    # Stress rolling features
    if 'stress_high_pct' in df.columns:
        try:
            df['stress_high_pct_3d'] = df['stress_high_pct'].rolling(3, min_periods=1).mean()
            df['stress_high_pct_7d'] = df['stress_high_pct'].rolling(7, min_periods=3).mean()
        except Exception as e:
            logger.warning(f"Stress rolling features failed: {e}")

    # Sleep needed
    if 'sleep_needed_hrs' not in df.columns and 'sleep_hours' in df.columns:
        df['sleep_needed'] = 7.5  # default if not available

    print(f"✅ Enriched dataset: {len(df)} days × {len(df.columns)} columns")
    try:
        df.to_csv(df.attrs.get('output_path', 'master_enriched.csv'), index=False)
    except Exception as e:
        logger.warning(f"Could not save enriched CSV: {e}")

    return df
