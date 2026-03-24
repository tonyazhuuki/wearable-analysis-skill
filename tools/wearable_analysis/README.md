# Wearable Data Analysis — Literature-First N=1 Pipeline

A scientific, hypothesis-driven framework for analyzing wearable health data (WHOOP, Oura, Garmin).

## What makes this different

Most wearable analysis tools find correlations and call them "insights." This pipeline does it backwards:

1. **Literature first**: 50 pre-registered hypotheses grounded in 16 scientific consensus references
2. **Then data**: Your wearable data tests these hypotheses using Bayesian N=1 methods
3. **Discovery**: All-pairs correlation scan finds what science doesn't predict
4. **Gap analysis**: Maps discoveries to hypothesis coverage — what's confirmed, what's new

## Quick Start

```bash
# 1. Copy and customize your config
cp user_config.template.yaml user_config.yaml
# Edit user_config.yaml: set profile (sex, age), data directory, enabled domains

# 2. Run the full portrait pipeline
python -m wearable_analysis portrait --data-dir path/to/whoop/data

# 3. Open health_portrait_output/health_portrait.html
```

## CLI Commands

```bash
# Full pipeline: ingest -> discovery -> population comparison -> HTML report
python -m wearable_analysis portrait --data-dir path/to/data --sex female --age 30

# Ingest only: produce master.csv from raw device data
python -m wearable_analysis ingest --data-dir path/to/data --device whoop

# Discovery only: run correlation scan on existing master.csv
python -m wearable_analysis discover --data-dir path/to/data

# With Telegram notification
python -m wearable_analysis portrait --data-dir path/to/data --notify

# Help
python -m wearable_analysis --help
```

## Output

- **`health_portrait.html`** — comprehensive report with grades, deep dives, scientific context
- **`data/master.csv`** — standardized dataset with derived features (lags, rolling averages, ratios)
- **`discovery/`** — correlation matrices, significant pairs mapped to hypotheses
- **`figures/`** — publication-quality plots (300 DPI)

## Architecture

```
wearable_analysis/
├── __main__.py          # CLI entry point (argparse subcommands)
├── __init__.py          # Package metadata
├── adapters.py          # Device adapters (WHOOP, Oura stub, Garmin stub, CSV)
├── config.py            # Schema registry, population norms, constants
├── ingest.py            # WHOOP MCP JSON → standardized DataFrame
├── eda.py               # Automated exploratory data analysis (STL, SHAP, changepoints)
├── discovery.py         # All-pairs correlation + hypothesis coverage mapping
├── hypothesis_test.py   # Bayesian N=1 hypothesis testing framework
├── causal.py            # Granger causality, mediation analysis
├── personalize.py       # Population comparison, anomaly detection
├── report.py            # Domain grading + HTML report generation
├── visualize.py         # Standardized figure generation (8 plot types)
├── generate_portrait.py # Legacy entry point (use __main__.py instead)
├── schema.yaml          # Single source of truth for all WHOOP field mappings
├── user_config.yaml     # Your personal configuration (git-ignored)
├── user_config.template.yaml  # Template with all options documented
└── hypotheses/          # Pre-registered hypothesis YAML files
    ├── recovery.yaml
    ├── sleep.yaml
    ├── training.yaml
    ├── stress.yaml
    ├── interactions.yaml
    └── cycle_hypotheses.yaml
```

## Hypothesis Database

50 hypotheses across 6 domains:

| Domain | Count | Examples |
|--------|-------|---------|
| Recovery | 8 | HRV-recovery coupling, RHR as bottleneck, sleep → recovery |
| Sleep | 10 | Duration, timing, architecture, debt accumulation |
| Training | 8 | Polarization, VO2max trends, steps, iron-HR shift |
| Stress | 6 | Autonomic coupling, cumulative load, stress × sleep |
| Interactions | 8 | Cross-domain compound effects (strain × sleep, bedtime × recovery) |
| Menstrual Cycle | 10 | Phase effects on RHR, HRV, skin temp, recovery |

Each hypothesis includes:
- **Literature source** and population effect size (Cohen's d or r)
- **Pre-registered statistical test** (Spearman, Mann-Whitney, regression, etc.)
- **Consensus reference IDs** (optional, for users with our research library)
- **Actionable interpretation** for both confirmed and refuted outcomes

## Requirements

Python 3.10+ with the following packages:

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
pyyaml
statsmodels
tabulate
```

Install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyyaml statsmodels tabulate
```

## Configuration

See `user_config.template.yaml` for all options. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `profile.sex` | Sex for population norms | `female` |
| `profile.age` | Age for percentile calculations | `30` |
| `device.type` | Device type | `whoop` |
| `device.data_dir` | Path to raw data | (required) |
| `domains.cycle` | Enable menstrual cycle hypotheses | `false` |
| `health_context` | Optional: conditions, meds, genetics | `{}` |

## Supported Devices

| Device | Status | Notes |
|--------|--------|-------|
| **WHOOP** | Full support | MCP JSON sync via `whoop_sync_mcp.py` |
| **Oura** | Schema ready, adapter planned | Use GenericCSVAdapter with CSV export |
| **Garmin** | Schema ready, adapter planned | Use GenericCSVAdapter with CSV export |
| **Generic CSV** | Supported | Any wearable data with date column |

## Telegram Notifications

The `--notify` flag on the `portrait` command sends:
1. A summary message with domain grades
2. The HTML report as a document attachment

Credentials are read from `~/Cursor/your-telegram-bot/.env` (same as other Second Brain tools).

## License

MIT
