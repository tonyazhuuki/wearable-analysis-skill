# Wearable Analysis — Literature-First N=1 Pipeline

A scientific, hypothesis-driven framework for analyzing wearable health data. Built as a [Claude Code](https://claude.ai/claude-code) skill.

## What makes this different

Most wearable tools find correlations and call them "insights." This pipeline does it backwards:

1. **Literature first**: 50 pre-registered hypotheses grounded in 16 scientific consensus references
2. **Then data**: Your wearable data tests these hypotheses using Bayesian N=1 methods
3. **Discovery**: All-pairs correlation scan finds what science doesn't predict
4. **Gap analysis**: Maps discoveries to hypothesis coverage — what's confirmed, what's new
5. **Deep dive report**: Human-readable HTML portrait with scientific context in Russian or English

## Quick Start

```bash
# 1. Clone into your Claude Code project
git clone https://github.com/tonyazhuuki/wearable-analysis-skill.git

# 2. Copy the skill file
cp wearable-analysis-skill/.claude/commands/wearable-analysis.md YOUR_PROJECT/.claude/commands/

# 3. Copy the Python pipeline
cp -r wearable-analysis-skill/tools/wearable_analysis YOUR_PROJECT/tools/

# 4. Configure
cd YOUR_PROJECT/tools/wearable_analysis
cp user_config.template.yaml user_config.yaml
# Edit user_config.yaml: set sex, age, device type, data directory

# 5. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn pyyaml statsmodels

# 6. Run
python -m wearable_analysis portrait --data-dir /path/to/whoop/json
```

Or use as a Claude Code skill:
```
/wearable-analysis whoop
```

## Output

- **Health Portrait HTML** — 9-section deep dive with grades, scientific references, recommendations
- **Correlation Discovery** — all significant pairs mapped to hypothesis coverage
- **New Hypotheses** — auto-generated from uncovered correlations

## Hypothesis Database

50 hypotheses across 6 domains, each with literature sources and consensus references:

| Domain | Hypotheses | Coverage |
|--------|-----------|----------|
| Recovery | 8 | HRV, RHR, recovery predictors |
| Sleep | 10 | Duration, timing, architecture, debt |
| Training | 8 | Polarization, VO2max, dose-response |
| Stress | 6 | Autonomic coupling, cumulative load |
| Interactions | 8 | Cross-domain compound effects |
| Menstrual Cycle | 10 | Phase effects on metrics (female) |

## Supported Devices

| Device | Status |
|--------|--------|
| **WHOOP** | Full support (MCP JSON) |
| **Oura** | Schema ready, adapter stub |
| **Garmin** | Schema ready, adapter stub |
| **CSV** | Generic import |

## Performance

- Full pipeline (664 days): **23.5 seconds**
- Hypothesis testing (50 hypotheses): **3.8 seconds**
- Handles partial/missing data gracefully

## Architecture

```
tools/wearable_analysis/
├── config.py              # Schema registry, population norms, user config
├── schema.yaml            # WHOOP data schema (150 metrics)
├── ingest.py              # Raw JSON/CSV → master DataFrame
├── discovery.py           # All-pairs correlation + hypothesis coverage
├── hypothesis_test.py     # 7 test types, Bayesian N=1
├── personalize.py         # Population norms, actionability scoring
├── report.py              # Deep dive HTML report generator
├── causal.py              # Granger causality, mediation
├── eda.py                 # SHAP, STL decomposition, changepoints
├── visualize.py           # 8 plot types
├── adapters.py            # Device adapter pattern
├── generate_portrait.py   # Pipeline orchestrator
├── __main__.py            # CLI entry point
└── hypotheses/            # 50 pre-registered hypotheses (6 YAML files)
```

## Related

- [Deep Research Skill](https://github.com/tonyazhuuki/deep-research-skill) — the research pipeline that generated the consensus references

## License

MIT
