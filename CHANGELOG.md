# Changelog

All notable changes to the Wearable Analysis skill.
Each release lists what users see — not internal refactors.

## [v2.1] — 2026-07-17 — privacy & portability cleanup

**In plain terms:** a hygiene pass so this is genuinely safe and clean for anyone
to fork and run. No changes to the analysis itself — same hypotheses, same stats.

### Fixed
- **Telegram notifications now read plain environment variables** —
  `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (`TELEGRAM_TOKEN` still accepted).
  Removed the old fallback that scraped a specific bot `.env` file and an
  `ALLOWED_USERS` list — those assumed the author's own setup.
- **Removed the legacy `generate_portrait.py`** entry point. Use
  `python -m wearable_analysis` (documented in the README).
- **Genericized internal references** in the skill spec that pointed at the
  author's private research folder — they now point at this repo's own
  `wearable_analysis/ingest.py` and `schema.yaml`.

### Added
- **`LICENSE`** — MIT, plus an explicit "not medical advice" notice.
- **This `CHANGELOG.md`.**
- **Hardened `.gitignore`** — also ignores `.env*`, token/secret patterns, and
  `.DS_Store`, on top of the existing `user_config.yaml` / data / output rules.

### Privacy reminder
Your `user_config.yaml`, your data, and all analysis output are git-ignored — they
never leave your machine. Fork freely.

## [v2.0] — 2026-04 — first public release

Literature-first N=1 wearable pipeline: ~50 pre-registered hypotheses across
recovery, sleep, training, stress, interactions, and menstrual cycle; Bayesian N=1
testing; all-pairs discovery scan; population comparison; HTML health-portrait
report. WHOOP / Oura / Garmin / generic-CSV support.
