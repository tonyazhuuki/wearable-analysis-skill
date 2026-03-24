# Wearable Data Deep Analysis — Literature-First N=1 Pipeline v2.0

Проведи глубокий анализ данных носимых устройств: **$ARGUMENTS**

> **Headless launch:** `claude --dangerously-skip-permissions -p "/wearable-analysis whoop sleep+recovery+training 8h"`
> Для фоновой работы запускать в tmux/screen.

## Quick Start

1. Copy `tools/wearable_analysis/user_config.template.yaml` → `tools/wearable_analysis/user_config.yaml`
2. Edit `user_config.yaml`: set your `profile.sex`, `profile.age`, `device.type`, `device.data_dir`
3. Run: `/wearable-analysis whoop` (or your device)

> First run will guide you through any missing configuration.

## Configuration

All user-specific settings live in `tools/wearable_analysis/user_config.yaml`:

| Setting | Description | Required |
|---------|------------|----------|
| `profile.sex` | `"female"` or `"male"` — for population norms | Yes |
| `profile.age` | Current age — for percentile calculations | Yes |
| `device.type` | `"whoop"`, `"oura"`, `"garmin"`, `"apple"`, `"csv"` | Yes |
| `device.data_dir` | Path to your device data directory | Yes |
| `game_sports` | List of your game/intermittent sports | Optional |
| `domains.cycle` | Enable menstrual cycle analysis (requires `sex: female`) | Optional |
| `health_context` | Conditions, medications, supplements, genetics | Optional |
| `consensus_dir` | Base path to consensus reference files | Optional |

See `user_config.template.yaml` for all options with documentation.

---

## Парсинг аргументов

Формат: `/wearable-analysis [device] [focus_areas] [hours]`

Примеры:
- `/wearable-analysis whoop` → полный анализ всех доменов, время=auto
- `/wearable-analysis whoop sleep+stress 4h` → фокус на sleep и stress, 4 часа
- `/wearable-analysis whoop training+recovery deep` → deep = extended analysis
- `/wearable-analysis oura sleep` → Oura ring, только sleep домен

### Устройства (adapters)

| Device | Adapter | Data source |
|--------|---------|-------------|
| **whoop** | WHOOP MCP → JSON → CSV | Set `device.data_dir` in `user_config.yaml` |
| **oura** | Oura API/export → CSV | Set `device.data_dir` in `user_config.yaml` |
| **garmin** | Garmin CSV export | Set `device.data_dir` in `user_config.yaml` |
| **apple** | Apple Health XML export | Set `device.data_dir` in `user_config.yaml` |
| **csv** | Generic CSV (user provides schema) | Provide path as argument |

### Домены анализа (focus_areas)

| Domain | Что анализируем | Ключевые переменные |
|--------|----------------|---------------------|
| **sleep** | Архитектура, качество, количество, timing | duration, efficiency, deep%, REM%, latency, wake events, bed/wake time, sleep debt, respiratory rate |
| **recovery** | Восстановление, HRV, RHR | recovery score, HRV, RHR, HRV:RHR ratio, recovery zones, autonomic balance |
| **training** | Нагрузка, зоны, объём, тип | strain, zone distribution, volume, activity types, Z2 volume, intensity distribution |
| **stress** | Стрессовая нагрузка | daily stress, stress by time-of-day, high/med/low distribution, stress × sleep, cumulative stress |
| **cardio** | Кардиореспираторная форма | VO2max trajectory, RHR trend, HRV trend, steps, cardio age |
| **body** | Телесные сигналы | skin temp, SpO2, weight, respiratory rate, healthspan age |
| **interactions** | Перекрёстные эффекты между доменами | все пары и тройки из вышеперечисленных |
| **all** (default) | Все домены + interactions | всё |

Если focus_areas не указан — используй `all`.

---

## Ключевое отличие от простого анализа

```
СТАРЫЙ ПОДХОД:  data → correlations → "interesting findings" → post-hoc literature
ЭТОТ СКИЛЛ:     literature → hypotheses → data → test hypotheses → Bayesian N=1 → actionable
```

**Literature-first** означает: СНАЧАЛА мы знаем что наука говорит о sleep/recovery/training, ПОТОМ проверяем это на персональных данных. Это даёт:
1. Интерпретируемость (не "r=-0.27", а "Föhr 2017 предсказывал r=-0.3, у тебя -0.27 — consistent")
2. Bayesian power (prior из литературы + likelihood из данных = posterior с узким CI)
3. Избежание p-hacking (гипотезы зафиксированы ДО анализа)
4. Действенность (каждый finding привязан к конкретному исследованию с dose-response)

---

## Шаг 0: Подготовка

1. Прочитай pipeline конфиг и гипотезы:
   - `tools/wearable_analysis/config.py` — схема данных, адаптеры
   - `tools/wearable_analysis/hypotheses/` — база гипотез по доменам

2. Прочитай контекст пользователя:
   - `tools/wearable_analysis/user_config.yaml` — profile (sex, age), device, health context
   - Optional: user's health records if provided in `health_context` config
   - Only enabled domains (`domains` in config) will be analyzed
   - Cycle hypotheses require `domains.cycle: true` and `profile.sex: "female"`

3. Создай рабочую папку (default: `output.base_dir` from `user_config.yaml`):
   ```
   [output_dir]/YYYY_MM_[device]_analysis/
   ├── data/          # processed datasets
   ├── figures/       # all visualizations (300 DPI PNG)
   ├── scripts/       # all Python scripts
   │   └── .venv/     # isolated venv
   └── reports/       # markdown reports by domain
   ```

4. Создай `_PROGRESS_LOG.md`

---

## Шаг 1: Data Ingestion (автоматическая)

### 1a. Запусти ingestion pipeline

```bash
cd "[рабочая папка]"
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn shap statsmodels
```

Создай и запусти `scripts/01_ingest.py`:

**Задача:** читает raw data → стандартизированный `data/master.csv`

Стандартная схема (universal across devices):

```python
CORE_SCHEMA = {
    # Time
    'date': 'datetime',

    # Recovery
    'recovery': 'float',      # 0-100%
    'hrv': 'float',           # ms (rMSSD)
    'rhr': 'float',           # bpm
    'resp_rate': 'float',     # brpm
    'spo2': 'float',          # %

    # Sleep
    'sleep_hours': 'float',
    'sleep_efficiency': 'float',  # %
    'deep_pct': 'float',
    'rem_pct': 'float',
    'light_pct': 'float',
    'wake_events': 'int',
    'sleep_latency_min': 'float',
    'bed_time_hour': 'float',     # decimal hours from midnight
    'wake_time_hour': 'float',
    'sleep_debt_hrs': 'float',
    'sleep_hr_avg': 'float',
    'sleep_hr_min': 'float',
    'sleep_stress_pct': 'float',

    # Strain / Training
    'strain': 'float',
    'steps': 'int',
    'calories': 'float',
    'zone1_min': 'float',
    'zone2_min': 'float',
    'zone3_min': 'float',
    'zone4_min': 'float',
    'zone5_min': 'float',
    'workout_duration_min': 'float',
    'n_workouts': 'int',

    # Cardio fitness
    'vo2max': 'float',

    # Stress
    'stress_high_min': 'float',
    'stress_med_min': 'float',
    'stress_low_min': 'float',
    'stress_high_pct': 'float',

    # Body
    'skin_temp': 'float',
    'skin_temp_deviation': 'float',
    'weight_kg': 'float',

    # Context
    'dow': 'int',              # 0=Mon
    'month': 'int',
    'year': 'int',
}
```

Для WHOOP: переиспользуй логику из `01_library/research/health/automated_reviews/2026_02_whoop_deep_analysis/scripts/preprocess_v2.py` — она уже обрабатывает все 20 MCP JSON файлов.

### 1b. Derived features (автоматические)

После базового `master.csv`, создай derived features:

```python
DERIVED_FEATURES = {
    # Lag features (critical for causal inference)
    'prev_1d_{col}': 'shift(1)',     # yesterday's value
    'prev_2d_{col}': 'shift(2)',     # 2 days ago
    'prev_3d_{col}': 'shift(3)',     # 3 days ago

    # Cumulative features (Van Dongen 2003: cumulative sleep debt matters)
    'sleep_debt_3d': 'rolling(3).sum(sleep_hours - sleep_needed)',
    'strain_3d': 'rolling(3).mean(strain)',
    'stress_3d': 'rolling(3).mean(stress_high_pct)',

    # Rolling averages
    '{col}_7d': 'rolling(7).mean()',
    '{col}_30d': 'rolling(30).mean()',

    # Ratios
    'hrv_rhr_ratio': 'hrv / rhr',
    'deep_rem_ratio': 'deep_pct / rem_pct',
    'z2_total_ratio': 'zone2_min / (zone1+zone2+zone3+zone4+zone5)',
    'sleep_vs_needed': 'sleep_hours / sleep_needed',

    # Interaction terms (for testing hypothesis interactions)
    'stress_x_sleep_debt': 'stress_high_pct * sleep_debt_hrs',
    'strain_x_sleep': 'strain * sleep_hours',
    'bedtime_x_strain': 'bed_time_hour * strain',

    # Change features
    'hrv_delta_1d': 'hrv - prev_1d_hrv',
    'rhr_delta_1d': 'rhr - prev_1d_rhr',
    'recovery_delta_1d': 'recovery - prev_1d_recovery',
}
```

Output: `data/master_enriched.csv` (core + derived, ~150-200 columns)

---

## Шаг 2: Literature Hypotheses (ДО анализа данных)

### 2a. Загрузи гипотезы

Прочитай файлы из `tools/wearable_analysis/hypotheses/`:
- `sleep_hypotheses.yaml`
- `recovery_hypotheses.yaml`
- `training_hypotheses.yaml`
- `stress_hypotheses.yaml`
- `cardio_hypotheses.yaml`
- `interactions_hypotheses.yaml`

Каждая гипотеза имеет формат:
```yaml
- id: SLEEP_001
  domain: sleep
  hypothesis: "Sleep duration 7.0-8.5h is optimal for recovery"
  source: "Buysse 2014; Hirshkowitz 2015 NSF guidelines"
  prediction: "recovery peaks at sleep_hours 7.5-8.5, with dose-response r >= 0.3"
  test: "dose_response(sleep_hours, recovery, bins=[5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5])"
  population_effect: "r = 0.30-0.40"
  variables: [sleep_hours, recovery]
  actionable: true
  action_if_confirmed: "Target 7.5-8.5h sleep; each +30min ≈ +3pp recovery"
```

### 2b. Зафиксируй гипотезы в Progress Log

Перед запуском анализа запиши в `_PROGRESS_LOG.md`:
- Список всех гипотез, которые будут тестироваться (по выбранным доменам)
- Ожидаемые effect sizes из литературы
- **Это pre-registration** — зафиксированные до данных предсказания

---

## Шаг 3: Automated EDA (exploratory)

Создай и запусти `scripts/02_eda.py`:

### 3a. Univariate profiles
- Для каждой переменной: mean, median, SD, min/max, missing%, histogram
- Flag outliers (>3 SD или IQR method)
- Trend detection (OLS slope, p-value)

### 3b. Correlation matrix (с лагами)
- Все пары переменных, lag 0-3 дня
- Highlight: |r| > 0.3 (moderate+)
- Partial correlations (controlling for top confounders)
- Output: `figures/correlation_matrix_lag0.png`, `figures/correlation_matrix_lag1.png`

### 3c. Interaction detection (automated)
- Random Forest: predict recovery from all features
- SHAP values: which features matter most? which interactions?
- Top-10 interactions by SHAP interaction value
- Output: `figures/shap_summary.png`, `figures/shap_interactions.png`, `data/feature_importance.csv`

### 3d. Time series decomposition
- STL decomposition for key variables (recovery, HRV, RHR, sleep_hours)
- Trend + seasonal + residual
- Changepoint detection (PELT algorithm or CUSUM)
- Output: `figures/stl_{variable}.png`, `data/changepoints.csv`

### 3e. Missing data analysis
- Pattern of missingness (MCAR/MAR/MNAR assessment)
- Visualization: `figures/missing_data_pattern.png`

Output: `reports/eda_report.md` (structured, with all figures embedded)

---

## Шаг 4: Hypothesis Testing (ядро)

Создай и запусти `scripts/03_hypothesis_test.py`:

### 4a. For each hypothesis from Step 2:

```python
for h in hypotheses:
    result = test_hypothesis(h, data)
    # result includes:
    #   - observed_effect (r, beta, OR, etc.)
    #   - p_value
    #   - confidence_interval
    #   - effect_size (Cohen's d, r², etc.)
    #   - vs_literature (concordant / discordant / null)
    #   - bayesian_posterior (prior=literature, likelihood=data)
    #   - dose_response_curve (if applicable)
    #   - figure_path
```

### 4b. Test types по типу гипотезы:

| Test type | When | Method |
|-----------|------|--------|
| **correlation** | "X correlates with Y" | Pearson/Spearman + bootstrap CI |
| **dose_response** | "X has optimal range for Y" | Binned means + nonlinear fit (GAM or piecewise linear) |
| **causal_lag** | "X predicts Y next day" | Granger causality + lagged regression |
| **mediation** | "X → M → Y" | Baron-Kenny + Sobel test |
| **interaction** | "X × Z modifies effect on Y" | Moderated regression + SHAP |
| **threshold** | "Below/above X threshold, Y changes" | Piecewise regression + changepoint |
| **temporal** | "X trend over time" | OLS + STL + Mann-Kendall |

### 4c. Bayesian N=1 framework

Для каждой гипотезы с population_effect:

```python
# Prior: from literature (e.g., r = 0.35 ± 0.10)
prior = Normal(mu=0.35, sigma=0.10)

# Likelihood: from personal data (e.g., r_observed = 0.32, SE = 0.05)
likelihood = Normal(mu=r_observed, sigma=se_observed)

# Posterior: Bayesian update
posterior = bayesian_update(prior, likelihood)
# → posterior_mean, posterior_ci, credible_interval

# Interpretation:
# "Literature predicts r=0.35; your data shows r=0.32;
#  Bayesian posterior: r=0.33 (95% CI: 0.22-0.44) — CONCORDANT"
```

### 4d. Causal inference

Для выбранных пар (based on SHAP top-10):
- **Granger causality**: does X predict Y (1-3 day lag)?
- **DAG estimation**: PC algorithm or expert-defined DAG → mediation paths
- **Mediation analysis**: X → M → Y (e.g., stress → sleep_quality → recovery)

Output: `reports/hypothesis_results.md`, `data/hypothesis_results.csv`, `figures/dose_response_*.png`

---

## Шаг 5: Personalization Layer

Создай и запусти `scripts/04_personalize.py`:

### 5a. Population norms comparison

Для VO2max, HRV, RHR, sleep duration — compare to age/sex percentiles:

```python
POPULATION_NORMS = {
    'vo2max': {'source': 'ACSM 2022', 'female_35-39': {
        'p10': 28, 'p25': 31, 'p50': 35, 'p75': 40, 'p90': 44, 'p95': 47
    }},
    'hrv': {'source': 'Shaffer 2017', 'female_35-39': {
        'p10': 25, 'p25': 40, 'p50': 60, 'p75': 85, 'p90': 110
    }},
    'rhr': {'source': 'AHA', 'general': {
        'excellent': '<60', 'good': '60-70', 'average': '71-80'
    }},
    # ... etc
}
```

Output: `figures/population_comparison.png`

### 5b. Anomaly detection (where you differ from population)

- Positive anomalies: better than expected (celebrate!)
- Negative anomalies: worse than expected (investigate)
- Neutral: population-typical

### 5c. Actionability scoring

For each confirmed hypothesis:
```python
actionability = effect_size * modifiability * confidence * (1 - current_compliance)
```

Where:
- `effect_size`: Cohen's d or r² from data
- `modifiability`: can user change this? (bedtime=high, genetics=zero)
- `confidence`: Bayesian posterior confidence
- `current_compliance`: how well user already follows (bedtime at optimal = low priority)

Output: `data/actionability_scores.csv`, `reports/personalization_report.md`

---

## Шаг 6: Synthesis

### 6a. Automated figures

`scripts/05_visualize.py`:

Required figures:
1. `population_comparison.png` — percentile chart
2. `shap_summary.png` — feature importance
3. `shap_interactions.png` — top interaction effects
4. `correlation_matrix.png` — heatmap with lag structure
5. `dose_response_{variable}.png` — per significant hypothesis
6. `timeline_dashboard.png` — key metrics over time (9-panel)
7. `changepoints.png` — detected regime changes
8. `monthly_scorecards.png` — month-by-month grades
9. `causal_dag.png` — estimated causal graph
10. `actionability_matrix.png` — effect size × modifiability

All 300 DPI, clean matplotlib style.

### 6b. Synthesis document

Write `synthesis.md` (EN) with sections:

1. **TL;DR** — top 5-8 actions ranked by actionability score
2. **Fitness Profile** — population percentiles, strengths, ceiling potential
3. **Hypothesis Results** — table: hypothesis → literature prediction → personal result → concordance → action
4. **Key Findings by Domain**
   - For each domain: top 3-5 findings, each with:
     - "Literature says: [citation, expected effect]"
     - "Your data shows: [observed effect, p, CI]"
     - "Bayesian conclusion: [posterior estimate]"
     - "Action: [specific recommendation]"
5. **Interaction Effects** — which variable combinations matter more than individuals
6. **Causal Model** — DAG with estimated paths, mediation results
7. **Dose-Response Curves** — key nonlinear relationships with personal optimal zones
8. **Temporal Dynamics** — trends, changepoints, phases
9. **What's Working** — explicit positives (not just problems!)
10. **Monitoring Plan** — KPIs, targets, frequencies
11. **Confidence Assessment** — per finding
12. **Limitations** — N=1 caveats, missing data, device accuracy

### 6c. Russian synthesis

Write `synthesis_ru.md` — full translation, not summary.

### 6d. Unknowns

Write `unknowns_and_next.md` — known unknowns, next data to collect, experiments to run.

---

## Шаг 7: Quality Gates

### 7a. Self-review checklist

- [ ] All hypotheses from Step 2 tested and reported
- [ ] No "interesting correlations" reported that weren't pre-registered (move to exploratory section)
- [ ] Effect sizes reported (not just p-values)
- [ ] Confidence intervals reported
- [ ] Population comparison included
- [ ] Bayesian posteriors calculated for key findings
- [ ] At least 1 causal analysis (Granger or mediation)
- [ ] Dose-response curves for top 3 actionable findings
- [ ] SHAP values for ML model
- [ ] "What's Working" section is substantial (not token)
- [ ] All figures have titles, labels, legends

### 7b. Finalization

```bash
bash tools/finalize_research.sh \
  --dir "[рабочая папка]" \
  --title "[Device] Deep Analysis — [focus]" \
  --message "Add [device] wearable analysis ([domains])"
```

---

## Гипотезы: Краткий обзор (полная база в tools/wearable_analysis/hypotheses/)

### Sleep (10 hypotheses)
- SLEEP_001: Duration 7.0-8.5h optimal for recovery (Buysse 2014)
- SLEEP_002: Consistency matters more than duration (Phillips 2017)
- SLEEP_003: Bedtime regularity > sleep duration for HRV (Huang 2020)
- SLEEP_004: Sleep debt accumulates over 3-5 days (Van Dongen 2003)
- SLEEP_005: Deep sleep >15% correlates with subjective recovery (Dijk 2010)
- SLEEP_006: Wake events >5 degrade recovery regardless of duration (Bonnet 2003)
- SLEEP_007: Late bedtime (>01:00) reduces deep sleep independently (Wittmann 2006)
- SLEEP_008: Sleep efficiency >85% threshold for adequate recovery (Buysse 2014)
- SLEEP_009: Respiratory rate elevation predicts illness 1-2 days before symptoms (Natarajan 2020)
- SLEEP_010: Sleep HR deviation from baseline predicts next-day recovery (Buchheit 2014)

### Recovery & HRV (8 hypotheses)
- REC_001: HRV is primary predictor of recovery (Buchheit 2014)
- REC_002: RHR is stronger predictor than HRV at daily resolution (Plews 2013)
- REC_003: HRV:RHR ratio more stable than either alone (Buchheit 2014)
- REC_004: HRV-guided training outperforms fixed periodization (Kiviniemi 2007)
- REC_005: Previous-day strain modulates recovery (OR ~0.85/SD) (Manresa-Rocamora 2021)
- REC_006: 2-day minimum for full recovery after strain >15 (Bishop 2008)
- REC_007: Iron deficiency suppresses HRV via RHR elevation (Yokusoglu 2007; Tuncer 2009)
- REC_008: Skin temperature deviation predicts illness/overreaching (Li 2017)

### Training (8 hypotheses)
- TRAIN_001: 80/20 polarization maximizes VO2max adaptation (Seiler 2010)
- TRAIN_002: Z2 minimum 120 min/wk for aerobic base (Seiler 2010)
- TRAIN_003: Game sports naturally produce 15-25% Z3 — not a training error (Foster 1995)
- TRAIN_004: VO2max is strongest predictor of all-cause mortality (Mandsager 2018)
- TRAIN_005: 150-300 min/wk moderate = 22-31% mortality reduction (Paluch 2022 meta)
- TRAIN_006: Diminishing returns above 300 min/wk for mortality (no J-curve) (Ekelund 2016)
- TRAIN_007: Steps 7000-10000/day optimal for mortality; marginal above (Paluch 2022)
- TRAIN_008: IDA reduces VO2max by 3-7%; correction restores it (Burden 2015)

### Stress (6 hypotheses)
- STRESS_001: Daily stress directly suppresses nocturnal HRV (Föhr 2017)
- STRESS_002: Cumulative stress (3+ days) has stronger effect than single-day (McEwen 1998)
- STRESS_003: High stress + short sleep = multiplicative recovery impairment (Åkerstedt 2007)
- STRESS_004: Stress timing matters — evening stress worse than morning (Brosschot 2010)
- STRESS_005: Physical stress (training) and psychological stress share HRV pathway (Hynynen 2011)
- STRESS_006: Chronic stress elevates RHR independently of training (Vrijkotte 2000)

### Interactions (8 hypotheses)
- INT_001: Strain × sleep interaction: high strain + <7h sleep = worst recovery (>15pp gap)
- INT_002: Stress × sleep debt: multiplicative, not additive (Åkerstedt 2007)
- INT_003: Bedtime × strain: late bedtime amplifies strain-recovery deficit
- INT_004: RHR × VO2max: as RHR drops, VO2max ceiling rises
- INT_005: Iron × VO2max: correction predicts 3-7% VO2max gain
- INT_006: Sleep debt × stress: 3-day cumulative debt amplifies stress impact
- INT_007: Deep sleep × strain: high strain + low deep% = prolonged recovery
- INT_008: Day-of-week × behavior: systematic patterns predict worst/best days

---

## Стиль

"Скуп" — data-first, конкретные числа. Для каждого finding:
- Что литература говорит (citation, expected effect)
- Что данные показывают (observed, CI)
- Что делать (конкретно, персонализированно)

**Обязательно:** секция "What's Working" — позитивные находки, сильные стороны, прогресс. Не только проблемы.

---

## Известные подводные камни

- **WHOOP MCP JSON structure** — dict-of-dicts with `_metadata`/`data` nesting or direct dict; Activities = dict-of-lists. Ссылка: `preprocess_v2.py`
- **Duration parsing** — WHOOP uses varied formats ("2h 30m", "1:30:00", "45m"). Нужен robust parser
- **PEP 668** — use venv, not system pip
- **SHAP on small N** — с 500-600 днями SHAP работает, но CI будут широкими. Используй TreeExplainer (быстрый)
- **Granger causality** — требует стационарный ряд. Дифференцируй если нужно (ADF test)
- **Multiple testing** — при 40+ гипотезах используй FDR correction (Benjamini-Hochberg), не Bonferroni
- **Bayesian priors** — используй weakly informative priors из литературы, не flat priors
- **N=1 caveats** — всегда упоминай: no randomization, no control, confounders possible
