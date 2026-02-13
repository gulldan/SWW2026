# Pipeline Granularity Refactoring: 9 → 17 Stages

> Approved 2026-02-13. See plan `1802d8f5` for full context.

## Goal

Split coarse pipeline stages into independently re-runnable steps
for easier debugging and selective re-execution.

## Stage Descriptions

### Stage 0 — `fetch`
**Script:** `s0_fetch/fetch_all.py`
**Depends on:** —
**Зачем:** Единственная стадия, требующая интернет. Скачивает все внешние данные:
PDB-структуры белков (RCSB), FASTA-последовательности (UniProt), MSA для Boltz-2
(ColabFold API), калибровочные SMILES (HuggingFace), активные молекулы из ChEMBL,
бинарник GNINA (GitHub releases), собирает Docker-образ FREED++.
**Выход:**
- `output/data/targets/{pair}/{A,B}/` — PDB, FASTA, PDBQT-рецепторы
- `output/data/calibration/smiles/` — калибровочные SMILES
- `output/data/chembl/` — ChEMBL активные молекулы
- `output/cache/boltz_msa/` — MSA-кэш
- `output/bin/gnina` — GNINA бинарник
- `docker/freedpp/data/` — данные для FREED++
- `output/data/targets/fetch_manifest.json` — манифест загрузки
**Зачем отдельно:** Всё остальное работает оффлайн. При сбое сети перезапускается
только fetch, не трогая вычисления.

### Stage 1 — `prepare`
**Script:** `s1_prepare/prepare_all.py`
**Depends on:** fetch
**Зачем:** Конвертирует сырые данные в формат, пригодный для дальнейших стадий.
PDB → PDBQT (рецепторы для UniDock), извлечение binding box, подготовка
базовых молекул (ZINC baseline → Parquet, 3D-embedding через RDKit).
**Выход:**
- `output/data/targets/{pair}/{A,B}/*_receptor.pdbqt` — рецепторы для докинга
- `output/data/targets/{pair}/{A,B}/binding_box.json` — координаты сайта связывания
- `output/data/baseline/zinc_sample.parquet` — бейзлайн для сравнения
**Зачем отдельно:** Подготовка данных — одноразовая операция. При добавлении новой
пары белков перезапускается только prepare, а не весь fetch.

### Stage 2 — `dock_cal`
**Script:** `s2_calibrate/dock_calibration.py`
**Depends on:** prepare
**Зачем:** Докинг калибровочного набора молекул (200–500 SMILES) против каждого
таргета для получения распределения скоров. Это необходимо для Z-нормализации
наград в RL: генератор FREED++ получает reward = normalized(dock_score), и без
калибровки reward-сигнал будет на разных шкалах для разных белков.
**Выход:**
- `output/data/calibration/scores/{target_id}.parquet` — сырые скоры калибровки
**Зачем отдельно:** Докинг калибровки — GPU-тяжёлая операция (~30 мин на таргет).
При сбое не нужно пересчитывать уже задокенные таргеты.

### Stage 3 — `normalize`
**Script:** `s2_calibrate/compute_normalization.py`
**Depends on:** dock_cal
**Зачем:** Агрегирует скоры калибровки в статистики (mean, std) для Z-нормализации.
Выполняет проверку стабильности: если std слишком мала или n_samples недостаточен,
таргет помечается как нестабильный и pipeline не продолжится.
**Выход:**
- `configs/normalization.yaml` — mean/std/n_samples/stable для каждого таргета
**Зачем отдельно:** Чисто CPU-вычисление (<1 сек). При изменении порогов стабильности
можно пересчитать нормализацию без повторного докинга.

### Stage 4 — `validate_cal`
**Scripts:** `s2_calibrate/validate_redocking.py` + `s2_calibrate/enrichment_benchmark.py`
**Depends on:** normalize
**Зачем:** Качественный контроль. Проверяет, что докинг-программа корректно работает
с данными белками:
1. **Redocking** (SCI-04): ко-кристаллический лиганд должен вернуться в нативную
   позу (RMSD ≤ 2.0 Å). Если нет — докинг ненадёжен для этого таргета.
2. **Enrichment** (SCI-05): AUROC > 0.6, BEDROC > 0.3 — докинг должен различать
   активные молекулы от декоев. Если нет — скоры не информативны.
**Выход:**
- `output/data/calibration/redocking_results.json` — RMSD для каждого таргета
- `output/data/calibration/enrichment_results.json` — AUROC/BEDROC
**Зачем отдельно:** Валидация — необязательный шаг (pipeline работает и без неё),
но блокирующий для публикации. Можно запускать независимо и итерировать
(например, поменять PDB-структуру и перепроверить).

### Stage 5 — `generate`
**Script:** `s3_generate/generate.py`
**Depends on:** normalize
**Зачем:** Ядро эксперимента. Запускает FREED++ в Docker-контейнере для генерации
молекул через reinforcement learning. Для каждой комбинации (pair × mode × seed)
тренирует RL-агента с dual-target reward и сэмплирует молекулы.
Моды: `single_a`, `single_b` (бейзлайны), `double` (основной), + exploratory.
**Выход (на каждый run):**
- `output/results/{pair}/{mode}/seed_{seed}/generated.parquet` — SMILES молекул
- `output/results/{pair}/{mode}/seed_{seed}/metadata.json` — провенанс
  (git commit, docker digest, diversity, filter stats, training curve)
**Зачем отдельно:** Самая долгая стадия (~2–8 ч на run, 240 runs = ~600 GPU-часов).
Каждый run независим; при сбое перезапускается только конкретный run,
а не весь скоринг.

### Stage 6 — `score`
**Script:** `s4_score/score.py`
**Depends on:** generate
**Зачем:** Оценка сгенерированных молекул суррогатным скорером Boltz-2.
Boltz-2 предсказывает аффинность связывания (log10 IC50 в μM) для каждой
молекулы против обоих таргетов. Это **evaluation surrogate** — независимый
от RL-reward (UniDock) метод оценки, что критично для валидности эксперимента.
**Выход (на каждый run):**
- `output/results/{pair}/{mode}/seed_{seed}/scores_A.parquet` — скоры таргета A
- `output/results/{pair}/{mode}/seed_{seed}/scores_B.parquet` — скоры таргета B
- `output/results/{pair}/{mode}/seed_{seed}/scored_joined.parquet` — объединённая
  таблица (smiles, affinity_a, affinity_b, is_hit_both, combined_min, ...)
- `output/results/{pair}/{mode}/seed_{seed}/scoring_context.json` — метаданные скорера
**Зачем отдельно:** GPU-тяжёлая операция. Независима от генерации: можно переоценить
молекулы другим скорером или с другими параметрами Boltz-2 без перегенерации.

### Stage 7 — `filter`
**Script:** **NEW** `s5_filter/apply_filters.py`
**Depends on:** score
**Зачем:** Аннотирует молекулы по drug-likeness критериям: Lipinski Rule of Five,
QED (≥0.5), PAINS (отсутствие pan-assay interference), SA score (≤5.0),
Veber rules (oral bioavailability). **Не удаляет молекулы** — только добавляет
колонки `passes_lipinski`, `passes_qed`, ..., `passes_all_filters`.
Это аналитическая аннотация: в отчёте мы показываем долю drug-like молекул
как secondary endpoint.
**Выход:**
- Обновлённый `scored_joined.parquet` с колонками фильтров
- `filter_stats.json` — агрегированная статистика (n_pass_lipinski, n_pass_all, ...)
**Зачем отдельно:** При изменении порогов в `configs/filters.yaml` (например,
MW max 500→600) можно перефильтровать за секунды без пересчёта Boltz-2 скоров.

### Stage 8 — `gnina`
**Script:** `s6_report/score_gnina.py`
**Depends on:** score
**Зачем:** Кросс-валидация с третьим независимым скорером. GNINA — CNN-based
скорер, ортогональный и UniDock (RL reward), и Boltz-2 (evaluation surrogate).
Согласие трёх скореров усиливает доказательность результата; расхождение
указывает на scorer-specific bias.
**Выход (на каждый run):**
- `output/results/{pair}/{mode}/seed_{seed}/gnina_scores_A.parquet`
- `output/results/{pair}/{mode}/seed_{seed}/gnina_scores_B.parquet`
- Колонки: `gnina_score`, `gnina_affinity`, `cnn_score` для каждого таргета
**Зачем отдельно:** Требует GNINA бинарник/Docker, может быть медленнее Boltz-2.
Опциональная стадия — pipeline работает и без неё, но для Q1 журнала
третий скорер сильно повышает credibility.

### Stage 9 — `aggregate`
**Script:** `s5_analyze/aggregate.py`
**Depends on:** score
**Зачем:** Собирает `scored_joined.parquet` из всех runs (pairs × modes × seeds)
в единый датафрейм. Применяет quality gate (отсев runs с >10% invalid scores
или >5% timeouts). Автоматически мержит GNINA-скоры, если они есть.
**Выход:**
- `report/data/all_results.parquet` — единая таблица всех результатов
**Зачем отдельно:** Быстрая CPU-операция (~сек). При добавлении новых runs
(например, дополнительный seed) можно переагрегировать без пересчёта метрик.

### Stage 10 — `metrics`
**Script:** `s5_analyze/compute_metrics.py`
**Depends on:** aggregate
**Зачем:** Главный статистический анализ по SAP (Statistical Analysis Plan):
1. Per-pair per-mode hit-rates (HR_A, HR_B, HR_both) с SEM по seeds
2. Hierarchical bootstrap (10 000 итераций): p-values для H1a, H1b
3. Cliff's delta effect size + 95% CI
4. BH-FDR коррекция на множественное тестирование
5. Degradation guard: Cliff's delta > −0.47 для per-target HR
6. Wild cluster bootstrap (sensitivity)
7. Power analysis
**Выход:**
- `report/metrics.json` — per-pair per-mode summary (means, SEM)
- `report/stats.json` — p-values, Cliff's delta, CI, significance, degradation,
  completeness, FDR-adjusted results
**Зачем отдельно:** Самая важная стадия для статьи. При изменении
статистических параметров (n_bootstrap, FDR threshold) пересчитывается
за минуты без переагрегации.

### Stage 11 — `sensitivity`
**Scripts:** `s5_analyze/threshold_sensitivity.py` + `s5_analyze/proxy_correlation.py`
**Depends on:** aggregate
**Зачем:** Два sensitivity analysis (SAP Section 9):
1. **Threshold sensitivity** (SCI-06): пересчёт hit-rates при T = −1, 0, +1
   (100 nM, 1 μM, 10 μM). Если результат устойчив — усиливает primary endpoint;
   если нет — ограничение.
2. **Proxy correlation** (SCI-07): Spearman ρ между UniDock (reward) и Boltz-2
   (evaluation). Высокая корреляция = surrogate bias risk; низкая = независимость.
**Выход:**
- `report/data/threshold_sensitivity.parquet` — hit-rates при разных порогах
- `report/data/proxy_correlation.parquet` — Spearman ρ per pair
**Зачем отдельно:** Оба анализа — secondary/sensitivity, не блокируют primary.
Можно запускать параллельно с metrics и итерировать пороги.

### Stage 12 — `summarize`
**Script:** `s5_analyze/summarize.py`
**Depends on:** metrics
**Зачем:** Генерирует человекочитаемый текстовый отчёт: per-pair breakdown,
top-N молекул, score distributions, statistical results, success criterion
evaluation. Это debug-артефакт — быстрый способ проверить результаты
без рендеринга фигур и LaTeX.
**Выход:**
- `report/summary.txt` — plain-text отчёт (~2–5 страниц)
**Зачем отдельно:** Мгновенная генерация. Удобно для итеративной проверки
результатов в терминале.

### Stage 13 — `plots`
**Script:** `s6_report/plot_results.py`
**Depends on:** metrics
**Зачем:** Визуализация результатов для статьи:
heatmap hit-rates, violin/box plots score distributions, scatter plots
(affinity_a vs affinity_b), bar charts effect sizes, forest plots CI.
**Выход:**
- `report/figures/heatmap_*.png`
- `report/figures/violin_*.png`
- `report/figures/scatter_*.png`
- `report/figures/forest_*.png`
**Зачем отдельно:** При изменении стиля/цветов/DPI перегенерируются только
фигуры за секунды, без пересчёта метрик.

### Stage 14 — `render`
**Script:** `s6_report/render_molecules.py`
**Depends on:** aggregate
**Зачем:** 2D-визуализация top-N молекул (RDKit + matplotlib): grid с
аннотациями скоров, цветовой маркировкой hit/miss. Для каждой пары и
моды — отдельная фигура.
**Выход:**
- `report/figures/molecules_{pair}_{mode}.png` — грид молекул
- `report/figures/top_hits_{pair}.png` — лучшие dual-hits
**Зачем отдельно:** Зависит только от aggregate (не от metrics), поэтому
может запускаться параллельно с metrics/sensitivity/plots.

### Stage 15 — `latex`
**Script:** `s6_report/generate_latex_report.py`
**Depends on:** plots, render
**Зачем:** Сборка LaTeX-рукописи: подставляет числа из `metrics.json`
и `stats.json` в шаблоны `report/chapters/*.tex`, компилирует PDF.
Режимы: `draft` (только .tex), `full` (.tex + pdflatex + bibtex).
**Выход:**
- `report/manuscript.tex` — обновлённый LaTeX
- `report/manuscript.pdf` — скомпилированный PDF (в режиме full)
**Зачем отдельно:** LaTeX-компиляция может сломаться по причинам, не связанным
с данными (пропущен пакет, битая ссылка). Перезапуск только этой стадии.

### Stage 16 — `md`
**Script:** `s6_md/run_md.py`
**Depends on:** score
**Зачем:** Молекулярная динамика (50 ns GROMACS) для top-3 молекул
каждого mode. Проверяет, что предсказанные хиты действительно стабильно
связываются с белком в full-atom MD: лиганд RMSD < 3.0 Å за последние
25 ns → "стабильный". Это **пост-хок валидация** — не влияет на primary
endpoint, но критична для credibility в Q1.
**Выход:**
- `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/rmsd_ligand.csv`
- `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/hbonds.csv`
- `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/interaction_energy.csv`
- `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/trajectory.xtc`
- `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/mmpbsa.csv` (optional)
- `output/md/md_summary.parquet` — сводка stability/energy
**Зачем отдельно:** Очень тяжёлая операция (часы–дни на молекулу). Требует
отдельный софт (GROMACS, AmberTools, acpype). Опциональна.

### Stage 17 — `audit`
**Script:** `s7_audit/claim_audit.py`
**Depends on:** metrics
**Зачем:** Финальная проверка traceability: каждое утверждение в рукописи
должно быть подтверждено числами из `stats.json`/`metrics.json`.
8 проверок по SAP:
1. stats.json существует
2. Все core пары имеют H1a + H1b результаты
3. Significance decisions корректны (p < 0.05 AND |Cliff's d| ≥ 0.33 AND CI)
4. Success criterion (≥3/5 core significant)
5. No-degradation guard
6. Completeness (missing data < 20%)
7. Claim language matches outcome scenario (6.1–6.6)
8. Evidence-to-claims mapping
**Выход:**
- `report/data/claim_audit.json` — структурированный аудит-трейл
- Exit code: 0 = pass, 1 = fail
**Зачем отдельно:** Последний gate перед submission. Если аудит не прошёл —
нельзя отправлять в журнал. Можно запускать многократно после правок в рукописи.

## Tasks

- [ ] **T1** Create `scripts/s5_filter/__init__.py` + `apply_filters.py`
  - Standalone filter: reads `scored_joined.parquet`, applies `configs/filters.yaml`,
    writes filter columns back + `filter_stats.json`
  - Register in `script_dispatch.py`

- [ ] **T2** Add `--skip-filters` flag to `scripts/s4_score/score.py`
  - Extract filter block (lines 282–304) into reusable utility
  - Default: still apply filters (backward compat)
  - `--skip-filters`: defer to standalone `filter` stage

- [ ] **T3** Rewrite `scripts/pipeline/run_stages.py`
  - `STAGE_ORDER`: 17 entries
  - `STAGE_DEPS`: dependency DAG per table above
  - `STAGE_DESCRIPTIONS`: one-line per stage
  - New `_run_*()` + preflight functions for each new stage

- [ ] **T4** Update `scripts/pipeline/run_post_pipeline.py`
  - Split `_commands()` into: aggregate → metrics → sensitivity → summarize → plots → render → latex

- [ ] **T5** Update `scripts/pipeline/run_full_experiment.py`
  - Split `_post_commands()` to match new stages
  - Add `--with-gnina` flag
  - Add `--with-filter` / `--skip-filter` flag

- [ ] **T6** Update `dualdocking/script_dispatch.py`
  - Add entries: `s5_filter/apply_filters.py`, `s2_calibrate/validate_redocking.py`,
    `s2_calibrate/enrichment_benchmark.py`, `s2_calibrate/power_analysis.py`,
    `s6_report/score_gnina.py`, `s6_md/run_md.py`

- [ ] **T7** Update `docs/pipeline.md`
  - Rewrite stage table and CLI examples for 17 stages
  - Remove `(planned)` from gnina and md

- [ ] **T8** Run linter + tests
  - `uv run ruff check scripts/ dualdocking/`
  - `uv run pytest tests/ -x -q`
