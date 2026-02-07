# Task Log

## Completed

- **KVA-255 / ENG-01**: Reward — исключить невалидный docking из обучения. Sentinel/error значения больше не участвуют как нормальные скоры; добавлены счётчики invalid/timeout в metadata.
- **KVA-256 / ENG-02**: Single_b — чистый baseline по target B. Primary receptor=B, A не влияет на training reward single_b.
- **KVA-257 / ENG-03**: Quality Gate — валидность reward-прогона. Конфиг-пороги, run_valid flag, Stage2 игнорирует invalid runs.
- **KVA-258 / ENG-04**: Objectives — синхронизация config и фактического расчёта. Активные objectives считаются автоматически из experiment.yaml.
- **KVA-259 / ENG-05**: Cache Key v2 — контекстный ключ скоринга. Boltz version + CLI args hash + MSA hash + docker digest в cache context.
- **KVA-260 / ENG-06**: Межпроцессный lock + atomic write для cache JSON. fcntl.flock + temp-write + rename; безопасно при parallel=4.
- **KVA-261 / ENG-07**: Метрики — hr_a/hr_b/hr_both/balance/missing_rate, completeness gate, n_valid_runs.
- **KVA-262 / ENG-08**: Логирование lifecycle + run manifest. Полный RUN/OK/FAIL/END per task, run_manifest.json.
- **KVA-264 / ENG-10**: Enforce human-only в runtime-валидации. Non-human блокируется, есть exploratory bypass флаг.
- **KVA-265 / SCI-01**: Pre-registration гипотез и decision rules. Создан `docs/preregistration.md` с frozen endpoints, thresholds, статтестами. Cross-referenced в `report/chapters/04_experimental_protocol.tex`.
- **KVA-266 / SCI-02**: Human-only набор пар и PDB freeze. `configs/pairs.yaml` полностью human-only, pair freeze version в master config.
- **KVA-267 / SCI-03**: Receptor QC sheet per target. 100% targets покрыты, `docs/receptor_qc.md`.
- **KVA-268 / SCI-04**: Redocking validation (RMSD). RMSD таблица, pass/fail threshold, `report/data/redocking_metrics.csv`.

## In Progress

- **KVA-269 / SCI-05**: Enrichment benchmark (actives vs decoys). AUROC, EF1%, BEDROC по мишеням. Скрипт и модуль готовы, ожидает GPU-прогон.
- **KVA-263 / ENG-09**: Docs hard-sync с кодом. README.md + docs/*.md синхронизация с кодом.

## Backlog

- **KVA-270 / SCI-06**: Hit-threshold calibration + sensitivity.
- **KVA-271 / SCI-07**: Proxy vs Final correlation (Uni-Dock reward vs Boltz-2).
- **KVA-272 / SCI-08**: SAP — Statistical Analysis Plan submission-grade.
- **KVA-273 / SCI-09**: Pilot matrix run (1 pair x 3 modes x 2 seeds).
- **KVA-274 / SCI-10**: Full matrix run + claim audit.

## Milestones

- **KVA-275 / M1**: P0 Engineering Closed.
- **KVA-276 / M2**: Scientific Protocol Validated.
- **KVA-277 / M3**: Pilot Passed.
- **KVA-278 / M4**: Submission Package Ready.
