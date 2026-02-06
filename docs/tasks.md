# План и статус

Обновлено: 2026-02-06
Статус: активное выполнение

## Цель недели

Проверить гипотезу, что режим `double` повышает вероятность dual-hit по двум мишеням относительно `single_a` и `single_b` на наборе из 2-5 пар белков.

## Каноничный пайплайн

1. Data deps (one-shot): `scripts/prepare_data_dependencies.py`
2. Data deps (точечные): `scripts/fetch_freed_data.py`, `scripts/fetch_baseline_zinc.py`
3. Подготовка таргетов: `scripts/prepare_targets.py`
4. Генерация: `scripts/generate.py` (`single_a`, `single_b`, `double`)
5. Скоринг: `scripts/score.py`
6. Агрегация: `scripts/aggregate.py`
7. Метрики и статистика: `scripts/compute_metrics.py`
8. Визуализация и сводка: `scripts/plot_results.py`, `scripts/render_molecules.py`, `scripts/summarize.py`
9. Статья/отчёт: `scripts/generate_latex_report.py` + модульные главы в `report/chapters/*.tex`
Структура статьи задаётся в `report/chapters/manifest.yaml` (уровень IMRaD + limitations + reproducibility appendix).
10. Оркестрация Stage 1: `scripts/run_full_experiment.py` (generate + score, опции `--prepare-freed-data`, `--prepare-baseline`)
11. Оркестрация Stage 2: `scripts/run_post_pipeline.py` (aggregate, metrics, plots, summary, LaTeX)

## Что уже сделано

- Калибровочный докинг выполнен, `configs/normalization.yaml` заполнен.
- `FREED++` reward path переведён на Uni-Dock + RDKit/Meeko.
- Добавлены скрипты загрузки data dependencies:
  - `scripts/fetch_freed_data.py`
  - `scripts/fetch_baseline_zinc.py`
  - `scripts/prepare_data_dependencies.py`
- Автосборка образа переведена на root-context:
  - `docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .`
- Исправлены проблемы со stale-output и несогласованностью hit-правила.
- Добавлены run-specific логи `report/experiment_<run_id>_<utc>.log`.
- Smoke-прогоны `single_b` и `double` проходят без аномалий `|z| > 10`.
- Обновлён кэш-ключ модели для отсечения старых артефактов.
- Проведена валидация `README.md` и `docs/*` на соответствие текущим CLI-скриптам.
- **ENG-01 (KVA-255)**: Reward — исключён невалидный docking из обучения.
  - `_capped_zscore()` ограничивает z-penalty для error_val (default 4.0, настраивается `INVALID_Z_PENALTY`).
  - Счётчики `invalid_a/b` ведутся в runtime, `reward_metadata.json` пишется при завершении (`atexit`).
  - Валидация `INVALID_Z_PENALTY > 0 && finite`, `mkdir -p` для `REWARD_METADATA_PATH`.
  - Smoke-артефакт: `docs/tasks/ENG-01_smoke.json`.
- **ENG-06 (KVA-260)**: Inter-process lock + atomic write для cache JSON (`cache_io.py`).

## Что в работе

- Полный прогон по всем парам/сидам.
- Финальные `report/metrics.json` и `report/stats.json` для сравнения режимов.
- Подготовка итогового отчёта и набора кандидатов.

## Критерии завершения

- Для большинства пар `hr_both(double)` выше `hr_both(single_a)` и `hr_both(single_b)`.
- Различия подтверждены статистически (`FDR-adjusted p < 0.05`, согласованный знак эффекта).
- Результаты воспроизводимы из `README.md` без ручных правок пайплайна.

## Cleanup (2026-02-05)

- Удалены неканоничные research-документы и устаревшие вспомогательные скрипты.
- Очищены `results/` и generated-артефакты в `report/`.
- `.gitignore` обновлён, чтобы generated-файлы не копились в репозитории.
