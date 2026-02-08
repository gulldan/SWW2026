# Документация: карта достоверности

Обновлено: 2026-02-08

Этот файл фиксирует, какие документы использовать как источник истины для запуска экспериментов, а какие являются исследовательскими заметками.

## Приоритет источников

1. `configs/*.yaml` и `scripts/*.py` (фактическое поведение кода)
2. `README.md` (каноничный runbook)
3. `docs/goal.md`, `docs/tasks.md` (цели, критерии, текущий статус выполнения)
4. `report/chapters/manifest.yaml` + `report/chapters/*.tex` (модульная структура и narrative-контент статьи)

## Статус файлов

| Файл | Статус | Назначение |
|---|---|---|
| `README.md` | Canonical | Пошаговое воспроизведение и команды |
| `docs/goal.md` | Operational spec | Гипотеза, метод, метрики, критерии успеха |
| `docs/tasks.md` | Living log | Журнал выполнения и план |
| `docs/selection_criteria.md` | Reference | Обоснование выбора пар; фактические пары берутся из `configs/pairs.yaml` |
| `docs/preregistration.md` | Frozen (amended v1.1) | Pre-registration: hypotheses, endpoints, decision rules, SAP summary |
| `docs/statistical_analysis_plan.md` | Frozen (amended v1.1) | SAP: inference, claim rules, pipeline-to-code mapping |
| `docs/README.md` | Index | Этот файл: карта достоверности и приоритетов |

## Правило разрешения конфликтов

Если команда/параметр в исследовательском документе расходится с кодом или `README.md`, использовать код и `README.md`.

**Примечание**: `AGENTS.md` является авторитетным источником для конвенций разработки (архитектура, стиль кода, тестирование). Данный файл (`docs/README.md`) определяет приоритеты для разрешения параметров эксперимента.
