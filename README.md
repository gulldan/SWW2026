# Воркшоп: Dual-Target Docking

Сравнение single-target baseline режимов (`single_a`, `single_b`) и `double` режима генерации с использованием **FREED++** (RL-генератор) и **Boltz-2** (structure-aware скорер).

## Гипотеза

RL-генерация, оптимизированная под worst-case цель (`max(score_A, score_B)` при *lower-is-better*), порождает молекулы с более высоким dual hit-rate, чем оптимизация под одну мишень.

## Структура проекта

```
├── configs/              # Конфигурация эксперимента
│   ├── pairs.yaml        # 9 пар белковых мишеней (5 core + 3 auxiliary + 1 blocked)
│   ├── metrics.yaml      # Пороги хитов и определения метрик
│   ├── experiment.yaml   # Бюджет, сиды, параметры запуска (вкл. freedpp секция)
│   ├── calibration.yaml  # Калибровочный набор (MOSES) и параметры докинга
│   ├── normalization.yaml # Per-target mean/std/CI из калибровки (generated)
│   └── master_config.yaml # Freeze: версия, SHA256-хэши конфигов, data sources
├── data/targets/         # Скачанные PDB-структуры и FASTA-последовательности
│   ├── <pair_id>/{A,B}/  # PDB файлы и метаданные (по парам)
│   ├── <target_id>.fasta # Последовательности белков (UniProt)
│   └── <target_id>.pdbqt # Receptor PDBQT для Uni-Dock (генерируется --prepare-all)
├── docker/freedpp/       # Docker-изоляция FREED++ (Python 3.10 / PyTorch 1.12 / DGL 0.9)
│   ├── Dockerfile        # CUDA 11.8, Uni-Dock, RDKit+Meeko, патч dual reward
│   ├── entrypoint.sh     # prepare-receptor / train / sample
│   ├── dual_reward_patch.py  # Двойной reward: worst-case (max для lower-is-better)
│   ├── run_with_patch.py     # Запуск FREED++ с auto-apply dual reward patch
│   ├── pdbfixer_add_h.py    # Протонирование PDB через PDBFixer
│   └── data/                 # zinc_crem.json + alert_collections.csv (fetched)
├── cache/                # Кеш (в .gitignore)
│   ├── <target>.{ctx[:8]}.json  # Кеш скоров Boltz-2 (context-segmented, cache key v2)
│   ├── boltz_msa/        # Pre-computed MSA для offline-скоринга
│   │   ├── <target_id>.csv           # MSA файл (копия для быстрого доступа)
│   │   └── <target_id>/              # Полный вывод Boltz MSA warmup
│   └── freedpp_models/   # Кеш обученных моделей FREED++ (по hash параметров)
│       └── <hash>/       # Обученная модель (checkpoints, config)
├── docs/                 # Проектная документация
│   ├── README.md         # Карта достоверности документации
│   ├── goal.md
│   ├── tasks.md          # Living log выполненных задач
│   ├── selection_criteria.md
│   ├── preregistration.md # Frozen pre-registration гипотез и decision rules (SCI-01)
│   ├── receptor_qc.md    # Receptor QC sheet per target (SCI-03)
│   └── statistical_analysis_plan.md # Frozen SAP: inference, claim rules, code mapping (SCI-08)
├── src/dualdocking/      # Основная библиотека
│   ├── objectives.py     # Целевые функции (min, harmonic, weighted_sum, geometric, Tchebycheff, Pareto)
│   ├── scorer_boltz2.py  # Скоринг Boltz-2 через CLI (subprocess + disk cache)
│   ├── generator_freedpp.py  # Обёртка генератора FREED++ (Docker subprocess)
│   ├── binding_box.py    # Извлечение binding box из PDB (лиганд / fallback)
│   ├── chem_sanitize.py  # Валидация и дедупликация SMILES
│   ├── statistical.py    # Иерархический bootstrap, FDR, effect sizes
│   ├── config.py         # Загрузка YAML-конфигов (вкл. organism gate)
│   ├── cache_io.py       # Атомарный disk I/O (fcntl + temp-and-replace)
│   ├── run_manifest.py   # Lifecycle-трекинг запусков (run_manifest.json)
│   ├── fairness.py       # Compute parity и fairness метрики
│   ├── baseline.py       # Сравнение с baseline
│   ├── latex_report.py   # Генерация LaTeX-отчёта
│   ├── diversity.py      # Метрики химического разнообразия
│   ├── enrichment.py        # Docking-based enrichment metrics (AUROC, BEDROC, EF)
│   └── chembl_validation.py  # Валидация мишеней через ChEMBL (fetch actives + Tanimoto + enrichment)
├── scripts/              # CLI-скрипты
│   ├── stage0_fetch.py       # Stage 0: единая точка входа для ВСЕХ загрузок (offline после)
│   ├── fetch_freed_data.py    # Загрузка docker/freedpp/data/{zinc_crem.json,alert_collections.csv}
│   ├── fetch_baseline_zinc.py # Скачивание baseline archive zinc-druglike-cano.tar.xz
│   ├── fetch_chembl_actives.py # Загрузка активных соединений из ChEMBL
│   ├── prepare_data_dependencies.py # One-shot подготовка data deps (FREED + baseline/parquet/sample)
│   ├── prepare_targets.py    # Скачивание PDB, FASTA, pre-compute MSA, PDB→PDBQT
│   ├── generate.py           # Запуск FREED++ (single_a/single_b/double)
│   ├── score.py              # Скоринг молекул через Boltz-2
│   ├── boltz_smoke.py        # Smoke-тест Boltz-2 (3 молекулы, --dry-run)
│   ├── smoke_eng01.py        # Smoke-тест ENG-01 reward patch
│   ├── rdkit_embed.py        # 3D embedding через RDKit
│   ├── validate_redocking.py    # SCI-04: re-docking validation (RMSD check)
│   ├── enrichment_benchmark.py  # SCI-05: enrichment benchmark (AUROC/BEDROC per target)
│   ├── threshold_sensitivity.py # SCI-06: hit-threshold sensitivity (T=-1,0,+1)
│   ├── proxy_final_correlation.py # SCI-07: proxy vs final correlation (docking↔Boltz-2)
│   ├── aggregate.py          # Объединение результатов по парам
│   ├── plot_results.py       # Генерация графиков сравнения
│   ├── compute_metrics.py    # Вычисление метрик и стат. тестов
│   ├── render_molecules.py   # 2D-визуализация молекул с аннотациями
│   ├── summarize.py          # Мини-аналитика: текстовый отчёт
│   ├── run_pilot.py          # SCI-09: pilot matrix run (1 pair x 3 modes x 2 seeds)
│   └── claim_audit.py        # SCI-10: claim traceability audit (SAP → stats.json)
├── data/enrichment/      # Enrichment benchmark artifacts
│   ├── scores/           # Per-target active docking scores ({TARGET}_actives.csv)
│   ├── receptors/        # Downloaded PDBs and PDBQTs for missing targets
│   └── enrichment_results.json  # Full metadata + results
├── tests/                # Юнит-тесты
├── results/              # Результаты запусков (в .gitignore)
├── report/               # Артефакты итогового отчёта
│   ├── data/             # Агрегированные данные (generated)
│   ├── figures/          # Сгенерированные графики (generated)
│   ├── references.bib    # Библиография для LaTeX
│   ├── experiment_<run_id>_<utc>.log  # Логи run_full_experiment.py (generated)
│   ├── run_manifest_<run_id>.json     # Lifecycle-манифест запуска (generated)
│   ├── metrics.json      # (generated)
│   ├── stats.json        # (generated)
│   └── summary.txt       # (generated)
```

## Установка

```bash
# Клонирование
git clone <repo-url> && cd SWW2026

# Создание окружения и установка (требуется uv >= 0.9)
uv sync --group dev

# Boltz-2 уже включён в pyproject.toml (boltz[cuda] via git source).
# requirements-boltz.txt нужен только при ручной установке вне uv sync:
# uv pip install -r requirements-boltz.txt

# Prerequisites для FREED++ генератора:
# - Docker >= 20.x
# - nvidia-container-toolkit (для --gpus all)
# Подготовить файлы, которые копируются в образ FREED++
uv run python scripts/fetch_freed_data.py

# Сборка Docker-образа FREED++ (из корня репозитория)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# Проверка
uv run pytest
uv run ruff check src/ scripts/ docker/
```

## Калибровочный набор для нормализации

Нормализация reward в FREED++ требует внешнего SMILES-набора для расчёта mean/std по каждому таргету.
Для этого используется конфиг `configs/calibration.yaml` и скрипт загрузки.

Команды:

```bash
# Скачать/подготовить все наборы из configs/calibration.yaml
uv run python scripts/fetch_calibration_sets.py

# Или только выбранный набор (selected)
uv run python scripts/fetch_calibration_sets.py --only-selected

# Прогнать докинг калибровочного набора (Uni-Dock GPU)
uv run python scripts/dock_calibration.py

# Проверить/пересчитать normalization.yaml
uv run python scripts/calibrate_normalization.py
```

По умолчанию выбран MOSES (ZINC Clean Leads). Для ZINC15 необходимо указать собственный URL
в `configs/calibration.yaml` и включить источник, так как на ZINC могут действовать ограничения
на перераспространение.

Параметры докинга (exhaustiveness/num_modes/energy_range/timeout) задаются в
`configs/calibration.yaml` и применяются одинаково для всех таргетов.

## Baseline ZINC → Parquet

Для базовой выборки удобно хранить исходный ZINC в Parquet.
Скрипт `scripts/convert_baseline_to_parquet.py` конвертирует CSV (в т.ч. внутри `tar.xz`)
стримингово и без загрузки всего файла в RAM.

### One-shot подготовка data dependencies

```bash
# Только FREED docker data (zinc_crem.json + alert_collections.csv)
uv run python scripts/prepare_data_dependencies.py

# Полный baseline цикл: скачать tar.xz + конвертировать в parquet + сделать sample .smi
uv run python scripts/prepare_data_dependencies.py \
  --fetch-baseline --baseline-url "https://huggingface.co/datasets/Pixelatory/ZINC20-Druglike/resolve/main/zinc-druglike-cano.tar.xz?download=true" \
  --convert-baseline --sample-baseline
```

Скачивание архива baseline:

```bash
# Вариант 1: передать URL явно
uv run python scripts/fetch_baseline_zinc.py --url "https://<host>/zinc-druglike-cano.tar.xz"

# Вариант 2: через переменную окружения
export ZINC_BASELINE_URL="https://<host>/zinc-druglike-cano.tar.xz"
uv run python scripts/fetch_baseline_zinc.py
```

Пример для архива `zinc-druglike-cano.tar.xz` (CSV с колонкой `smiles`):

```bash
uv run python scripts/convert_baseline_to_parquet.py \
  --input data/baseline/zinc-druglike-cano.tar.xz \
  --output data/baseline/zinc-druglike-cano.parquet \
  --column smiles
```

Проверка:

```bash
uv run python - <<'PY'
import pyarrow.parquet as pq
pf = pq.ParquetFile("data/baseline/zinc-druglike-cano.parquet")
print("rows", pf.metadata.num_rows)
print("schema", pf.schema)
PY
```

### Baseline SMILES из Parquet

Чтобы получить baseline‑подвыборку (seed=42) из большого Parquet:

```bash
uv run python scripts/sample_baseline_from_parquet.py \
  --input data/baseline/zinc-druglike-cano.parquet \
  --output data/baseline/zinc_druglike_baseline_1000_seed42.smi \
  --n 1000 --seed 42 --column smiles
```

## Архитектура скоринга Boltz-2

Boltz-2 не имеет публичного Python API — взаимодействие только через CLI (`boltz predict`).

## LaTeX отчёт

Для генерации PDF нужен установленный LaTeX (pdflatex + bibtex).
На Arch Linux:

```bash
sudo pacman -S --needed texlive-core texlive-latexextra texlive-bibtexextra
```

Генерация:

```bash
# Черновик (.tex), главы берутся из report/chapters + manifest.yaml
uv run python scripts/generate_latex_report.py --mode draft --chapters-dir report/chapters --chapters-manifest report/chapters/manifest.yaml

# Полная компиляция (.pdf)
uv run python scripts/generate_latex_report.py --mode full
```

Выходные файлы: `report/manuscript.tex`, `report/manuscript.pdf`.

Модульность статьи:
- структура и порядок секций: `report/chapters/manifest.yaml`
- текст секций: `report/chapters/*.tex`
- численные таблицы и фигуры подставляются автоматически из `report/metrics.json`, `report/stats.json`, `report/figures/*.png`.

### Принцип: максимальное кеширование, работа без интернета

Все внешние данные выкачиваются один раз при `scripts/stage0_fetch.py` и кешируются локально:

| Данные | Расположение | Источник |
|--------|-------------|----------|
| Protein sequences | `data/targets/{target_id}.fasta` | UniProt REST API |
| PDB structures | `data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | RCSB |
| MSA данные | `cache/boltz_msa/{target_id}.csv` | ColabFold API через Boltz |
| Boltz checkpoints | `~/.boltz/` | Автоматически при первом запуске |
| Scoring cache | `cache/{target}.{ctx[:8]}.json` | Результаты Boltz per molecule (context-segmented) |

После `prepare_targets.py --prepare-all` весь пайплайн скоринга работает **полностью offline**.

### Organism gate

`load_pairs()` блокирует пары с не-человеческими мишенями (`organism != "Homo sapiens"`). Поле `organism` обязательно в `configs/pairs.yaml`. Пара COX1_COX2 заблокирована по этому критерию (COX1 PDB = Ovis aries, COX2 PDB = Mus musculus).

### Run manifest

Каждый запуск `run_full_experiment.py` создаёт lifecycle-манифест `report/run_manifest_{run_id}.json`, фиксирующий параметры, статус задач, таймстемпы начала/завершения и хэши конфигурации. Модуль `src/dualdocking/run_manifest.py` предоставляет API для записи и обновления манифеста.

### Конвенция скоров

- Boltz выдаёт `affinity_pred_value` = log10(IC50 в μM), меньше = сильнее связывание
- Наш пайплайн: **меньше = лучше** (конвенция совместима с docking-энергиями)
- Пример: IC50=1 nM = 0.001 μM → log10(IC50 μM) = -3
- Перевод в pIC50 (если нужно): `pIC50 = 6 - log10(IC50 μM)`
- В таблицах/коде `combined_min` = worst-case агрегация, т.е. `max(score_A, score_B)` для lower-is-better

### Входной YAML для Boltz

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MENFQKVEKIGE..."
      msa: cache/boltz_msa/CDK2.csv    # pre-computed MSA → offline
  - ligand:
      id: B
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'
properties:
  - affinity:
      binder: B
```

### Параметры CLI

```bash
# OFFLINE скоринг (MSA указан в YAML):
uv run boltz predict input.yaml \
  --out_dir cache/boltz_runs/{target_id}/{hash} \
  --model boltz2 --accelerator gpu \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 4 \
  --sampling_steps_affinity 200 --diffusion_samples_affinity 5 --override

# ONLINE (для prepare_targets.py --prepare-all):
uv run boltz predict warmup.yaml \
  --out_dir cache/boltz_msa/{target_id} \
  --model boltz2 --accelerator gpu --use_msa_server
```

### Защита от ошибок

| Проблема | Защита |
|----------|--------|
| Таймаут | 600 сек на молекулу, subprocess timeout |
| OOM GPU | Детекция "CUDA out of memory" в stderr → NaN |
| Большие лиганды | Проверка ≤ 128 атомов перед вызовом |
| Невалидный SMILES | RDKit проверка → NaN |
| Нет сети при скоринге | MSA pre-computed, сеть не нужна |
| Нестабильный MSA сервер | Retry с backoff при prepare (3 попытки) |
| Прерывание | Инкрементальный кеш после каждой молекулы |

## FREED++ (генератор молекул)

FREED++ — RL-генератор молекул (SAC + фрагментная сборка), работающий через Docker из-за несовместимости зависимостей с основным окружением проекта.

### Prerequisites

- Docker >= 20.x
- nvidia-container-toolkit (`--gpus all` для GPU-ускорения обучения)

### Сборка Docker-образа

```bash
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

Если менялись файлы в `docker/freedpp/` (например `dual_reward_patch.py` или `run_with_patch.py`), образ нужно пересобрать перед следующим запуском генерации.

### Архитектура

```
scripts/generate.py
  → generator_freedpp.generate(config)
      1. _ensure_docker_image()       # собрать образ при отсутствии
      2. _prepare_receptor()          # PDB → PDBQT + binding box
      3. _run_freedpp()               # docker run --gpus all ... python main.py
      4. _parse_output_smiles()       # CSV из смонтированного volume
      → list[str] SMILES
```

- **Reward при обучении**: Uni-Dock через патч `dual_reward_patch.py` (backend подготовки лиганда: RDKit+Meeko)
- **Интерпретация**: docking affinity в ккал/моль; более отрицательные значения = более сильное связывание
- **Финальный скоринг**: Boltz-2 (`scorer_boltz2.py`, GPU; время зависит от параметров и железа)

### Single vs Double режим

- **single_a**: receptor A is primary (generation + reward); B is not scored during training
- **single_b**: receptor B is primary (generation + reward); A is not scored during training (clean baseline)
- **double**: receptor A is primary for generation; both A and B scored; reward = worst-case `max(z_a, z_b)` (via `dual_reward_patch.py`)

### Кеширование обученных моделей

Обученные модели сохраняются в `cache/freedpp_models/{hash}/`. Ключ хэша включает версию схемы кэша, режим reward, таргеты и параметры обучения (`epochs`, `steps_per_epoch`, `num_sub_proc`, `seed`, `vina_program`, `starting_smile`). При повторном запуске с теми же параметрами используется `--commands "sample"` (минуты вместо часов).

### Smoke test

```bash
uv run python scripts/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10
```

## Stage 0: Fetch (единая точка входа для всех загрузок)

**Всё что требует интернета** собрано в одном скрипте `scripts/stage0_fetch.py`.
После успешного выполнения дальнейшие стадии работают **полностью offline**.

```bash
# Полная загрузка всех зависимостей (интернет нужен только здесь)
uv run python scripts/stage0_fetch.py

# С baseline ZINC (опционально, нужен URL)
uv run python scripts/stage0_fetch.py --baseline-url "https://<host>/zinc-druglike-cano.tar.xz"

# Без Docker-сборки (если образ уже есть)
uv run python scripts/stage0_fetch.py --skip-docker-build

# Только проверить что всё скачано
uv run python scripts/stage0_fetch.py --verify-only

# Dry-run: показать команды без выполнения
uv run python scripts/stage0_fetch.py --dry-run
```

### Что скачивает Stage 0

| Шаг | Что | Откуда | Куда | Требуется |
|-----|-----|--------|------|-----------|
| 1 | Python-пакеты | PyPI, GitHub | `.venv/` | `uv sync` |
| 2 | FREED data files | GitHub raw | `docker/freedpp/data/` | Для Docker build |
| 3 | Docker-образ FREED++ | NVIDIA, PyPI, conda-forge, GitHub | Docker cache | GPU + nvidia-toolkit |
| 4 | Calibration SMILES | HuggingFace | `data/calibration/*.smi` | Для нормализации reward |
| 5 | Baseline ZINC | URL (user) | `data/baseline/*.tar.xz` | Опционально |
| 6 | PDB структуры | RCSB | `data/targets/` | Для всех 9 пар |
| 7 | FASTA | UniProt | `data/targets/*.fasta` | Для MSA |
| 8 | MSA | ColabFold API | `cache/boltz_msa/*.csv` | Для offline Boltz-2 |
| 9 | PDB→PDBQT | Docker (local) | `data/targets/*.pdbqt` | Для UniDock GPU |

### Pilot Run (SCI-09)

Quick end-to-end check with 1 pair x 3 modes x 2 seeds before committing to the full matrix:

```bash
# Default: CDK2_ERK2, seeds 42+137, Stage 1+2 + artifact verification
uv run python scripts/run_pilot.py

# Dry run
uv run python scripts/run_pilot.py --dry-run

# Custom pair/seeds
uv run python scripts/run_pilot.py --pair AChE_BChE --seeds 42,256

# Verify-only (skip generation, just check existing artifacts)
uv run python scripts/run_pilot.py --verify-only
```

`run_full_experiment.py` now also supports `--pair` and `--seeds` filters:

```bash
# Run only CDK2_ERK2 with 2 seeds
uv run python scripts/run_full_experiment.py --pair CDK2_ERK2 --seeds 42,137 --resume --with-post
```

### После Stage 0: полный pipeline offline

```bash
# Stage 0.5: Calibration + Validation (offline, нужен GPU)
uv run python scripts/dock_calibration.py
uv run python scripts/calibrate_normalization.py
uv run python scripts/validate_redocking.py        # SCI-04
uv run python scripts/enrichment_benchmark.py       # SCI-05

# Stage 1: Generate + Score (offline)
EXPERIMENT_RUN_ID=run01 uv run python scripts/run_full_experiment.py --resume --parallel 2

# Stage 2: Analysis (offline)
EXPERIMENT_RUN_ID=run01 uv run python scripts/run_post_pipeline.py --latex-mode draft
```

## Воспроизведение (end-to-end для одной пары)

```bash
# 0. Stage 0: скачать всё необходимое (ОНЛАЙН)
uv run python scripts/stage0_fetch.py

# 1. Подготовить таргеты (уже включено в Stage 0, можно пропустить)
# uv run python scripts/prepare_targets.py --pair CDK2_ERK2 --prepare-all

# 2. Проверить что данные на месте
ls data/targets/CDK2.fasta data/targets/ERK2.fasta
ls cache/boltz_msa/CDK2.csv cache/boltz_msa/ERK2.csv

# 3. Smoke-тест Boltz-2 (3 молекулы, OFFLINE)
uv run python scripts/boltz_smoke.py --target CDK2

# 3a. Dry-run: проверить YAML без запуска Boltz
uv run python scripts/boltz_smoke.py --target CDK2 --dry-run

# 4. Генерация молекул (single-target A)
uv run python scripts/generate.py --pair CDK2_ERK2 --mode single_a

# 5. Скоринг по обеим мишеням (OFFLINE)
uv run python scripts/score.py --pair CDK2_ERK2 --mode single_a

# 6. Генерация молекул (single-target B)
uv run python scripts/generate.py --pair CDK2_ERK2 --mode single_b

# 7. Скоринг single_b молекул
uv run python scripts/score.py --pair CDK2_ERK2 --mode single_b

# 8. Генерация молекул (double-target)
uv run python scripts/generate.py --pair CDK2_ERK2 --mode double

# 9. Скоринг double-target молекул
uv run python scripts/score.py --pair CDK2_ERK2 --mode double

# 10. Агрегация и анализ
uv run python scripts/aggregate.py
uv run python scripts/compute_metrics.py
uv run python scripts/plot_results.py
uv run python scripts/render_molecules.py
uv run python scripts/summarize.py
```

Оркестрация разделена на 2 этапа:

```bash
# Stage 1: generate + score
# По умолчанию запускает scripts/prepare_data_dependencies.py (минимум: FREED data files)
EXPERIMENT_RUN_ID=run_2026-02-05 uv run python scripts/run_full_experiment.py --resume --parallel 4

# Опционально: вместе со Stage 1 скачать baseline archive
EXPERIMENT_RUN_ID=run_2026-02-05 uv run python scripts/run_full_experiment.py --resume --parallel 4 \
  --prepare-baseline --baseline-url "https://<host>/zinc-druglike-cano.tar.xz"

# Stage 2: aggregate + metrics + plots + summary + LaTeX draft
EXPERIMENT_RUN_ID=run_2026-02-05 uv run python scripts/run_post_pipeline.py --latex-mode draft
```

Полезные режимы:

```bash
# Stage 1 + Stage 2 одной командой (опционально)
EXPERIMENT_RUN_ID=run_2026-02-05 uv run python scripts/run_full_experiment.py --resume --parallel 4 --with-post --latex-mode draft

# Полная сборка PDF на этапе отчёта (требуется LaTeX в системе)
EXPERIMENT_RUN_ID=run_2026-02-05 uv run python scripts/run_post_pipeline.py --latex-mode full
```

Логи:
- Stage 1: `report/experiment_<EXPERIMENT_RUN_ID>_<UTC_TIMESTAMP>.log`
- Stage 2: `report/post_<EXPERIMENT_RUN_ID>_<UTC_TIMESTAMP>.log`

### Зачем несколько seed

По умолчанию берутся сиды из `configs/experiment.yaml` (`[42, 137, 256, 512, 1024]`):
- это снижает риск случайного вывода из одного удачного/неудачного запуска;
- метрики и статистика (`scripts/compute_metrics.py`) считаются по независимым сидам.

Для smoke/диагностики используйте один сид и маленький бюджет (`--epochs`, `--n-molecules`, `--steps-per-epoch`).

## Troubleshooting

### FREED++: повторяющиеся `|z_a| > 10` / `|z_b| > 10`

Симптом:
- в логе `freedpp_workspace/freedpp_*.log` много строк вида `dual_reward_patch: |z_a| > 10 (...)`
- reward деградирует, результаты выглядят неинформативно

Типичные причины:
- локальный Docker-образ не пересобран после изменений в `docker/freedpp/*`
- используется устаревший кэш модели, обученный на старом reward backend

Проверка и исправление:

```bash
# 1) Пересобрать образ после любых правок docker/freedpp/*
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# 2) Проверить, что у образа есть локальный image id (для локальных билдов это нормально)
docker image inspect dualdocking/freedpp:latest --format '{{.Id}}'

# 3) Для диагностического прогона отключить кэш моделей
# configs/experiment.yaml -> generator.freedpp.use_model_cache: false
```

Примечание:
- если `docker image inspect ... --format '{{index .RepoDigests 0}}'` пустой, это ожидаемо для локально собранного образа без push в registry.

## Запуск по всем парам

```bash
# Подготовка (однократно, нужен интернет)
uv run python scripts/prepare_targets.py --prepare-all

# Генерация и скоринг (offline)
for pair in CDK2_ERK2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 EGFR_HER2 BACE1_CDK5 AChE_MAOB; do
  for mode in single_a single_b double; do
    uv run python scripts/generate.py --pair $pair --mode $mode
    uv run python scripts/score.py --pair $pair --mode $mode
  done
done

uv run python scripts/aggregate.py
uv run python scripts/compute_metrics.py
uv run python scripts/plot_results.py
uv run python scripts/render_molecules.py
uv run python scripts/summarize.py
```

## Структура результатов

```
results/
├── smoke/                    # Результаты smoke-теста
│   ├── boltz_score.json      # Скоры 3 молекул
│   └── dry_run/              # YAML файлы (--dry-run)
├── CDK2_ERK2/
│   ├── single_a/
│   │   └── ...
│   ├── single_b/
│   │   └── ...
│   └── double/
│       └── ...               # generated.csv, scores_A.csv, scores_B.csv, scored_joined.csv
├── AChE_BChE/ ...
├── BACE1_GSK3B/ ...
├── AChE_BACE1/ ...
├── GSK3B_CDK5/ ...
├── EGFR_HER2/ ...           # auxiliary
├── BACE1_CDK5/ ...           # auxiliary
└── AChE_MAOB/ ...            # auxiliary
# COX1_COX2 blocked (non-human organisms) — не генерируется

report/
├── chapters/                  # Редактируемые главы статьи (.tex), не generated
│   ├── manifest.yaml
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methods_notes.tex
│   ├── 04_experimental_protocol.tex
│   ├── 05_results_scope.tex
│   ├── 06_limitations.tex
│   └── 07_conclusion.tex
├── data/
│   ├── all_results.parquet
│   ├── redocking_metrics.csv      # SCI-04 re-docking RMSD per target
│   └── enrichment_metrics.csv     # SCI-05 AUROC/BEDROC/EF per target
├── figures/
│   ├── enrichment/
│   │   ├── roc_{TARGET}.png           # Per-target ROC curves
│   │   └── enrichment_summary.png     # AUROC + BEDROC bar chart
│   ├── {pair}_molecules_single_a.png
│   ├── {pair}_molecules_single_b.png
│   ├── {pair}_molecules_double.png
│   └── {pair}_molecules_top_hits.png
├── metrics.json
├── stats.json
└── summary.txt              # Текстовый отчёт с аналитикой
```

## Validation Pipeline (SCI-03 → SCI-05)

Перед запуском экспериментов выполняется серия проверок качества docking setup:

### SCI-03: Receptor QC

Документирует качество каждой PDB-структуры (разрешение, организм, наличие лиганда, кофактора).

### SCI-04: Re-docking Validation

Валидация docking setup: со-кристаллизованный лиганд докируется обратно, RMSD ≤ 2.0 Å = pass.

```bash
uv run python scripts/validate_redocking.py
uv run python scripts/validate_redocking.py --target-id BACE1 --dry-run
```

Выход: `data/validation/redocking_results.json`, `report/data/redocking_metrics.csv`.

### SCI-05: Enrichment Benchmark

Проверяет, что UniDock способен отличить ChEMBL-активные соединения от druglike декоев (MOSES).
Для каждого таргета вычисляется AUROC, BEDROC(α=20) и EF@1%/5%.

Пороги pass/fail задаются в `configs/metrics.yaml` → `enrichment_quality`:
- AUROC > 0.6 (выше random = 0.5)
- BEDROC(α=20) > 0.3 (значимое раннее обогащение)

```bash
# Dry run: проверить доступность данных
uv run python scripts/enrichment_benchmark.py --dry-run

# Полный запуск (Docker + GPU)
uv run python scripts/enrichment_benchmark.py

# Конкретные таргеты
uv run python scripts/enrichment_benchmark.py --target-id CDK2 --target-id BACE1

# Только метрики (без докинга, по уже имеющимся скорам)
uv run python scripts/enrichment_benchmark.py --skip-docking
```

Выход:
- `data/enrichment/enrichment_results.json` — полный JSON с метаданными
- `report/data/enrichment_metrics.csv` — таблица (target, AUROC, BEDROC, EF, status)
- `report/figures/enrichment/roc_{TARGET}.png` — ROC-кривые
- `report/figures/enrichment/enrichment_summary.png` — сводная гистограмма

### SCI-07: Proxy vs Final Correlation

Spearman ρ и Pearson r между UniDock docking (proxy reward) и Boltz-2 affinity (final scorer) per pair/mode/target. Молекулы из FREED++ training logs join-ятся с Boltz-2 scores по каноническому SMILES.

```bash
uv run python scripts/proxy_final_correlation.py
```

Выход:
- `report/data/proxy_final_corr.csv` — плоская таблица (pair × mode × target × correlation)
- `report/data/proxy_final_corr.json` — JSON со сводкой proxy suitability

### SCI-06: Threshold Sensitivity

Проверяет устойчивость primary conclusion к выбору порога хита.
Пре-зарегистрированные пороги: T = -1.0 (100 nM), 0.0 (1 μM), +1.0 (10 μM).

```bash
uv run python scripts/threshold_sensitivity.py
```

Выход:
- `report/data/threshold_sensitivity.csv` — плоская таблица (threshold × pair × mode)
- `report/data/threshold_sensitivity.json` — структурированный JSON со стат-тестами и robustness summary

## Ключевые ссылки

- **FREED++**: [arXiv:2401.09840](https://arxiv.org/abs/2401.09840) -- RL-генерация молекул с docking-наградой
- **FREED (original)**: [arXiv:2110.01219](https://arxiv.org/abs/2110.01219) -- Fragment-based RL
- **Репозиторий FREED++**: [AIRI-Institute/FFREED](https://github.com/AIRI-Institute/FFREED)
- **Boltz-2**: [boltz-community](https://github.com/jwohlwend/boltz) -- Совместное предсказание структуры и аффинности связывания (CLI only)
- **RDKit**: [rdkit.org](https://www.rdkit.org/) -- Хемоинформатика
