# Воркшоп: Dual-Target Docking

Сравнение single-target baseline режимов (`single_a`, `single_b`) и `double` режима генерации с использованием **FREED++** (RL-генератор) и **Boltz-2** (structure-aware скорер).

## Гипотеза

RL-генерация, оптимизированная под worst-case цель (`max(score_A, score_B)` при *lower-is-better*), порождает молекулы с более высоким dual hit-rate, чем оптимизация под одну мишень.

## Структура проекта

```
├── configs/              # Конфигурация эксперимента
│   ├── pairs.yaml        # 9 пар белковых мишеней (4 non-AD + 5 AD)
│   ├── metrics.yaml      # Пороги хитов и определения метрик
│   └── experiment.yaml   # Бюджет, сиды, параметры запуска (вкл. freedpp секция)
├── data/targets/         # Скачанные PDB-структуры и FASTA-последовательности
│   ├── <pair_id>/{A,B}/  # PDB файлы и метаданные (по парам)
│   ├── <target_id>.fasta # Последовательности белков (UniProt)
│   └── <target_id>.pdbqt # Receptor PDBQT для Uni-Dock (генерируется --prepare-all)
├── docker/freedpp/       # Docker-изоляция FREED++ (Python 3.10 / PyTorch 1.12 / DGL 0.9)
│   ├── Dockerfile        # CUDA 11.8, Uni-Dock, RDKit+Meeko, патч dual reward
│   ├── entrypoint.sh     # prepare-receptor / train / sample
│   └── dual_reward_patch.py  # Двойной reward: worst-case (max для lower-is-better)
├── cache/                # Кеш (в .gitignore)
│   ├── <target_id>.json  # Кеш скоров Boltz-2 (per molecule)
│   ├── boltz_msa/        # Pre-computed MSA для offline-скоринга
│   │   ├── <target_id>.csv           # MSA файл (копия для быстрого доступа)
│   │   └── <target_id>/              # Полный вывод Boltz MSA warmup
│   └── freedpp_models/   # Кеш обученных моделей FREED++ (по hash параметров)
│       └── <hash>/       # Обученная модель (checkpoints, config)
├── docs/                 # Проектная документация
│   ├── README.md         # Карта достоверности документации
│   ├── goal.md
│   ├── tasks.md
│   ├── selection_criteria.md
│   └── ...               # При необходимости: исследовательские заметки
├── src/dualdocking/      # Основная библиотека
│   ├── objectives.py     # Объединённые целевые функции (min, harmonic, Pareto)
│   ├── scorer_boltz2.py  # Скоринг Boltz-2 через CLI (subprocess + disk cache)
│   ├── generator_freedpp.py  # Обёртка генератора FREED++ (Docker subprocess)
│   ├── binding_box.py    # Извлечение binding box из PDB (лиганд / fallback)
│   ├── chem_sanitize.py  # Валидация и дедупликация SMILES
│   ├── statistical.py    # Иерархический bootstrap, FDR, effect sizes
│   └── config.py         # Загрузка YAML-конфигов
├── scripts/              # CLI-скрипты
│   ├── stage0_fetch.py       # Stage 0: единая точка входа для ВСЕХ загрузок (offline после)
│   ├── fetch_freed_data.py    # Загрузка docker/freedpp/data/{zinc_crem.json,alert_collections.csv}
│   ├── fetch_baseline_zinc.py # Скачивание baseline archive zinc-druglike-cano.tar.xz
│   ├── prepare_data_dependencies.py # One-shot подготовка data deps (FREED + baseline/parquet/sample)
│   ├── prepare_targets.py    # Скачивание PDB, FASTA, pre-compute MSA, PDB→PDBQT
│   ├── generate.py           # Запуск FREED++ (single_a/single_b/double)
│   ├── score.py              # Скоринг молекул через Boltz-2
│   ├── boltz_smoke.py        # Smoke-тест Boltz-2 (3 молекулы, --dry-run)
│   ├── aggregate.py          # Объединение результатов по парам
│   ├── plot_results.py       # Генерация графиков сравнения
│   ├── compute_metrics.py    # Вычисление метрик и стат. тестов
│   ├── render_molecules.py   # 2D-визуализация молекул с аннотациями
│   └── summarize.py          # Мини-аналитика: текстовый отчёт
├── tests/                # Юнит-тесты
├── results/              # Результаты запусков (в .gitignore)
├── report/               # Артефакты итогового отчёта
│   ├── data/             # Агрегированные данные
│   ├── figures/          # Сгенерированные графики
│   ├── experiment_<run_id>_<utc>.log  # Логи run_full_experiment.py
│   ├── metrics.json
│   ├── stats.json
│   └── summary.txt
```

## Установка

```bash
# Клонирование
git clone <repo-url> && cd SWW2026

# Создание окружения и установка (требуется uv >= 0.9)
uv sync --extra dev

# Установка Boltz-2 (отдельно, из-за пина scipy==1.13.1)
uv pip install -r requirements-boltz.txt

# Prerequisites для FREED++ генератора:
# - Docker >= 20.x
# - nvidia-container-toolkit (для --gpus all)
# Подготовить файлы, которые копируются в образ FREED++
uv run python scripts/fetch_freed_data.py

# Сборка Docker-образа FREED++ (из корня репозитория)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# Проверка
uv run pytest
uv run ruff check src/ scripts/ tests/
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
| Scoring cache | `cache/{target_id}.json` | Результаты Boltz per molecule |

После `prepare_targets.py --prepare-all` весь пайплайн скоринга работает **полностью offline**.

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
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
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

### После Stage 0: полный pipeline offline

```bash
# Stage 0.5: Calibration (offline, нужен GPU)
uv run python scripts/dock_calibration.py
uv run python scripts/calibrate_normalization.py

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
for pair in CDK2_ERK2 EGFR_HER2 COX1_COX2 AChE_BChE; do
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
├── EGFR_HER2/ ...
├── COX1_COX2/ ...
└── AChE_BChE/ ...

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
│   └── all_results.parquet
├── figures/
│   ├── {pair}_molecules_single_a.png
│   ├── {pair}_molecules_single_b.png
│   ├── {pair}_molecules_double.png
│   └── {pair}_molecules_top_hits.png
├── metrics.json
├── stats.json
└── summary.txt              # Текстовый отчёт с аналитикой
```

## Ключевые ссылки

- **FREED++**: [arXiv:2401.09840](https://arxiv.org/abs/2401.09840) -- RL-генерация молекул с docking-наградой
- **FREED (original)**: [arXiv:2110.01219](https://arxiv.org/abs/2110.01219) -- Fragment-based RL
- **Репозиторий FREED++**: [AIRI-Institute/FFREED](https://github.com/AIRI-Institute/FFREED)
- **Boltz-2**: [boltz-community](https://github.com/jwohlwend/boltz) -- Совместное предсказание структуры и аффинности связывания (CLI only)
- **RDKit**: [rdkit.org](https://www.rdkit.org/) -- Хемоинформатика
