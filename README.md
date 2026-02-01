# Воркшоп: Dual-Target Docking

Сравнение single-target и double-target молекулярной генерации с использованием **FREED++** (RL-генератор) и **Boltz-2** (structure-aware скорер).

## Гипотеза

RL-генерация, оптимизированная под worst-case цель (`max(score_A, score_B)` при *lower-is-better*), порождает молекулы с более высоким dual hit-rate, чем оптимизация под одну мишень.

## Структура проекта

```
├── configs/              # Конфигурация эксперимента
│   ├── pairs.yaml        # 4 пары белковых мишеней
│   ├── metrics.yaml      # Пороги хитов и определения метрик
│   └── experiment.yaml   # Бюджет, сиды, параметры запуска (вкл. freedpp секция)
├── data/targets/         # Скачанные PDB-структуры и FASTA-последовательности
│   ├── <pair_id>/{A,B}/  # PDB файлы и метаданные (по парам)
│   ├── <target_id>.fasta # Последовательности белков (UniProt)
│   └── <target_id>.pdbqt # Receptor PDBQT для QuickVina2 (генерируется --prepare-all)
├── docker/freedpp/       # Docker-изоляция FREED++ (Python 3.7 / PyTorch 1.12)
│   ├── Dockerfile        # CUDA 11.3, Conda, DGL 0.9, OpenBabel, QuickVina2
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
│   ├── goal.md
│   ├── acceleration.md
│   ├── resources.md
│   ├── selection_criteria.md
│   └── multi_objective_review.md
├── src/dualdocking/      # Основная библиотека
│   ├── objectives.py     # Объединённые целевые функции (min, harmonic, Pareto)
│   ├── scorer_boltz2.py  # Скоринг Boltz-2 через CLI (subprocess + disk cache)
│   ├── generator_freedpp.py  # Обёртка генератора FREED++ (Docker subprocess)
│   ├── binding_box.py    # Извлечение binding box из PDB (лиганд / fallback)
│   ├── chem_sanitize.py  # Валидация и дедупликация SMILES
│   ├── metrics.py        # Метрики двойных хитов и стат. тесты
│   └── config.py         # Загрузка YAML-конфигов
├── scripts/              # CLI-скрипты
│   ├── prepare_targets.py    # Скачивание PDB, FASTA, pre-compute MSA, PDB→PDBQT
│   ├── generate.py           # Запуск FREED++ (single/double режим)
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
│   └── molecules/        # 2D-рендер молекул
└── cache/                # Кеш скоров и MSA (в .gitignore)
```

## Установка

```bash
# Клонирование
git clone <repo-url> && cd dualdocking

# Создание окружения и установка (требуется uv >= 0.9)
uv sync --extra dev

# Установка Boltz-2 (отдельно, из-за пина scipy==1.13.1)
uv pip install -r requirements-boltz.txt

# Prerequisites для FREED++ генератора:
# - Docker >= 20.x
# - nvidia-container-toolkit (для --gpus all)
# Сборка Docker-образа FREED++:
docker build -t dualdocking/freedpp:latest docker/freedpp/

# Проверка
uv run pytest
uv run ruff check src/ scripts/ tests/
```

## Архитектура скоринга Boltz-2

Boltz-2 не имеет публичного Python API — взаимодействие только через CLI (`boltz predict`).

### Принцип: максимальное кеширование, работа без интернета

Все внешние данные выкачиваются один раз при `prepare_targets.py --prepare-all` и кешируются локально:

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

FREED++ — RL-генератор молекул (SAC + фрагментная сборка), использующий QuickVina2 docking score как reward. Работает через Docker из-за несовместимости зависимостей (Python 3.7 / PyTorch 1.12 / DGL 0.9).

### Prerequisites

- Docker >= 20.x
- nvidia-container-toolkit (`--gpus all` для GPU-ускорения обучения)

### Сборка Docker-образа

```bash
docker build -t dualdocking/freedpp:latest docker/freedpp/
```

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

- **Reward при обучении**: QuickVina2 (встроен в FREED++, ~1 сек/мол, CPU)
- **Интерпретация**: docking affinity в ккал/моль; более отрицательные значения = более сильное связывание
- **Финальный скоринг**: Boltz-2 (`scorer_boltz2.py`, ~58 сек/мол, GPU)

### Single vs Double режим

- **Single**: один receptor → стандартный FREED++ reward
- **Double**: два receptors → worst-case reward `max(score_a, score_b)` для *lower-is-better* (через `dual_reward_patch.py`, monkey-patch `DockingVina.__call__`)

### Кеширование обученных моделей

Обученные модели сохраняются в `cache/freedpp_models/{hash}/`, где hash = SHA256 от (target_id, second_target_id, reward_mode, epochs, seed, starting_smile). При повторном запуске с теми же параметрами используется `--commands "sample"` (минуты вместо часов).

### Smoke test

```bash
uv run python scripts/generate.py --pair CDK2_ERK2 --mode single --epochs 2 --n-molecules 10
```

## Воспроизведение (end-to-end для одной пары)

```bash
# 0. Собрать Docker-образ FREED++ (однократно)
docker build -t dualdocking/freedpp:latest docker/freedpp/

# 1. Подготовить всё (ОНЛАЙН: PDB, FASTA, MSA, Boltz checkpoints, PDB→PDBQT)
uv run python scripts/prepare_targets.py --pair CDK2_ERK2 --prepare-all

# 2. Проверить что данные на месте
ls data/targets/CDK2.fasta data/targets/ERK2.fasta
ls cache/boltz_msa/CDK2.csv cache/boltz_msa/ERK2.csv

# 3. Smoke-тест Boltz-2 (3 молекулы, OFFLINE)
uv run python scripts/boltz_smoke.py --target CDK2

# 3a. Dry-run: проверить YAML без запуска Boltz
uv run python scripts/boltz_smoke.py --target CDK2 --dry-run

# 4. Генерация молекул (single-target)
uv run python scripts/generate.py --pair CDK2_ERK2 --mode single

# 5. Скоринг по обеим мишеням (OFFLINE)
uv run python scripts/score.py --pair CDK2_ERK2 --mode single

# 6. Генерация молекул (double-target)
uv run python scripts/generate.py --pair CDK2_ERK2 --mode double

# 7. Скоринг double-target молекул
uv run python scripts/score.py --pair CDK2_ERK2 --mode double

# 8. Агрегация и анализ
uv run python scripts/aggregate.py
uv run python scripts/compute_metrics.py
uv run python scripts/plot_results.py
uv run python scripts/render_molecules.py
uv run python scripts/summarize.py
```

## Запуск по всем парам

```bash
# Подготовка (однократно, нужен интернет)
uv run python scripts/prepare_targets.py --prepare-all

# Генерация и скоринг (offline)
for pair in CDK2_ERK2 EGFR_HER2 COX1_COX2 AChE_BChE; do
  for mode in single double; do
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
│   ├── single/
│   │   ├── generated.csv     # SMILES
│   │   ├── scores_A.csv      # Скоры Boltz-2 по мишени A (log10 IC50, μM)
│   │   ├── scores_B.csv      # Скоры Boltz-2 по мишени B (log10 IC50, μM)
│   │   └── scored_joined.csv # Объединённая таблица
│   └── double/
│       └── ...               # Та же структура
├── EGFR_HER2/ ...
├── COX1_COX2/ ...
└── AChE_BChE/ ...

report/
├── data/
│   └── all_results.parquet
├── figures/
│   ├── {pair}_molecules_single.png
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
