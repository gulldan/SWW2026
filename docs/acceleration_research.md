# Исследование ускорения и визуализации dual-docking пайплайна

> Дата: 2026-02-01. Источники: GitHub issues, arXiv, бенчмарки 2024-2026.

---

## Содержание

1. [Диагностика: почему Uni-Dock не помогает](#1-диагностика)
2. [Корневая причина: FREED++ вызывает докинг последовательно](#2-корневая-причина)
3. [Подготовка молекул: OpenBabel vs RDKit+Meeko](#25-подготовка-молекул-openbabel-vs-rdkitmeeko)
4. [Решение A: Батчирование reward в FREED++](#3-решение-a-батчирование)
5. [Решение B: Замена генератора (Saturn, REINVENT 4, GraphGA)](#4-решение-b-замена-генератора)
6. [Ускорение Boltz-2 скоринга (Boltzina + иерархический пайплайн)](#5-ускорение-boltz-2)
7. [Визуализация dual-docking результатов](#6-визуализация)
8. [Рекомендации и приоритеты](#7-рекомендации)

---

## 1. Диагностика

### Текущее состояние

Uni-Dock **уже интегрирован** в проект:
- Установлен в Docker (`/usr/local/bin/unidock`)
- Настроен как дефолт в `configs/experiment.yaml`: `vina_program: unidock`
- `generator_freedpp.py` поддерживает переключение `qvina02` / `unidock`

**Но скорость не выросла.** GPU по-прежнему простаивает. Почему?

### Замеренные bottleneck'и

| Этап | Инструмент | Скорость | Где работает | Статус |
|------|-----------|----------|-------------|--------|
| RL reward (FREED++) | Uni-Dock (настроен) | ~1 сек/мол* | GPU **недоиспользуется** | **Bottleneck** |
| 3D конформер | RDKit (ETKDG) | ~0.1-0.5 сек/мол | CPU only | Второстепенный |
| Финальный скоринг | Boltz-2 | ~16-58 сек/мол | GPU | Медленно для >1000 мол |
| SMILES валидация | RDKit | < 0.01 сек/мол | CPU | Не проблема |

*Uni-Dock вызывается по одной молекуле — GPU kernel launch overhead доминирует.

---

## 2. Корневая причина

### FREED++ вызывает докинг **по одной молекуле** последовательно

Из `dual_reward_patch.py`:
```python
def _dual_call(self, *args, **kwargs):
    score_a = _original_call(self, *args, **kwargs)  # 1 молекула → Uni-Dock
    # ... swap receptor ...
    score_b = _original_call(self, *args, **kwargs)  # 1 молекула → Uni-Dock
    return max(score_a, score_b)
```

**Проблема**: Uni-Dock даёт 1000x ускорение при **батчах 64-256+ молекул**. При вызове по одной молекуле:
- GPU kernel launch overhead (~10-50 мс) сопоставим со временем самого докинга
- GPU загружен на <5% — основное время уходит на запуск/остановку CUDA kernels
- Фактически получается скорость как у CPU Vina или даже хуже (overhead)

Бенчмарки Uni-Dock подтверждают: throughput **0.10 сек/мол** достигается при батчах 100+ молекул. При батче из 1 молекулы — overhead до ~0.5-1 сек.

### Дополнительные последовательные операции

Помимо докинга, для каждой молекулы FREED++ последовательно выполняет:
1. **RDKit** 3D generation (CPU, ~0.1-0.5 сек)
2. **PDBQT конвертация** лиганда (CPU)
3. **Uni-Dock** вызов (GPU, но по 1 молекуле — не эффективно)
4. В dual mode — всё × 2 (два таргета)

Итого: ~2-4 сек на молекулу в dual mode, при том что GPU простаивает ~95% времени.

---

## 2.5 Подготовка молекул: OpenBabel vs RDKit+Meeko

**Короткий вывод для проекта:** RDKit+Meeko надёжнее для конверсии SMILES → PDBQT и воспроизводимости, а OpenBabel остаётся полезен для протонирования и быстрых конверсий. Практичный компромисс — гибридный пайплайн: protonation (OpenBabel/Dimorphite‑DL) → RDKit (3D) → Meeko (PDBQT).

**Почему:**
- Meeko прямо отмечает, что восстановление bond orders из PDBQT ненадёжно: формат не хранит bond order и H на углеродах, из‑за чего обратные конверсии могут быть неоднозначными.
- В подготовке рецепторов Meeko требует точного совпадения тяжёлых атомов с шаблонами и различает протонированные состояния (HIS, N‑term и др.), поэтому нужен контролируемый препроцессинг PDB.
- В DockStream RDKit используется как базовый инструмент подготовки и хранения лигандов, что отражает современную практику пайплайнов.
- SwissDock 2024 использует OpenBabel и RDKit для конвертации входных форматов (PDBQT/SMILES/Mol2), подтверждая жизнеспособность гибридного стека в продакшене.
- OpenBabel может выдавать предупреждения о проблемах кекулизации отдельных ароматических структур, что добавляет неопределённость при конвертациях.

## 3. Решение A: Батчирование reward в FREED++

### Суть: накапливать молекулы и скорить батчами

```
# Текущий подход (по 1 молекуле):
for mol in generated_mols:              # последовательно
    score = unidock(mol, receptor)      # GPU launch overhead доминирует
    # ~1 сек/мол из-за overhead

# Нужный подход (батчами):
batch = collect_mols(generated_mols, batch_size=128)
scores = unidock_batch(batch, receptor)  # 1 GPU launch на 128 мол
# ~0.1 сек/мол — реальный throughput Uni-Dock
```

### Сложность реализации: ВЫСОКАЯ

FREED++ использует SAC (Soft Actor-Critic) — online RL алгоритм. Изменение с per-step reward на batched reward требует:
1. Модификации RL-цикла FREED++ (Python 3.7, legacy код)
2. Накопление experience buffer перед скорингом
3. Перестройки dual_reward_patch для батчей
4. RDKit 3D generation тоже нужно батчировать

**Вывод**: батчирование reward внутри FREED++ — возможно, но требует глубокого рефакторинга legacy-кода. Более практичный путь — замена генератора.

---

## 4. Решение B: Замена генератора

### 4.1 Сравнение альтернатив

| Генератор | Тип | Sample efficiency | Батчирование oracle | Multi-objective | Лицензия |
|----------|-----|-------------------|-------------------|----------------|---------|
| **FREED++** (текущий) | Fragment RL (SAC) | Средняя (~180K вызовов) | Нет (per-molecule) | Через патч | MIT |
| **Saturn** | Mamba + Augmented Memory | **Лучшая** (1K вызовов) | **Да** (batch scoring) | Нет (single) | MIT |
| **REINVENT 4** | RNN/Transformer + RL | Высокая (#1-2 PMO) | **Да** (batch scoring) | **Да** (встроено) | Apache 2.0 |
| **GraphGA** | Генетический алгоритм | Высокая (#1-2 PMO) | **Да** (популяция = батч) | Базовая | MIT |
| **DrugEx v3** | Graph Transformer + RL | Средняя | Да | **Парето** | MIT |

### 4.2 Saturn — лучший вариант для дорогого oracle

| Параметр | Значение |
|----------|----------|
| **Архитектура** | Mamba (5.2M params) + Augmented Memory (experience replay) |
| **Ключевое преимущество** | Достигает качества FREED++ с **1,000 вызовов oracle** вместо 180,000 |
| **Почему решает проблему** | (1) Вызовов oracle в 180x меньше → меньше времени на докинг. (2) Scoring можно делать батчами → Uni-Dock работает на полной мощности GPU |
| **PMO бенчмарк** | Outperforms 22 из 25 методов, включая REINVENT |
| **GitHub** | [schwallergroup/saturn](https://github.com/schwallergroup/saturn) |

**Расчёт для dual-target**: 1000 молекул × Uni-Dock батч × 2 таргета = ~200 сек на GPU. Вместо ~50 часов с FREED++.

### 4.3 REINVENT 4 — production-ready с multi-objective из коробки

| Параметр | Значение |
|----------|----------|
| **Поддержка** | AstraZeneca, активная разработка |
| **Батчирование** | Scoring components вызываются батчами — Uni-Dock будет эффективен |
| **Multi-objective** | Встроено: множественные scoring components с весами |
| **Dual-target** | Два docking score как два компонента → нативная поддержка |
| **Режимы** | De novo, scaffold decoration (LibINVENT), linker design (LinkINVENT) |
| **PMO бенчмарк** | #1-2 (в паре с GraphGA) |
| **GitHub** | [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4) |

**Преимущество для проекта**: не нужен `dual_reward_patch.py` — dual-target реализуется через конфигурацию двух scoring components.

### 4.4 GraphGA — быстрый baseline

- Генетический алгоритм, **не требует GPU** для генератора
- Популяция = естественный батч для Uni-Dock
- На PMO бенчмарке — **на уровне REINVENT** (лучший RL-метод)
- [mol_ga GitHub](https://github.com/AustinT/mol_ga)

### 4.5 RL + Active Learning (применимо к любому генератору)

Chemical Science (2024): комбинация RL + active learning даёт:
- **5-66x больше хитов** при том же бюджете oracle
- **4-64x сокращение CPU-времени**
- Применимо к Saturn, REINVENT, GraphGA

---

## 5. Ускорение Boltz-2

### 5.1 Boltz-2: текущие ограничения

- ~16.5-20 сек/мол на H100, ~58 сек/мол на менее мощных GPU
- Bottleneck: диффузионный процесс + 5 recycling итераций Trunk Module
- GitHub issue #627: кеширование MSA **деградирует качество** ipTM
- GitHub issue #207: на CPU — драматически медленно

### 5.2 Boltzina — 7-12x ускорение Boltz-2

| Параметр | Значение |
|----------|----------|
| **Что делает** | Пропускает structure prediction, предсказывает аффинность из Vina-позы |
| **Скорость** | 1.4 сек/мол (cycle=1) — 2.3 сек/мол (стандарт) на H100 |
| **Speedup** | **7-12x** vs полный Boltz-2 |
| **Точность** | mAP 0.056 vs 0.084 (Boltz-2) на MF-PCBA. Vina/GNINA ≈ 0 |
| **GitHub** | [ohuelab/boltzina](https://github.com/ohuelab/boltzina) |

### 5.3 Альтернативные скореры

| Инструмент | Тип | Время/мол | Точность | Применение |
|-----------|-----|----------|---------|-----------|
| **Uni-Dock batch** | Физика (Vina) | 0.1 сек | Средняя | Массовый скрининг |
| **Boltzina** | ML (Boltz-2 trunk) | 1.4-2.3 сек | Высокая | Rescoring top кандидатов |
| **Boltz-2** | ML (full) | 16-58 сек | Наивысшая | Финальная валидация |
| **TankBind** | ML (geometric) | 0.03 сек | Низкая | Pre-filter |
| **DiffDock-L** | Diffusion | ~18 сек | Высокая (позы) | Blind docking |
| **GNINA** | CNN rescoring | ~16 сек | Высокая (VS) | Rescoring |

### 5.4 Иерархический скоринг (рекомендуемый подход)

| Стадия | Инструмент | Входных | Выходных | Время |
|--------|-----------|---------|---------|-------|
| 1. Быстрый скрининг | Uni-Dock fast (батч) | 1000 | top 100 (10%) | ~100 сек |
| 2. Rescoring | Boltzina | 100 | top 10 (1%) | ~140 сек |
| 3. Финальная валидация | Boltz-2 (full) | 10 | финальный ранг | ~200 сек |
| **Итого** | | 1000 мол | 10 лучших | **~7 мин** |

Текущий подход (Boltz-2 на все 1000) = ~16,000 сек = **~4.5 часа**.

---

## 6. Визуализация

### 6.1 Рекомендуемый стек

| Задача | Инструмент | Установка |
|--------|-----------|-----------|
| 3D визуализация в Jupyter | **py3Dmol** | `pip install py3Dmol` |
| Публикационные 3D-рендеры | **PyMOL** (headless) | `conda install pymol-open-source` |
| 2D структуры с аннотациями | **RDKit** (уже установлен) | — |
| Тепловые карты на молекулах | **rdkit_heatmaps** | `pip install rdkit-heatmaps` |
| Парето-фронт | **pymoo** или **Optuna** | `pip install pymoo` |
| SAR-анализ (десктоп) | **DataWarrior** | Бесплатная загрузка |

### 6.2 Web-дашборд: React/TypeScript + Mol*

Вместо Streamlit — полноценное React-приложение с TypeScript.

#### Библиотеки для молекулярной визуализации в React

| Библиотека | Назначение | npm |
|-----------|-----------|-----|
| **Mol\*** (Molstar) | 3D белок-лиганд визуализация | `molstar` |
| **3Dmol.js** | Лёгкая 3D визуализация | `3dmol` |
| **@rdkit/rdkit** | 2D структуры, SMILES→SVG | `@rdkit/rdkit` |
| **Plotly.js** / **recharts** | Интерактивные графики | `plotly.js` / `recharts` |
| **@tanstack/react-table** | Таблица молекул | `@tanstack/react-table` |

#### Mol* в React

```typescript
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';

// Инициализация Mol* viewer
const plugin = await createPluginUI({
  target: document.getElementById('viewer')!,
  layoutIsExpanded: false,
});

// Загрузка белка + лиганда
await plugin.builders.data.download(
  { url: '/api/structures/CDK2.pdb' }, { state: { isGhost: false } }
);
```

#### RDKit WASM для 2D в браузере

```typescript
import initRDKitModule from '@rdkit/rdkit';

const RDKit = await initRDKitModule();
const mol = RDKit.get_mol('CC(=O)Oc1ccccc1C(=O)O');
const svg = mol.get_svg();  // SVG строка для отрисовки
```

#### Архитектура дашборда

```
┌──────────────────────────────────────────────────────────────┐
│  React App (TypeScript + Vite)                               │
├──────────────────────────────────────────────────────────────┤
│  Header: [Pair selector ▼] [Mode: single|double] [Seed ▼]   │
├────────────────────────┬─────────────────────────────────────┤
│  Mol* Viewer           │  Mol* Viewer                        │
│  Target A + Ligand     │  Target B + Ligand                  │
│  (3D, interactive)     │  (3D, interactive)                   │
├────────────────────────┴─────────────────────────────────────┤
│  Plotly Scatter: Score A vs Score B                          │
│  - Single mode (blue) vs Double mode (red)                   │
│  - Pareto front overlay                                      │
│  - Threshold lines (hit criteria)                            │
│  - Click point → select molecule → update 3D viewers         │
├──────────────────────────────────────────────────────────────┤
│  Metrics Panel:                                              │
│  [Dual hit rate: 15%] [KS p-value: 0.003] [Enrichment@10]   │
├──────────────────────────────────────────────────────────────┤
│  Molecule Table (@tanstack/react-table):                     │
│  | # | 2D (RDKit SVG) | Score A | Score B | Combined | Hit? │
│  | 1 | [structure]     | -2.1    | -1.8    | -1.8     | ✓   │
│  | 2 | [structure]     | -1.5    | -0.3    | -0.3     | ✗   │
│  Click row → update 3D viewers + highlight on scatter        │
├──────────────────────────────────────────────────────────────┤
│  Distribution Panel:                                         │
│  [Violin: single vs double] [ECDF overlay] [Box plots]       │
└──────────────────────────────────────────────────────────────┘
```

#### Backend API (Python FastAPI или данные как static JSON)

Два варианта:
1. **Static JSON** — просто экспортировать `report/` в JSON, React загружает файлы
2. **FastAPI backend** — динамический API для фильтрации, пагинации, и подгрузки структур

Минимальный вариант (static):
```
public/
  data/
    all_results.json     # из report/data/all_results.parquet
    metrics.json         # из report/metrics.json
    stats.json           # из report/stats.json
  structures/
    CDK2.pdb, ERK2.pdb, ...
    ligands/             # SDF файлы поз
```

### 6.3 Dual-target scatter plot (Plotly.js в React)

```typescript
import Plotly from 'plotly.js-dist-min';

const traces = [
  {
    x: singleData.map(d => d.score_a),
    y: singleData.map(d => d.score_b),
    mode: 'markers',
    name: 'Single',
    marker: { color: 'rgba(66, 133, 244, 0.6)' },
    text: singleData.map(d => d.smiles),
    hovertemplate: '%{text}<br>A: %{x:.2f}<br>B: %{y:.2f}',
  },
  {
    x: doubleData.map(d => d.score_a),
    y: doubleData.map(d => d.score_b),
    mode: 'markers',
    name: 'Double',
    marker: { color: 'rgba(234, 67, 53, 0.6)' },
    text: doubleData.map(d => d.smiles),
    hovertemplate: '%{text}<br>A: %{x:.2f}<br>B: %{y:.2f}',
  },
];

Plotly.newPlot('scatter', traces, {
  xaxis: { title: 'Score Target A (log₁₀ IC₅₀)' },
  yaxis: { title: 'Score Target B (log₁₀ IC₅₀)' },
  // Click handler → update Mol* viewers
});
```

### 6.4 3D dual-target визуализация (py3Dmol для Jupyter)

```python
import py3Dmol

view = py3Dmol.view(width=1200, height=600, viewergrid=(1, 2), linked=False)

# Target A + ligand
view.addModel(open('target_A.pdb').read(), 'pdb', viewer=(0, 0))
view.setStyle({'cartoon': {'color': 'spectrum'}}, viewer=(0, 0))
view.addModel(open('ligand_in_A.sdf').read(), 'sdf', viewer=(0, 0))
view.setStyle({'model': -1}, {'stick': {'colorscheme': 'magentaCarbon'}}, viewer=(0, 0))

# Target B + ligand
view.addModel(open('target_B.pdb').read(), 'pdb', viewer=(0, 1))
view.setStyle({'cartoon': {'color': 'spectrum'}}, viewer=(0, 1))
view.addModel(open('ligand_in_B.sdf').read(), 'sdf', viewer=(0, 1))
view.setStyle({'model': -1}, {'stick': {'colorscheme': 'cyanCarbon'}}, viewer=(0, 1))

view.zoomTo(viewer=(0, 0))
view.zoomTo(viewer=(0, 1))
view.show()
```

### 6.5 Публикационные фигуры

- **PyMOL-PUB** ([BGI-SynBio/PyMOL-PUB](https://github.com/BGI-SynBio/PyMOL-PUB)) — headless publication-quality рендеры
- **MolViewSpec** ([molstar/mol-view-spec](https://github.com/molstar/mol-view-spec)) — viewer-agnostic описание молекулярных сцен

---

## 7. Рекомендации и приоритеты

### Фаза 1: Решить проблему GPU underutilization

| # | Действие | Эффект | Сложность |
|---|---------|--------|-----------|
| 1 | **Saturn** вместо FREED++ | 180x меньше вызовов oracle + батчи → GPU загружен | Средняя |
| 2 | **Boltzina** как промежуточный скорер | 7-12x ускорение финального скоринга | Простая |
| 3 | Иерархический скоринг (Uni-Dock батч → Boltzina → Boltz-2) | 1000 мол за 7 мин вместо 4.5 ч | Средняя |

### Фаза 2: Альтернативные генераторы для сравнения

| # | Действие | Эффект | Сложность |
|---|---------|--------|-----------|
| 4 | **REINVENT 4** с dual scoring components | Production-ready + dual-target из коробки | Средняя |
| 5 | **GraphGA** baseline | Sanity-check, естественное батчирование | Простая |
| 6 | RL + Active Learning поверх любого генератора | 5-66x больше хитов | Средняя |

### Фаза 3: Визуализация и анализ

| # | Действие | Эффект | Сложность |
|---|---------|--------|-----------|
| 7 | React/TypeScript дашборд с Mol* + Plotly | Интерактивный анализ dual-target | Средняя |
| 8 | py3Dmol в Jupyter для dual-target позирования | 3D понимание связывания | Простая |
| 9 | PyMOL-PUB для финальных рендеров | Publication-quality | Средняя |

### Оценка времени: текущий vs оптимизированный пайплайн

| | Текущий (FREED++ + Uni-Dock per-mol) | Оптимизированный (Saturn + Uni-Dock batch) |
|--|--------------------------------------|-------------------------------------------|
| Генерация (1000 мол) | ~50 ч (180K вызовов × ~1 сек) | ~3 мин (1K вызовов × батч 0.1 сек) |
| Скоринг (1000 мол × 2 таргета) | ~32 ч (Boltz-2 на всё) | ~7 мин (иерархический) |
| **1 пара, 1 сид** | **~82 часа** | **~10 минут** |
| **4 пары × 5 сидов** | **~68 дней** | **~3.5 часа** |

---

### Источники

- [Uni-Dock (JCTC)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) — GPU docking, 1000x speedup при батчах
- [Boltzina (arXiv:2508.17555)](https://arxiv.org/abs/2508.17555) — 7-12x ускорение Boltz-2
- [Saturn (arXiv:2405.17066)](https://arxiv.org/abs/2405.17066) — sample-efficient генерация
- [REINVENT 4 (J. Cheminf. 2024)](https://link.springer.com/article/10.1186/s13321-024-00812-5) — production RL генератор
- [PMO Benchmark (NeurIPS 2022)](https://arxiv.org/abs/2206.12411) — сравнение генераторов
- [RL + Active Learning (Chem. Sci. 2024)](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc04653b) — 5-66x ускорение
- [Vina-GPU 2.1 (IEEE TCBB)](https://www.biorxiv.org/content/10.1101/2023.11.04.565429v1.full)
- [GNINA 1.3 (J. Cheminf. 2025)](https://link.springer.com/article/10.1186/s13321-025-00973-x)
- [DiffDock (ICLR 2023)](https://arxiv.org/abs/2210.01776)
- [PyMOL-PUB (Bioinformatics 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10950480/)
- [MolPAL (Digital Discovery 2024)](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d3dd00227f) — Pareto для dual-target
- [ParetoDrug (Nature Comms Bio 2024)](https://www.nature.com/articles/s42003-024-06746-w)
- [Boltz-2 MSA caching issue #627](https://github.com/jwohlwend/boltz/issues/627)
- [Boltz-2 CPU issue #207](https://github.com/jwohlwend/boltz/issues/207)
- [Molstar React integration](https://github.com/molstar/molstar)
- [@rdkit/rdkit WASM](https://www.npmjs.com/package/@rdkit/rdkit)
- [Meeko docs: RDKit ↔ PDBQT](https://meeko.readthedocs.io/en/latest/guide_rdkit_mol_from_smiles_and_pdbqt.html)
- [Meeko docs: receptor preparation](https://meeko.readthedocs.io/en/latest/cli_rec_prep.html)
- [SwissDock 2024 (NAR)](https://academic.oup.com/nar/article/52/W1/W324/7660078)
- [DockStream (J. Cheminf. 2021)](https://link.springer.com/article/10.1186/s13321-021-00563-7)
- [OpenBabel issue: kekulize warnings](https://github.com/openbabel/openbabel/issues/2600)
