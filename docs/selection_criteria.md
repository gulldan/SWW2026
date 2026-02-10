# Критерии выбора пар белков

## Требования к парам (primary analysis)

### Для FREED++ (UniDock GPU-генерация)

1. **PDB-структура**: разрешение <= 2.5 Å с со-кристаллизованным лигандом (определяет grid box center). Исключение: EGFR 1M17 (2.6 Å) — auxiliary pair, допущена как единственный Human EGFR co-crystal с erlotinib; на primary analysis не влияет.
2. **Чёткий binding site**: однозначный связывающий карман для корректного reward signal.
3. **Размер**: белок <= 600 остатков (в `double` режиме UniDock GPU вызывается дважды на каждый шаг RL-эпизода).
4. **Биологическое обоснование**: пара должна иметь обоснование в multi-target дизайне лекарств.

### Для Boltz-2 (валидация)

5. **Совместимость**: растворимые белки без кофакторов в binding pocket (Boltz-2 affinity head не моделирует кофакторы). Пары с кофакторами (AChE_MAOB) допускаются только в exploratory analysis с `affinity_probability_binary` и помечаются `warning` в таблице пар.
6. **Последовательность**: UniProt FASTA достаточно (PDB не нужен для Boltz-2).
7. **Размер**: <= 1000 токенов (ограничение Boltz-2).

### Кофакторы — критичное ограничение Boltz-2

> Boltz-2 paper (Limitations): *"the affinity module does not explicitly handle such cofactors, including ions, water, or multimeric binding partners"*
> Rowan FAQ: *"affinity predictions will be unreliable"* для targets с кофакторами в binding pocket.

Пары с кофакторами (COX1_COX2 — гем, AChE_MAOB — FAD): ось «Cofactor: Present» относится к **биологии мишени и надёжности Boltz-2**, а не к состоянию docking-рецептора. Протокол подготовки рецептора (`removeHeterogens(keepWater=False)`) удаляет **все** HETATM, включая кофакторы; таким образом, UniDock докирует к **cofactor-free** рецептору для всех мишеней без исключения. Это означает, что docking-скоры для COX/MAO-B отражают связывание к «очищенному» карману и могут быть менее репрезентативны биологически. **Boltz-2 affinity** для мишеней с кофакторами также ненадёжен (affinity head не моделирует кофакторы). Для MAO-B рекомендуется `affinity_probability_binary` вместо `affinity_pred_value`.

## Сводная таблица всех 9 пар

Классификация по трём осям: **Disease** (AD / Non-AD), **Structure** (Homo / Hetero), **Cofactor** (Clean / Present). Подробнее: см. [Факторный дизайн](#факторный-дизайн-эксперимента).

| # | Пара | Disease | Structure | Cofactor | Set | PDB A / PDB B | Размер* | Boltz-2 |
|---|------|---------|-----------|----------|-----|---------------|---------|---------|
| 1 | CDK2_ERK2 | Non-AD | Homo | Clean | **Core** | 1HCL¹ / 4QTA | ~629 | ok |
| 2 | EGFR_HER2 | Non-AD | Homo | Clean | Auxiliary | 1M17 / 3PP0 | ~671 | ok |
| 3 | COX1_COX2 | Non-AD | Homo | Present | Blocked | 1EQG / 3LN1 | ~1120 | warning (гем) |
| 4 | AChE_BChE | AD | Homo | Clean | **Core** | 4EY7 / 4BDS | ~1073 | ok |
| 5 | **BACE1_GSK3B** | AD | Hetero | Clean | **Core** | 6EQM / 5KPK | ~724 | ok |
| 6 | **AChE_BACE1** | AD | Hetero | Clean | **Core** | 4EY7 / 6EQM | ~943 | ok |
| 7 | **GSK3B_CDK5** | AD | Hetero | Clean | **Core** | 5KPK / 1UNL | ~642 | ok |
| 8 | **BACE1_CDK5** | AD | Hetero | Clean | Auxiliary | 6EQM / 1UNL | ~692 | ok |
| 9 | AChE_MAOB | AD | Hetero | Present | Auxiliary | 4EY7 / 2V5Z | ~1063 | warning (FAD) |

\* **Размер** — приблизительное суммарное число остатков обоих белков после обработки PDBFixer (trimming, удаление HETATM). Отличия ±20–50 aa от RCSB deposited sequence возможны из-за missing residues, non-standard residue replacement и removal of terminal tags.

## Human-Only Enforcement

Starting with ENG-10, `load_pairs()` validates that all targets have `organism: "Homo sapiens"`.
Non-human targets (e.g. COX1 — Ovis aries) are blocked by default.

- **Bypass**: Set `DUALDOCKING_ALLOW_NON_HUMAN=true` for exploratory runs
- **COX1_COX2**: Blocked in production. Available for exploratory analysis only.

## Факторный дизайн эксперимента

Выбор 9 пар определяется заранее зафиксированным факторным дизайном по трём осям:

- **Structural relation**: Homo (одно семейство, высокая гомология) / Hetero (разные семейства)
- **Cofactor**: Clean (нет кофактора в кармане биологически) / Present (кофактор присутствует биологически; влияет на интерпретацию docking и Boltz-2 — см. [Кофакторы](#кофакторы--критичное-ограничение-boltz-2))
- **Disease context**: AD (болезнь Альцгеймера) / Non-AD (контрольные)

### Матрица дизайна

| # | Пара | Structure | Cofactor | Disease | Set | ML-гипотеза |
|---|------|-----------|----------|---------|-----|-------------|
| 1 | CDK2_ERK2 | Homo | Clean | Non-AD | **Core** | При высокой гомологии double-dock должен легко находить dual ингибиторы — тест «лёгкого» случая |
| 2 | EGFR_HER2 | Homo | Clean | Non-AD | Auxiliary | Второй homo-контроль: обобщаемость на другое киназное подсемейство (ErbB) |
| 3 | COX1_COX2 | Homo | Present | Non-AD | Blocked | Non-human PDB: COX1 (Ovis aries), COX2 (Mus musculus); cofactor (гем). Только exploratory |
| 4 | AChE_BChE | Homo | Clean | AD | **Core** | AD-relevant homo-пара: сохраняет ли double-dock преимущество для крупных белков (~1073 остатка) |
| 5 | BACE1_GSK3B | Hetero | Clean | AD | **Core** | **Центральный тест**: может ли double-dock находить молекулы, активные на структурно различных мишенях |
| 6 | AChE_BACE1 | Hetero | Clean | AD | **Core** | Максимум ChEMBL данных (3524 dual-compound) — гипотеза + валидация через химическое сходство |
| 7 | GSK3B_CDK5 | Hetero | Clean | AD | **Core** | Dual tau-киназы при минимальном compute (~642 остатка). Тест функциональной близости vs структурной |
| 8 | BACE1_CDK5 | Hetero | Clean | AD | Auxiliary | Robustness: сохраняется ли выигрыш double-dock при замене GSK3B→CDK5 как tau-киназы |
| 9 | AChE_MAOB | Hetero | Present | AD | Auxiliary | Sensitivity: влияние кофактора (FAD) на reward signal. Boltz-2 affinity только exploratory |

### Core vs Auxiliary

**Core set (5 пар)** — основная проверка гипотезы. Статистические выводы делаются по этим парам:
- Покрывают оба уровня гомологии (homo: CDK2_ERK2, AChE_BChE; hetero: BACE1_GSK3B, AChE_BACE1, GSK3B_CDK5)
- Все без кофакторов → Boltz-2 affinity рассматривается как достаточно надёжная ортогональная метрика при наличии согласованности с docking и ChEMBL-валидацией
- Каждый белок участвует не более чем в 2 core-парах (AChE в 2, BACE1 в 2, GSK3B в 2; CDK2, ERK2, BChE, CDK5 — в 1)

**Auxiliary set (3 пары + 1 blocked)** — robustness и sensitivity анализ:
- EGFR_HER2: второй homo Non-AD контроль (не в core, т.к. CDK2_ERK2 уже покрывает)
- BACE1_CDK5: тест замены tau-киназы (перекрывается с BACE1_GSK3B и GSK3B_CDK5)
- AChE_MAOB: cofactor-sensitive пара, только docking + binary Boltz-2

**Blocked:** COX1_COX2 (non-human PDB structures: COX1 — Ovis aries, COX2 — Mus musculus). Не входит в основной и вспомогательный анализ.

> Основная гипотеза проверяется на 5 заранее зафиксированных core-парах; auxiliary пары используются для robustness/sensitivity анализов. Это деление зафиксировано **до начала экспериментов** для исключения cherry-picking.

### Перекрытие таргетов и статистическая независимость

Одни и те же белки (AChE, BACE1, GSK3B) участвуют в нескольких парах.
Это полезно для анализа влияния второй мишени, но наивное «N=9 пар» не является
9 независимыми наблюдениями.

- Primary statistical tests проводятся на **core set** (5 пар), где перекрытие минимизировано
- Auxiliary пары используются как **within-target sensitivity analysis**, а не как дополнительные независимые наблюдения
- В limitations явно указывается: *«поскольку core-пары разделяют ряд белков, размер эффекта трактуется с осторожностью»*

## Non-AD контроль (пары 1-3)

- **Киназные пары** (CDK2/ERK2, EGFR/HER2): хорошо изученные связывающие карманы, множество известных ингибиторов, обширные бенчмарки докинга.
- **COX-пара**: золотой стандарт задачи селективности в медицинской химии. Гем в активном сайте — проблема для Boltz-2. Blocked (non-human PDB: COX1 — Ovis aries, COX2 — Mus musculus).

## AD-гомологичная пара (пара 4)

- **AChE_BChE**: формально AD-relevant (оба таргета — мишени для AD-терапии), но используется также как **homology control** — единственная AD-пара с высокой структурной гомологией. Служит «мостом» между Non-AD homo-контролем (CDK2_ERK2, EGFR_HER2) и AD hetero-парами (BACE1_GSK3B и др.).

## AD-специфичные пары (пары 5-9)

### BACE1_GSK3B — лучшая пара

- **Биология**: GSK3B регулирует экспрессию BACE1 через NF-kB (Ly et al. 2013, J Clin Invest). Notopterol — валидированный in vivo dual ингибитор (Jiang et al. 2020, APP/PS1 мыши).
- **Структуры**: BACE1 6EQM (1.35 Å, CNP520 IC50 11 nM) + GSK3B 5KPK (2.4 Å, BRD0209).
- **ChEMBL**: >8000 записей BACE1, >4000 записей GSK3B.
- **Compute**: суммарно ~750 остатков — быстрый UniDock GPU, комфортный для Boltz-2.

### AChE_BACE1 — классическая пара

- **Биология**: Самая изученная dual-target комбинация в AD. Stern et al. 2022 — от in silico до in vivo.
- **ChEMBL**: 3524 dual-compound записей (максимальный overlap для валидации).
- **Compute**: суммарно ~943 остатка — на грани, но работает.

### GSK3B_CDK5 — оптимальная для compute

- **Биология**: Dual tau-киназы. GSK3B фосфорилирует Ser396/Ser404, CDK5 — Ser202/Thr205 (AT8 epitope). Оба — причина NFT в AD.
- **Структуры**: GSK3B 5KPK (2.4 Å) + CDK5 1UNL (2.2 Å, Roscovitine IC50 0.2 μM).
- **Compute**: суммарно ~642 остатка — **самая маленькая пара**, самый быстрый `double` run.

### BACE1_CDK5 — robustness test (auxiliary)

- **Биология**: Amyloid (BACE1) + Tau (CDK5). p25/CDK5 complex регулирует BACE1 активность. Менее изученная пара — оригинальность.
- **Compute**: суммарно ~692 остатка — быстро.
- **Роль**: Auxiliary pair для **robustness analysis** — проверяет, сохраняется ли выигрыш double-dock при замене tau-киназы GSK3B→CDK5. Перекрывается с core-парами BACE1_GSK3B и GSK3B_CDK5, поэтому не входит в core set.

### AChE_MAOB — с оговоркой

- **Биология**: Клинический прецедент — ladostigil (Phase II). Rivastigmine + rasagiline фармакофоры.
- **Оговорка**: MAO-B содержит ковалентный FAD биологически. Однако протокол подготовки рецептора удаляет все HETATM (`removeHeterogens`), поэтому UniDock докирует к **cofactor-free** рецептору MAO-B. Это означает, что docking-скоры для MAO-B отражают связывание к очищенному карману и могут не учитывать стерические/электростатические эффекты FAD. Boltz-2 affinity для MAO-B также **ненадёжен** (affinity head не моделирует кофакторы) — интерпретировать с осторожностью; рекомендуется `affinity_probability_binary`.

## Протокол Boltz-2 скоринга

### Два режима использования

Boltz-2 используется как **ортогональный scorer** (независимый от UniDock reward) в двух режимах:

| Режим | Когда применяется | Метрика | Статус |
|-------|-------------------|---------|--------|
| **Primary (quantitative)** | Все пары без кофакторов: 5 core + 2 auxiliary (7 из 8 активных) | `affinity_pred_value` (log10 IC50-like, μM) | Primary endpoint |
| **Exploratory (binary)** | Пары с кофактором (AChE_MAOB) | `affinity_probability_binary` (hit/non-hit) | Exploratory only |

### Primary режим: `affinity_pred_value`

Для всех пар **без кофакторов** в binding pocket:
- Непрерывная метрика: предсказание log10(IC50-like, μM) — surrogate, обученный на смеси Ki/Kd/IC50/EC50
- Используется для ранжирования молекул и вычисления hr_both, enrichment@K
- Параметры стандартизированы в `configs/master_config.yaml`:
  `recycling_steps: 3, sampling_steps: 200, diffusion_samples: 4,
   sampling_steps_affinity: 200, diffusion_samples_affinity: 5`

### Exploratory режим: `affinity_probability_binary`

Для пар с кофакторами (AChE_MAOB — FAD):
- Бинарная метрика: вероятность связывания (hit/non-hit)
- **Не входит в primary endpoint** — только для качественного анализа
- Причина: Boltz-2 affinity head не моделирует кофакторы (Passaro et al. 2025, Limitations)

> Для AChE_MAOB основная метрика — docking score (UniDock GPU). Boltz-2 используется
> только как вспомогательный бинарный индикатор. Результаты по этой паре
> **не входят в primary statistical analysis**.

### In-domain bias и интерпретация

Boltz-2 обучен на масштабных датасетах структур и комплексов белок-лиганд.
С высокой вероятностью используемые PDB-структуры (или их близкие аналоги)
входили в train set.

Поэтому:
- Метрики Boltz-2 интерпретируются как **relative ranking** между стратегиями
  (single vs double), а не как абсолютные предсказания экспериментальной IC50
- Основная ценность Boltz-2 в данном проекте — **ортогональность** к UniDock:
  два независимых метода (physics-based docking vs learned affinity) должны
  давать согласованный сигнал для повышения доверия к результатам
- Для пар с обширными ChEMBL данными (BACE1, AChE, GSK3B) дополнительно
  проводится enrichment по Tanimoto-сходству к известным активам — это
  третий, независимый от обоих скореров, источник валидации

## Протокол подготовки рецептора

Для всех мишеней используется **идентичный** протокол подготовки,
исключающий скрытую вариабельность между парами.

### Шаг 1: PDB → Fixed PDB (`pdbfixer_add_h.py`)

Выполняется в Docker-контейнере (`dualdocking/freedpp`) с PDBFixer + OpenMM:

1. `findMissingResidues()` — обнаружение пропущенных остатков
2. `findNonstandardResidues()` + `replaceNonstandardResidues()` — замена нестандартных аминокислот
3. `removeHeterogens(keepWater=False)` — удаление **всех** HETATM (вода, ионы, лиганды, кофакторы). **Важно**: это означает, что кофакторы (HEM для COX, FAD для MAO-B) также удаляются; docking выполняется к cofactor-free рецептору для всех мишеней без исключения
4. `findMissingAtoms()` + `addMissingAtoms()` — достройка пропущенных атомов
5. `addMissingHydrogens(pH=7.0)` — протонирование при pH 7.0

Post-processing для совместимости с Meeko:
- HIS → HID/HIE/HIP в зависимости от протонирования (HD1, HE2)
- CYS → CYX для дисульфидных мостиков (SSBOND или SG-SG ≤ 2.3 Å)
- Удаление терминальных атомов H2/H3/OXT

### Шаг 2: Fixed PDB → PDBQT (`mk_prepare_receptor.py`)

Meeko конвертирует PDB в формат PDBQT для AutoDock/UniDock:
- Назначение парциальных зарядов (Gasteiger)
- Назначение типов атомов AutoDock
- Флаг `--skip_gpf` (grid parameter файл не нужен — box из co-ligand)

### Шаг 3: Binding box

Автоматическое определение из co-crystallized ligand (HETATM) **исходного PDB** (до `removeHeterogens`):
- Координаты лиганда извлекаются из оригинального PDB-файла модулем `scoring/binding_box.py`
- Центр = центроид тяжёлых атомов лиганда
- Размер = extent + `box_padding` (10.0 Å по умолчанию)
- Fallback: центр масс белка + куб 30×30×30 Å

> **Порядок операций**: binding box определяется из исходного PDB → затем `pdbfixer_add_h.py` очищает PDB (удаляя HETATM) → Meeko конвертирует очищенный PDB в PDBQT. Таким образом, координаты бокса берутся до удаления лиганда/кофакторов.

### Валидация: re-docking

Для каждой мишени проводится re-docking co-crystallized ligand:
- Лиганд извлекается из PDB, SMILES получается из RCSB
- Docking с теми же параметрами, что и в эксперименте
- RMSD ≤ 2.0 Å между docked и crystal позой = PASS
- Скрипт: `scripts/s2_calibrate/validate_redocking.py`

> Идентичность протокола подготовки для всех мишеней — критичная гарантия
> того, что различия в результатах между парами не являются артефактами
> разной обработки рецепторов.

## Общие свойства всех пар

- PDB-структуры с со-кристаллизованными лигандами (кроме 1HCL — см. ¹)
- Обширные данные по активности в ChEMBL
- Управляемые размеры белков (< 600 остатков на мишень)

> ¹ **1HCL (CDK2)** — apo-структура (без лиганда). Binding box определяется по координатам ATP-связывающего кармана из PDB 4GCJ (CDK2 с лигандом) или через fallback (центр масс белка + 30×30×30 Å куб). Влияние на docking-качество минимально: CDK2 ATP-карман хорошо определён структурно, что подтверждается re-docking валидацией (RMSD ≤ 2.0 Å, см. `scripts/s2_calibrate/validate_redocking.py`).

## Ключевые источники

- Ly et al. (2013) J Clin Invest — GSK3B регулирует BACE1
- Stern et al. (2022) IJMS — AChE/BACE1 dual-target
- Ivanova et al. (2020) Molecules 25:1846 — AD multi-target
- Jiang et al. (2020) Clin Transl Med — Notopterol dual BACE1/GSK3B in vivo (DOI: 10.1002/ctm2.50)
- Boltz-2 (Passaro et al. 2025) — cofactor limitations
