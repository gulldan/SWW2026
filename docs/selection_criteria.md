# Критерии выбора пар белков

## Требования к каждой паре

### Для FREED++ (UniDock GPU-генерация)

1. **PDB-структура**: разрешение <= 2.5 A с со-кристаллизованным лигандом (определяет grid box center).
2. **Чёткий binding site**: однозначный связывающий карман для корректного reward signal.
3. **Размер**: белок <= 600 остатков (в `double` режиме UniDock GPU вызывается дважды на каждый шаг RL-эпизода).
4. **Биологическое обоснование**: пара должна иметь обоснование в multi-target дизайне лекарств.

### Для Boltz-2 (валидация)

5. **Совместимость**: растворимые белки без кофакторов в binding pocket (Boltz-2 affinity head не моделирует кофакторы).
6. **Последовательность**: UniProt FASTA достаточно (PDB не нужен для Boltz-2).
7. **Размер**: <= 1000 токенов (ограничение Boltz-2).

### Кофакторы — критичное ограничение Boltz-2

> Boltz-2 paper (Limitations): *"the affinity module does not explicitly handle such cofactors, including ions, water, or multimeric binding partners"*
> Rowan FAQ: *"affinity predictions will be unreliable"* для targets с кофакторами в binding pocket.

Пары с кофакторами (COX1_COX2 — гем, AChE_MAOB — FAD): docking-скоры OK (UniDock GPU обрабатывает кофакторы при подготовке рецептора), но **Boltz-2 affinity интерпретировать с осторожностью**. Для MAO-B рекомендуется `affinity_probability_binary` вместо `affinity_pred_value`.

## Сводная таблица всех 9 пар

| # | Пара | Тип | PDB A / PDB B | Размер (сумма) | Boltz-2 | Роль в эксперименте |
|---|------|-----|--------------|----------------|---------|---------------------|
| 1 | CDK2_ERK2 | Non-AD | 1HCL / 4QTA | ~626 | ok | Контроль: гомологичные киназы |
| 2 | EGFR_HER2 | Non-AD | 1M17 / 3PP0 | ~860 | ok | Контроль: семейство ErbB |
| 3 | COX1_COX2 | Non-AD | 1EQG / 3LN1 | ~1120 | warning (гем) | Контроль: гомологичные + кофактор |
| 4 | AChE_BChE | AD-homo | 4EY7 / 4BDS | ~1073 | ok | AD лёгкий: высокая гомология |
| 5 | **BACE1_GSK3B** | **AD-hetero** | **6EQM / 5KPK** | **~750** | ok | **AD главная: разные семейства** |
| 6 | **AChE_BACE1** | **AD-hetero** | **4EY7 / 6EQM** | **~943** | ok | **AD классическая: max данные** |
| 7 | **GSK3B_CDK5** | **AD-hetero** | **5KPK / 1UNL** | **~642** | ok | **AD fast: min compute** |
| 8 | **BACE1_CDK5** | **AD-hetero** | **6EQM / 1UNL** | **~692** | ok | **AD оригинальная: amyloid x tau** |
| 9 | AChE_MAOB | AD-hetero | 4EY7 / 2V5Z | ~1063 | warning (FAD) | AD clinical: ladostigil прецедент |

## Non-AD контроль (пары 1-4)

- **Киназные пары** (CDK2/ERK2, EGFR/HER2): хорошо изученные связывающие карманы, множество известных ингибиторов, обширные бенчмарки докинга.
- **COX-пара**: золотой стандарт задачи селективности в медицинской химии. Гем в активном сайте — проблема для Boltz-2.
- **Холинэстеразная пара**: multi-target дизайн для нейродегенерации, высокая гомология AChE/BChE.

## AD-специфичные пары (пары 5-9)

### BACE1_GSK3B — лучшая пара

- **Биология**: GSK3B регулирует экспрессию BACE1 через NF-kB (Ly et al. 2013, J Clin Invest). Notopterol — валидированный in vivo dual ингибитор (Xia et al. 2019, APP/PS1 мыши).
- **Структуры**: BACE1 6EQM (1.67 A, CNP520 IC50 11 nM) + GSK3B 5KPK (2.04 A, BRD0209).
- **ChEMBL**: >8000 записей BACE1, >4000 записей GSK3B.
- **Compute**: суммарно ~750 остатков — быстрый UniDock GPU, комфортный для Boltz-2.

### AChE_BACE1 — классическая пара

- **Биология**: Самая изученная dual-target комбинация в AD. Stern et al. 2022 — от in silico до in vivo.
- **ChEMBL**: 3524 dual-compound записей (максимальный overlap для валидации).
- **Compute**: суммарно ~943 остатка — на грани, но работает.

### GSK3B_CDK5 — оптимальная для compute

- **Биология**: Dual tau-киназы. GSK3B фосфорилирует Ser396/Ser404, CDK5 — Ser202/Thr205 (AT8 epitope). Оба — причина NFT в AD.
- **Структуры**: GSK3B 5KPK (2.04 A) + CDK5 1UNL (2.20 A, Roscovitine IC50 0.2 uM).
- **Compute**: суммарно ~642 остатка — **самая маленькая пара**, самый быстрый `double` run.

### BACE1_CDK5 — оригинальная

- **Биология**: Amyloid (BACE1) + Tau (CDK5). p25/CDK5 complex регулирует BACE1 активность. Менее изученная пара — оригинальность.
- **Compute**: суммарно ~692 остатка — быстро.

### AChE_MAOB — с оговоркой

- **Биология**: Клинический прецедент — ladostigil (Phase II). Rivastigmine + rasagiline фармакофоры.
- **Оговорка**: MAO-B содержит ковалентный FAD. FREED++/UniDock GPU — OK (FAD включается при подготовке рецептора). Boltz-2 affinity для MAO-B — **ненадёжен**, интерпретировать с осторожностью.

## Общие свойства всех пар

- PDB-структуры с со-кристаллизованными лигандами
- Обширные данные по активности в ChEMBL
- Управляемые размеры белков (< 600 остатков на мишень)

## Ключевые источники

- Ly et al. (2013) J Clin Invest — GSK3B регулирует BACE1
- Stern et al. (2022) IJMS — AChE/BACE1 dual-target
- Ivanova et al. (2020) Molecules 25:1846 — AD multi-target
- Xia et al. (2019) — Notopterol dual BACE1/GSK3B in vivo
- Boltz-2 (Passaro et al. 2025) — cofactor limitations
