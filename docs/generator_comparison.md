# Обзор методов молекулярной генерации для замены FREED++

Актуализировано 2026-02-01. Все ссылки и данные о доступности кода проверены.

---

## 1. Постановка задачи

Текущий pipeline проекта SWW2026 использует **FREED++** [Telepov et al., TMLR 2024](https://arxiv.org/abs/2401.09840) как молекулярный генератор с докингом в качестве reward. Проблема: FREED++ требует **~180 000 вызовов оракула** на один запуск, что делает невозможным использование дорогих скореров (Boltz-2: ~20 с/мол, GNINA: ~16 с/мол). При двухтаргетной задаче (две мишени, два докинга) wall-clock время становится неприемлемым.

**Требования к замене:**
- Поддержка пользовательских scoring-функций (докинг)
- Multi-objective оптимизация (≥2 мишени одновременно)
- Sample efficiency: ≤10K–50K оракульных вызовов
- Доступный, поддерживаемый open-source код
- Генерация drug-like молекул (QED, SA, отсутствие PAINS)

---

## 2. Методология отбора

Проведён систематический поиск по публикациям 2022–2026 гг. (NeurIPS, ICLR, ICML, Nature Communications, JACS Au, J. Cheminformatics, Chemical Science, Advanced Science, KDD, PLOS Computational Biology, bioRxiv, ChemRxiv) и GitHub-репозиториям. Из ~40 изначально рассмотренных методов отсеяны:
- Методы без публичного кода (MT-Mol, ReMol, MOLLM, SFG-Drug, FragDockRL)
- Методы с нерабочим / критически устаревшим кодом (FREED original, MTMol-GPT)
- Методы, не поддерживающие custom scoring (чистые VAE без RL)

Осталось **18 методов** с доступным кодом, которые оцениваются ниже.

---

## 3. Аудит доступности кода (проверено 2026-02-01)

| # | Метод | Репозиторий | Stars | Последний коммит | Лицензия | Usability |
|---|-------|------------|-------|-----------------|----------|-----------|
| 1 | REINVENT4 | [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4) | 674 | Nov 2025 | Apache-2.0 | **Отлично** |
| 2 | Saturn | [schwallergroup/saturn](https://github.com/schwallergroup/saturn) | 70 | Jul 2024 | Присутствует | Хорошо |
| 3 | FREED++ | [AIRI-Institute/FFREED](https://github.com/AIRI-Institute/FFREED) | 8 | Dec 2023 | MIT | Удовл. (стагнация) |
| 4 | Graph GA (mol_ga) | [AustinT/mol_ga](https://github.com/AustinT/mol_ga) | — | Активный | MIT | **Отлично** |
| 5 | DrugEx v3 | [CDDLeiden/DrugEx](https://github.com/CDDLeiden/DrugEx) | 166 | Dec 2024 | MIT | Хорошо |
| 6 | POLYGON | [bpmunson/polygon](https://github.com/bpmunson/polygon) | 42 | Feb 2024 | MIT | Хорошо |
| 7 | MOLLEO | [zoom-wang112358/MOLLEO](https://github.com/zoom-wang112358/MOLLEO) | 53 | Jun 2024 | Не указана | Удовл.–Хор. (неполный код) |
| 8 | Chemlactica | [yerevann/chemlactica](https://github.com/yerevann/chemlactica) | 36 | Sep 2024 | Не указана | Удовл.–Хор. (неполная документация) |
| 9 | SynFlowNet | [mirunacrt/synflownet](https://github.com/mirunacrt/synflownet) | 114 | ~2024 | MIT | Хорошо |
| 10 | RxnFlow | [SeonghwanSeo/RxnFlow](https://github.com/SeonghwanSeo/RxnFlow) | 31 | ~2024 | MIT | Удовл.–Хор. (**WARNING: код не воспроизводит статью**) |
| 11 | PMMG | [Liuyifeii/PMMG](https://github.com/Liuyifeii/PMMG) | 12 | Aug 2024 | MIT | Удовл. (Python 3.7, TF 2.5 — устарело) |
| 12 | ParetoDrug | [CNDOTA/ParetoDrug](https://github.com/CNDOTA/ParetoDrug) | 10 | ~2024 | MIT | Удовл. (Python 3.7, внешние данные) |
| 13 | AIxFuse | [biomed-AI/AIxFuse](https://github.com/biomed-AI/AIxFuse) | 11 | Oct 2024 | Apache-2.0 | Удовл. (**требует Schrodinger**) |
| 14 | DiffSBDD | [arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD) | 484 | Oct 2024 | MIT | Хорошо (авторы перешли на DrugFlow) |
| 15 | DiffGui | [QiaoyuHu89/DiffGui](https://github.com/QiaoyuHu89/DiffGui) | 63 | Jul 2025 | MIT | Хорошо |
| 16 | GFlowNet lib | [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) | 311 | Dec 2024 | Apache-2.0 | **Отлично** |
| 17 | HN-GFN | [violet-sto/HN-GFN](https://github.com/violet-sto/HN-GFN) | — | ~2023 | — | Удовл. |
| 18 | GEAM | [SeulLee05/GEAM](https://github.com/SeulLee05/GEAM) | — | ~2024 | — | Удовл.–Хор. |

**Scoring-инструменты (не генераторы, но критичные компоненты):**

| Инструмент | Репозиторий | Stars | Последний коммит | Usability |
|------------|------------|-------|-----------------|-----------|
| MolScore | [MorganCThomas/MolScore](https://github.com/MorganCThomas/MolScore) | 222 | Jan 2026 | **Отлично** |
| Boltz-2 | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) | 3700 | Sep 2025 | **Отлично** |
| Uni-Dock | [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock) | 274 | Jan 2025 | **Отлично** |
| GNINA 1.3 | [gnina/gnina](https://github.com/gnina/gnina) | 841 | Jan 2025 | **Отлично** |

---

## 4. Классификация методов по парадигме

### 4.1 SMILES-based RL (авторегрессия на строках)

| Метод | Backbone | RL-алгоритм | Предобучение | Ссылка |
|-------|----------|-------------|-------------|--------|
| **REINVENT4** | RNN (LSTM) / Transformer | REINFORCE + DAP | ChEMBL ~1.8M мол. | [Loeffler et al., *J. Cheminf.* 2024](https://link.springer.com/article/10.1186/s13321-024-00812-5) |
| **Saturn** | Mamba (SSM) | Augmented Memory (replay + aug.) | ChEMBL | [Guo & Schwaller, 2024](https://arxiv.org/abs/2405.17066) |
| **DrugEx v3** | Transformer + Pareto RL | Multi-obj RL | ChEMBL + target data | [Liu et al., *J. Cheminf.* 2023](https://link.springer.com/article/10.1186/s13321-023-00694-z) |
| **POLYGON** | SMILES LSTM + RL | Policy gradient + embedding | ChEMBL | [Bai et al., *Nat. Commun.* 2024](https://www.nature.com/articles/s41467-024-47120-y) |

### 4.2 Генетические / эволюционные алгоритмы

| Метод | Парадигма | Ссылка |
|-------|-----------|--------|
| **Graph GA (mol_ga)** | Crossover + mutation на графах | [Jensen, *Chem. Sci.* 2019](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c) |
| **MOLLEO** | LLM (BioT5/DeepSeek) как операторы мутации/кроссовера + GA | [Ye et al., ICLR 2025](https://openreview.net/forum?id=p5VDaa8aIY) |
| **Chemlactica** | Galactica/Gemma fine-tuned + GA + rejection sampling | [Tyszkiewicz et al., ICLR 2025](https://arxiv.org/abs/2407.18897) |

### 4.3 GFlowNet-методы (синтезориентированные)

| Метод | Действие | Venue | Ссылка |
|-------|----------|-------|--------|
| **SynFlowNet** | Реакции + buyable reactants | ICLR 2025 Spotlight | [Cretu et al.](https://arxiv.org/abs/2405.01155) |
| **RxnFlow** | 1.2M building blocks + 71 шаблон реакций | ICLR 2025 | [Seo et al.](https://github.com/SeonghwanSeo/RxnFlow) |
| **HN-GFN** | Hypernetwork для Pareto front | NeurIPS 2023 | [Zhu et al.](https://github.com/violet-sto/HN-GFN) |

### 4.4 MCTS-методы (Парето-ориентированные)

| Метод | Targets tested | Venue | Ссылка |
|-------|---------------|-------|--------|
| **PMMG** | EGFR + HER2 (7 целей) | *Adv. Sci.* 2025 | [Liu et al.](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202410640) |
| **ParetoDrug** | Множественные | *Commun. Biol.* 2024 | [DOI](https://www.nature.com/articles/s42003-024-06746-w) |

### 4.5 Фрагментные методы

| Метод | Парадигма | Ссылка |
|-------|-----------|--------|
| **FREED++** | GNN + SAC на фрагментах | [Telepov et al., 2024](https://arxiv.org/abs/2401.09840) |
| **GEAM** | GIB + SAC + GA на фрагментах | [Lee et al., ICML 2024](https://github.com/SeulLee05/GEAM) |
| **AIxFuse** | Dual self-play RL + pharmacophore fusion | [Li et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11234869/) |

### 4.6 3D-диффузионные модели

| Метод | Protein-aware | Venue | Ссылка |
|-------|--------------|-------|--------|
| **DiffSBDD** | Да (native) | *Nat. Comp. Sci.* 2024 | [Schneuing et al.](https://www.nature.com/articles/s43588-024-00737-x) |
| **DiffGui** | Да (native) | *Nat. Commun.* 2025 | [Hu et al.](https://www.nature.com/articles/s41467-025-63245-0) |

---

## 5. Сравнительная оценка по критериям

### 5.1 Sample Efficiency (оракульный бюджет)

Это **ключевой критерий** для замены FREED++ — именно 180K вызовов делают его неприемлемым.

| Метод | Бюджет (типичный) | PMO AUC (если есть) | Источник |
|-------|--------------------|---------------------|----------|
| **Saturn** | **1K–10K** | SOTA (превосходит 22 модели) | [Guo & Schwaller, 2024](https://arxiv.org/abs/2405.17066) |
| **Augmented Memory** (precursor Saturn) | 10K | Предыдущий SOTA (14/23 побед vs REINVENT) | [Guo et al., *JACS Au* 2024](https://pubs.acs.org/doi/10.1021/jacsau.4c00066) |
| **Chemlactica-1.3B** | 10K | +8% vs предыдущий SOTA | [Tyszkiewicz et al., ICLR 2025](https://arxiv.org/abs/2407.18897) |
| **REINVENT4** | 10K–100K | Стабильный top-3 | [Gao et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/8644353f7d307baaf29bc1e56fe8e0ec-Paper-Datasets_and_Benchmarks.pdf) |
| **Graph GA** | 10K | Стабильный top-3 (наравне с REINVENT) | [Gao et al., 2022] |
| **MOLLEO-D** | 10K | AUC sum = 12.23 | [ICLR 2025] |
| **DrugEx v3** | 10K–50K | Не оценён на PMO напрямую | [Liu et al., 2023] |
| **POLYGON** | ~часы (не PMO format) | N/A (свой бенчмарк) | [Bai et al., 2024] |
| **PMMG** | Не сообщено | N/A | [Liu et al., 2025] |
| **SynFlowNet** | ~10K–100K | N/A | [Cretu et al., 2025] |
| **RxnFlow** | ~10K–100K | N/A (pocket-conditional) | [Seo et al., 2025] |
| **FREED++** | **~180K** | **Не оценён** | [Telepov et al., 2024] |
| **DiffSBDD/DiffGui** | Нефиксированный (sampling) | N/A (другая парадигма) | [Schneuing et al., 2024] |

**Факт:** PMO-бенчмарк (10K budget) показал, что REINVENT и Graph GA — сильнейшие baseline'ы, которые «новые модели часто не превосходят» [Gao et al., 2022]. Saturn впервые существенно превзошёл этот baseline через Augmented Memory.

### 5.2 Multi-Objective поддержка

| Метод | Тип | Двухтаргетный докинг | Нативная Pareto | Примечания |
|-------|-----|---------------------|----------------|------------|
| **REINVENT4** | Weighted scalarization (TOML) | **Два компонента в конфиге, без кода** | Нет | 50+ встроенных scoring-компонентов |
| **Saturn** | Произведение нормализованных скоров | Модификация scoring-функции | Нет | Гибкий, но требует кода |
| **DrugEx v3** | **Pareto RL (native)** | Да, через Pareto-оптимизацию | **Да** | Специально спроектирован для multi-target |
| **POLYGON** | RL + dual reward | **Да — validated MEK1+mTOR** | Нет (скаляризация) | 32 соединения синтезированы и валидированы |
| **PMMG** | **Pareto MCTS** | **Да — EGFR+HER2, 7 целей** | **Да** | 51.65% success по 7 целям |
| **ParetoDrug** | **Pareto MCTS** | Да | **Да** | Target-aware generation |
| **HN-GFN** | **Hypernetwork Pareto** | Да | **Да** | Нативное Pareto-сэмплирование |
| **Graph GA** | Любой callable | Да (внешний) | Нет | Простейшая интеграция |
| **MOLLEO** | GA + LLM | Через objective function | Нет | LLM-мутации |
| **Chemlactica** | GA + rejection sampling | Через objective function | Нет | LLM-мутации |
| **GEAM** | Property-conditioned fragments | Да | Нет | SAC + GA |
| **AIxFuse** | Dual self-play RL | **Да — GSK3β+JNK3, RORγt+DHODH** | Нет | **Требует Schrodinger** |
| **SynFlowNet** | Через reward | Через reward composition | Нет | Синтезоориентированный |
| **DiffSBDD/DiffGui** | Post-hoc filtering | Ограниченно | Нет | 3D generation |
| **FREED++** | Один скаляр | **Модификация кода** | Нет | Нет multi-obj |

**Критический вывод:** Из методов с реальным кодом, **нативную Pareto-оптимизацию** для multi-target предлагают: DrugEx v3, PMMG, ParetoDrug, HN-GFN. REINVENT4 решает ту же задачу через scalarization. POLYGON — единственный экспериментально валидированный dual-target генератор.

### 5.3 Интеграция пользовательского скоринга (докинг)

| Метод | Как подключить внешний докинг | Сложность | Batch | Кэш |
|-------|------------------------------|-----------|-------|-----|
| **REINVENT4** | **ExternalProcess** (CLI) или plugin (Python) | **Без модификации ядра** | **Да** (v4.6+) | **Да** (SMILES-level) |
| **Saturn** | Модификация scoring-функции в коде | Средняя | Стандартный | Да (oracle cache) |
| **Graph GA** | Передать Python callable | **Минимальная** | Зависит от callable | Нет |
| **DrugEx v3** | Модификация reward | Средняя | Стандартный | Нет |
| **POLYGON** | Модификация reward | Средняя | Нет | Нет |
| **MOLLEO/Chemlactica** | Через objective function (PMO API) | Низкая–Средняя | Зависит от backend | Нет |
| **SynFlowNet** | Reward function (продемонстрировано с Boltz-2) | Средняя | Да (async workers) | Replay buffer |
| **PMMG** | Модификация scoring | Средняя–Высокая | Нет | Нет |
| **FREED++** | Замена QuickVina2 в коде | Высокая (hardcoded) | 12 подпроцессов | Нет |
| **DiffSBDD/DiffGui** | Post-hoc re-docking | N/A (не RL) | N/A | N/A |

### 5.4 Качество генерируемых молекул (docking scores)

Прямое сравнение затруднено — разные статьи используют разные мишени и метрики. Приведены данные из тех же публикаций:

**Из FREED++ paper (USP7, QuickVina2, convention: выше = лучше):**

| Метод | Avg DS | Top-5% DS | Uniqueness |
|-------|--------|-----------|------------|
| FREED++ | 9.66 ± 3.28 | **13.30 ± 0.13** | Низкая |
| REINVENT | 9.69 ± 0.39 | 12.05 ± 0.73 | Выше |

**Из Saturn paper (hit ratio %: DS threshold + QED > 0.5 + SA < 5):**

| Мишень | Saturn | GEAM |
|--------|--------|------|
| PARP1 | **57.98 ± 18.54** | 45.16 ± 2.41 |
| 5HT1B | **68.19 ± 3.40** | 47.66 ± 1.20 |

**Из POLYGON paper (MEK1+mTOR, AutoDock Vina):**
- 32 соединения синтезированы, валидированы in vitro. Joint dual-target hit rate не сообщён как числовой бенчмарк.

**Из PMMG paper (EGFR+HER2):**
- 2.96% молекул превзошли lapatinib по обеим мишеням одновременно.

**Из Thomas et al. (A₂A receptor, REINVENT + MolScore):**
- **88% hit rate** по связыванию, 50% функциональная активность, 3 наномолярных лиганда [*Nat. Commun.* 2025](https://www.nature.com/articles/s41467-025-60629-0).

**Замечание:** FREED++ показывает лучшие **raw docking scores**, но при 180K вызовах и низкой uniqueness. REINVENT + sophisticated scoring (MolScore) при 10K–50K вызовах достигает экспериментально подтверждённых хитов. Saturn при 10K вызовах достигает высоких hit ratio по MPO-критериям.

### 5.5 Синтезируемость

| Метод | SA score | Ретросинтез в loop | Guaranteed synth. | Ссылка |
|-------|---------|-------------------|-------------------|--------|
| **Saturn** | Компонент scoring | **Да** (AiZynth, RetroKNN, Graph2Edits, RootAligned) | Нет | [Guo & Schwaller, *Chem. Sci.* 2025](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc01476j) |
| **SynFlowNet** | N/A | N/A | **Да** (реакции + buyable reactants) | [Cretu et al., ICLR 2025](https://arxiv.org/abs/2405.01155) |
| **RxnFlow** | N/A | N/A | **Да** (1.2M blocks + templates) | [Seo et al., ICLR 2025](https://github.com/SeonghwanSeo/RxnFlow) |
| **REINVENT4** | Компонент scoring | Через ExternalProcess | Нет | [Loeffler et al., 2024] |
| **POLYGON** | Ease-of-synthesis reward | Нет | Нет, но 32 молекулы реально синтезированы | [Bai et al., 2024] |
| **FREED++** | Не контролируется | Нет | Нет | [Telepov et al., 2024] |
| **DiffSBDD/DiffGui** | Низкая (проблема 7-чл. циклов) | Нет | Нет | [Schneuing et al., 2024] |

### 5.6 Устойчивость к reward hacking

Yoshizawa et al. [*Nat. Commun.* 2025](https://www.nature.com/articles/s41467-025-57582-3) показали, что multi-objective RL-генерация подвержена reward hacking (предиктивные модели некорректно экстраполируют вне тренировочного распределения).

| Метод | Механизм защиты |
|-------|----------------|
| **REINVENT4** | DAP: σ·log(Prior/Agent) ограничивает drift от prior; curriculum learning |
| **Saturn** | Selective Memory Purge; oracle caching |
| **Graph GA** | Нет prior → нет drift; но нет и регуляризации |
| **DrugEx v3** | Pareto RL с diversity pressure |
| **POLYGON** | Embedding-based reward |
| **3D-diffusion** | Нет RL → нет reward hacking; но проблемы с реалистичностью |

### 5.7 Зрелость экосистемы

| Метод | Орг. | Индустрия | Эксп. валидация (in vitro) | Активность |
|-------|------|-----------|---------------------------|------------|
| **REINVENT4** | AstraZeneca | **Да** | **88% hit rate A₂A** [Thomas et al., 2025]; GAL+ESMACS [Loeffler et al., 2024] | Очень высокая (674★, Nov 2025) |
| **Saturn** | EPFL | Нет | Нет in vitro, но generative VS из 142B библиотеки | Средняя (70★, Jul 2024) |
| **DrugEx v3** | Leiden Univ. | Нет | Упомянута in silico валидация | Хорошая (166★, Dec 2024) |
| **POLYGON** | Harvard/Dana-Farber | Нет | **32 молекулы синтезированы и валидированы** (MEK1+mTOR) | Низкая (42★, Feb 2024) |
| **Graph GA** | Независимый | Нет | Нет | Высокая (pip install) |
| **MOLLEO** | Академия | Нет | Нет | Средняя (53★) |
| **PMMG** | Академия | Нет | In silico (EGFR+HER2) | Низкая (12★, устаревшие зависимости) |
| **SynFlowNet** | Mila / Recursion | Частично | Boltz-2 + ABFE валидация (TYK2) | Средняя (114★) |
| **FREED++** | AIRI | Нет | Нет | **Стагнация** (8★, Dec 2023) |

### 5.8 Wall-clock время при реалистичных скорерах

Двухтаргетная задача: 2 × docking на молекулу.

| Скорер | Скорость | FREED++ (180K×2) | REINVENT4 (50K×2) | Saturn (10K×2) | Graph GA (10K×2) |
|--------|----------|-----------------|-------------------|---------------|-----------------|
| **Uni-Dock** (GPU) | ~0.1 с/мол | ~10 ч | ~2.8 ч | ~0.6 ч | ~0.6 ч |
| **GNINA 1.3** (CPU rescore) | ~16 с/мол | ~1600 ч (67 дней) | ~444 ч (18.5 дней) | ~89 ч (3.7 дня) | ~89 ч |
| **Boltz-2** (1 GPU) | ~20 с/мол | ~2000 ч (83 дня) | ~556 ч (23 дня) | ~111 ч (4.6 дня) | ~111 ч |
| **QuickVina2** (CPU) | ~10 с/мол | ~1000 ч | ~278 ч | ~56 ч | ~56 ч |

**Вывод:** При Uni-Dock разница между 10K и 50K бюджетами несущественна (минуты vs часы). При дорогих скорерах (Boltz-2, GNINA) бюджет ≤10K критичен — преимущество Saturn.

---

## 6. Дифференциальная оценка: сводная матрица

Критерии оцениваются по шкале 1–5 (5 = лучше). Веса отражают приоритеты задачи SWW2026.

| Критерий | Вес | REINVENT4 | Saturn | DrugEx v3 | Graph GA | POLYGON | MOLLEO | SynFlowNet | PMMG | FREED++ |
|----------|-----|-----------|--------|-----------|----------|---------|--------|------------|------|---------|
| Sample efficiency (≤10K) | 0.20 | 3 | **5** | 3 | 4 | 2 | 3 | 3 | 2 | **1** |
| Multi-obj / dual-target | 0.20 | 4 | 3 | **5** | 3 | **5** | 3 | 3 | **5** | 1 |
| Custom scoring integration | 0.15 | **5** | 4 | 3 | **5** | 3 | 3 | 3 | 2 | 2 |
| Код: зрелость + поддержка | 0.15 | **5** | 3 | 4 | **5** | 3 | 2 | 3 | 1 | 1 |
| Docking score качество | 0.10 | 4 | 4 | 3 | 3 | 4 | 3 | 3 | 4 | **5** |
| Синтезируемость | 0.05 | 3 | **5** | 3 | 2 | 3 | 2 | **5** | 2 | 2 |
| Экспериментальная валидация | 0.05 | **5** | 2 | 2 | 1 | **5** | 1 | 3 | 1 | 1 |
| Reward hacking устойчивость | 0.05 | 4 | 4 | 3 | 2 | 3 | 2 | 3 | 3 | 3 |
| Документация / onboarding | 0.05 | **5** | 3 | 3 | **5** | 3 | 2 | 3 | 1 | 2 |
| | | | | | | | | | | |
| **Взвешенная оценка** | | **4.10** | **3.80** | **3.55** | **3.75** | **3.40** | **2.60** | **3.15** | **2.55** | **1.55** |

### Рейтинг

| Место | Метод | Оценка | Сильные стороны | Слабые стороны |
|-------|-------|--------|----------------|----------------|
| 1 | **REINVENT4** | 4.10 | Зрелость, экосистема, custom scoring без правки кода, curriculum learning, AstraZeneca backing, экспериментальная валидация A₂A | Не лучшая sample efficiency; нет нативного Pareto |
| 2 | **Saturn** | 3.80 | Лучшая sample efficiency (1K–10K), ретросинтез в loop, SOTA на PMO | Требует модификации кода для dual-target; менее зрелая экосистема |
| 3 | **Graph GA** | 3.75 | Предельная простота, pip install, любой callable как objective, стабильный baseline | Нет RL — нет transfer learning; нет prior для drug-likeness |
| 4 | **DrugEx v3** | 3.55 | Нативный Pareto RL для multi-target; scaffold-constrained | Менее тестирован на PMO; неполная документация |
| 5 | **POLYGON** | 3.40 | Единственный с in vitro валидацией dual-target (MEK1+mTOR, 32 synth.) | Медленный; старый код (Feb 2024); не PMO-формат |
| 6 | **SynFlowNet** | 3.15 | Guaranteed synthesizability; Boltz-2 интеграция показана | Фиксированное пространство реакций; не для нестандартной химии |
| 7 | **MOLLEO** | 2.60 | LLM-мутации; не требует training | Дорогой inference; неполный код; нет лицензии |
| 8 | **PMMG** | 2.55 | Нативный Pareto; 7 целей одновременно; EGFR+HER2 | Python 3.7, TF 2.5 — критически устарело |
| 9 | **FREED++** | 1.55 | Лучшие raw docking scores | 180K вызовов; стагнация; нет multi-obj |

---

## 7. Анализ top-3 кандидатов для SWW2026

### 7.1 REINVENT4 — детальный разбор

**За:**
- **ExternalProcess**: любой скорер (Uni-Dock, Boltz-2, GNINA) вызывается как CLI без правки ядра кода [Loeffler et al., 2024].
- **Curriculum learning**: Stage 1 → дешёвые фильтры (QED, SA, PAINS), Stage 2 → Uni-Dock target A, Stage 3 → добавить target B. Это экономит оракульный бюджет на ранних стадиях.
- **Промышленная валидация**: AstraZeneca внутренне использует; AMD оптимизировала для MI300X [AMD ROCm Blog 2025](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html); 88% hit rate на A₂A-рецепторе [Thomas et al., *Nat. Commun.* 2025](https://www.nature.com/articles/s41467-025-60629-0).
- **674 stars, 194 forks, активные discussions** — крупнейшее сообщество среди молекулярных генераторов.
- **MolScore-совместим** [Thomas et al., *J. Cheminf.* 2024](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w) — стандартизированная оценка.

**Против:**
- Sample efficiency **не лучшая**: при 10K бюджете уступает Saturn и Graph GA на PMO.
- Нет нативного Pareto: двухтаргетная задача решается через scalarization (weighted product/sum), что может упускать часть Pareto-фронта.
- DAP-алгоритм: σ-параметр требует тюнинга; при неверном σ модель может коллапсировать к prior или уйти в режим mode collapse.

### 7.2 Saturn — детальный разбор

**За:**
- **SOTA sample efficiency**: PMO benchmark top-1; 1K вызовов достаточно для ретросинтез-оптимизации [Guo & Schwaller, *Chem. Sci.* 2025](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc01476j).
- **Augmented Memory**: multiple updates per oracle call — критично при дорогих скорерах.
- **Selective Memory Purge**: защита от mode collapse.
- **Ретросинтез в loop**: единственный генератор с прямой оптимизацией синтезируемости (AiZynth, RetroKNN и др.).
- **Mamba backbone**: эффективнее LSTM по памяти и скорости на длинных последовательностях.

**Против:**
- **Менее зрелая экосистема**: 70 stars, 22 коммита, нет активной поддержки сообществом.
- **Модификация кода** для dual-target scoring — нет ExternalProcess, нет TOML-конфига.
- **Нет экспериментальной (in vitro) валидации** сгенерированных молекул.
- **Зависимости**: GLIBCXX issues, mamba-ssm может быть проблемным в установке.
- **Дисперсия высокая**: Saturn hit ratio на PARP1 = 57.98 ± **18.54** — большой разброс между запусками.

### 7.3 Graph GA (mol_ga) — детальный разбор

**За:**
- **Предельная простота**: `pip install mol_ga`, objective = любая Python-функция.
- **Стабильный baseline**: на PMO наравне с REINVENT, несмотря на простоту [Gao et al., 2022].
- **Нет training/prior** — нет проблемы reward hacking через drift от prior.
- **Быстрый прототип**: можно запустить dual-target experiment за часы.

**Против:**
- **Нет prior из ChEMBL** — генерируемые молекулы могут быть менее drug-like.
- **Нет transfer learning** — не учится между запусками.
- **Нет curriculum learning** — все scoring-компоненты сразу.
- **Ограниченная diversity**: crossover на SMILES графах может застревать в локальных оптимумах.

---

## 8. Обзор scoring-инструментов для pipeline

| Скорер | Тип | Скорость | Точность | Batch GPU | Как reward в RL | Ссылка |
|--------|-----|----------|----------|-----------|----------------|--------|
| **Uni-Dock** | Empirical (Vina) | **~0.1 с/мол** (V100) | Средняя | **Да** (1000+) | Идеально: быстрый, batch | [Yu et al., *JCTC* 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145) |
| **Boltz-2** | AI foundation model | ~20 с/мол | Близко к FEP (Pearson 0.62) | Средний | Хорошо, но медленно; дебаты о generalization | [Wohlwend et al., *bioRxiv* 2025](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1) |
| **GNINA 1.3** | CNN rescore | ~16 с/мол (CPU) | Лучше Vina (Top1: 73% vs 58%) | Нет | Хорошо; CNN scoring → меньше reward hacking vs Vina | [McNutt et al., *J. Cheminf.* 2025](https://link.springer.com/article/10.1186/s13321-025-00973-x) |
| **QuickVina2** | Empirical | ~10 с/мол (CPU) | Средняя | Нет | ОК; используется FREED++ и Saturn по умолчанию | Alhossary et al., 2015 |

---

## 9. Ограничения данного обзора

1. **PMO-бенчмарк не покрывает docking-задачи** — он оценивает proxy-оракулы (QED, LogP, GSK3β predictors), а не реальный докинг. Методы, лучшие на PMO, не обязательно лучшие при реальном докинге.
2. **Сравнение docking scores между статьями невалидно** — разные мишени, разные конвенции (kcal/mol vs нормализованные), разные инструменты (QuickVina2 vs AutoDock Vina vs Glide).
3. **Отсутствие единого бенчмарка для dual-target** — каждая статья использует свои пары мишеней.
4. **Код ≠ воспроизводимость** — наличие репозитория не гарантирует воспроизводимость (см. RxnFlow WARNING).
5. **Экспериментальная валидация крайне ограничена** — только REINVENT (A₂A, 2025) и POLYGON (MEK1+mTOR, 2024) имеют in vitro данные.
6. **LLM-методы** (MOLLEO, Chemlactica, MT-Mol) — новая парадигма без достаточной проверки на docking-задачах.

---

## 10. Источники

### Генераторы
- [REINVENT4] Loeffler, H.H. et al. *J. Cheminformatics* 16, 20 (2024). [DOI](https://link.springer.com/article/10.1186/s13321-024-00812-5)
- [Saturn] Guo, J. & Schwaller, P. arXiv:2405.17066 (2024). [DOI](https://arxiv.org/abs/2405.17066)
- [FREED++] Telepov, A. et al. *TMLR* 2024. [DOI](https://arxiv.org/abs/2401.09840)
- [Augmented Memory] Guo, J. et al. *JACS Au* 4(6), 2024. [DOI](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)
- [Graph GA] Jensen, J.H. *Chem. Sci.* 10, 3567 (2019). [DOI](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)
- [DrugEx v3] Liu, X. et al. *J. Cheminformatics* 15, 17 (2023). [DOI](https://link.springer.com/article/10.1186/s13321-023-00694-z)
- [POLYGON] Bai, Q. et al. *Nat. Commun.* 15, 3826 (2024). [DOI](https://www.nature.com/articles/s41467-024-47120-y)
- [MOLLEO/Chemlactica] Ye, H. et al. / Tyszkiewicz, M. et al. ICLR 2025. [arXiv:2407.18897](https://arxiv.org/abs/2407.18897)
- [SynFlowNet] Cretu, M. et al. ICLR 2025 Spotlight. [arXiv:2405.01155](https://arxiv.org/abs/2405.01155)
- [RxnFlow] Seo, S. et al. ICLR 2025. [GitHub](https://github.com/SeonghwanSeo/RxnFlow)
- [PMMG] Liu, Y. et al. *Adv. Sci.* 2025. [DOI](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202410640)
- [ParetoDrug] *Commun. Biol.* 2024. [DOI](https://www.nature.com/articles/s42003-024-06746-w)
- [AIxFuse] Li, Z. et al. *PMC* 2024. [DOI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11234869/)
- [DiffSBDD] Schneuing, A. et al. *Nat. Comp. Sci.* 2024. [DOI](https://www.nature.com/articles/s43588-024-00737-x)
- [DiffGui] Hu, Q. et al. *Nat. Commun.* 2025. [DOI](https://www.nature.com/articles/s41467-025-63245-0)
- [GEAM] Lee, S. et al. ICML 2024. [GitHub](https://github.com/SeulLee05/GEAM)
- [HN-GFN] Zhu, H. et al. NeurIPS 2023. [GitHub](https://github.com/violet-sto/HN-GFN)
- [MTMol-GPT] Guofei, T. et al. *PLOS Comp. Biol.* 2024. [DOI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012229)

### Бенчмарки и frameworks
- [PMO] Gao, W. et al. *NeurIPS* 2022. [PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/8644353f7d307baaf29bc1e56fe8e0ec-Paper-Datasets_and_Benchmarks.pdf)
- [MolScore] Thomas, M. et al. *J. Cheminformatics* 2024. [DOI](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w)

### Валидация
- [A₂A] Thomas, M. et al. *Nat. Commun.* 16, 5485 (2025). [DOI](https://www.nature.com/articles/s41467-025-60629-0)
- [GAL+ESMACS] Loeffler, H.H. et al. *JCTC* 20(18), 8308–8328 (2024). [DOI](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00576)
- [ADMETrix] Mourdoukoutas, N. et al. ICANN 2025 / LNCS 2026. [ChemRxiv](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-3x5nq-v3)
- [Saturn+Retro] Guo, J. & Schwaller, P. *Chem. Sci.* 2025, 16, 6943–6956. [DOI](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc01476j)

### Скоринг
- [Boltz-2] Wohlwend, J. et al. *bioRxiv* 2025. [DOI](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
- [Uni-Dock] Yu, Y. et al. *JCTC* 2023. [DOI](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145)
- [GNINA 1.3] McNutt, A.T. et al. *J. Cheminformatics* 2025. [DOI](https://link.springer.com/article/10.1186/s13321-025-00973-x)

### Reward hacking
- [Yoshizawa et al.] *Nat. Commun.* 2025. [DOI](https://www.nature.com/articles/s41467-025-57582-3)

### Hardware
- [AMD MI300X] AMD ROCm Blog, 2025. [URL](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html)
- [RL-AL] Sundin, I. et al. *Chem. Sci.* 2024. [DOI](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d3sc04653b)
