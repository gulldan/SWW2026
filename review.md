ПОЛНЫЙ ЛИТЕРАТУРНЫЙ ОБЗОР И GAP-АНАЛИЗ

  Исследовательский вопрос:

  Can dual-pocket-conditioned de novo molecular generation produce more selective ligands for structurally similar binding sites compared to single-target generation with subsequent off-target filtering, without compromising drug-likeness
  properties?

  ---
  I. ЧТО УЖЕ СУЩЕСТВУЕТ (по категориям)

  A. Dual/Multi-target генерация (полифармакология — binding BOTH targets)

  Эти работы генерируют молекулы, которые связывают оба таргета одновременно. Это НЕ selectivity, а polypharmacology:

  ┌────────────────────────────────────────────────────────────────────────────────┬──────┬────────────────────┬─────────────────────────┬─────────────────────────────────────────────────┐
  │                                     Метод                                      │ Год  │       Журнал       │         Таргеты         │                    Валидация                    │
  ├────────────────────────────────────────────────────────────────────────────────┼──────┼────────────────────┼─────────────────────────┼─────────────────────────────────────────────────┤
  │ https://arxiv.org/abs/2306.13957                                               │ 2024 │ IEEE               │ D2R + 5-HT1A            │ In silico                                       │
  ├────────────────────────────────────────────────────────────────────────────────┼──────┼────────────────────┼─────────────────────────┼─────────────────────────────────────────────────┤
  │ https://www.nature.com/articles/s41467-024-47120-y                             │ 2024 │ Nat Comms          │ MEK1 + mTOR             │ 32 соединения синтезированы, >50% ингибирование │
  ├────────────────────────────────────────────────────────────────────────────────┼──────┼────────────────────┼─────────────────────────┼─────────────────────────────────────────────────┤
  │ https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012229 │ 2024 │ PLOS Comp Biol     │ DRD2 + HTR1A            │ Docking                                         │
  ├────────────────────────────────────────────────────────────────────────────────┼──────┼────────────────────┼─────────────────────────┼─────────────────────────────────────────────────┤
  │ https://doi.org/10.1101/2023.12.10.571029                                      │ 2023 │ bioRxiv → Chem Sci │ GSK3b/JNK3, RORgt/DHODH │ FEP, 5x vs SOTA                                 │
  ├────────────────────────────────────────────────────────────────────────────────┼──────┼────────────────────┼─────────────────────────┼─────────────────────────────────────────────────┤
  │ https://doi.org/10.1101/2025.11.04.686584                                      │ 2025 │ bioRxiv            │ JNK3/GSK3b, PI3K/PARP1  │ In silico                                       │
  └────────────────────────────────────────────────────────────────────────────────┴──────┴────────────────────┴─────────────────────────┴─────────────────────────────────────────────────┘

  B. Selectivity через post-hoc оптимизацию (single-target + filtering/optimization)

  ┌───────────────────────────────────────────────────────────────────────────────────────────────┬──────┬──────────────┬─────────────────────────────────────────┬────────────────────┬───────────────────────────────────────────────┐
  │                                             Метод                                             │ Год  │    Журнал    │                Стратегия                │      Таргеты       │                   Результат                   │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://www.nature.com/articles/s43588-024-00737-x                                            │ 2024 │ Nat Comp Sci │ Negative design через modified sampling │ BIKE vs MPSK1      │ On-target: -7.2→-13.9; Off-target: -10.8→-8.7 │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://pmc.ncbi.nlm.nih.gov/articles/PMC11350878/                                            │ 2024 │ JCIM         │ Evolutionary latent search              │ ATM vs Vps34       │ -9.3 vs -6.1 kcal/mol                         │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://www.nature.com/articles/s42003-025-07840-3                                            │ 2025 │ Comms Biol   │ Pharmacophore matching                  │ PARP1 vs PARP2     │ >787x selectivity, wet-lab                    │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202502702                           │ 2025 │ Adv Sci      │ NCI pattern guidance                    │ EGFR mutant vs WT  │ 0.69 kcal/mol median difference               │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://proceedings.neurips.cc/paper/2020/hash/2d16ad1968844a4300e9a490588ff9f8-Abstract.html │ 2020 │ NeurIPS      │ VAE + CLaSS multi-attribute             │ SARS-CoV-2 targets │ 87-95% docking <-6 kcal/mol                   │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ https://www.nature.com/articles/s41467-024-47613-w                                            │ 2024 │ Nat Comms    │ Post-hoc scoring                        │ PPARg vs PPARd     │ Wet-lab validated, crystal structure          │
  ├───────────────────────────────────────────────────────────────────────────────────────────────┼──────┼──────────────┼─────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┤
  │ REACTOR                                                                                       │ 2021 │ ACS Omega    │ RL penalty                              │ DRD2 vs D1/D3      │ Off-target penalty                            │
  └───────────────────────────────────────────────────────────────────────────────────────────────┴──────┴──────────────┴─────────────────────────────────────────┴────────────────────┴───────────────────────────────────────────────┘

  C. Критическое открытие от BInD: существующие модели НЕ обеспечивают selectivity

  https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202502702 (2025) провёл прямое тестирование и обнаружил, что TargetDiff и Pocket2Mol не способны генерировать pocket-селективные молекулы — разница аффинности между двумя pocket'ами
  статистически незначима. Это прямое доказательство того, что single-pocket conditioning не решает задачу selectivity.

  D. Benchmarks для selectivity

  ┌───────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────┬───────────────────────────────────────────────┐
  │                                   Benchmark                                   │            Что измеряет            │                  Ограничения                  │
  ├───────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┤
  │ https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w (2024) │ 5-HT2A/D2 dual docking selectivity │ Selectivity "barely improved" при оптимизации │
  ├───────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┤
  │ DOCKSTRING                                                                    │ JAK2/LCK selectivity tasks         │ Ограниченный набор                            │
  ├───────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┤
  │ https://arxiv.org/abs/2406.03403                                              │ Single-target binding only         │ НЕТ selectivity metrics                       │
  ├───────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────┼───────────────────────────────────────────────┤
  │ https://pubs.acs.org/doi/10.1021/acs.jcim.4c01598 (2024)                      │ 32 targets, 2D/3D metrics          │ Single-target only                            │
  └───────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────┴───────────────────────────────────────────────┘

  E. Данные ChEMBL: target families с богатыми selectivity данными

  ┌─────────────┬─────────────────┬───────────────────┬─────────────────────────────────────────────┐
  │ Target pair │   ChEMBL IDs    │ IC50 measurements │           Selectivity achievable            │
  ├─────────────┼─────────────────┼───────────────────┼─────────────────────────────────────────────┤
  │ JAK1/JAK2   │ CHEMBL2835/2971 │ >15,000 combined  │ 3x biochemical, 75x cellular (upadacitinib) │
  ├─────────────┼─────────────────┼───────────────────┼─────────────────────────────────────────────┤
  │ CDK4/CDK6   │ CHEMBL331/2508  │ 1000+             │ Практически нет — unmet need                │
  ├─────────────┼─────────────────┼───────────────────┼─────────────────────────────────────────────┤
  │ PI3Ka/PI3Kd │ CHEMBL4005/3130 │ 5000+             │ >10x (alpelisib), >50x (idelalisib)         │
  ├─────────────┼─────────────────┼───────────────────┼─────────────────────────────────────────────┤
  │ EGFR WT/mut │ CHEMBL203       │ 10000+            │ Mutant-selective possible                   │
  └─────────────┴─────────────────┴───────────────────┴─────────────────────────────────────────────┘

  F. Клинические испытания: selectivity-by-design работает

  - https://clinicaltrials.gov/ct2/show/NCT06982521 (Relay Therapeutics) — mutant-selective PI3Ka inhibitor, Phase 3, n=540. Самый продвинутый computationally designed selectivity-optimized drug
  - 7 distinct mutant-selective PI3Ka programs в клинике (STX-478, OKI-219, CGT6297, ETX-636)
  - 20+ AI-designed compounds в клинике от Insilico, Exscientia, Relay
  - INS018_055 (Insilico) — Phase 2 completed с результатами

  G. HuggingFace: ML-landscape

  Из ~15 ведущих SBDD моделей (MolCRAFT, MolJO, DrugFlow, DrugGPS, DecompOpt, Apo2Mol, 3DMolFormer и др.) — ни одна не поддерживает dual-pocket conditioning для selectivity. Все работают с single pocket.

  ---
  II. ОПРЕДЕЛЁННЫЙ GAP

  Что НИКТО не сделал:

  1. Нет прямого head-to-head сравнения: dual-pocket conditioning vs single-target + off-target filtering на одном benchmark'е
  2. Нет модели с a priori dual-pocket conditioning: DiffSBDD делает negative design post-hoc (через sampling), DrugHIVE оптимизирует последовательно, CMD-GEN сравнивает pharmacophore clouds — но ни одна модель не conditioning на обоих pocket'ах
  одновременно во время генерации
  3. Нет стандартизированного selectivity benchmark: CrossDocked2020 и POKMOL-3D не включают selectivity metrics. MolScore имеет только 5-HT2A/D2
  4. BInD доказал, что существующие модели fail на selectivity: TargetDiff и Pocket2Mol не различают pocket'ы — это прямая мотивация для вашей работы
  5. Drug-likeness trade-off не изучен: Неизвестно, как dual conditioning влияет на QED/SA/Lipinski vs single-target + filtering

  ---
  III. ВЕРДИКТ

  Можете писать статью — gap чёткий и значимый

  Ваш вопрос попадает точно в пробел между:
  - Polypharmacology (DiffDTM, POLYGON — binding both)
  - Post-hoc selectivity (DiffSBDD, DrugHIVE — generate then optimize)
  - Failure of current models (BInD showed TargetDiff/Pocket2Mol fail at selectivity)

  ---
  IV. ПЛАН ЭКСПЕРИМЕНТОВ ДЛЯ СТАТЬИ

  1. Selectivity benchmark (новый вклад)

  Построить benchmark из 4-6 пар структурно-похожих таргетов:
  - JAK1/JAK2 (gold standard, >15K IC50 в ChEMBL)
  - CDK4/CDK6 (unmet need — нет selective drugs)
  - PI3Ka/PI3Kd (clinical validation exists)
  - EGFR WT/mutant (BInD уже тестировал)
  - PARP1/PARP2 (CMD-GEN уже тестировал, wet-lab)
  - D2R/D3R (классика GPCR selectivity)

  2. Методы для сравнения (3 парадигмы)

  - Baseline A: Single-pocket generation (TargetDiff/DiffSBDD/MolCRAFT) + post-hoc docking filter
  - Baseline B: DiffSBDD negative design (iterative noising/denoising + dual scoring)
  - Proposed: True dual-pocket conditioned generation (conditioning на target + anti-target во время generation)

  3. Метрики (comprehensive)

  ┌───────────────┬──────────────────────────────────────────────────────────┐
  │   Категория   │                         Метрики                          │
  ├───────────────┼──────────────────────────────────────────────────────────┤
  │ Selectivity   │ ΔVina score, selectivity ratio, hit rate (% > threshold) │
  ├───────────────┼──────────────────────────────────────────────────────────┤
  │ Drug-likeness │ QED, SA, Lipinski, logP                                  │
  ├───────────────┼──────────────────────────────────────────────────────────┤
  │ Quality       │ Validity, uniqueness, novelty, PoseBusters pass rate     │
  ├───────────────┼──────────────────────────────────────────────────────────┤
  │ Diversity     │ Tanimoto distance, scaffold diversity                    │
  ├───────────────┼──────────────────────────────────────────────────────────┤
  │ Efficiency    │ Time per molecule, compute cost                          │
  └───────────────┴──────────────────────────────────────────────────────────┘

  4. Wet-lab validation (для high-impact)

  - Синтез 10-20 top candidates для 1-2 target pairs
  - Binding assay on-target + off-target
  - Selectivity index measurement

  5. Целевые журналы

  - Nature Computational Science (DiffSBDD published there)
  - Nature Communications (POLYGON, PMDM, DRAGONFLY, CMD-GEN)
  - Advanced Science (BInD published there)
  - NeurIPS/ICML (if ML methodology focus)

  ---
  V. КЛЮЧЕВЫЕ ССЫЛКИ ДЛЯ RELATED WORK

  Обязательно цитировать:

  Selectivity в SBDD:
  - DiffSBDD negative design — https://www.nature.com/articles/s43588-024-00737-x
  - BInD selectivity failure analysis — https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202502702
  - CMD-GEN PARP1/2 selectivity — https://www.nature.com/articles/s42003-025-07840-3
  - DrugHIVE ATM/Vps34 — https://pubs.acs.org/doi/10.1021/acs.jcim.4c01193

  Multi-target generation:
  - POLYGON — https://www.nature.com/articles/s41467-024-47120-y
  - DiffDTM — https://arxiv.org/abs/2306.13957
  - AIxFuse — https://doi.org/10.1101/2023.12.10.571029

  Benchmarks:
  - MolScore selectivity — https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC8310368/ — Cournia et al., 2020
  - ATOM-GMD H1 selectivity — https://doi.org/10.1101/2023.02.14.528391

  Clinical translation:
  - RLY-2608 mutant-selective PI3Ka — Phase 3 (https://clinicaltrials.gov/ct2/show/NCT06982521)
