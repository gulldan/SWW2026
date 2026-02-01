# Multi-Objective / Multi-Target Optimization in Molecular Generation and Drug Design

**Обзор современных подходов (2023-2026)**

---

## 1. Pareto-based RL для молекул

### 1.1 CPRL — Clustered Pareto-based Reinforcement Learning (2024)

- **Paper:** [Neural Networks, August 2024](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005203)
- **Суть:** CPRL предобучает генеративную модель на молекулярных данных, затем применяет кластеризацию сгенерированных молекул на "сбалансированные" и "несбалансированные" группы, после чего строит Pareto-фронты для ранжирования по нескольким целям. Специальная exploration-стратегия совместно контролирует распределение вероятностей через exploration model и agent.
- **Преимущества над worst-case (max для lower-is-better):** Кластеризация позволяет удалить несбалансированные молекулы до ранжирования, а Pareto-фронт находит компромисс без фиксированных весов. Desirability = 0.9551, validity = 0.9923.
- **Применимость к dual-target docking:** Высокая. Два docking score могут быть двумя objectives в Pareto-ранжировании. Кластеризация помогает фильтровать молекулы, хорошие только по одному таргету.

### 1.2 MLPS — Learning Latent Pareto Set (ICLR 2024)

- **Paper:** [OpenReview / ICLR 2024](https://openreview.net/forum?id=S62iZf0cba)
- **Суть:** Первый метод обучения отображения из пространства предпочтений (direction vectors) в полное Pareto-множество. Encoder-decoder трансформирует дискретное химическое пространство в непрерывное латентное, локальное BO ищет оптимумы, глобальная модель учится покрывать весь Pareto-фронт.
- **Преимущества над worst-case:** Позволяет decision-maker'у динамически настраивать предпочтения и сразу получать соответствующие молекулы без переобучения. Покрывает весь Pareto-фронт, а не одну точку.
- **Применимость к dual-target docking:** Прямая. Два docking score задают 2D objective space; предпочтения задают баланс между таргетами.

### 1.3 DrugEx v2/v3 — Pareto-based Multi-Objective RL in Polypharmacology

- **Paper v2:** [J. Cheminformatics, 2021](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00561-9)
- **Paper v3:** [J. Cheminformatics, 2023](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00694-z)
- **GitHub:** [CDDLeiden/DrugEx](https://github.com/CDDLeiden/DrugEx)
- **Суть:** DrugEx v2 перенёс концепции mutation и crossover из эволюционных алгоритмов в RL, интегрировав Pareto ranking для обработки противоречивых целей. v3 добавил scaffold-constrained генерацию на Graph Transformer с гарантией химической валидности на каждом шаге.
- **Преимущества над worst-case:** Pareto ranking позволяет генерировать разнообразные молекулы по всему фронту, а не только в одной точке компромисса. Эволюционные операторы увеличивают diversity.
- **Применимость к dual-target docking:** Прямая — DrugEx изначально разработан для polypharmacology (несколько таргетов одновременно). Продемонстрирован на A2AAR + QED.

### 1.4 ParetoDrug — Pareto MCTS (2024)

- **Paper:** [Communications Biology, 2024](https://www.nature.com/articles/s42003-024-06746-w)
- **GitHub:** [CNDOTA/ParetoDrug](https://github.com/CNDOTA/ParetoDrug)
- **Суть:** Использует Monte Carlo Tree Search в химическом пространстве с pretrained atom-by-atom autoregressive model. Scheme ParetoPUCT балансирует exploration/exploitation при выборе следующего атома. Оптимизирует docking score, LogP, QED, SA, NP-likeness синхронно.
- **Преимущества над worst-case:** MCTS с Pareto PUCT ищет по всему фронту без фиксации весов. Более высокие docking scores, чем REINVENT 4 при joint optimization.
- **Применимость к dual-target docking:** Прямая — метод target-aware и multi-objective by design.

### 1.5 PMMG — Pareto MCTS Molecular Generation (2025)

- **Paper:** [Advanced Science, 2025](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202410640)
- **Суть:** RNN-генератор + MCTS для поиска Pareto Front в высокоразмерном пространстве целей. Успешно оптимизирует 7 целей одновременно с success rate 51.65%, что в 2.5x лучше SOTA.
- **Преимущества над worst-case:** Масштабируется на many-objective (7+), ищет весь фронт, не требует весов.
- **Применимость к dual-target docking:** Валидирован на dual-target задаче EGFR + HER2.

---

## 2. Scalarization vs. Pareto: Сравнение подходов агрегации скоров

### 2.1 Методы агрегации скоров

| Метод | Формула | Плюсы | Минусы |
|-------|---------|-------|--------|
| **Weighted Sum** | `w_A * s_A + w_B * s_B` | Простота, дифференцируемость | Не находит non-convex Pareto front; требует выбора весов |
| **Weighted Product** | `s_A^w_A * s_B^w_B` | Мультипликативный эффект, подавляет outlier'ы | Чувствительность к масштабу |
| **Worst-case (Pessimistic)** | `max(s_A, s_B)` при lower-is-better | Гарантия по всем таргетам | Игнорирует выигрыш по другим objectives |
| **Tchebycheff** | `max_i(w_i * |f_i - z_i*|)` | Находит non-convex front | Нужна утопическая точка z* |
| **Harmonic Mean** | `2/(1/s_A + 1/s_B)` | Баланс, наказывает экстремальные значения | Чувствительность к нулям |
| **Geometric Mean** | `sqrt(s_A * s_B)` | Мягче worst-case, жёстче sum | Нужна трансформация при отрицательных значениях |
| **Pareto (EHVI)** | Hypervolume improvement | Находит весь фронт, нет весов | Вычислительно дороже |

### 2.2 Pareto vs. Scalarization: эмпирические результаты

- **Paper:** [Pareto optimization to accelerate multi-objective virtual screening, Digital Discovery (RSC), 2024](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00227f)
- **Paper:** [Bayesian Optimization for Molecules Should Be Pareto-Aware, arXiv 2025](https://arxiv.org/html/2507.13704)
- **Вывод:** EHVI (Expected Hypervolume Improvement) consistently outperforms scalarized EI по покрытию Pareto front, скорости сходимости и химическому разнообразию. Даже сильные scalarization-варианты проигрывают в low-data режиме.
- **Weighted sum не находит non-convex участки Pareto front** — это фундаментальное ограничение.

### 2.3 Tchebycheff в Multi-Objective BO

- **Paper:** [ASME J. Mech. Des., 2022](https://asmedigitalcollection.asme.org/mechanicaldesign/article/144/1/011703/1114631)
- **Суть:** Weighted Tchebycheff формулирует acquisition function через max отклонение от утопической точки; находит и convex, и non-convex участки Pareto front. Утопическая точка оценивается итеративно через MLR.
- **Применимость к dual-target docking:** Подходит напрямую — z* = (лучший score_A, лучший score_B), Tchebycheff минимизирует максимальное отклонение.

---

## 3. Conditional Generation по нескольким свойствам

### 3.1 Conditional VAE (CVAE) для Multi-Property Generation

- **Paper:** [Enhancing De Novo Drug Design across Multiple Therapeutic Targets with CVAE, ACS Omega, 2024](https://pubs.acs.org/doi/10.1021/acsomega.4c08027)
- **Paper:** [MGCVAE: Multi-Objective Inverse Design via Molecular Graph CVAE, JCIM, 2022](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00487)
- **Суть:** CVAE расширяет VAE condition-вектором (MW, LogP, HBD, HBA, TPSA и др.), позволяя генерировать молекулы с заданными свойствами за один forward pass. MGCVAE работает на графовом представлении — 25.89% оптимизированных молекул vs 0.66% у обычного VAE.
- **Преимущества над worst-case:** Прямое управление каждым свойством независимо, а не единый скалярный reward. Можно зафиксировать одни свойства и варьировать другие.
- **Применимость к dual-target docking:** Можно conditionировать на предсказанные binding affinity к двум таргетам.

### 3.2 Graph Diffusion Transformer (Graph DiT) — NeurIPS 2024

- **Paper:** [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f6931a9e339a012a9909306d7c758b4-Abstract-Conference.html)
- **Суть:** Condition encoder обучает representation для числовых и категориальных свойств, Transformer-based graph denoiser выполняет denoising молекулярного графа под условиями. Graph-dependent noise model для точной оценки noise в молекулах.
- **Преимущества над worst-case:** Натурально поддерживает произвольное число condition'ов; denoising process направляется всеми свойствами одновременно.
- **Применимость к dual-target docking:** Два docking score или binding affinity могут быть числовыми condition'ами.

### 3.3 DiffMC-Gen — Dual Denoising Diffusion для Multi-Conditional Generation (2025)

- **Paper:** [Advanced Science, 2025](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202417726)
- **Суть:** Интегрирует дискретные и непрерывные features для восприятия 3D-структуры. Multi-objective optimization одновременно по binding affinity, drug-likeness, synthesizability, toxicity. SOTA по novelty и uniqueness для LRRK2, HPK1, GLP-1R.
- **Преимущества над worst-case:** Мульти-модальный conditioning + 3D awareness позволяют генерировать по нескольким целям с учётом структуры белка.
- **Применимость к dual-target docking:** Прямая — протестирован на 3 protein targets.

### 3.4 Text-Guided Multi-Property Molecular Optimization (2024)

- **Paper:** [arXiv, 2024](https://arxiv.org/html/2410.13597v3)
- **Суть:** Diffusion language model с текстовым описанием желаемых свойств. Оптимизирует сгенерированные молекулы по нескольким objectives через guided denoising.
- **Применимость к dual-target docking:** Текстовый prompt: "bind to target A and target B with high affinity".

---

## 4. Pareto-Optimal Molecular Design: Evolutionary Methods

### 4.1 NSGA-II/III с SELFIES (CEC 2024)

- **Paper:** [arXiv:2405.00401, IEEE CEC 2024](https://arxiv.org/abs/2405.00401)
- **Суть:** Применение NSGA-II, NSGA-III и MOEA/D с SELFIES-представлением для оптимизации по GuacaMol multi-objective задачам + QED + SA score. Все три алгоритма сходятся и находят novel candidates.
- **Преимущества над worst-case:** Pareto-set содержит множество разнообразных кандидатов; NSGA-III эффективнее NSGA-II по числу функциональных оценок.
- **Применимость к dual-target docking:** 2 docking score — стандартная bi-objective задача для NSGA-II.

### 4.2 Graph-based NSGA-II/III (Verhellen, 2022)

- **Paper:** [Graph-based molecular Pareto optimisation, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9241971/)
- **Суть:** Open-source реализация NSGA-II и NSGA-III на графовом представлении молекул. 4-5 objectives (MW, logP, similarity, affinity, BBB). Оба метода превосходят weighted sum по hypervolume indicator.
- **Применимость к dual-target docking:** Графовое представление натурально для молекул; 2 targets — частный случай.

### 4.3 MOMO — Evolutionary Multiobjective Molecule Optimization (2024)

- **Paper:** [JCIM, 2024](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00031)
- **Суть:** Pareto-based multi-property evaluation strategy на уровне молекулярных последовательностей в implicit chemical space. Превосходит 5 SOTA методов по diversity, novelty и оптимизированным свойствам.
- **Применимость к dual-target docking:** Evolutionary search без ограничений на число objectives.

### 4.4 DEL — Deep Evolutionary Learning

- **Обзор:** [Frontiers in Chemistry, 2023](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2023.1288626/full)
- **Суть:** Комбинация Fragment VAE + NSGA-II компонент (non-domination rank + crowding distance) для de novo design в непрерывном латентном пространстве.
- **Применимость к dual-target docking:** Латентное пространство + NSGA-II позволяет эффективно оптимизировать 2+ objectives.

---

## 5. Hypervolume-based Reward в RL

### 5.1 Текущее состояние

- **Обзор:** [Fromer & Coley, Cell Patterns, 2023](https://www.sciencedirect.com/science/article/pii/S2666389923000016)
- **Hypervolume в MORL:** [Van Moffaert et al., EMO 2013](https://link.springer.com/chapter/10.1007/978-3-642-37140-0_28)
- **HVO для Text Summarization:** [arXiv, 2025](https://arxiv.org/abs/2510.19325)
- **Hypervolume DRL для инженерных задач:** [MDPI, 2024](https://www.mdpi.com/2673-2688/5/4/85)

**Суть:** Hypervolume indicator измеряет объём dominated space в objective space. Его можно использовать как reward: agent получает reward пропорционально приросту hypervolume при добавлении нового решения к Pareto set.

**Статус в молекулярной генерации:** Hypervolume широко используется как **метрика оценки** (evaluation metric) Pareto front в молекулярной оптимизации, но **прямое использование hypervolume improvement как reward в RL для молекул** остаётся **research gap**. Fromer & Coley (2023) прямо указывают, что "emerging workflows will benefit from evaluation metrics that measure the dominated hypervolume."

**Преимущества над worst-case:**
- Hypervolume reward поощряет diversity по всему фронту, а не схлопывание в одну точку
- Автоматически балансирует все objectives без весов
- Incentivizes finding new non-dominated solutions

**Применимость к dual-target docking:** Высокая — 2D hypervolume легко вычислим; reward = прирост area под Pareto front при добавлении (score_A, score_B).

---

## 6. Multi-Property Optimization Benchmarks

### 6.1 PMO — Practical Molecular Optimization (NeurIPS 2022)

- **Paper:** [NeurIPS 2022](https://arxiv.org/abs/2206.12411)
- **GitHub:** [wenhao-gao/mol_opt](https://github.com/wenhao-gao/mol_opt)
- **Суть:** 25 методов на 23 single-objective задачах с лимитом 10K oracle calls. Фокус на sample efficiency (AUC top-10 vs oracle calls). Многие "SOTA" методы не превосходят предшественников при ограниченном бюджете.
- **Multi-objective расширения:** DyMol (IJCAI 2024) — первый метод для dynamic many-objective оптимизации на PMO.

### 6.2 Therapeutics Data Commons (TDC)

- **GitHub:** [mims-harvard/TDC](https://github.com/mims-harvard/TDC)
- **Paper TDC-2:** [bioRxiv, 2024](https://www.biorxiv.org/content/10.1101/2024.06.12.598655v2)
- **Суть:** 66 ML-ready datasets из 22 learning tasks. Oracles: GSK3B, JNK3, DRD2, QED и др. TDC-2 (2024) добавляет 1000+ мульти-модальных датасетов, ~85M клеток, embeddings от 5 single-cell models.
- **Multi-objective:** GuacaMol oracles (7 drugs с multi-property objectives) интегрированы в TDC.

### 6.3 GuacaMol

- **Paper:** [Brown et al., JCIM, 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839)
- **Суть:** Benchmark для 2D molecular generation с фокусом на validity, novelty, uniqueness. Multi-objective task sets включают оптимизацию по нескольким physicochemical properties одновременно (MW, LogP, TPSA, similarity и др.) для 7 известных лекарств.
- **Использование:** Стандарт для сравнения NSGA-II/III, DrugEx, REINVENT и др.

### 6.4 CMOMO — Constrained Multi-Objective Molecular Optimization (2025)

- **Paper:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240737/)
- **Суть:** Deep multi-objective optimization framework с dynamic constraint handling и latent vector fragmentation для evolutionary reproduction. Превосходит 5 SOTA методов.

---

## 7. Specific Papers / Tools

### 7.1 REINVENT 4 (AstraZeneca, 2024)

- **Paper:** [J. Cheminformatics, Feb 2024](https://link.springer.com/article/10.1186/s13321-024-00812-5)
- **GitHub:** [MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4) (Apache 2.0)
- **Суть:** 4 генератора (REINVENT/RNN, Libinvent, Linkinvent, Mol2Mol), multi-stage RL. Multi-component scoring function с weighted arithmetic или geometric mean. Staged learning для поэтапного включения дорогих scoring functions (например, docking после drug-likeness фильтров).
- **Multi-objective подход:** A priori scalarization — weighted mean всех components. Каждому endpoint назначается вес.
- **Преимущества:** Mature, production-ready. Staged learning позволяет эффективно использовать дорогие scores.
- **Ограничения:** Scalarization — не находит non-convex Pareto front. ParetoDrug показал более высокие docking scores при multi-objective оптимизации.
- **Применимость к dual-target docking:** Два docking score как components с весами. Geometric mean лучше arithmetic для dual-target (наказывает дисбаланс).

### 7.2 MolDQN (Google, 2019)

- **Paper:** [Scientific Reports, 2019](https://www.nature.com/articles/s41598-019-47148-x)
- **arXiv:** [1810.08678](https://arxiv.org/abs/1810.08678)
- **Суть:** Double Q-learning + randomized value functions для модификации молекул. MDP-формулировка с гарантией 100% химической валидности. Multi-objective: 2D reward vector (QED + Tanimoto similarity).
- **Преимущества:** Value-based RL (стабильнее policy gradient), без pre-training, гарантия валидности.
- **Ограничения:** Авторы сами отмечают, что benchmarks неадекватны для реальных задач drug discovery.
- **Применимость к dual-target docking:** Reward vector можно расширить на (docking_A, docking_B, QED), но метод устарел.

### 7.3 FREED++ (2024)

- **Paper:** [arXiv:2401.09840](https://arxiv.org/abs/2401.09840)
- **OpenReview:** [FREED++](https://openreview.net/forum?id=YVPb6tyRJu)
- **Суть:** Улучшенная воспроизводимая версия FREED — RL agent на SAC (Soft Actor-Critic) для fragment-based molecule generation. Docking score как reward. Авторы исправили множество багов оригинала, упростили модель и увеличили качество.
- **Преимущества:** Fragment-based approach обеспечивает synthesizability; SAC + Maximum Entropy даёт стабильное обучение; стабильно превосходит конкурентов.
- **Ограничения:** Single-target (docking score как единственный reward). Multi-reward потребует адаптации.
- **Применимость к dual-target docking:** SAC framework позволяет multi-reward; нужно модифицировать reward на комбинацию двух docking scores.

### 7.4 Saturn (2024)

- **Paper:** [arXiv:2405.17066](https://arxiv.org/abs/2405.17066), NeurIPS AI4Mat 2024 Spotlight
- **GitHub:** schwallergroup/saturn
- **Суть:** Первое применение Mamba architecture для генеративного молекулярного дизайна. Augmented Memory (experience replay + data augmentation) для sample efficiency. Replay buffer из top-100 SMILES по reward. Превосходит 22 модели на multi-parameter optimization tasks.
- **Преимущества:** Sample efficiency позволяет рассматривать прямую оптимизацию high-fidelity oracles (вместо proxy). Mamba синергично эксплуатирует experience replay.
- **Применимость к dual-target docking:** Multi-parameter optimization нативно; docking scores как objectives. Sample efficiency критична для дорогих docking oracles.

### 7.5 POLYGON — POLYpharmacology Generative Optimization Network (2024)

- **Paper:** [Nature Communications, 2024](https://www.nature.com/articles/s41467-024-47120-y)
- **Суть:** VAE + generative RL для polypharmacology. Reward: predicted inhibition двух protein targets + drug-likeness + ease-of-synthesis. Протестирован на 10 парах белков с документированной co-dependency.
- **Экспериментальная валидация:** 32 синтезированных соединения для MEK1 + mTOR. Большинство дали >50% reduction in activity обоих белков при 1-10 uM.
- **Преимущества:** Единственный метод с экспериментально подтверждённой dual-target генерацией. Точность распознавания polypharmacology interactions: 82.5%.
- **Применимость к dual-target docking:** **Прямая и подтверждённая.** Это ближайший аналог нашей задачи.

### 7.6 MTMol-GPT (2024)

- **Paper:** [PLOS Computational Biology, 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012229)
- **Суть:** GPT + generative adversarial imitation learning (GAIL) с dual discriminator через Inverse RL для concurrent multi-target molecular generation. Продемонстрирован на парах DRD2 + HTR1A, EGFR + Src.
- **Применимость к dual-target docking:** Прямая — разработан именно для multi-target задач.

### 7.7 MDRL — 3D Diffusion + RL для Multi-Target (2025)

- **Paper:** [J. Cheminformatics, 2025](https://link.springer.com/article/10.1186/s13321-025-01035-y)
- **Суть:** 3D diffusion model + RL для генерации multi-target compounds. AutoDock Vina для docking по двум targets. Примеры: BRD4 + PARP1 (-8.6/-9.9 kcal/mol), CDK7 + PRMT5 (-8.5/-8.6 kcal/mol).
- **Применимость к dual-target docking:** **Прямая.** 3D-aware + dual docking rewards.

### 7.8 SFG-Drug — Dual-Target MCTS + GRU (2025)

- **Paper:** [MDPI Inventions, 2025](https://www.mdpi.com/2411-5134/11/1/12)
- **Суть:** MCTS + GRU для dual-target generation (MEK1 + mTOR). DigFrag digital fragmentation + low-frequency masking для diversity. Docking scores как rewards. Validity = 1.0, uniqueness = 1.0, novelty = 1.0 на MOSES.
- **Применимость к dual-target docking:** Прямая — разработан именно для dual-target.

### 7.9 AIxFuse — Structure-aware Dual-Target Drug Design (2024)

- **Paper:** [PMC / Nature Machine Intelligence area, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11234869/)
- **Суть:** Первый метод, который учит pharmacophore fusion patterns для удовлетворения dual-target structural constraints через molecular docking. Два self-play RL agents учатся pharmacophore selection и fusion через feedback от dual-target docking scores.
- **Преимущества:** Structure-aware, generalizability не зависит от размера training dataset (используется docking, а не ML predictor).
- **Применимость к dual-target docking:** **Прямая и primary use case.**

### 7.10 Mol-Zero-GAN (2023)

- **Paper:** [PMC, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10714197/)
- **GitHub:** [cucpbioinfo/Mol-Zero-GAN](https://github.com/cucpbioinfo/Mol-Zero-GAN)
- **Суть:** Zero-shot adaptation GAN для конкретных protein targets. Bayesian optimization оптимизирует singular values весов GAN (SVD), генерируя drug candidates без дополнительных данных.
- **Применимость к dual-target docking:** Можно адаптировать BO objective на два targets.

### 7.11 MOLLM — Multi-Objective LLM для Molecular Design (2025)

- **Paper:** [arXiv, 2025](https://arxiv.org/html/2502.12845v1)
- **Суть:** Genetic algorithms + LLM с in-context learning и prompt engineering. Без дополнительного обучения. SOTA на unconstrained molecular optimization.
- **Применимость к dual-target docking:** LLM + GA framework натурально multi-objective.

### 7.12 Mol-MoE — Mixture-of-Experts для Multi-Objective (2025)

- **Paper:** [arXiv, 2025](https://arxiv.org/html/2502.05633v1)
- **Суть:** Mixture-of-Experts подход: dynamic steering молекулярной генерации по нескольким objectives через activation-space manipulation без полного переобучения.
- **Применимость к dual-target docking:** Разные experts для разных targets; routing по предпочтениям.

---

## 8. Boltz-2 / Structure-Based Scoring как Reward в RL-Loop

### 8.1 Boltz-2 + SynFlowNet (Recursion / MIT, June 2025)

- **Paper:** [bioRxiv, 2025](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full)
- **GitHub:** [recursionpharma/synflownet-boltz](https://github.com/recursionpharma/synflownet-boltz) (open-source)
- **GitHub (Boltz):** [jwohlwend/boltz](https://github.com/jwohlwend/boltz) (MIT license)

**Суть:** Boltz-2 — foundation model для предсказания structure + binding affinity. Первая AI-модель, приближающая FEP по точности (Pearson r=0.66 на 4-target FEP subset), но в 1000x быстрее. SynFlowNet (GFlowNet) использует Boltz-2 как reward:

1. SynFlowNet генерирует молекулы из synthesizable chemical space (Enamine 76B REAL)
2. Batches асинхронно отправляются в Boltz-2 workers (~20s/molecule, рекомендуется ~50 workers)
3. Reward = binding likelihood + affinity (Boltz-2) + medicinal chemistry filters
4. Annotated samples возвращаются в Replay Buffer для off-policy training

**Результаты на TYK2:**
- 117K Boltz-2 evaluations (vs 460K для HLL screen)
- Все 10 de novo кандидатов предсказаны как binders
- Higher affinity на average vs fixed library screens
- Новые compounds не имеют значительного сходства с известными TYK2 binders

**Преимущества над worst-case:**
- Boltz-2 предсказывает и структуру, и affinity; можно использовать для нескольких таргетов
- GFlowNet loss позволяет sampling из multi-modal distributions
- Asynchronous architecture масштабируется

**Применимость к dual-target docking:**
- Boltz-2 affinity head может скорить по любому target → reward = f(affinity_A, affinity_B)
- GFlowNet натурально подходит для multi-modal reward (разные binding modes для двух targets)
- **Это наиболее перспективный подход для нашей задачи**, но требует значительных compute ресурсов (~50 GPU workers)

### 8.2 Другие Structure-Based Scoring в RL

- **FREED++** — AutoDock Vina docking score как reward в SAC
- **ParetoDrug** — pretrained structure-aware generative model + docking
- **MDRL** — AutoDock Vina для dual-target
- **AIxFuse** — dual-target docking feedback для RL agents

---

## 9. Сводная таблица: Применимость к Dual-Target Docking

| Метод | Multi-Obj | Pareto | Structure-aware | Dual-target tested | Open-source | Year |
|-------|-----------|--------|-----------------|-------------------|-------------|------|
| **POLYGON** | RL reward | No (scalarized) | Docking validation | MEK1+mTOR (exp!) | Partial | 2024 |
| **AIxFuse** | RL + docking | No | Pharmacophore fusion | Yes (primary) | ? | 2024 |
| **MDRL** | Diffusion+RL | No | 3D + Vina | BRD4+PARP1, CDK7+PRMT5 | ? | 2025 |
| **ParetoDrug** | MCTS | Pareto PUCT | Docking score | Yes | [GitHub](https://github.com/CNDOTA/ParetoDrug) | 2024 |
| **PMMG** | MCTS | Pareto | RNN+docking | EGFR+HER2 | ? | 2025 |
| **DrugEx v3** | RL | Pareto ranking | ML predictor | A2AAR+QED | [GitHub](https://github.com/CDDLeiden/DrugEx) | 2023 |
| **REINVENT 4** | RL | No (weighted mean) | Staged docking | Configurable | [GitHub](https://github.com/MolecularAI/REINVENT4) | 2024 |
| **Saturn** | RL (Augmented Memory) | No | PMO oracles | Multi-param | GitHub | 2024 |
| **CPRL** | RL | Clustered Pareto | Objective functions | Multi-obj | ? | 2024 |
| **MLPS** | BO + Pareto Set | Full Pareto | Latent space | Multi-obj | ? | 2024 |
| **Boltz-2+SynFlowNet** | GFlowNet | Multi-modal | Structure + Affinity | TYK2 (extensible) | [GitHub](https://github.com/recursionpharma/synflownet-boltz) | 2025 |
| **MTMol-GPT** | GAIL | No | Docking validation | DRD2+HTR1A, EGFR+Src | ? | 2024 |
| **SFG-Drug** | MCTS+GRU | No | Docking reward | MEK1+mTOR | ? | 2025 |
| **NSGA-II/III+SELFIES** | EA | NSGA Pareto | GuacaMol | Multi-obj | ? | 2024 |

---

## 10. Рекомендации для Dual-Target Docking проекта

### Tier 1 — Наиболее релевантные и проверенные подходы:
1. **POLYGON** — единственный с экспериментальной валидацией dual-target
2. **AIxFuse** — structure-aware dual-target, pharmacophore fusion
3. **Boltz-2 + SynFlowNet** — самый мощный scoring (structure + affinity), но дорогой
4. **ParetoDrug** — Pareto MCTS, target-aware, open-source

### Tier 2 — Хорошие frameworks для адаптации:
5. **REINVENT 4** — production-ready, легко настроить dual-target scoring
6. **DrugEx v3** — Pareto RL, scaffold-constrained
7. **Saturn** — sample-efficient, Augmented Memory
8. **MDRL** — 3D diffusion + dual-target docking

### Агрегация скоров: что использовать вместо worst-case (max для lower-is-better):
- **Geometric mean** — простой, наказывает дисбаланс, совместим с REINVENT 4
- **Tchebycheff** — находит non-convex front, нужна утопическая точка
- **Pareto ranking** — находит весь фронт, нужен Pareto-aware algorithm (DrugEx, ParetoDrug)
- **EHVI** — лучший теоретически обоснованный подход, но вычислительно дороже
- **Boltz-2 affinity** — заменяет классический docking, 1000x быстрее FEP
