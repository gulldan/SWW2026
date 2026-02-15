 Деструктивный анализ: «Dual-target RL generation improves polypharmacology hit-rate for Alzheimer's multi-target drug design»                                                                                                                      
                                                                                                                                                                                                                                                       
  УРОВЕНЬ 1: FATAL — проблемы, которые гарантируют reject                                                                                                                                                                                            

  1. Отсутствие экспериментальной валидации на фоне конкурентов с wet-lab

  Isigkeit et al. (Nat Comms 2024) — 7 из 12 dual-target молекул подтверждены экспериментально. POLYGON (Nat Comms 2024) — 32 молекулы синтезированы. CMD-GEN — >787x PARP1/PARP2 селективность подтверждена.

  Рецензент Nature Comms/JCIM скажет: «Why should we publish another purely computational study when the field has moved to experimental validation?» Ваш текущий 06_limitations.tex (строка 79-82) просто говорит «No wet-lab validation: all results
  are computational. Experimental IC50 confirmation is beyond the scope of this workshop» — это недостаточный ответ для журнала уровня JCIM и выше.

  Что нужно: Минимум — MD-валидация (GROMACS/OpenMM) для top-10 молекул per pair. Код для этого заявлен в pipeline (dualdocking/validation/md.py), но это Stage 7, которая судя по всему не выполнена. В идеале — FEP/MMPBSA хотя бы на 1-2 парах.

  2. FREED++ — устаревший генератор (2022), не SOTA

  FREED++ (Telepov et al., TMLR 2023, arXiv 2401.09840) — fragment-based RL. Текущие SOTA для генерации:
  - MolCRAFT (ICML 2024): autoregressive 3D, pocket-conditioned
  - MolJO (NeurIPS 2024): joint 2D+3D optimization
  - BindGPT (2024): Transformer + RL, pocket-conditioned

  Рецензент скажет: «You use a 2022-era fragment-based generator while the field has moved to 3D pocket-conditioned models. Your results may simply reflect limitations of FREED++, not properties of dual-target RL.»

  Ваш контраргумент в novelty_report.md (строка 297): «FREED++ is the only open-source fragment-based RL generator supporting dual-target via monkey-patching» — это слабый аргумент. Если FREED++ — единственное, что поддерживает dual-target, это
  говорит о том, что dual-target нужно имплементировать в лучшем генераторе, а не оправдывать использование устаревшего.

  Что нужно: Либо (a) запустить REINVENT4 как второй генератор (код есть, multi-objective scoring поддерживается), либо (b) чётко обосновать выбор FREED++ в контексте fragment-based chemical space coverage и показать, что выводы
  generator-agnostic.

  3. m=5 seeds — статистическая мощность крайне недостаточна

  5 RL seeds при bootstrap — это 25 возможных комбинаций при ресамплинге уровня 1 (5^5 = 3125 с повторением, но уникальных наборов мало). Cameron/Gelbach/Miller (2008) показали, что wild cluster bootstrap помогает с 5 кластерами, но не решает
  проблему.

  Cliff's delta с n=5 vs n=5: возможные значения delta = {-1, -0.84, -0.68, ..., 0, ..., 0.68, 0.84, 1} — дискретная шкала с шагом 0.16. Ваш порог «medium effect» ≥ 0.33 технически достижим при 5 vs 5, но один seed-переворот меняет вывод.
  Рецензент посчитает power analysis и покажет, что при 5 seeds у вас мощность ~30-40% для обнаружения среднего эффекта.

  Что нужно: Минимум — power analysis в SAP (его там нет). Или увеличить seeds до 10-20.

  УРОВЕНЬ 2: MAJOR — проблемы, которые потребуют major revision

  4. Target overlap в core set — зависимые наблюдения

  5 core pairs, но:
  - AChE участвует в 2 парах (AChE_BChE, AChE_BACE1)
  - BACE1 участвует в 2 парах (BACE1_GSK3B, AChE_BACE1)
  - GSK3B участвует в 2 парах (BACE1_GSK3B, GSK3B_CDK5)

  Из 5 core pairs только CDK2_ERK2 не разделяет таргет ни с одним другим. Фактически у вас не 5 независимых наблюдений, а ~3. Критерий «≥3 из 5 пар» теряет смысл, потому что если AChE-содержащие эксперименты систематически лучше/хуже, две пары
  коррелируют.

  Что нужно: Sensitivity analysis: повторить primary test на подмножестве без shared targets (CDK2_ERK2, AChE_BChE, BACE1_GSK3B = 3 пары, все targets уникальны). Добавить корреляционный анализ между парами, разделяющими таргеты.

  5. Drug-likeness — код есть, но не в endpoints

  dualdocking/data/filters.py вычисляет Lipinski, QED, PAINS, SA, Veber. Данные записываются в scored_joined.parquet. Но ни один из этих показателей не входит в primary, secondary или даже exploratory endpoints (см. configs/metrics.yaml).
  computed_metrics — это dual_hit_rate, hit_rate_a, hit_rate_b, enrichment_at_k, ks_test, mannwhitney_u, rank_biserial_r. Ни QED, ни SA, ни Lipinski.

  Рецензент скажет: «You optimize for docking scores only. The generated molecules may have terrible drug-likeness. Where is the QED/SA comparison between modes?» Это стандартная проблема всех docking-driven RL генераторов (reward hacking:
  молекулы растут в размере, теряют drug-likeness).

  Что нужно: Добавить drug-likeness как secondary endpoint: среднее QED, доля Lipinski-pass, средний SA score — для каждого mode, с bootstrap тестом double vs single.

  6. Нет контрольной группы — random/ZINC baseline

  Вы сравниваете double vs single_a vs single_b. Но нет random baseline: что если случайная выборка из ZINC drug-like даёт сопоставимый hr_both? ZINC baseline скачивается (cli/s0_fetch/fetch_zinc_baseline.py, файл zinc-druglike-cano.tar.xz),
  сэмплируется (cli/s1_prepare/sample_baseline.py, 1000 молекул), но нигде не используется как control arm в primary analysis.

  Рецензент скажет: «Without a random baseline, we don't know if your RL modes genuinely improve over chance. If ZINC drug-like molecules have 5% hr_both and your double mode gives 8%, that's interesting. If they give 7%, your result is trivial.»

  Что нужно: Прогнать ZINC baseline через Boltz-2, вычислить hr_both, добавить в таблицу результатов.

  7. Гетерологичные пары — тривиальный успех?

  BACE1 (aspartyl protease) + GSK3B (kinase) имеют совершенно разные pocket geometries. Для fragment-based генератора тривиально найти молекулу, которая стыкуется в оба pocket'а — достаточно иметь kinase hinge binder + hydrophobic core
  достаточного размера. AChE_BACE1 тоже гетерологичная.

  Рецензент скажет: «For heterologous pairs, dual docking success may simply reflect that the generator produces large, flexible molecules that fit anywhere. This is a known problem with docking-based metrics (DUD decoy bias).»

  Что нужно: Анализ размера молекул (MW, heavy atoms) по modes. Если double-mode молекулы значимо крупнее — это reward hacking, а не полифармакология. Также: анализ по гомо vs гетеро парам раздельно.

  8. Boltz-2 как evaluation oracle — нерешённые вопросы

  Bret et al. (2026, JCIM) показали, что Boltz-2 affinity «insensitive to key binding-site mutations». Ваш 06_limitations.tex это упоминает, но не количественно анализирует. Если Boltz-2 не различает мутации в одном binding site, как он различит
  разные pocket'ы двух разных белков?

  Более серьёзная проблема: все ваши PDB структуры (4EY7, 6EQM, 5KPK, 1UNL, 4BDS, 1HCL, 4QTA) гарантированно в Boltz-2 training data (PDB cutoff 06/2023). Boltz-2 может просто вернуть «memorized» affinity для любого лиганда, близкого к
  co-crystallized ligand. Это invalidates orthogonal evaluation: если Boltz-2 тоже learned the same structures, оно не «orthogonal» к UniDock.

  Что нужно: Per-target Spearman rho (docking vs Boltz-2) — заявлен как exploratory, но критически важен. Если rho ≈ 0 → Boltz-2 не видит того же, что видит docking → orthogonality подтверждена, но concordance нулевая → не ясно, кому верить. Если
  rho > 0.6 → Boltz-2 фактически подтверждает docking → orthogonality сомнительна.

  9. Compute asymmetry не решена

  Double mode делает 2x docking calls per molecule (dock_A + dock_B для reward). Single_a делает 1x. При 200 epochs и 1000 molecules double-mode видит ~2x больше docking landscape. Ваш primary comparison — fixed molecule count (1000), не fixed
  compute.

  preregistration.md (строка 161): «Equal docking calls: Single modes with 400 RL epochs (2x training budget); same n_molecules = 1000» — заявлено как sensitivity only, не как primary. Рецензент скажет: «Your primary comparison is unfair. Double
  mode had twice the compute budget for training.»

  Что нужно: Сделать equal-compute comparison primary или equal-weight: показать оба в основном тексте, а не прятать в sensitivity.

  УРОВЕНЬ 3: MINOR — будут подняты, но не заблокируют

  10. Alzheimer framing — слабая связь

  Заголовок включает «Alzheimer's multi-target drug design», но:
  - CDK2_ERK2 — не AD-relevant (kinase selectivity control)
  - EGFR_HER2 — онкология
  - COX1_COX2 — blocked

  Из 5 core pairs только 3 AD-relevant (AChE_BChE, BACE1_GSK3B, AChE_BACE1; GSK3B_CDK5 borderline). Рецензент AD-журнала скажет: «This is a computational methods paper, not an Alzheimer's paper.» Рецензент methods-журнала скажет: «The Alzheimer's
  framing is misleading.»

  Что нужно: Либо (a) фреймить как general polypharmacology method paper с AD case study, либо (b) убрать CDK2_ERK2 из core и заменить на AD pair.

  11. Scalarization sensitivity — только offline

  6 scalarization functions в коде (MIN, HARMONIC, GEOMETRIC, WEIGHTED_SUM, WEIGHTED_PRODUCT, TCHEBYCHEFF), но primary experiment использует только MIN (worst-case). Остальные — «offline recomputation» (preregistration строка 162).

  Offline recomputation — это пост-хок переоценка, а не настоящая sensitivity. RL-trajectory при MIN reward и при WEIGHTED_SUM reward будет совершенно разной. Рецензент скажет: «Offline recomputation of alternative scalarizations is not equivalent
   to training with those scalarizations.»

  Что нужно: Хотя бы 1-2 дополнительных scalarization (WEIGHTED_SUM, HARMONIC) как полноценные RL runs на 1-2 парах.

  12. Diversity metrics не связаны с endpoints

  dualdocking/analysis/diversity.py вычисляет uniqueness, scaffold diversity, internal Tanimoto diversity, SNN. Но эти метрики нигде не сравниваются между modes. Рецензент скажет: «Does double mode collapse chemical space compared to single?» —
  это важный вопрос, на который нет ответа.

  13. 1HCL (CDK2) — apo structure

  docs/selection_criteria.md упоминает, что 1HCL — apo structure, binding box определён через fallback. Это нестандартно и может дать нереалистичные docking scores. Рецензент-structural-biologist это заметит.

  УРОВЕНЬ 4: ОТСУТСТВУЮЩИЕ ЭЛЕМЕНТЫ

  ┌───────────────────────────────────────────┬─────────────────────────────┬───────────────────┐
  │                  Элемент                  │           Статус            │    Критичность    │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Power analysis (заранее, не post-hoc)     │ Отсутствует в SAP           │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ REINVENT4 или другой generator comparison │ Отсутствует                 │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ ZINC random baseline в primary table      │ Код есть, не используется   │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ MD validation (top-N molecules)           │ Код заявлен, не выполнен    │ Fatal для JCIM+   │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Per-mode MW/heavy-atom/QED distribution   │ Данные есть, анализ нет     │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Drug-likeness endpoints                   │ Фильтры есть, endpoints нет │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Mode collapse diversity comparison        │ Код есть, не в analysis     │ Minor             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Equal-compute as primary                  │ Заявлен as sensitivity      │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Independent-target core subset test       │ Не реализован               │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Docking-Boltz-2 concordance analysis      │ Заявлен, критически важен   │ Major             │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Abstract/intro written                    │ Template placeholder        │ N/A (not started) │
  ├───────────────────────────────────────────┼─────────────────────────────┼───────────────────┤
  │ Synthesis feasibility (retro-synthesis)   │ Не реализован               │ Minor             │
  └───────────────────────────────────────────┴─────────────────────────────┴───────────────────┘

  ИТОГ: ВЕРДИКТ

  Текущее состояние: Отлично спроектированный pipeline с pre-registered SAP, 450+ тестов, хорошая инфраструктура. Но недостаточно для публикации в JCIM-level journal без:

  1. Обязательно: ZINC baseline, drug-likeness endpoints, power analysis, equal-compute в основном тексте, per-mode MW analysis, independent-target sensitivity
  2. Крайне желательно: REINVENT4 comparison, MD validation на top-N молекулах, diversity mode-comparison
  3. Для Nature-level: Wet-lab validation (хотя бы binding assay на 5-10 top compounds)

  Целевой журнал при текущем состоянии: Workshop paper (NeurIPS/ICML Workshop on ML for Drug Discovery). Для JCIM/J. Cheminformatics — нужны пункты из «обязательно». Для Chem. Sci./Nat Comms — нужно всё из «крайне желательно» + partial
  experimental validation.
