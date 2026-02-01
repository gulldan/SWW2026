# Ускорение dual-docking пайплайна

## 1. Boltz-2: ускорение инференса

### 1.1 NVIDIA NIM + TensorRT (рекомендуемый путь)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Полный инференс Boltz-2 (structure prediction + affinity prediction) |
| **Speedup** | **1.2x-2.3x** через TensorRT; до **2.5x** с cuEquivariance BF16 vs PyTorch FP32 |
| **Источник** | [NVIDIA NIM for Boltz-2](https://docs.nvidia.com/nim/bionemo/boltz2/latest/performance.html), [NVIDIA cuEquivariance Blog](https://developer.nvidia.com/blog/accelerated-molecular-modeling-with-nvidia-cuequivariance-and-nvidia-nim-microservices/) |
| **Сложность** | **Средне** - требуется NVIDIA GPU >= 48 GB VRAM, Docker-контейнер NIM |

### 1.2 BFloat16 (mixed precision)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Снижение потребления памяти GPU, более крупные структуры |
| **Speedup** | Позволяет обрабатывать ~1400 остатков вместо ~1000 на RTX 4090 (24 GB) |
| **Источник** | [GitHub Issue #263](https://github.com/jwohlwend/boltz/issues/263) |
| **Сложность** | **Просто** - параметр в конфигурации, уже поддерживается |

Внимание: на CPU bfloat16 работает в 6 раз медленнее. Только для GPU Ampere+.

### 1.3 torch.compile

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | JIT-компиляция графа вычислений, слияние ядер CUDA |
| **Speedup** | Типично **1.5x-3x** для трансформерных моделей |
| **Сложность** | **Сложно** - Boltz использует trifast ядра и диффузионную архитектуру |

### 1.4 Batched inference

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Множественный докинг одного белка с разными лигандами |
| **Speedup** | Линейный от числа GPU; кеширование MSA устраняет повторную генерацию |
| **Источник** | [GitHub Issue #522](https://github.com/jwohlwend/boltz/issues/522), [Nebius: Boltz-2 at scale](https://nebius.com/blog/posts/running-boltz-2-inference-at-scale) |
| **Сложность** | **Средне** - кеширование MSA и распределение по GPU |

---

## 2. FREED++: ускорение генерации

### 2.1 Фрагментная генерация (встроенное)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Пространство поиска молекул (фрагменты вместо атомов) |
| **Speedup** | Экспоненциальное сокращение пространства поиска |
| **Источник** | [FREED++ (arXiv:2401.09840)](https://arxiv.org/abs/2401.09840), [GitHub AIRI-Institute/FFREED](https://github.com/AIRI-Institute/FFREED) |
| **Сложность** | **Уже интегрировано** |

### 2.2 Ускорение reward-функции

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Вычисление docking score как reward - основной bottleneck |
| **Speedup** | Замена CPU Vina на Uni-Dock - до **1000x** |
| **Сложность** | **Сложно** - требуется переписать reward pipeline |

---

## 3. Молекулярный докинг

### 3.1 Uni-Dock (GPU docking) - ПРИОРИТЕТ

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | AutoDock Vina-совместимый докинг на GPU |
| **Speedup** | **1000x-1627x** по сравнению с Vina на 1 CPU-ядре |
| **Источник** | [Uni-Dock (JCTC)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145), [GitHub dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock) |
| **Сложность** | **Просто** - совместим с Vina/Vinardo/AD4, Apache 2.0 |

### 3.2 Deep Docking (ML-суррогат)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Предсказание docking score без полного докинга |
| **Speedup** | **50x** ускорение виртуального скрининга |
| **Источник** | [Deep Docking (ACS Central Science)](https://pubs.acs.org/doi/10.1021/acscentsci.0c00229) |
| **Сложность** | **Сложно** - обучение суррогатной модели |

### 3.3 Кеширование результатов

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Повторные запросы к одинаковым парам белок-лиганд |
| **Speedup** | Полное устранение повторов; 2x-10x при итеративных циклах |
| **Сложность** | **Просто** - LRU-кеш по ключу (protein_hash, smiles) |

---

## 4. PyTorch: общие оптимизации

### 4.1 FlashAttention / FlashAttention-3

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Attention-слои: скорость и память |
| **Speedup** | FA-2: **2x-4x**; FA-3: **1.5x-2x** vs FA-2 (до 740 TFLOPS на H100) |
| **Источник** | [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) |
| **Сложность** | **Средне** - замена attention-модулей |

### 4.2 Mixed Precision (BF16/FP16)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Все матричные операции; снижение памяти в 2x |
| **Speedup** | **1.5x-2x** на Ampere+ GPU |
| **Сложность** | **Просто** - `torch.autocast("cuda", dtype=torch.bfloat16)` |

### 4.3 CUDA Graphs

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Устранение CPU-оверхеда при мелких CUDA-ядрах |
| **Speedup** | **До 5x** на графированных участках |
| **Сложность** | **Средне** - `torch.compile(mode="reduce-overhead")` |

---

## 5. RDKit: ускорение обработки

### 5.1 datamol (параллельная обработка)

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Парсинг, стандартизация, фильтрация SMILES |
| **Speedup** | Линейный по числу ядер CPU |
| **Источник** | [datamol GitHub](https://github.com/datamol-io/datamol) |
| **Сложность** | **Просто** - `dm.parallelized(fn, items, n_jobs=-1)` |

### 5.2 RDKit Pickle Cache

| Параметр | Значение |
|----------|----------|
| **Что ускоряет** | Повторная загрузка молекул без парсинга |
| **Speedup** | **~15x** (pickle vs SDF парсинг) |
| **Сложность** | **Просто** |

---

## Приоритеты внедрения

| # | Метод | Speedup | Сложность | Этап |
|---|-------|---------|-----------|------|
| 1 | Uni-Dock (замена Vina) | 1000x+ | Просто | Докинг |
| 2 | BF16 mixed precision | 1.5x-2x | Просто | Boltz-2 |
| 3 | datamol parallelized | Nx (N = ядра) | Просто | SMILES |
| 4 | Кеширование (MSA + docking) | 2x-10x | Просто | Все |
| 5 | FlashAttention | 2x-16x | Средне | Boltz-2 attention |
| 6 | NVIDIA NIM + TensorRT | 1.5x-2.3x | Средне | Boltz-2 инференс |
| 7 | torch.compile | 1.5x-3.5x | Средне | Все PyTorch модели |
| 8 | CUDA Graphs | 1.7x-5x | Средне | Мелкие ядра |
| 9 | Deep Docking (суррогат) | 50x | Сложно | Скрининг |
| 10 | Batched RL (FREED++) | 2x-5x | Сложно | Генерация |

**Рекомендация**: начинать с пунктов 1-4 (максимальный эффект при минимальной сложности), затем 5-8, и только потом 9-10.
