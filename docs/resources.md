# Dual-Docking Workshop: Tool & Target Research

---

## 1. FREED++ (Fragment-based Reinforcement-learning for Exploratory Drug Design)

### Overview
FREED++ is a reinforcement-learning-based molecular generation framework that uses
docking scores (QuickVina2) as the reward signal to guide the generation
of drug-like molecules toward a specific protein target. Uses SAC (Soft Actor-Critic)
with fragment-based molecular assembly via CReM library.

### GitHub Repository
- **URL:** https://github.com/AIRI-Institute/FFREED
- Contains FREED++ implementation with QuickVina2 integration.

### Paper Reference
- **FREED++:** arXiv:2401.09840 — improved fragment libraries, reward shaping, stable RL.
- **FREED (original):** arXiv:2110.01219 — foundational fragment RL approach.

### Installation (Docker-based)

FREED++ требует Python 3.7 / PyTorch 1.12 / DGL 0.9 — несовместимо с проектом
(Python ≥3.11). Изоляция через Docker:

```bash
# Сборка Docker-образа (однократно)
docker build -t dualdocking/freedpp:latest docker/freedpp/

# Проверка: PDB → PDBQT
docker run --rm -v $(pwd)/data:/workspace dualdocking/freedpp:latest \
  prepare-receptor /workspace/targets/CDK2_ERK2/A/1hcl.pdb /workspace/targets/CDK2.pdbqt
```

### Key Dependencies (inside Docker)
- Python 3.7, PyTorch 1.12+cu113, DGL 0.9
- `rdkit` 2022.03 (molecule manipulation, SMILES)
- `crem` 0.2.9 (fragment-based molecular assembly)
- RDKit + Meeko (PDB/PDBQT/ligand prep)
- QuickVina2 binary (`ffreed/env/qvina02`)

### Key CLI Usage Patterns (via `python main.py`)

Все операции через единый entry point `main.py` с аргументами из `ffreed/args.py`:

```bash
# Training + sampling (single target)
docker run --rm --gpus all -v $(pwd)/workspace:/workspace \
  dualdocking/freedpp:latest \
  --commands "train,sample" \
  --receptor /workspace/receptor.pdbqt \
  --box_center "12.3,4.5,6.7" --box_size "25.0,20.0,22.0" \
  --fragments /opt/freed/data/zinc_crem.json \
  --alert_collections /opt/freed/data/alert_collections.csv \
  --vina_program /opt/freed/ffreed/env/qvina02 \
  --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1" \
  --exp_root /workspace/output --name run \
  --epochs 200 --num_sub_proc 12 --seed 42 \
  --num_mols 1000 --reward_version soft

# Sampling only (from cached model)
docker run --rm --gpus all -v $(pwd)/workspace:/workspace \
  dualdocking/freedpp:latest \
  --commands "sample" \
  --receptor /workspace/receptor.pdbqt \
  --box_center "12.3,4.5,6.7" --box_size "25.0,20.0,22.0" \
  --fragments /opt/freed/data/zinc_crem.json \
  --alert_collections /opt/freed/data/alert_collections.csv \
  --vina_program /opt/freed/ffreed/env/qvina02 \
  --exp_root /workspace/output --name run \
  --num_mols 1000
```

### Output Format

CSV файлы в `{exp_root}/{name}/mols/sample_NNNNNN.csv`:
```csv
smiles,score
c1ccc(CC(=O)O)cc1,-7.5
CC(=O)Nc1ccccc1,-6.2
```

### Dual-Target Integration

Для двойного режима (`reward_mode=double`):
- `dual_reward_patch.py` — monkey-patch `DockingVina.__call__()` при наличии env `DUAL_RECEPTOR_B`
- Скорит молекулу по двум receptor PDBQT (A и B)
- Возвращает worst-case reward: `max(score_a, score_b)` для lower-is-better
- Box параметры второго receptor из `/workspace/dual_config.json`

### Notes for Dual-Docking Workshop
- QuickVina2 reward (~1 сек/мол, CPU) используется при обучении RL-агента
- Boltz-2 (~58 сек/мол, GPU) — для финального скоринга сгенерированных молекул
- Кеширование обученных моделей: `cache/freedpp_models/{hash}/`

---

## 2. Boltz-2 (MIT / Boltz)

### Overview
Boltz is an open-source structure prediction model. Boltz-1 focused on biomolecular
complex structure prediction (proteins, ligands, nucleic acids). Boltz-2 extends this
to jointly model complex 3D structures AND predict binding affinities, making it
highly relevant for scoring protein-ligand interactions without a separate docking step.

### GitHub Repository
- **URL:** https://github.com/jwohlwend/boltz
- Developed by Jeremy Wohlwend and collaborators at MIT.
- Boltz-1 was the initial release; Boltz-2 adds affinity prediction capabilities.

### Paper Reference
- **Boltz-1:** Wohlwend, J., Corso, G., Passaro, S., Getz, M., et al.
  *"Boltz-1: Democratizing Biomolecular Interaction Modeling"*
  Preprint (2024). https://doi.org/10.1101/2024.11.19.624167
- **Boltz-2:** Wohlwend, J., Passaro, S., Corso, G., et al.
  *"Boltz-2: Jointly Modeling Complex Structures and Binding Affinities"*
  (2025). Check the repository for the latest citation.

### Installation
```bash
pip install boltz
# Or from source:
git clone https://github.com/jwohlwend/boltz.git
cd boltz
pip install -e .
```

### Key Python Dependencies
- `torch` (>= 2.0)
- `pytorch-lightning`
- `einops`
- `biopython`
- `numpy`, `scipy`
- `pandas`
- `hydra-core` / `omegaconf` (configuration management)
- Model weights are downloaded automatically on first use.

### Key CLI Usage Patterns

Boltz-2 **не имеет публичного Python API** — взаимодействие только через CLI.

```bash
# Predict structure + affinity from a YAML input
boltz predict input.yaml --out_dir results/ --model boltz2 --accelerator gpu

# С pre-computed MSA (offline):
boltz predict input.yaml --out_dir results/ --model boltz2 --accelerator gpu --override

# С online MSA (для первичной подготовки):
boltz predict input.yaml --out_dir results/ --model boltz2 --accelerator gpu --use_msa_server
```

#### Входной YAML формат
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MENFQKVEKIGE..."
      msa: cache/boltz_msa/CDK2.csv    # опционально, для offline
  - ligand:
      id: B
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'
properties:
  - affinity:
      binder: B
```

#### Выходной формат (affinity)
Путь: `{out_dir}/boltz_results_{stem}/predictions/{stem}/affinity_{stem}.json`
```json
{
    "affinity_pred_value": -1.0,
    "affinity_probability_binary": 0.85
}
```
`affinity_pred_value` = log10(IC50 в μM). В пайплайне используется напрямую (меньше = лучше).

#### Кеширование и offline-режим
- MSA pre-compute при подготовке: `scripts/prepare_targets.py --prepare-all`
- MSA кешируется в `cache/boltz_msa/{target_id}.csv`
- Boltz checkpoints (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`) в `~/.boltz/`
- Scoring cache per molecule: `cache/{target_id}.json`

### Notes for Dual-Docking Workshop
- Boltz-2 serves as a fast surrogate for docking: predict structure + affinity
  in one forward pass (~18 sec/molecule on GPU), avoiding expensive AutoDock Vina calls.
- Used as a scoring oracle in the FREED++ reward loop via subprocess calls.
- No Python API — all integration through `uv run boltz predict` CLI.
- MSA must be pre-computed for offline operation (see `prepare_targets.py --prepare-all`).

---

## 3. RDKit (Cheminformatics Toolkit)

### Overview
RDKit is the de facto open-source cheminformatics library for Python. It handles
SMILES parsing, molecular validation, sanitization, property calculation (QED, SA,
logP), substructure search, fingerprinting, and 3D coordinate generation.

### GitHub Repository
- **URL:** https://github.com/rdkit/rdkit
- Official website: https://www.rdkit.org/

### Paper / Reference
- Greg Landrum et al. *"RDKit: Open-source cheminformatics"*
  https://www.rdkit.org/
- Cite as: RDKit: Open-source cheminformatics. https://www.rdkit.org

### Installation
```bash
# Recommended: conda (most reliable for RDKit)
conda install -c conda-forge rdkit

# Also available via pip (since ~2021)
pip install rdkit
# or the older package name:
pip install rdkit-pypi
```

### Key Python Dependencies
- `numpy`
- `boost` (C++ backend, bundled in conda/pip packages)
- `matplotlib` (optional, for molecule drawing)
- `pillow` / `PIL` (optional, for image output)
- `pandas` (optional, for PandasTools integration)

### Key API Usage Patterns
```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Draw

# Parse and validate a SMILES string
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
if mol is None:
    print("Invalid molecule!")

# Sanitize a molecule
Chem.SanitizeMol(mol)

# Compute drug-likeness properties
qed_score = QED.qed(mol)
logp = Descriptors.MolLogP(mol)
mol_wt = Descriptors.MolWt(mol)

# Generate 3D coordinates (needed before docking)
mol_h = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol_h, AllChem.ETML())
AllChem.MMFFOptimizeMolecule(mol_h)

# Write to SDF for docking input
writer = Chem.SDWriter("ligand.sdf")
writer.write(mol_h)
writer.close()

# Morgan fingerprint for similarity
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Canonical SMILES
canonical = Chem.MolToSmiles(mol, canonical=True)
```

### Notes for Dual-Docking Workshop
- Essential for validating FREED++-generated molecules before docking.
- Use RDKit to convert SMILES -> 3D SDF for Vina or Boltz-2 input.
- Filter molecules by Lipinski's Rule of Five, QED, synthetic accessibility.

---

## 4. PDB / UniProt Access (Protein Structure & Sequence Retrieval)

### Overview
Two complementary databases:
- **PDB (Protein Data Bank):** Experimental 3D structures (X-ray, cryo-EM, NMR).
  URL: https://www.rcsb.org/
- **UniProt:** Protein sequences, functional annotations, cross-references.
  URL: https://www.uniprot.org/

### GitHub Repositories / Tools
- **Biopython:** https://github.com/biopython/biopython
  General bioinformatics toolkit with PDB and UniProt access.
- **pypdb:** https://github.com/williamgilpin/pypdb
  Lightweight Python client for the RCSB PDB REST API.
- **rcsbsearchapi:** https://github.com/rcsb/py-rcsbsearchapi
  Official RCSB search API Python client.
- **UniProt REST API:** No special library needed; use `requests`.

### Installation
```bash
pip install biopython       # PDB file download, parsing, UniProt
pip install pypdb            # PDB search & metadata
pip install rcsbsearchapi    # Official RCSB search
pip install requests         # For direct REST API calls
```

### Key Python Dependencies
- `biopython` (Bio.PDB, Bio.SeqIO)
- `requests`
- `numpy`

### Key API Usage Patterns

#### Download a PDB structure
```python
from Bio.PDB import PDBList

pdbl = PDBList()
# Downloads to current directory, returns file path
pdb_file = pdbl.retrieve_pdb_file("2HYY", pdir="structures/", file_format="pdb")

# Or using requests directly (mmCIF format):
import requests
url = "https://files.rcsb.org/download/2HYY.cif"
response = requests.get(url)
with open("2HYY.cif", "w") as f:
    f.write(response.text)
```

#### Parse a PDB file
```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structures/2HYY.pdb")
model = structure[0]
for chain in model:
    print(f"Chain {chain.id}: {len(list(chain.get_residues()))} residues")
```

#### Search PDB by keyword
```python
from pypdb import Query
from pypdb import describe_pdb

# Text search
results = Query("CDK2 inhibitor").search()
print(f"Found {len(results)} structures")

# Get metadata
desc = describe_pdb("2HYY")
```

#### Fetch UniProt entry
```python
import requests

uniprot_id = "P24941"  # CDK2
url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
response = requests.get(url)
data = response.json()
print(data["proteinDescription"]["recommendedName"]["fullName"]["value"])

# Get FASTA sequence
fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
fasta = requests.get(fasta_url).text
```

### Notes for Dual-Docking Workshop
- Download target protein structures for both targets in the dual-docking setup.
- Extract binding sites, prepare receptor PDBQT files for AutoDock Vina.
- Cross-reference UniProt to ensure correct isoforms and active-site annotations.

---

## 5. Dual-Target Protein Pairs for Benchmarking

### Pair 1: CDK2 / CDK4 (Cyclin-Dependent Kinase Selectivity)

| Property | CDK2 | CDK4 |
|----------|------|------|
| **UniProt** | P24941 | P11802 |
| **PDB ID** | 2HYY (with inhibitor) | 2W96 (with inhibitor) |
| **Alt PDB** | 1H1Q, 4GCJ | 2W9Z, 6GUB |

**Rationale:** CDK2 and CDK4 are both cyclin-dependent kinases driving cell cycle
progression. They share a highly conserved ATP-binding pocket but differ in selectivity
determinants. Designing molecules that selectively inhibit one over the other (or
dual-inhibit both) is a classic medicinal chemistry challenge in oncology.
Palbociclib (CDK4/6-selective) and clinical CDK2 inhibitors illustrate the
therapeutic relevance.

---

### Pair 2: EGFR / HER2 (ErbB Receptor Tyrosine Kinase Family)

| Property | EGFR | HER2 |
|----------|------|------|
| **UniProt** | P00533 | P04626 |
| **PDB ID** | 1M17 (with erlotinib) | 3PP0 (with lapatinib-analog) |
| **Alt PDB** | 4HJO, 4ZAU | 3RCD, 6OGE |

**Rationale:** EGFR and HER2 are members of the ErbB family of receptor tyrosine
kinases. Both are validated oncology targets (NSCLC, breast cancer). Lapatinib is
an approved dual EGFR/HER2 inhibitor, demonstrating the feasibility and clinical
value of dual-targeting. The binding pockets are similar but with key residue
differences enabling selectivity studies.

---

### Pair 3: COX-1 / COX-2 (Cyclooxygenase Selectivity)

| Property | COX-1 | COX-2 |
|----------|-------|-------|
| **UniProt** | P23219 | P35354 |
| **PDB ID** | 3KK6 (with celecoxib-analog) | 3LN1 (with celecoxib) |
| **Alt PDB** | 1EQG | 1CX2, 5IKR |

**Rationale:** COX-1 and COX-2 are prostaglandin-endoperoxide synthases central to
inflammation. COX-2-selective inhibitors (celecoxib, rofecoxib) were developed to
reduce GI side-effects of non-selective NSAIDs. This is one of the most classic
selectivity problems in drug design, with extensive SAR data available. The binding
sites differ primarily in the size of a hydrophobic side pocket (Val523 in COX-2
vs. Ile523 in COX-1).

---

### Pair 4: AChE / BuChE (Cholinesterase Dual-Target for Neurodegeneration)

| Property | AChE | BuChE |
|----------|------|-------|
| **UniProt** | P22303 | P06276 |
| **PDB ID** | 4EY7 (with donepezil) | 4BDS (with tacrine derivative) |
| **Alt PDB** | 1EVE, 4M0E | 1P0I, 6QAA |

**Rationale:** Acetylcholinesterase (AChE) and butyrylcholinesterase (BuChE) are
both targets for Alzheimer's disease. As disease progresses, BuChE activity
increases while AChE decreases, making dual AChE/BuChE inhibitors attractive for
sustained therapeutic effect. Donepezil (AChE-selective) and rivastigmine
(dual-inhibitor) are approved drugs. The catalytic gorges share structural
homology but differ in residue composition at the peripheral anionic site.

---

### Pair 5: JAK1 / JAK2 (Janus Kinase Selectivity for Autoimmune/Oncology)

| Property | JAK1 | JAK2 |
|----------|------|------|
| **UniProt** | P23458 | O60674 |
| **PDB ID** | 6N7A (with upadacitinib) | 4AQC (with ruxolitinib) |
| **Alt PDB** | 6BBU, 7T6F | 6VNE, 3FUP |

**Rationale:** JAK1 and JAK2 are members of the Janus kinase family involved in
cytokine signaling. Selective JAK1 inhibition (upadacitinib) is preferred for
autoimmune diseases (rheumatoid arthritis), while JAK2 inhibition (ruxolitinib)
targets myeloproliferative neoplasms. Dual JAK1/2 inhibitors like baricitinib also
exist. The ATP-binding pockets are ~85% similar, making selectivity a nuanced
design challenge ideal for benchmarking generative models.

---

## Summary Table

| Tool | Repo | Install | Primary Use in Workshop |
|------|------|---------|------------------------|
| FREED++ | github.com/AIRI-Institute/FFREED | Docker (Python 3.7) | RL-based molecule generation with QuickVina2 reward |
| Boltz-2 | github.com/jwohlwend/boltz | pip install boltz | Structure + affinity prediction (scoring oracle) |
| RDKit | github.com/rdkit/rdkit | conda install rdkit | Molecule validation, sanitization, properties |
| PDB/UniProt | BioPython + REST APIs | pip install biopython | Download & parse protein targets |

| Target Pair | Disease Area | Design Challenge |
|-------------|-------------|-----------------|
| CDK2 / CDK4 | Cancer | Selectivity within kinase subfamily |
| EGFR / HER2 | Cancer | Dual-inhibition of ErbB family |
| COX-1 / COX-2 | Inflammation | Classic selectivity benchmark |
| AChE / BuChE | Alzheimer's | Dual-inhibition for neurodegeneration |
| JAK1 / JAK2 | Autoimmune/Cancer | Selectivity in JAK family |
