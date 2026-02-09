# Documentation Index

Source priority and conflict resolution rules.

Updated: 2026-02-09

## Source Priority (higher wins)

1. `docs/goal.md` — supreme authority: hypothesis, endpoints, success criteria, notation. **Read-only without user approval.**
2. `configs/*.yaml` + `src/dualdocking/` — runtime parameters and computation logic
3. `docs/preregistration.md`, `docs/statistical_analysis_plan.md` — frozen analysis contracts
4. `docs/selection_criteria.md` — pair rationale
5. `report/chapters/*.tex` — narrative (must not contradict above)
6. `README.md` — operational guide (may simplify, must not misrepresent)

## Document Status

| File | Status | Purpose |
|------|--------|---------|
| [`goal.md`](goal.md) | Operational spec | Hypothesis, method, metrics, success criteria |
| [`setup.md`](setup.md) | Canonical | Installation, Docker, calibration, baseline |
| [`pipeline.md`](pipeline.md) | Canonical | Full experiment pipeline: fetch -> run -> analyze |
| [`architecture.md`](architecture.md) | Reference | Boltz-2, FREED++, caching, score conventions, troubleshooting |
| [`selection_criteria.md`](selection_criteria.md) | Reference | Pair rationale; actual pairs from `configs/pairs.yaml` |
| [`preregistration.md`](preregistration.md) | Frozen (v1.1) | Pre-registration: hypotheses, endpoints, decision rules |
| [`statistical_analysis_plan.md`](statistical_analysis_plan.md) | Frozen (v1.1) | SAP: inference, claim rules, code mapping |
| [`receptor_qc.md`](receptor_qc.md) | Reference | Per-target structural quality control |
| [`tasks.md`](tasks.md) | Living log | Task journal and execution plan |

## Conflict Resolution

If a command/parameter in a document conflicts with code or `README.md`, use code and `README.md`.

See `AGENTS.md` for development conventions (architecture, code style, testing).
