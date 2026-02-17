# Reward Modes Configuration

This document explains how `configs/reward_modes.yaml` works and how reward
definitions are resolved at runtime.

## Why This Exists

The project now separates reward logic into two layers:

1. **Evaluator kinds (code)**: implemented in `dualdocking/reward/selectivity.py`
2. **Mode declarations (config)**: declared in `configs/reward_modes.yaml`

This removes reward duplication across generators (RL and non-RL backends) and
makes default mode parameters auditable in one place.

## Runtime Flow

When `dualdocking.reward.selectivity` is imported:

1. It tries to load mode declarations from YAML.
2. Each YAML mode is validated and registered in the in-process registry.
3. Optional plugin modules are loaded from `DUALDOCKING_REWARD_PLUGINS`.

Calls such as `resolve_reward_mode(name, params)` and
`evaluate_reward_mode(name, ...)` always use this registry.

## Config Path Resolution

By default, the first existing path is used:

1. `DUALDOCKING_REWARD_MODES_PATH` (explicit override)
2. `<module_dir>/reward_modes.yaml`
3. `<repo_root>/configs/reward_modes.yaml`
4. `<cwd>/configs/reward_modes.yaml`

In FREED++ Docker fallback, the file is copied to `/opt/reward_modes.yaml`,
which matches rule 2 for `/opt/reward_selectivity.py`.

## YAML Schema

`configs/reward_modes.yaml`:

```yaml
version: 1
modes:
  selective_weighted:
    kind: selective_weighted
    requires_secondary_target: true
    requires_dual_docking: true
    default_params:
      lambda_off: 1.0
```

### Field Semantics

- `kind`: built-in evaluator identifier (implemented in code)
- `requires_secondary_target`:
  - `true` => mode needs target-B normalization stats
- `requires_dual_docking`:
  - `true` => mode scores both targets per reward call
- `default_params`:
  - numeric defaults merged with runtime overrides

`requires_dual_docking=true` with `requires_secondary_target=false` is rejected.

## Built-in Evaluator Kinds

Implemented in `dualdocking/reward/selectivity.py`:

- `single_a`: `z_a`
- `single_b`: `z_b`
- `double`: `max(z_a, z_b)`
- `linear_equal`: `z_a + z_b`
- `biased_a`: `z_a + weight_b * z_b`
- `selective_a`: `z_a - z_b`
- `selective_weighted`: `z_a - lambda_off * z_b`

Where `z_x` is z-normalized docking score for target X.

## Runtime Parameter Overrides

`resolve_reward_mode(name, params)` merges:

1. YAML `default_params`
2. runtime `params` override

For FREED++ patch (`docker/freedpp/dual_reward_patch.py`), overrides come from:

- `REWARD_PARAMS_JSON` (JSON mapping)
- `REWARD_LAMBDA_OFF` fallback for `lambda_off`
- `REWARD_WEIGHT_B` fallback for `weight_b`

The patch writes resolved metadata to `reward_metadata.json`, including:

- `reward_mode`
- `reward_params`
- `primary_target`

## Docker Passthrough Is Configurable

Reward-mode config passthrough is backend-specific and can be toggled.
For FREED++ (`generator.freedpp` in `configs/experiment.yaml`):

```yaml
generator:
  freedpp:
    pass_reward_modes_to_container: true
    reward_modes_path_in_container: /opt/reward_modes.yaml
```

- `pass_reward_modes_to_container: true`:
  sets `DUALDOCKING_REWARD_MODES_PATH` inside the container.
- `pass_reward_modes_to_container: false`:
  does not set this env var; the model/container resolves reward config on its
  own.

This is useful when different model images have different internal config
layouts.

## Plugins vs YAML

YAML can only reference built-in `kind` values.
For new formulas, use a plugin:

1. Implement Python module with `register_reward_modes()`
2. Call `register_reward_mode(...)` inside it
3. Set `DUALDOCKING_REWARD_PLUGINS="package.module[:callable]"`

Use YAML for parameterization and mode wiring.
Use plugins for new math.

## Scientific Caveats

- `selective_weighted` (`z_a - lambda_off*z_b`) is sensitive to calibration
  quality of both targets.
- A single global `lambda_off` can be suboptimal across heterogeneous pairs.
- Any change in mode semantics should be recorded with exact config values to
  preserve reproducibility.
