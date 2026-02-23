"""REINVENT4 TOML configuration generator.

Translates :class:`Reinvent4GenerationConfig` + docking target metadata into
a TOML file consumable by the ``reinvent`` CLI.

The generator produces a single-stage RL config using the DAP learning
strategy with one ``DualDocking`` scoring component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualdocking.generation.reinvent4.scoring import DockingTarget


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Reinvent4TOMLConfig:
    """All parameters needed to write a REINVENT4 TOML config file.

    Attributes:
        output_dir: REINVENT4 output directory (logs, checkpoints, results).
        prior_model: Path to the prior model (empty string = built-in ChEMBL prior).
        agent_model: Path to the agent model (empty string = copy from prior).
        algorithm: Learning strategy (``"dap"`` recommended).
        batch_size: Molecules sampled per RL step.
        n_steps: Total RL steps in this stage.
        sigma: DAP sigma parameter.
        learning_rate: Agent learning rate.
        seed: Random seed.
        use_diversity_filter: Whether to apply scaffold diversity filter.
        target_a: Primary docking target config.
        target_b: Optional secondary docking target config.
        reward_mode: Reward mode name.
        vina_program: Docking binary.
        device: CUDA device string.
    """

    output_dir: Path = field(default_factory=lambda: Path("output"))
    prior_model: str = ""
    agent_model: str = ""
    algorithm: str = "dap"
    batch_size: int = 128
    n_steps: int = 500
    sigma: int = 128
    learning_rate: float = 0.0001
    seed: int = 42
    use_diversity_filter: bool = True
    target_a: DockingTarget | None = None
    target_b: DockingTarget | None = None
    reward_mode: str = "single_a"
    vina_program: str = "unidock"
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------


def _quote(value: str) -> str:
    """Quote a string for TOML, escaping backslashes and quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_list_str(values: list[str]) -> str:
    """Format a list of strings as TOML inline array."""
    return "[" + ", ".join(_quote(v) for v in values) + "]"


def _toml_list_float(values: list[float]) -> str:
    """Format a list of floats as TOML inline array."""
    return "[" + ", ".join(str(v) for v in values) + "]"


def generate_toml(config: Reinvent4TOMLConfig) -> str:
    """Generate a REINVENT4 TOML configuration string.

    Args:
        config: All parameters for the REINVENT4 run.

    Returns:
        TOML-formatted configuration string.

    Raises:
        ValueError: If ``target_a`` is not set.
    """
    if config.target_a is None:
        msg = "target_a is required for TOML generation"
        raise ValueError(msg)

    # REINVENT4 prior registry: ".reinvent" resolves to bundled ChEMBL prior.
    # Empty string would resolve to CWD via Path("").resolve(), causing errors.
    prior = config.prior_model or ".reinvent"
    agent = config.agent_model or prior

    # Diversity filter section
    if config.use_diversity_filter:
        df_section = """\
[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4
"""
    else:
        df_section = ""

    # Target B parameters (empty strings/zeros when absent)
    ta = config.target_a
    tb = config.target_b
    receptor_b = str(tb.pdbqt_path) if tb else ""
    center_b = tb.center if tb else ""
    size_b = tb.size if tb else ""
    norm_mean_b = tb.norm_mean if tb else 0.0
    norm_std_b = tb.norm_std if tb else 1.0

    out_dir = str(config.output_dir)
    toml = f"""\
# Auto-generated REINVENT4 config — do not edit manually.
run_type = "staged_learning"
device = {_quote(config.device)}
seed = {config.seed}

[parameters]
prior_file = {_quote(prior)}
agent_file = {_quote(agent)}
summary_csv_prefix = {_quote(out_dir + "/scaffold_memory")}
batch_size = {config.batch_size}
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = {_quote(config.algorithm)}
sigma = {config.sigma}
rate = {config.learning_rate}

{df_section}
[[stage]]
max_score = 1.0
max_steps = {config.n_steps}
chkpt_file = {_quote(out_dir + "/agent.ckpt")}

[stage.scoring]
type = "custom_sum"

[[stage.scoring.component]]
[stage.scoring.component.DualDockingComponent]

[[stage.scoring.component.DualDockingComponent.endpoint]]
name = "docking_reward"
weight = 1

[stage.scoring.component.DualDockingComponent.endpoint.params]
receptor_a = {_quote(str(ta.pdbqt_path))}
center_a = {_quote(ta.center)}
size_a = {_quote(ta.size)}
norm_mean_a = {ta.norm_mean}
norm_std_a = {ta.norm_std}
receptor_b = {_quote(receptor_b)}
center_b = {_quote(center_b)}
size_b = {_quote(size_b)}
norm_mean_b = {norm_mean_b}
norm_std_b = {norm_std_b}
reward_mode = {_quote(config.reward_mode)}
vina_program = {_quote(config.vina_program)}
"""
    return toml


def write_toml(config: Reinvent4TOMLConfig, path: Path) -> Path:
    """Generate TOML config and write to file.

    Args:
        config: REINVENT4 configuration.
        path: Output file path.

    Returns:
        The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_toml(config))
    return path
