"""Hook composition and monkey patch application for dual reward mode."""
# ruff: noqa: SLF001

from __future__ import annotations

import atexit
from collections.abc import Callable
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import NamedTemporaryFile
from typing import Any

import dual_reward_patch_gpu as gpu
import dual_reward_patch_state as state
from dual_reward_patch_deps import prepare_ligand_pdbqt


def _build_rdkit_docking(docking_vina_cls: Any) -> Callable[..., float]:
    """Build ``DockingVina.docking`` replacement using RDKit+Meeko prep."""

    def _rdkit_docking(  # noqa: PLR0913
        smile: str,
        *,
        vina_program: str,
        receptor: str,
        box_center: Any,
        box_size: Any,
        error_val: float,
        seed: int,
        num_modes: int,
        exhaustiveness: int,
        timeout_dock: int,
        timeout_gen3d: int,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> float:
        state._set_last_docking_timed_out(value=False)
        receptor_name = state._receptor_label(receptor)
        short_smile = state._short_value(smile)

        cache_key = (receptor, smile)
        if cache_key in gpu._gpu_dock_cache:
            state.logger.debug(
                "dual_reward_patch: stage=rdkit_docking cache_hit receptor=%s smile=%s",
                receptor_name,
                short_smile,
            )
            return gpu._gpu_dock_cache[cache_key]
        state.logger.debug(
            "dual_reward_patch: stage=rdkit_docking cache_miss receptor=%s smile=%s",
            receptor_name,
            short_smile,
        )

        center = state._coords(box_center)
        size = state._coords(box_size)

        with (
            NamedTemporaryFile(mode="w", suffix=".pdbqt") as ligand,
            NamedTemporaryFile(mode="r+t", suffix=".pdbqt") as docking_file,
        ):
            if not prepare_ligand_pdbqt(
                smile,
                ligand.name,
                seed=seed,
                max_heavy=state._MAX_HEAVY_ATOMS,
                max_rotatable=state._MAX_ROTATABLE_BONDS,
            ):
                state.logger.warning(
                    "dual_reward_patch: stage=rdkit_docking ligand_prep_failed receptor=%s smile=%s",
                    receptor_name,
                    short_smile,
                )
                return error_val

            cmd = [
                vina_program,
                "--receptor",
                receptor,
                "--ligand",
                ligand.name,
                "--out",
                docking_file.name,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--num_modes",
                str(num_modes),
                "--exhaustiveness",
                str(exhaustiveness),
                "--seed",
                str(seed),
            ]

            try:
                result = run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_dock,
                    check=False,
                )
            except TimeoutExpired:
                state._set_last_docking_timed_out(value=True)
                state.logger.warning(
                    "dual_reward_patch: stage=rdkit_docking timeout receptor=%s timeout=%ss smile=%s",
                    receptor_name,
                    timeout_dock,
                    short_smile,
                )
                return error_val

            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()[-500:]
                if tail:
                    state.logger.debug("dual_reward_patch: unidock rc=%s: %s", result.returncode, tail)
                state.logger.warning(
                    "dual_reward_patch: stage=rdkit_docking failed rc=%d receptor=%s smile=%s",
                    result.returncode,
                    receptor_name,
                    short_smile,
                )
                return error_val

            parsed = docking_vina_cls.parse_output(result.stdout, error_val)
            if parsed != error_val:
                if not (-20.0 <= parsed <= 10.0):
                    state.logger.warning(
                        "corrupt docking score %.3f from CPU path: %s",
                        parsed,
                        smile,
                    )
                    return error_val
                state.logger.debug(
                    "dual_reward_patch: stage=rdkit_docking parsed_stdout receptor=%s score=%.3f smile=%s",
                    receptor_name,
                    parsed,
                    short_smile,
                )
                return parsed

            parsed_from_file = state._parse_result_from_pdbqt(Path(docking_file.name))
            if parsed_from_file is not None:
                state.logger.debug(
                    "dual_reward_patch: stage=rdkit_docking parsed_file receptor=%s score=%.3f smile=%s",
                    receptor_name,
                    parsed_from_file,
                    short_smile,
                )
                return parsed_from_file
            state.logger.warning(
                "dual_reward_patch: stage=rdkit_docking parse_failed receptor=%s smile=%s",
                receptor_name,
                short_smile,
            )
            return error_val

    return _rdkit_docking


def _build_dual_call(
    *,
    original_call: Callable[..., float],
    context: state._PatchContext,
) -> Callable[..., float]:
    """Build ``DockingVina.__call__`` replacement for configured reward mode."""

    def _dual_call(self: Any, *args: Any, **kwargs: Any) -> float:
        state._call_stats["total"] += 1
        call_idx = state._call_stats["total"]
        mode = context.reward_mode
        state._log_call_debug(
            "dual_reward_patch: stage=reward_call start idx=%d mode=%s",
            call_idx,
            mode.name,
        )

        if mode.requires_secondary_target and not mode.requires_dual_docking:
            if context.stats_b is None:
                raise RuntimeError(f"stats_b missing for mode '{mode.name}'")
            score_b = original_call(self, *args, **kwargs)
            timed_out_b = state._consume_last_docking_timed_out()
            error_val_b = float(self.config.get("error_val", 100.0))
            z_b, invalid_b = state._evaluate_target_score(
                label="b",
                score=score_b,
                timed_out=timed_out_b,
                error_val=error_val_b,
                stats=context.stats_b,
            )
            state._log_call_debug(
                "dual_reward_patch: stage=reward_call done idx=%d mode=single_b score_b=%.3f z_b=%.3f invalid_b=%s timeout_b=%s",
                call_idx,
                score_b,
                z_b,
                invalid_b,
                timed_out_b,
            )
            return mode.evaluate(z_b=z_b)

        score_a = original_call(self, *args, **kwargs)
        timed_out_a = state._consume_last_docking_timed_out()
        error_val_a = float(self.config.get("error_val", 100.0))
        z_a, invalid_a = state._evaluate_target_score(
            label="a",
            score=score_a,
            timed_out=timed_out_a,
            error_val=error_val_a,
            stats=context.stats_a,
        )

        if not mode.requires_secondary_target:
            reward_single = mode.evaluate(z_a=z_a)
            state._log_call_debug(
                "dual_reward_patch: stage=reward_call done idx=%d mode=%s "
                "score_a=%.3f z_a=%.3f reward=%.3f invalid_a=%s timeout_a=%s",
                call_idx,
                mode.name,
                score_a,
                z_a,
                reward_single,
                invalid_a,
                timed_out_a,
            )
            return reward_single

        if context.dual_cfg is None:
            raise RuntimeError(f"dual_cfg missing for mode '{mode.name}'")
        if context.stats_b is None:
            raise RuntimeError(f"stats_b missing for mode '{mode.name}'")

        score_b = state._score_secondary_target(self, original_call, context.dual_cfg, *args, **kwargs)
        timed_out_b = state._consume_last_docking_timed_out()
        error_val_b = float(self.config.get("error_val", 100.0))
        z_b, invalid_b = state._evaluate_target_score(
            label="b",
            score=score_b,
            timed_out=timed_out_b,
            error_val=error_val_b,
            stats=context.stats_b,
        )

        reward = mode.evaluate(z_a=z_a, z_b=z_b)
        state._log_call_debug(
            "dual_reward_patch: stage=reward_call done idx=%d mode=%s "
            "z_a=%.3f z_b=%.3f reward=%.3f invalid_a=%s invalid_b=%s timeout_a=%s timeout_b=%s",
            call_idx,
            mode.name,
            z_a,
            z_b,
            reward,
            invalid_a,
            invalid_b,
            timed_out_a,
            timed_out_b,
        )
        return reward

    return _dual_call


def _patch_cache_and_pool(docking_vina_cls: Any, context: state._PatchContext) -> None:
    """Patch CacheAndPool to precompute docking with UniDock GPU batch."""
    if not gpu._GPU_BATCH_ENABLED:
        state.logger.info("dual_reward_patch: GPU batch docking DISABLED (GPU_BATCH_DOCK=0)")
        return

    try:
        from ffreed.utils import CacheAndPool
    except (ImportError, AttributeError):
        state.logger.debug("dual_reward_patch: ffreed.utils.CacheAndPool not available, GPU batch disabled")
        CacheAndPool = None  # type: ignore[assignment,misc]  # noqa: N806

    if CacheAndPool is None:
        state.logger.info("dual_reward_patch: GPU batch docking DISABLED (CacheAndPool unavailable)")
        return

    original_cap_call = CacheAndPool.__call__

    def _gpu_aware_cap_call(self: Any, args: list[str]) -> list[Any]:
        unseen = list(set(args).difference(self.results))
        mode = context.reward_mode
        state.logger.info(
            "dual_reward_patch: stage=gpu_cache_pool start requested=%d unseen=%d cached=%d",
            len(args),
            len(unseen),
            len(self.results),
        )
        if unseen and hasattr(self.func, "property") and isinstance(self.func.property, docking_vina_cls):
            config = self.func.property.config
            state.logger.info(
                "dual_reward_patch: stage=gpu_cache_pool precompute primary=%s mode=%s",
                state._receptor_label(config["receptor"]),
                mode.name,
            )
            gpu._precompute_gpu_batch_for_receptor(
                unseen,
                config,
                config["receptor"],
                config["box_center"],
                config["box_size"],
            )
            if mode.requires_dual_docking and context.dual_cfg is not None:
                state.logger.info(
                    "dual_reward_patch: stage=gpu_cache_pool precompute secondary=%s",
                    state._receptor_label(context.dual_cfg.receptor_b),
                )
                gpu._precompute_gpu_batch_for_receptor(
                    unseen,
                    config,
                    context.dual_cfg.receptor_b,
                    context.dual_cfg.center_b,
                    context.dual_cfg.size_b,
                )

            values = list(map(self.func, unseen))
            new_results = dict(zip(unseen, values))
            self.results = {**self.results, **new_results}
            state.logger.info(
                "dual_reward_patch: stage=gpu_cache_pool scored unseen=%d total_cached=%d",
                len(unseen),
                len(self.results),
            )

            dual_receptor_b = context.dual_cfg.receptor_b if context.dual_cfg is not None else None
            gpu._cleanup_gpu_cache_entries(config["receptor"], dual_receptor_b, unseen)
            state.logger.debug(
                "dual_reward_patch: stage=gpu_cache_pool cleaned_cache_entries smiles=%d",
                len(unseen),
            )
            return [self.results[arg] for arg in args]

        state.logger.debug("dual_reward_patch: stage=gpu_cache_pool fallback=original_cache_and_pool")
        return original_cap_call(self, args)

    CacheAndPool.__call__ = _gpu_aware_cap_call
    state.logger.info("dual_reward_patch: GPU batch docking ENABLED (CacheAndPool patched)")


def _patch_docking_vina(docking_vina_cls: Any, context: state._PatchContext) -> None:
    """Apply DockingVina monkey patches for docking and reward modes."""
    original_call = docking_vina_cls.__call__

    docking_vina_cls.docking = staticmethod(_build_rdkit_docking(docking_vina_cls))
    state.logger.info("dual_reward_patch: stage=patching replaced DockingVina.docking backend=rdkit_meeko")

    docking_vina_cls.__call__ = _build_dual_call(original_call=original_call, context=context)
    atexit.register(state._write_reward_metadata)
    state.logger.info("dual_reward_patch: stage=patching replaced DockingVina.__call__ and registered metadata writer")

    _patch_cache_and_pool(docking_vina_cls, context)


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina for dual-target reward calculation."""
    context = state._load_patch_context(gpu_batch_enabled=gpu._GPU_BATCH_ENABLED)

    # ffreed must already be on sys.path (set by run_with_patch.py)
    from ffreed.env.docking import DockingVina

    state.logger.info("dual_reward_patch: stage=patching hook_target=DockingVina")
    _patch_docking_vina(DockingVina, context)
    state.logger.info(
        "dual_reward_patch: patched DockingVina"
        " (reward_mode=%s, ligand_backend=rdkit_meeko,"
        " invalid_z_penalty=%.1f, max_z_abs=%.1f)",
        context.reward_mode.name,
        state._INVALID_Z_PENALTY,
        state._MAX_Z_ABS,
    )


__all__ = [
    "_apply_dual_patch",
    "_build_dual_call",
    "_build_rdkit_docking",
    "_patch_cache_and_pool",
    "_patch_docking_vina",
]
