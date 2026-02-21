"""Generate publication-quality MD validation figures.

Produces a multi-panel figure suitable for Nature/JACS/JMC:
  Panel A: 3D binding pose (PyMOL ray-traced, cartoon + sticks + surface)
  Panel B: RMSD time series (backbone + ligand)
  Panel C: Per-residue RMSF with binding site annotation
  Panel D: 2D protein-ligand interaction fingerprint (ProLIF)
  Panel E: Radius of gyration + H-bonds
  Panel F: Summary dashboard (2x2)

Usage:
    uv run python scripts/s6_md/render_md_figures.py --pair CDK2_ERK2 --mode single_a --mol 0
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Style ────────────────────────────────────────────────────────────
COLORS = {
    "A": "#2563eb",  # blue
    "B": "#dc2626",  # red
}
# Okabe-Ito colorblind-safe
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

matplotlib.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ── Panel A: PyMOL 3D binding pose ──────────────────────────────────
def render_binding_pose(
    gro_path: Path,
    xtc_path: Path,
    out_png: Path,
    target_label: str,
    *,
    frame: int = -1,
    width: int = 1800,
    height: int = 1400,
) -> None:
    """Ray-trace a publication-quality binding pose with PyMOL."""
    from pymol import cmd

    cmd.reinitialize()
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadow", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_gain", 0.1)
    cmd.set("ambient", 0.4)
    cmd.set("specular", 0.15)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    # Load reference structure
    cmd.load(str(gro_path), "complex")

    # Load trajectory if available and select frame
    if xtc_path.exists():
        cmd.load_traj(str(xtc_path), "complex")
        if frame == -1:
            n_states = cmd.count_states("complex")
            frame = max(1, n_states * 3 // 4)  # 75% through trajectory
        cmd.frame(frame)

    # Select protein and ligand
    cmd.select("prot", "polymer.protein")
    cmd.select("lig", "resn LIG")
    cmd.select("pocket", "byres (lig around 5.0) and prot")

    # Remove water and ions for clarity
    cmd.remove("resn SOL or resn NA or resn CL or resn HOH or resn WAT")

    # Protein: cartoon (light blue)
    cmd.hide("everything", "prot")
    cmd.show("cartoon", "prot")
    cmd.color("skyblue", "prot")
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("cartoon_fancy_helices", 1)

    # Pocket residues: sticks (gray carbons)
    cmd.show("sticks", "pocket and sidechain")
    cmd.color("gray70", "pocket and elem C")
    cmd.color("tv_blue", "pocket and elem N")
    cmd.color("tv_red", "pocket and elem O")
    cmd.color("tv_yellow", "pocket and elem S")

    # Ligand: sticks (orange carbons)
    cmd.show("sticks", "lig")
    cmd.color("orange", "lig and elem C")
    cmd.color("tv_blue", "lig and elem N")
    cmd.color("tv_red", "lig and elem O")
    cmd.color("tv_yellow", "lig and elem S")
    cmd.set("stick_radius", 0.2, "lig")

    # Transparent surface around pocket
    cmd.create("pocket_surf", "pocket")
    cmd.show("surface", "pocket_surf")
    cmd.set("transparency", 0.7, "pocket_surf")
    cmd.color("gray90", "pocket_surf")

    # H-bond dashes
    cmd.distance("hbonds", "lig", "pocket", mode=2, cutoff=3.5)
    cmd.hide("labels", "hbonds")
    cmd.set("dash_color", "black", "hbonds")
    cmd.set("dash_width", 2.0)
    cmd.set("dash_gap", 0.2)

    # Label pocket residues
    cmd.set("label_font_id", 7)  # sans-serif
    cmd.set("label_size", 14)
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.label("pocket and name CA", '"%s%s" % (resn, resi)')

    # Center and orient
    cmd.center("lig")
    cmd.zoom("lig", 8.0)
    cmd.turn("y", 15)

    # Ray-trace and save
    cmd.ray(width, height)
    cmd.png(str(out_png))
    cmd.delete("all")
    print(f"  [PyMOL] Saved binding pose: {out_png}")


# ── Panel B: RMSD time series ───────────────────────────────────────
def plot_rmsd(targets: dict[str, Path], out_dir: Path) -> None:
    """RMSD time series for backbone and ligand."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=False)

    for label, d in targets.items():
        color = COLORS[label]
        target_name = _load_target_name(d)

        # Backbone
        df = pd.read_csv(d / "rmsd_backbone.csv")
        ax1.plot(df["time_ns"], df["rmsd_A"], color=color, lw=0.6, label=f"{target_name}", alpha=0.85)

        # Ligand
        df = pd.read_csv(d / "rmsd_ligand.csv")
        ax2.plot(df["time_ns"], df["rmsd_A"], color=color, lw=0.6, label=f"{target_name}", alpha=0.85)

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Backbone RMSD (\u00c5)")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False)

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Ligand RMSD (\u00c5)")
    ax2.axhline(3.0, color="gray", ls="--", lw=0.5, zorder=0)
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=False)

    _add_panel_labels(fig, [ax1, ax2], ["B", "C"])
    plt.tight_layout()
    fig.savefig(out_dir / "panel_BC_rmsd.png")
    fig.savefig(out_dir / "panel_BC_rmsd.svg")
    plt.close(fig)
    print(f"  Saved RMSD panels: {out_dir / 'panel_BC_rmsd.png'}")


# ── Panel C: RMSF ───────────────────────────────────────────────────
def plot_rmsf(targets: dict[str, Path], out_dir: Path) -> None:
    """Per-residue RMSF with binding site annotation."""
    fig, ax = plt.subplots(figsize=(5.5, 2.0))

    for label, d in targets.items():
        color = COLORS[label]
        target_name = _load_target_name(d)
        df = pd.read_csv(d / "rmsf.csv")
        ax.plot(df["residue"], df["rmsf_A"], color=color, lw=0.5, label=f"{target_name}", alpha=0.8)

    ax.set_xlabel("Residue number")
    ax.set_ylabel("RMSF (\u00c5)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _add_panel_labels(fig, [ax], ["D"])
    plt.tight_layout()
    fig.savefig(out_dir / "panel_D_rmsf.png")
    fig.savefig(out_dir / "panel_D_rmsf.svg")
    plt.close(fig)
    print(f"  Saved RMSF panel: {out_dir / 'panel_D_rmsf.png'}")


# ── Panel D: ProLIF interaction fingerprint ──────────────────────────
def plot_prolif_interactions(
    gro_path: Path,
    xtc_path: Path,
    out_dir: Path,
    target_label: str,
) -> None:
    """2D protein-ligand interaction fingerprint from MD trajectory."""
    try:
        import MDAnalysis as mda
        import prolif
    except ImportError:
        print("  [SKIP] ProLIF not available")
        return

    target_name = target_label

    u = mda.Universe(str(gro_path), str(xtc_path))
    prot = u.select_atoms("protein")
    lig = u.select_atoms("resname LIG")

    if lig.n_atoms == 0:
        print(f"  [SKIP] No LIG atoms found for {target_name}")
        return

    # Run fingerprint analysis (sample every 10th frame for speed)
    fp = prolif.Fingerprint()
    fp.run(u.trajectory[::10], lig, prot)

    df = fp.to_dataframe()
    if df.empty:
        print(f"  [SKIP] No interactions found for {target_name}")
        return

    # Interaction barcode plot
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    try:
        fp.plot_barcode(ax=ax)
        ax.set_title(f"Protein\u2013Ligand Interactions \u2014 {target_name}")
        _add_panel_labels(fig, [ax], ["E"])
        plt.tight_layout()
        fig.savefig(out_dir / f"panel_E_prolif_{target_label}.png")
        fig.savefig(out_dir / f"panel_E_prolif_{target_label}.svg")
        plt.close(fig)
        print(f"  Saved ProLIF barcode: panel_E_prolif_{target_label}.png")
    except Exception as exc:
        print(f"  [WARN] ProLIF barcode failed: {exc}")
        plt.close(fig)

    # Ligand interaction network (2D diagram)
    try:
        net = fp.plot_lignetwork(lig)
        net.save(str(out_dir / f"lignetwork_{target_label}.html"))
        print(f"  Saved LigNetwork: lignetwork_{target_label}.html")
    except Exception as exc:
        print(f"  [WARN] LigNetwork failed: {exc}")


# ── Panel E: Rg + H-bonds ───────────────────────────────────────────
def plot_rg_hbonds(targets: dict[str, Path], out_dir: Path) -> None:
    """Radius of gyration and H-bond count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2))

    for label, d in targets.items():
        color = COLORS[label]
        target_name = _load_target_name(d)

        # Rg
        df = pd.read_csv(d / "radius_gyration.csv")
        ax1.plot(df["time_ps"] / 1000, df["rg_A"], color=color, lw=0.6, label=f"{target_name}", alpha=0.85)

        # H-bonds
        df = pd.read_csv(d / "hbonds.csv")
        if not df.empty and "time_ns" in df.columns:
            ax2.plot(df["time_ns"], df["n_hbonds"], color=color, lw=0.5, label=f"{target_name}", alpha=0.7)

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Radius of Gyration (\u00c5)")
    ax1.legend(frameon=False)

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("H-bonds (count)")
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=False)

    _add_panel_labels(fig, [ax1, ax2], ["F", "G"])
    plt.tight_layout()
    fig.savefig(out_dir / "panel_FG_rg_hbonds.png")
    fig.savefig(out_dir / "panel_FG_rg_hbonds.svg")
    plt.close(fig)
    print(f"  Saved Rg + H-bonds: {out_dir / 'panel_FG_rg_hbonds.png'}")


# ── Summary table ────────────────────────────────────────────────────
def render_summary_table(targets: dict[str, Path], out_dir: Path) -> None:
    """Render summary metrics as a styled table image."""
    rows = []
    for label, d in targets.items():
        result_file = d / "md_result.json"
        if not result_file.exists():
            continue
        r = json.loads(result_file.read_text())
        rows.append(
            {
                "Target": f"{r['target']} ({label})",
                "Ligand RMSD (\u00c5)": f"{r['mean_rmsd']:.2f} \u00b1 {r['max_rmsd']:.2f}",
                "Backbone RMSD (\u00c5)": f"{r['backbone_rmsd_mean']:.2f}",
                "RMSF pocket (\u00c5)": f"{r['rmsf_binding_pocket']:.2f}",
                "Rg (\u00c5)": f"{r['rg_mean']:.1f} \u00b1 {r['rg_std']:.2f}",
                "H-bonds": f"{r['mean_hbonds']:.1f}",
                "Stable": "\u2713" if r["stable"] else "\u2717",
                "Converged": "\u2713" if r["converged"] else "\u2717",
                "MM-GBSA (kcal/mol)": f"{r['mmpbsa_dg']:.1f}" if not np.isnan(r.get("mmpbsa_dg", float("nan"))) else "\u2014",
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 0.3 + 0.3 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor("#e2e8f0")
        cell.set_text_props(fontweight="bold")

    fig.savefig(out_dir / "summary_table.png")
    fig.savefig(out_dir / "summary_table.svg")
    plt.close(fig)
    print(f"  Saved summary table: {out_dir / 'summary_table.png'}")


# ── Helpers ──────────────────────────────────────────────────────────
def _load_target_name(target_dir: Path) -> str:
    result_file = target_dir / "md_result.json"
    if result_file.exists():
        return json.loads(result_file.read_text()).get("target", target_dir.name)
    return target_dir.name


def _add_panel_labels(fig: plt.Figure, axes: list, labels: list[str]) -> None:
    for ax, lbl in zip(axes, labels):
        ax.text(
            -0.12,
            1.08,
            lbl,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )


# ── Main ─────────────────────────────────────────────────────────────
@click.command()
@click.option("--pair", required=True, help="Pair ID (e.g., CDK2_ERK2)")
@click.option("--mode", default="single_a", help="Generation mode")
@click.option("--mol", default=0, type=int, help="Molecule index")
@click.option("--output-dir", default=None, help="Output directory for figures")
def main(pair: str, mode: str, mol: int, output_dir: str | None) -> None:
    """Render publication-quality MD validation figures."""
    base = PROJECT_ROOT / "output" / "md" / pair / mode / f"mol_{mol}"

    if output_dir is None:
        out = PROJECT_ROOT / "output" / "md_plots" / pair / mode / f"mol_{mol}"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = {}
    for label in ["A", "B"]:
        d = base / f"target_{label}"
        if d.exists() and (d / "md_result.json").exists():
            targets[label] = d

    if not targets:
        print(f"No completed MD results found in {base}")
        return

    print(f"=== Rendering MD figures for {pair}/{mode}/mol_{mol} ===")
    print(f"Output: {out}")
    print(f"Targets: {list(targets.keys())}")

    # Panel A: 3D binding poses
    for label, d in targets.items():
        gro = d / "reference.gro"
        if not gro.exists():
            gro = d / "reference_npt.gro"
        xtc = d / "trajectory.xtc"
        if gro.exists():
            target_name = _load_target_name(d)
            render_binding_pose(
                gro,
                xtc,
                out / f"panel_A_pose_{target_name}.png",
                target_name,
            )

    # Panel B+C: RMSD
    plot_rmsd(targets, out)

    # Panel D: RMSF
    plot_rmsf(targets, out)

    # Panel E: ProLIF interactions
    for label, d in targets.items():
        gro = d / "reference.gro"
        if not gro.exists():
            gro = d / "reference_npt.gro"
        xtc = d / "trajectory.xtc"
        if gro.exists() and xtc.exists():
            target_name = _load_target_name(d)
            plot_prolif_interactions(gro, xtc, out, target_name)

    # Panel F+G: Rg + H-bonds
    plot_rg_hbonds(targets, out)

    # Summary table
    render_summary_table(targets, out)

    print(f"\n=== Done. Figures saved to: {out} ===")
    print("Files:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
