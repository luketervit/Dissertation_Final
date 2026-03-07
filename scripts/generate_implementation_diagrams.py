#!/usr/bin/env python3
"""Generate clean implementation pipeline diagrams for dissertation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon


# Visual system
BG = "#F7F9FC"
INK = "#26344D"
MUTED = "#4A5A78"
C_INPUT = "#E8EEF7"
C_PROCESS = "#EAF3E8"
C_MODEL = "#F5EED9"
C_OUTPUT = "#F3E9EE"


def _setup_ax(figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded_box(ax, x, y, w, h, title, body_lines, fill):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=2.2,
        edgecolor=INK,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h - 0.018,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=INK,
        zorder=3,
    )

    ax.text(
        x + w / 2,
        y + h * 0.34,
        "\n".join(body_lines),
        ha="center",
        va="center",
        fontsize=11,
        color="#2B2F36",
        linespacing=1.35,
        zorder=3,
    )


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color=INK, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.02,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED,
            fontweight="bold",
            zorder=5,
        )


def draw_personality_pipeline(out_path: Path) -> None:
    fig, ax = _setup_ax()

    ax.text(
        0.5,
        0.95,
        "Personality Extraction Pipeline",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )

    # Top row
    rounded_box(
        ax,
        0.03,
        0.60,
        0.20,
        0.22,
        "Raw Tweet Chunks",
        ["may_july_chunk_N.csv", "aug_chunk_N.csv"],
        C_INPUT,
    )
    rounded_box(
        ax,
        0.27,
        0.60,
        0.20,
        0.22,
        "Load + Clean",
        ["parse chunked CSVs", "extract user_id fields", "checkpoint every 1,000 tweets"],
        C_PROCESS,
    )
    rounded_box(
        ax,
        0.51,
        0.60,
        0.20,
        0.22,
        "Per-Tweet Classification",
        ["political leaning", "emotion + sentiment", "hate + offensive"],
        C_MODEL,
    )
    rounded_box(
        ax,
        0.75,
        0.60,
        0.20,
        0.22,
        "Tweet-Level DNA Rows",
        ["labels + confidence", "view/reply engagement", "per-tweet feature record"],
        C_PROCESS,
    )

    # Model family callout
    callout = Rectangle(
        (0.50, 0.585),
        0.22,
        0.25,
        linewidth=2,
        linestyle="--",
        edgecolor=MUTED,
        facecolor="none",
        zorder=1,
    )
    ax.add_patch(callout)
    ax.text(
        0.61,
        0.845,
        "Specialized Twitter Models",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=MUTED,
        fontweight="bold",
    )

    # Bottom row
    rounded_box(
        ax,
        0.54,
        0.24,
        0.28,
        0.20,
        "User-Level Aggregation",
        ["mean: continuous scores", "mode: categorical labels", "sum: engagement features"],
        C_PROCESS,
    )
    rounded_box(
        ax,
        0.22,
        0.24,
        0.30,
        0.20,
        "Agent DNA Outputs",
        ["processed_agents_chunk_N.csv", "agents_for_thread.csv"],
        C_OUTPUT,
    )

    # Arrows
    arrow(ax, 0.23, 0.71, 0.27, 0.71)
    arrow(ax, 0.47, 0.71, 0.51, 0.71)
    arrow(ax, 0.71, 0.71, 0.75, 0.71)
    arrow(ax, 0.85, 0.60, 0.68, 0.44)
    arrow(ax, 0.54, 0.34, 0.52, 0.34)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_simulation_pipeline(out_path: Path) -> None:
    fig, ax = _setup_ax()

    ax.text(
        0.5,
        0.95,
        "Simulation Pipeline (0-Shot Thread Generation)",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )

    rounded_box(
        ax,
        0.12,
        0.81,
        0.76,
        0.13,
        "Inputs",
        ["thread_metadata.json | agents_for_thread.csv | config.yaml"],
        C_INPUT,
    )

    rounded_box(
        ax,
        0.12,
        0.62,
        0.76,
        0.13,
        "Initialize ThreadModel",
        ["load root tweet + few-shot context", "compute mean aggression and instantiate Mesa agents"],
        C_PROCESS,
    )

    # Loop container
    loop = Rectangle(
        (0.11, 0.18),
        0.78,
        0.42,
        linewidth=2,
        linestyle="--",
        edgecolor=MUTED,
        facecolor="none",
        zorder=1,
    )
    ax.add_patch(loop)
    ax.text(
        0.13,
        0.603,
        "Synchronous Per-Round Simulation Loop",
        ha="left",
        va="bottom",
        fontsize=13,
        color=MUTED,
        fontweight="bold",
        bbox=dict(facecolor=BG, edgecolor="none", pad=0.2),
    )

    rounded_box(
        ax,
        0.12,
        0.45,
        0.76,
        0.13,
        "Round t - Stage 1: Generate",
        [
            "sample reply probability and select target post",
            "generate pending reply via persona-conditioned prompt (Dolphin-Llama3/Ollama)",
        ],
        C_MODEL,
    )

    rounded_box(
        ax,
        0.12,
        0.27,
        0.76,
        0.13,
        "Round t - Stage 2: Commit",
        ["commit all pending replies simultaneously", "update thread history and per-round metrics"],
        C_PROCESS,
    )

    # Decision diamond
    cx, cy = 0.50, 0.245
    dw, dh = 0.18, 0.10
    diamond = Polygon(
        [[cx, cy + dh / 2], [cx + dw / 2, cy], [cx, cy - dh / 2], [cx - dw / 2, cy]],
        closed=True,
        facecolor="#FFFFFF",
        edgecolor=INK,
        linewidth=2.2,
        zorder=2,
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, "More rounds\nremaining?", ha="center", va="center", fontsize=12.5, color="#2B2F36")

    rounded_box(
        ax,
        0.12,
        0.04,
        0.76,
        0.12,
        "Export Simulation Artifacts",
        ["simulated_thread_metadata.json | thread_history.json", "simulation_timeseries.csv | run summary metrics"],
        C_OUTPUT,
    )

    # Flow arrows
    arrow(ax, 0.50, 0.81, 0.50, 0.75)
    arrow(ax, 0.50, 0.62, 0.50, 0.58)
    arrow(ax, 0.50, 0.45, 0.50, 0.40)
    arrow(ax, 0.50, 0.27, 0.50, 0.295)

    # Decision yes-loop back to Stage 1
    ax.annotate(
        "",
        xy=(0.88, 0.52),
        xytext=(0.59, 0.245),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.2,
            color=INK,
            connectionstyle="angle,angleA=0,angleB=90,rad=0",
        ),
        zorder=4,
    )
    ax.text(0.80, 0.26, "yes", fontsize=12.5, color=MUTED, fontweight="bold")

    # Decision no to export
    arrow(ax, 0.50, 0.195, 0.50, 0.16, label="no")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "dissertation_figures" / "implementation"
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_personality_pipeline(out_dir / "personality_extraction_pipeline.png")
    draw_simulation_pipeline(out_dir / "simulation_pipeline.png")

    print(f"Wrote: {out_dir / 'personality_extraction_pipeline.png'}")
    print(f"Wrote: {out_dir / 'simulation_pipeline.png'}")


if __name__ == "__main__":
    main()
