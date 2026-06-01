from pathlib import Path

import matplotlib.pyplot as plt

from pc_pricing_diagnostic.config import OUTPUT_FIGURES


NAVY = "#172033"
BLUE = "#2f5f98"
BAR_BLUE = "#2f80b7"
SOFT_BLUE = "#eef5fb"
GREEN = "#5f8f72"
ORANGE = "#c47f2c"
RED = "#a94442"
MUTED = "#5f6b7a"
GRID = "#d9e2ec"

WARNING_COLORS = {
    "Escalate": RED,
    "Investigate": ORANGE,
    "Monitor": BLUE,
    "No immediate action": GREEN,
}


def apply_axis_style(ax) -> None:
    """
    Apply the shared chart style used by executive and monitoring outputs.
    """
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.45, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    ax.tick_params(axis="both", colors=NAVY, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def save_figure(
    fig,
    file_name: str,
    output_dir: Path = OUTPUT_FIGURES,
    dpi: int = 220,
) -> None:
    """
    Save a chart with consistent export settings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / file_name,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)