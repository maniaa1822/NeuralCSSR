"""Render the seven-state machine diagram as a PNG using Matplotlib."""

from __future__ import annotations

from itertools import count

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


# Node coordinates chosen to mirror the presentation layout
NODE_POSITIONS = {
    "BB": (0.0, 0.0),
    "BA": (3.0, 2.3),
    "BAA": (6.4, 3.8),
    "AAA": (9.2, 2.2),
    "BAAB": (6.2, 0.2),
    "AAAB": (9.4, 5.2),
    "BAB": (3.1, -1.9),
}


def add_node(ax, label: str, xy: tuple[float, float]) -> None:
    circle = Circle(xy, radius=0.65, edgecolor="black", facecolor="white", linewidth=2)
    ax.add_patch(circle)
    ax.text(*xy, label, ha="center", va="center", fontsize=12, fontweight="bold")


def add_edge(
    ax,
    start: str,
    end: str,
    label: str,
    text_offset: tuple[float, float] = (0.0, 0.0),
    curvature: float = 0.0,
    dist: float = 0.0,
    shrink: float = 0.7,
) -> None:
    sx, sy = NODE_POSITIONS[start]
    ex, ey = NODE_POSITIONS[end]
    arrow = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color="black",
        connectionstyle=f"arc3,rad={curvature}",
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(arrow)
    label_x = (sx + ex) / 2 + text_offset[0]
    label_y = (sy + ey) / 2 + text_offset[1]
    ax.text(label_x, label_y, label, fontsize=9, fontweight="bold", ha="center", va="center")


def add_loop(
    ax,
    node: str,
    label: str,
    start_offset: tuple[float, float],
    end_offset: tuple[float, float],
    curvature: float,
    label_offset: tuple[float, float],
) -> None:
    x, y = NODE_POSITIONS[node]
    sx, sy = x + start_offset[0], y + start_offset[1]
    ex, ey = x + end_offset[0], y + end_offset[1]
    loop = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color="black",
        connectionstyle=f"arc3,rad={curvature}",
    )
    ax.add_patch(loop)
    ax.text(x + label_offset[0], y + label_offset[1], label, fontsize=9, fontweight="bold", ha="center")


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    for name, xy in NODE_POSITIONS.items():
        add_node(ax, name, xy)

    add_edge(ax, "BB", "BA", "A | 0.9375", (0.0, 0.35))
    add_edge(ax, "BA", "BAA", "A | 0.5625", (0.05, 0.35), curvature=0.2)
    add_edge(ax, "BAA", "AAA", "A | 0.5625", (0.4, 0.4), curvature=0.2)
    add_edge(ax, "BAB", "BA", "A | 0.2500", (0.0, -0.35), curvature=-0.35)
    add_edge(ax, "BAAB", "BA", "A | 0.7500", (0.1, -0.4), curvature=-0.3)
    add_edge(ax, "AAAB", "BA", "A | 0.4375", (-0.55, 0.0), curvature=0.35)

    add_edge(ax, "BA", "BAB", "B | 0.4375", (0.35, -0.1), curvature=-0.45)
    add_edge(ax, "BAA", "BAAB", "B | 0.4375", (0.25, -0.35), curvature=-0.35)
    add_edge(ax, "AAA", "AAAB", "B | 0.8125", (0.35, 0.0), curvature=-0.2)
    add_edge(ax, "BAB", "BB", "B | 0.7500", (-0.5, -0.45), curvature=-0.2)
    add_edge(ax, "BAAB", "BB", "B | 0.2500", (-0.9, -0.4), curvature=-0.3)
    add_edge(ax, "AAAB", "BB", "B | 0.5625", (-1.1, 0.35), curvature=0.32)

    add_loop(ax, "AAA", "A | 0.1875", start_offset=(-0.2, 0.9), end_offset=(0.2, 0.9), curvature=1.2, label_offset=(0.0, 1.5))
    add_loop(ax, "BB", "B | 0.0625", start_offset=(-0.2, -0.9), end_offset=(0.2, -0.9), curvature=-1.2, label_offset=(0.0, -1.45))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2, 12)
    ax.set_ylim(-4.5, 7)

    fig.tight_layout()
    fig.savefig("7statemachine.png", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
