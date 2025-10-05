from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

@dataclass
class PlotFormatter:
    linewidth: float = 1.2
    alpha: float = 0.95
    grid_alpha: float = 0.2
    legend: bool = True
    title_prefix: str = ""
    fontsize: int = 10

    def __post_init__(self) -> None:
        pass

    def apply_axes_basics(self, ax: Axes, *, title: str, xlabel: str, ylabel: str, percent_y: bool = False) -> None:
        ax.set_title(f"{self.title_prefix}{title}" if self.title_prefix else title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=self.grid_alpha)
        if percent_y:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if self.legend:
            ax.legend(loc="best", frameon=False)
