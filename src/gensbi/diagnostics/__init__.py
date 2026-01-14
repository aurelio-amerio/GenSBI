"""This module adapts part of the sbi.diagnostics package for use in GenSBI.

See individual files for license and modification notices.
"""

from .distribution_wrapper import PosteriorWrapper

from .sbc import check_sbc, run_sbc, sbc_rank_plot
from .tarp import check_tarp, run_tarp, plot_tarp
from .lc2st import LC2ST, plot_lc2st


__all__ = [
    "PosteriorWrapper",
    "check_sbc",
    "run_sbc",
    "sbc_rank_plot",
    "check_tarp",
    "run_tarp",
    "plot_tarp",
    "LC2ST",
    "plot_lc2st",
]
