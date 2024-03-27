# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for specifying parameters that control the optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast


@dataclass
class OptimizationSettings:
    """Specify the parameters that control the optimization.

    ``max_gamma`` specifies a constraint on the maximum value of gamma that a
    solution is allowed to have to be considered feasible. If a solution exists
    but the associated gamma exceeds ``max_gamma``, :func:`.greedy_best_first_search`,
    which is used to warm start the search engine will still return a valid albeit
    typically suboptimal solution.

    ``engine_selections`` is a dictionary that defines the selection
    of search engines for the optimization.

    ``max_backjumps`` specifies any constraints on the maximum number of backjump
    operations that can be performed by the search algorithm.

    ``seed`` is a seed used to provide a repeatable initialization
    of the pesudorandom number generators used by the optimization.
    If None is used as the random seed, then a seed is obtained using an
    operating-system call to achieve an unrepeatable randomized initialization.

    NOTE: The current release only supports LO gate and wire cuts. LOCC
    flags have been incorporated with an eye towards future releases.
    """

    max_gamma: float = 1024
    max_backjumps: None | int = 10000
    seed: int | None = None
    LO: bool = True
    LOCC_ancillas: bool = False
    LOCC_no_ancillas: bool = False
    engine_selections: dict[str, str] | None = None

    def __post_init__(self):
        """Post-init method for the data class."""
        if self.max_gamma < 1:
            raise ValueError("max_gamma must be a positive definite integer.")
        if self.max_backjumps is not None and self.max_backjumps < 0:
            raise ValueError("max_backjumps must be a positive semi-definite integer.")

        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas

        self.wire_cut_LO = self.LO
        self.wire_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.wire_cut_LOCC_no_ancillas = self.LOCC_no_ancillas
        if self.engine_selections is None:
            self.engine_selections = {"CutOptimization": "BestFirst"}

    @property
    def get_max_gamma(self) -> float:
        """Return the constraint on the maxiumum allowed value of gamma."""
        return self.max_gamma

    @property
    def get_max_backjumps(self) -> None | int:
        """Return the maximum number of allowed search backjumps.

        `None` denotes that there is no such restriction in place.
        """
        return self.max_backjumps

    @property
    def get_seed(self) -> int | None:
        """Return the seed used to generate the pseudorandom numbers used in the optimizaton."""
        return self.seed

    def get_engine_selection(self, stage_of_optimization: str) -> str:
        """Return the name of the search engine to employ."""
        self.engine_selections = cast(dict, self.engine_selections)
        return self.engine_selections[stage_of_optimization]

    def set_engine_selection(
        self, stage_of_optimization: str, engine_name: str
    ) -> None:
        """Set the name of the search engine to employ."""
        self.engine_selections = cast(dict, self.engine_selections)
        self.engine_selections[stage_of_optimization] = engine_name

    def set_gate_cut_types(self) -> None:
        """Select which gate-cut types to include in the optimization.

        The default is to only include LO gate cuts, which are the
        only cut types supported in this release.
        """
        self.gate_cut_LO = self.LO
        self.gate_cut_LOCC_with_ancillas = self.LOCC_ancillas

    def set_wire_cut_types(self) -> None:
        """Select which wire-cut types to include in the optimization.

        The default is to only include LO wire cuts, which are the
        only cut types supported in this release.
        """
        self.wire_cut_LO = self.LO
        self.wire_cut_LOCC_with_ancillas = self.LOCC_ancillas
        self.wire_cut_LOCC_no_ancillas = self.LOCC_no_ancillas

    def get_cut_search_groups(self) -> list[None | str]:
        """Return a list of action groups to include in the optimization."""
        out: list
        out = [None]

        if self.gate_cut_LO or self.gate_cut_LOCC_with_ancillas:
            out.append("GateCut")

        if (
            self.wire_cut_LO
            or self.wire_cut_LOCC_with_ancillas
            or self.wire_cut_LOCC_no_ancillas
        ):
            out.append("WireCut")

        return out


@dataclass
class OptimizationParameters:
    """Specify parameters that control the optimization.

    The other attributes of :class:`OptimizationSettings` are taken
    to be private.
    """

    seed: int | None = OptimizationSettings().seed
    max_gamma: float = OptimizationSettings().max_gamma
    max_backjumps: None | int = OptimizationSettings().max_backjumps
