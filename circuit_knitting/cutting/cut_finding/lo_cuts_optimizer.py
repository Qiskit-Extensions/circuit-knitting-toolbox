# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the wrapper class for optimizing LO gate and wire cuts."""
from __future__ import annotations

from .cut_optimization import CutOptimization
from .cut_optimization import disjoint_subcircuit_actions
from .cut_optimization import cut_optimization_next_state_func
from .cut_optimization import cut_optimization_goal_state_func
from .cut_optimization import cut_optimization_min_cost_bound_func
from .cut_optimization import cut_optimization_upper_bound_cost_func
from .search_space_generator import SearchFunctions, SearchSpaceGenerator

import numpy as np
from numpy.typing import NDArray
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .quantum_device_constraints import DeviceConstraints
from .optimization_settings import OptimizationSettings
from .circuit_interface import SimpleGateList


# Functions for generating the cut optimization search space
cut_optimization_search_funcs = SearchFunctions(
    cost_func=cut_optimization_upper_bound_cost_func,  # Valid choice only for LO cuts.
    upperbound_cost_func=cut_optimization_upper_bound_cost_func,
    next_state_func=cut_optimization_next_state_func,
    goal_state_func=cut_optimization_goal_state_func,
    mincost_bound_func=cut_optimization_min_cost_bound_func,
)


class LOCutsOptimizer:
    """Optimize circuit cuts for the case in which only LO decompositions are employed.

    The ``search_engine_config`` dictionary that configures the optimization
    algorithms must be specified in the constructor. For flexibility, the
    circuit_interface, optimization_settings, and device_constraints can
    be specified either in the constructor or in :meth:`LOCutsOptimizer.optimize`.
    In the latter case, the values provided overwrite the previous values.

    ``circuit_interface``, an instance of :class:`CircuitInterface`, defines the circuit to be cut.
    The circuit_interface object that is passed to the :meth:`LOCutsOptimizer.optimize`
    is updated to reflect the optimized circuit cuts that were
    identified.

    :meth:`LOCutsOptimizer.optimize` returns ``best_result``, an instance of :class:`DisjointSubcircuitsState`,
    which is the lowest-cost :class:`DisjointSubcircuitsState` instance identified in the search.
    """

    def __init__(
        self,
        circuit_interface=None,
        optimization_settings=None,
        device_constraints=None,
        search_engine_config={
            "CutOptimization": SearchSpaceGenerator(
                functions=cut_optimization_search_funcs,
                actions=disjoint_subcircuit_actions,
            )
        },
    ):
        """Initialize :class:`LOCutsOptimizer with the specified configuration variables."""
        self.circuit_interface = circuit_interface
        self.optimization_settings = optimization_settings
        self.device_constraints = device_constraints
        self.search_engine_config = search_engine_config
        self.cut_optimization = None
        self.best_result = None

    def optimize(
        self,
        circuit_interface: SimpleGateList | None = None,
        optimization_settings: OptimizationSettings | None = None,
        device_constraints: DeviceConstraints | None = None,
    ) -> DisjointSubcircuitsState | None:
        """Optimize the cutting of a circuit by calling :meth:`CutOptimization.optimization_pass`.

        Args:
        ``circuit_interface``: defines the circuit to be
        cut. This object is then updated with the optimized cuts that
        were identified.

        ``optimization_settings``: defines the settings
        to be used for the optimization.

        ``device_constraints``:  the capabilties of
        the target quantum hardware.

        Returns:
        The lowest-cost instance of :class:`DisjointSubcircuitsState`
        identified in the search, or None if no solution could be found.
        In case of the former, the circuit_interface object is also
        updated as a side effect to incorporate the cuts found.
        """
        if circuit_interface is not None:
            self.circuit_interface = circuit_interface

        if optimization_settings is not None:
            self.optimization_settings = optimization_settings

        if device_constraints is not None:
            self.device_constraints = device_constraints

        assert self.circuit_interface is not None, "circuit_interface cannot be None"

        assert (
            self.optimization_settings is not None
        ), "optimization_settings cannot be None"

        assert self.device_constraints is not None, "device_constraints cannot be None"

        self.cut_optimization = CutOptimization(
            self.circuit_interface,
            self.optimization_settings,
            self.device_constraints,
            search_engine_config=self.search_engine_config,
        )

        out_1 = list()

        while True:
            state, cost = self.cut_optimization.optimization_pass()
            if state is None:
                break
            out_1.append((cost, state))

        min_cost = min(out_1, key=lambda x: x[0], default=None)

        if min_cost is not None:
            self.best_result = min_cost[-1]
            self.best_result.export_cuts(self.circuit_interface)
        else:  # pragma: no cover
            self.best_result = None

        return self.best_result

    def get_results(self) -> DisjointSubcircuitsState | None:
        """Return the optimization results."""
        return self.best_result

    def get_stats(self, penultimate=False) -> dict[str, NDArray[np.int_]]:
        """Return a dictionary containing optimization results.

        The value is a Numpy array containing the number of states visited
        (dequeued), the number of next-states generated, the number of
        next-states that are enqueued after cost pruning, and the number
        of backjumps performed. Return None if no search is performed.
        If the bool penultimate is set to True, return the stats that
        correspond to the penultimate step in the search.
        """
        return {
            "CutOptimization": self.cut_optimization.get_stats(penultimate=penultimate)
        }

    def minimum_reached(self) -> bool:
        """Return a Boolean flag indicating whether the global minimum was reached."""
        return self.cut_optimization.minimum_reached()


def print_state_list(
    state_list: list[DisjointSubcircuitsState],
) -> None:  # pragma: no cover
    """Call :meth:`print` defined for a :class:`DisjointSubcircuitsState` instance."""
    for x in state_list:
        print()
        x.print(simple=True)
