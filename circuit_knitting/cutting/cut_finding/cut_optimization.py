# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes required to search for optimal cut locations."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import cast
from numpy.typing import NDArray
from .search_space_generator import ActionNames
from .cco_utils import select_search_engine, greedy_best_first_search
from .cutting_actions import disjoint_subcircuit_actions
from .search_space_generator import (
    get_action_subset,
    SearchFunctions,
    SearchSpaceGenerator,
)
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .circuit_interface import SimpleGateList, GateSpec
from .optimization_settings import OptimizationSettings
from .quantum_device_constraints import DeviceConstraints


@dataclass
class CutOptimizationFuncArgs:
    """Collect arguments for passing to the search-space generating methods in :class:`CutOptimization`."""

    entangling_gates: list[GateSpec] | None = None
    search_actions: ActionNames | None = None
    max_gamma: float | int | None = None
    qpu_width: int | None = None


def cut_optimization_cost_func(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> tuple[float, int]:
    """Return the cost function.

    The particular cost function chosen here aims to minimize the (square root)
    of the classical overhead, :math:`gamma`, while also (secondarily) giving preference
    to circuit partitionings that balance the sizes of the resulting partitions, by
    minimizing the maximum width across subcircuits.
    """
    # pylint: disable=unused-argument
    return (state.lower_bound_gamma(), state.get_max_width())


def cut_optimization_upper_bound_cost_func(
    goal_state, func_args: CutOptimizationFuncArgs
) -> tuple[float, float]:
    """Return the value of :math:`gamma` computed assuming all LO cuts."""
    # pylint: disable=unused-argument
    return (goal_state.upper_bound_gamma(), np.inf)


def cut_optimization_min_cost_bound_func(
    func_args: CutOptimizationFuncArgs,
) -> tuple[float, float] | None:
    """Return the a priori min-cost bound defined in the optimization settings."""
    if func_args.max_gamma is None:  # pragma: no cover
        return None

    return (func_args.max_gamma, np.inf)


def cut_optimization_next_state_func(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> list[DisjointSubcircuitsState]:
    """Generate a list of next states from the input state."""
    assert func_args.entangling_gates is not None
    assert func_args.search_actions is not None

    # Get the entangling gate spec that is to be processed next based
    # on the search level of the input state.

    gate_spec = func_args.entangling_gates[state.get_search_level()]

    # Determine which cutting actions can be performed, taking into
    # account any user-specified constraints that might have been
    # placed on how the current entangling gate is to be handled.

    gate = gate_spec.gate
    if len(gate.qubits) == 2:
        action_list = func_args.search_actions.get_group("TwoQubitGates")
    else:
        raise ValueError(
            "The input circuit must contain only single and two-qubits gates. Found "
            f"{len(gate.qubits)}-qubit gate: ({gate.name})."
        )

    gate_actions = gate_spec.cut_constraints
    action_list = get_action_subset(action_list, gate_actions)

    # Apply the search actions to generate a list of next states.
    next_state_list = []
    assert action_list is not None
    for action in action_list:
        func_args.qpu_width = cast(int, func_args.qpu_width)
        next_state_list.extend(action.next_state(state, gate_spec, func_args.qpu_width))
    return next_state_list


def cut_optimization_goal_state_func(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> bool:
    """Return True if the input state is a goal state."""
    func_args.entangling_gates = cast(list, func_args.entangling_gates)
    return state.get_search_level() >= len(func_args.entangling_gates)


# Global variable that holds the search-space functions for generating
# the cut optimization search space.
cut_optimization_search_funcs = SearchFunctions(
    cost_func=cut_optimization_cost_func,
    upperbound_cost_func=cut_optimization_upper_bound_cost_func,
    next_state_func=cut_optimization_next_state_func,
    goal_state_func=cut_optimization_goal_state_func,
    mincost_bound_func=cut_optimization_min_cost_bound_func,
)


def greedy_cut_optimization(
    circuit_interface: SimpleGateList,
    optimization_settings: OptimizationSettings,
    device_constraints: DeviceConstraints,
    search_space_funcs: SearchFunctions = cut_optimization_search_funcs,
    search_actions: ActionNames = disjoint_subcircuit_actions,
) -> DisjointSubcircuitsState | None:
    """Peform a first pass at cut optimization using greedy best first search.

    This step is effectively used to warm start our algorithm. It ignores the user
    specified constraint ``max_gamma``. Its primary purpose is to estimate an upper
    bound on the actual minimum gamma. Its secondary purpose is to provide a guaranteed
    "anytime" solution (`<https://en.wikipedia.org/wiki/Anytime_algorithm>`).
    """
    func_args = CutOptimizationFuncArgs()
    func_args.entangling_gates = circuit_interface.get_multiqubit_gates()
    func_args.search_actions = search_actions
    func_args.max_gamma = optimization_settings.get_max_gamma
    func_args.qpu_width = device_constraints.get_qpu_width()

    start_state = DisjointSubcircuitsState(
        circuit_interface.get_num_qubits(), max_wire_cuts_circuit(circuit_interface)
    )
    return greedy_best_first_search(start_state, search_space_funcs, func_args)


class CutOptimization:
    """Implement cut optimization whereby qubits are not reused.

    Because of the condition of no qubit reuse, it is assumed that
    there is no circuit folding (i.e., when mid-circuit measurement and active
    reset are not available). Cuts are placed with the goal of finding
    separable subcircuits.

    Member Variables:
    ``circuit`` (:class:`CircuitInterface`) is the interface for the circuit
    to be cut.

    ``settings`` (:class:`OptimizationSettings`) contains the settings that
    control the optimization process.

    ``constraints`` (:class:`DeviceConstraints`) contains the device constraints
    that solutions must obey.

    ``search_funcs`` (:class:`SearchFunctions`) holds the functions needed to generate
    and explore the cut optimization search space.

    ``func_args`` (:class:`CutOptimizationFuncArgs`) contains the necessary device constraints
    and optimization settings parameters that are needed by the cut optimization
    search-space function.

    ``search_actions`` (:class:`ActionNames`) contains the allowed actions that are used to
    generate the search space.

    ``search_engine`` (:class`BestFirstSearch`) implements the search algorithm.
    """

    def __init__(
        self,
        circuit_interface,
        optimization_settings,
        device_constraints,
        search_engine_config={
            "CutOptimization": SearchSpaceGenerator(
                functions=cut_optimization_search_funcs,
                actions=disjoint_subcircuit_actions,
            )
        },
    ):
        """Assign member variables."""
        generator = search_engine_config["CutOptimization"]
        search_space_funcs = generator.functions
        search_space_actions = generator.actions

        # Extract the subset of allowed actions as defined in the settings object
        cut_groups = optimization_settings.get_cut_search_groups()
        cut_actions = search_space_actions.copy(cut_groups)

        self.circuit = circuit_interface
        self.settings = optimization_settings
        self.constraints = device_constraints
        self.search_funcs = search_space_funcs
        self.search_actions = cut_actions

        self.func_args = CutOptimizationFuncArgs()
        self.func_args.entangling_gates = self.circuit.get_multiqubit_gates()
        self.func_args.search_actions = self.search_actions
        self.func_args.max_gamma = self.settings.get_max_gamma
        self.func_args.qpu_width = self.constraints.get_qpu_width()

        # Perform an initial greedy best-first search to determine an upper
        # bound for the optimal gamma
        self.greedy_goal_state = greedy_cut_optimization(
            self.circuit,
            self.settings,
            self.constraints,
            search_space_funcs=self.search_funcs,
            search_actions=self.search_actions,
        )

        # Use the upper bound for the optimal gamma to determine the maximum
        # number of wire cuts that can be performed.
        max_wire_cuts = max_wire_cuts_circuit(self.circuit)

        if self.greedy_goal_state is not None:
            mwc = max_wire_cuts_gamma(self.greedy_goal_state.upper_bound_gamma())
            max_wire_cuts = min(max_wire_cuts, mwc)

        # The elif block below covers a rare edge case
        # which would need a clever circuit to get tested.
        # Excluded from test coverage for now.
        elif self.func_args.max_gamma is not None:  # pragma: no cover
            mwc = max_wire_cuts_gamma(self.func_args.max_gamma)
            max_wire_cuts = min(max_wire_cuts, mwc)

        # Push the start state onto the search_engine
        start_state = DisjointSubcircuitsState(
            self.circuit.get_num_qubits(), max_wire_cuts
        )

        sq = select_search_engine(
            "CutOptimization",
            self.settings,
            self.search_funcs,
            stop_at_first_min=False,
        )
        sq.initialize([start_state], self.func_args)

        # Use the upper bound from the initial greedy search to constrain the
        # subsequent search.
        if self.greedy_goal_state is not None:
            sq.update_upperbound_goal_state(self.greedy_goal_state, self.func_args)

        self.search_engine = sq
        self.goal_state_returned = False

    def optimization_pass(self) -> tuple[DisjointSubcircuitsState, float]:
        """Produce, at each call, a goal state representing a distinct set of cutting decisions.

        None is returned once no additional choices of cuts can be made
        without exceeding the minimum upper bound across all cutting
        decisions previously returned.
        """
        state, cost = self.search_engine.optimization_pass(self.func_args)
        if state is None and not self.goal_state_returned:
            state = self.greedy_goal_state
            cost = self.search_funcs.cost_func(state, self.func_args)

        self.goal_state_returned = True

        return state, cost

    def minimum_reached(self) -> bool:
        """Return True if the optimization reached a global minimum."""
        return self.search_engine.minimum_reached()

    def get_stats(self, penultimate: bool = False) -> NDArray[np.int_]:
        """Return the search-engine statistics.

        This is a Numpy array containing the number of states visited
        (dequeued), the number of next-states generated, the number of
        next-states that are enqueued after cost pruning, and the number
        of backjumps performed. Return None if no search is performed.
        If the bool penultimate is set to True, return the stats that
        correspond to the penultimate step in the search.
        """
        return self.search_engine.get_stats(penultimate=penultimate)

    def get_upperbound_cost(self) -> tuple[float, float]:
        """Return the current upperbound cost."""
        return self.search_engine.get_upperbound_cost()

    def update_upperbound_cost(self, cost_bound: tuple[float, float]) -> None:
        """Update the cost upper bound based on an input cost bound."""
        self.search_engine.update_upperbound_cost(cost_bound)


def max_wire_cuts_circuit(circuit_interface: SimpleGateList) -> int:
    """Calculate an upper bound on the maximum possible number of wire cuts.

    This is constrained by the total number of inputs to multiqubit gates in
    the circuit.

    NOTE: There is no advantage gained by cutting wires that
    only have single qubit gates acting on them, so without
    loss of generality we can assume that wire cutting is
    performed only on the inputs to multiqubit gates.
    """
    multiqubit_wires = [
        len(x.gate.qubits) for x in circuit_interface.get_multiqubit_gates()
    ]
    return sum(multiqubit_wires)


def max_wire_cuts_gamma(max_gamma: float | int) -> int:
    """Calculate an upper bound on the maximum number of wire cuts that can be made, given the maximum allowed gamma."""
    return int(np.ceil(np.log2(max_gamma + 1) - 1))
