# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes needed to implement the actions involved in circuit cutting."""


from __future__ import annotations

from abc import ABC, abstractmethod
from .circuit_interface import SimpleGateList
from .search_space_generator import ActionNames
from typing import Hashable, cast
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .circuit_interface import GateSpec

# Global variable that holds action names for constructing disjoint subcircuits
disjoint_subcircuit_actions = ActionNames()


class DisjointSearchAction(ABC):
    """Base class for search actions for constructing disjoint subcircuits."""

    @abstractmethod
    def get_name(self):
        """Return the look-up name of the associated instance of :class:`DisjointSearchAction`."""

    @abstractmethod
    def get_group_names(self):
        """Return the group name of the associated instance of :class:`DisjointSearchAction`."""

    @abstractmethod
    def next_state_primitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying the associated instance of :class:`DisjointSearchAction`."""

    def next_state(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return list of states resulting from applying associated instance of :class:`DisjointSearchAction` to ``gate_spec``.

        This is subject to the constraint that the number of resulting qubits (wires)
        in each subcircuit cannot exceed ``max_width``.
        """
        next_list = self.next_state_primitive(state, gate_spec, max_width)

        for next_state in next_list:
            next_state.set_next_level(state)

        return next_list


class ActionApplyGate(DisjointSearchAction):
    """Implement the action of applying a two-qubit gate without decomposition."""

    def get_name(self) -> None:
        """Return the look-up name of :class:`ActionApplyGate`."""
        return None

    def get_group_names(self) -> list[None | str]:
        """Return the group name of :class:`ActionApplyGate`."""
        return [None, "TwoQubitGates"]

    def next_state_primitive(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return the new state that results from applying the gate given by ``gate_spec``."""
        gate = gate_spec.gate

        # extract the root wire for the first qubit
        # acted on by the given 2-qubit gate.
        r1 = state.find_qubit_root(gate.qubits[0])

        # extract the root wire for the second qubit
        # acted on by the given 2-qubit gate.
        r2 = state.find_qubit_root(gate.qubits[1])

        # If applying the gate would cause the number of qubits to exceed
        # the qubit limit, then do not apply the gate
        assert state.width is not None
        if r1 != r2 and state.width[r1] + state.width[r2] > max_width:
            return list()

        # If the gate cannot be applied because it would violate the
        # merge constraints, then do not apply the gate
        if state.check_donot_merge_roots(r1, r2):
            return list()

        new_state = state.copy()

        if r1 != r2:
            new_state.merge_roots(r1, r2)

        new_state.add_action(self, gate_spec)
        return [new_state]


# Add ActionApplyGate to the global variable ``disjoint_subcircuit_actions``
disjoint_subcircuit_actions.define_action(ActionApplyGate())


class ActionCutTwoQubitGate(DisjointSearchAction):
    """Cut a two-qubit gate."""

    def get_name(self) -> str:
        """Return the look-up name of :class:`ActionCutTwoQubitGate`."""
        return "CutTwoQubitGate"

    def get_group_names(self) -> list[str]:
        """Return the group name of :class:`ActionCutTwoQubitGate`."""
        return ["GateCut", "TwoQubitGates"]

    def next_state_primitive(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return the state that results from cutting the gate given by ``gate_spec``."""
        gate = gate_spec.gate

        # Cutting of multi-qubit gates is not supported in this release.
        if len(gate.qubits) != 2:  # pragma: no cover
            raise ValueError(
                "In the current version, only the cutting of two qubit gates is supported."
            )

        gamma_LB, num_bell_pairs, gamma_UB = self.get_cost_params(gate_spec)

        if gamma_LB is None:  # pragma: no cover
            return list()

        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.get_wire(q1)
        w2 = state.get_wire(q2)
        r1 = state.find_qubit_root(q1)
        r2 = state.find_qubit_root(q2)

        if r1 == r2:
            return list()

        new_state = state.copy()

        new_state.assert_donot_merge_roots(r1, r2)

        new_state.gamma_LB = cast(float, new_state.gamma_LB)
        new_state.gamma_LB *= gamma_LB

        for k in range(num_bell_pairs):  # pragma: no cover
            new_state.bell_pairs = cast(list, new_state.bell_pairs)
            new_state.bell_pairs.append((r1, r2))

        gamma_UB = cast(float, gamma_UB)
        new_state.gamma_UB = cast(float, new_state.gamma_UB)
        new_state.gamma_UB *= gamma_UB

        new_state.add_action(self, gate_spec, ((1, w1), (2, w2)))

        return [new_state]

    @staticmethod
    def get_cost_params(
        gate_spec: GateSpec,
    ) -> tuple[float | None, int, float | None]:
        """
        Get the cost parameters for gate cuts.

        This method returns a tuple of the form:
            (<gamma_lower_bound>, <num_bell_pairs>, <gamma_upper_bound>)

        Since CKT does not support LOCC at the moment, these tuples will be of
        the form (gamma, 0, gamma).
        """
        gate = gate_spec.gate
        gamma = gate.gamma
        return (gamma, 0, gamma)

    def export_cuts(
        self,
        circuit_interface: SimpleGateList,
        wire_map: list[Hashable],
        gate_spec: GateSpec,
        args,
    ) -> None:
        """Insert an LO gate cut into the input circuit for the specified gate and cut arguments."""
        # pylint: disable=unused-argument
        circuit_interface.insert_gate_cut(gate_spec.instruction_id, "LO")


# Add ActionCutTwoQubitGate to the global variable disjoint_subcircuit_actions
disjoint_subcircuit_actions.define_action(ActionCutTwoQubitGate())


class ActionCutLeftWire(DisjointSearchAction):
    """Cut the left (first input) wire of a two-qubit gate."""

    def get_name(self) -> str:
        """Return the look-up name of :class:`ActionCutLeftWire`."""
        return "CutLeftWire"

    def get_group_names(self) -> list[str]:
        """Return the group name of :class:`ActionCutLeftWire`."""
        return ["WireCut", "TwoQubitGates"]

    def next_state_primitive(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return the state that results from cutting the left (first input) wire of the gate given by ``gate_spec``."""
        gate = gate_spec.gate

        # Cutting of multi-qubit gates is not supported in this release.
        if len(gate.qubits) != 2:  # pragma: no cover
            raise ValueError(
                "In the current version, only the cutting of two qubit gates is supported."
            )

        # If the wire-cut limit would be exceeded, return the empty list.
        if not state.can_add_wires(1):
            return list()

        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.get_wire(q1)
        r1 = state.find_qubit_root(q1)
        r2 = state.find_qubit_root(q2)

        if r1 == r2:
            return list()

        if not state.can_expand_subcircuit(r2, 1, max_width):
            return list()

        new_state = state.copy()

        rnew = new_state.new_wire(q1)
        new_state.merge_roots(rnew, r2)
        new_state.assert_donot_merge_roots(r1, r2)  # Because r2 < rnew

        new_state.bell_pairs = cast(list, new_state.bell_pairs)
        new_state.bell_pairs.append((r1, r2))
        new_state.gamma_UB = cast(float, new_state.gamma_UB)
        new_state.gamma_UB *= 4

        new_state.add_action(self, gate_spec, (1, w1, rnew))

        return [new_state]

    def export_cuts(
        self,
        circuit_interface: SimpleGateList,
        wire_map: list[Hashable],
        gate_spec: GateSpec,
        cut_args,
    ) -> None:
        """Insert an LO wire cut into the input circuit for the specified gate and cut arguments."""
        insert_all_lo_wire_cuts(circuit_interface, wire_map, gate_spec, cut_args)


# Add ActionCutLeftWire to the global variable disjoint_subcircuit_actions
disjoint_subcircuit_actions.define_action(ActionCutLeftWire())


def insert_all_lo_wire_cuts(
    circuit_interface: SimpleGateList,
    wire_map: list[Hashable],
    gate_spec: GateSpec,
    cut_args,
) -> None:
    """Insert LO wire cuts into the input circuit for the specified gate and all cut arguments."""
    gate_ID = gate_spec.instruction_id
    for input_ID, wire_ID, new_wire_ID in cut_args:
        circuit_interface.insert_wire_cut(
            gate_ID, input_ID, wire_map[wire_ID], wire_map[new_wire_ID], "LO"
        )


class ActionCutRightWire(DisjointSearchAction):
    """Cut the right (second input) wire of a two-qubit gate."""

    def get_name(self) -> str:
        """Return the look-up name of :class:`ActionCutRightWire`."""
        return "CutRightWire"

    def get_group_names(self) -> list[str]:
        """Return the group name of :class:`ActionCutRightWire`."""
        return ["WireCut", "TwoQubitGates"]

    def next_state_primitive(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return the state that results from cutting the right (second input) wire of the gate given by ``gate_spec``."""
        gate = gate_spec.gate

        # Cutting of multi-qubit gates is not supported in this release.
        if len(gate.qubits) != 2:  # pragma: no cover
            raise ValueError(
                "In this release, only the cutting of two qubit gates is supported."
            )

        # If the wire-cut limit would be exceeded, return the empty list
        if not state.can_add_wires(1):
            return list()

        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w2 = state.get_wire(q2)
        r1 = state.find_qubit_root(q1)
        r2 = state.find_qubit_root(q2)

        if r1 == r2:
            return list()

        if not state.can_expand_subcircuit(r1, 1, max_width):
            return list()

        new_state = state.copy()

        rnew = new_state.new_wire(q2)
        new_state.merge_roots(r1, rnew)
        new_state.assert_donot_merge_roots(r1, r2)  # Because r1 < rnew

        new_state.gamma_UB = cast(float, new_state.gamma_UB)
        new_state.bell_pairs = cast(list, new_state.bell_pairs)
        new_state.bell_pairs.append((r1, r2))
        new_state.gamma_UB *= 4

        new_state.add_action(self, gate_spec, (2, w2, rnew))

        return [new_state]

    def export_cuts(
        self,
        circuit_interface: SimpleGateList,
        wire_map: list[Hashable],
        gate_spec: GateSpec,
        cut_args,
    ) -> None:  # pragma: no cover
        """Insert an LO wire cut into the input circuit for the specified gate and cut arguments."""
        insert_all_lo_wire_cuts(circuit_interface, wire_map, gate_spec, cut_args)


# Add ActionCutRightWire to the global variable disjoint_subcircuit_actions
disjoint_subcircuit_actions.define_action(ActionCutRightWire())


class ActionCutBothWires(DisjointSearchAction):
    """Cut both input wires of a two-qubit gate."""

    def get_name(self) -> str:
        """Return the look-up name of :class:`ActionCutBothWires`."""
        return "CutBothWires"

    def get_group_names(self) -> list[str]:
        """Return the group name of :class:`ActionCutBothWires`."""
        return ["WireCut", "TwoQubitGates"]

    def next_state_primitive(
        self,
        state: DisjointSubcircuitsState,
        gate_spec: GateSpec,
        max_width: int,
    ) -> list[DisjointSubcircuitsState]:
        """Return the new state that results from cutting both input wires of the gate given by ``gate_spec``."""
        gate = gate_spec.gate

        # Cutting of multi-qubit gates is not supported in this release.
        if len(gate.qubits) != 2:  # pragma: no cover
            raise ValueError(
                "In the current version, only the cutting of two qubit gates is supported."
            )

        # If the wire-cut limit would be exceeded, do not cut.
        if not state.can_add_wires(2):  # pragma: no cover
            return list()

        # If the maximum width is less than two, do not cut.
        if max_width < 2:
            return list()

        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.get_wire(q1)
        w2 = state.get_wire(q2)
        r1 = state.find_qubit_root(q1)
        r2 = state.find_qubit_root(q2)

        new_state = state.copy()

        rnew_1 = new_state.new_wire(q1)
        rnew_2 = new_state.new_wire(q2)
        new_state.merge_roots(rnew_1, rnew_2)
        new_state.assert_donot_merge_roots(r1, rnew_1)  # Because r1 < rnew_1
        new_state.assert_donot_merge_roots(r2, rnew_2)  # Because r2 < rnew_2

        new_state.bell_pairs = cast(list, new_state.bell_pairs)
        new_state.gamma_UB = cast(float, new_state.gamma_UB)
        new_state.bell_pairs.append((r1, rnew_1))
        new_state.bell_pairs.append((r2, rnew_2))
        new_state.gamma_UB *= 16

        new_state.add_action(self, gate_spec, ((1, w1, rnew_1), (2, w2, rnew_2)))

        return [new_state]

    def export_cuts(
        self,
        circuit_interface: SimpleGateList,
        wire_map: list[Hashable],
        gate_spec: GateSpec,
        cut_args,
    ) -> None:  # pragma: no cover
        """Insert LO wire cuts into the input circuit for the specified gate and cut arguments."""
        insert_all_lo_wire_cuts(circuit_interface, wire_map, gate_spec, cut_args)


# Add ActionCutBothWires to the global variable disjoint_subcircuit_actions
disjoint_subcircuit_actions.define_action(ActionCutBothWires())
