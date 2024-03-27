# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes needed for representing search-space states when cutting circuits."""

from __future__ import annotations

import copy
import numpy as np
from numpy.typing import NDArray
from collections import Counter
from .circuit_interface import SimpleGateList, GateSpec
from typing import (
    Hashable,
    Iterable,
    TYPE_CHECKING,
    no_type_check,
    cast,
    NamedTuple,
    Sequence,
)

if TYPE_CHECKING:  # pragma: no cover
    from .cutting_actions import DisjointSearchAction


class Action(NamedTuple):
    """Named tuple for specification of search (cutting) action."""

    action: DisjointSearchAction
    gate_spec: GateSpec
    args: list


class GateCutLocation(NamedTuple):
    """Named tuple for specification of gate cut location."""

    instruction_id: int
    gate_name: str
    qubits: Sequence


class WireCutLocation(NamedTuple):
    """Named tuple for specification of wire cut location.

    Wire cuts are identified through the gates whose input wires are cut.
    """

    instruction_id: int
    gate_name: str
    qubits: Sequence
    input: int


class CutIdentifier(NamedTuple):
    """Named tuple for specification of location of :class:`CutTwoQubitGate` or :class:`CutBothWires` instances."""

    cut_action: DisjointSearchAction
    gate_cut_location: GateCutLocation


class OneWireCutIdentifier(NamedTuple):
    """Named tuple for specification of location of :class:`CutLeftWire` or :class:`CutRightWire` instances."""

    cut_action: DisjointSearchAction
    wire_cut_location: WireCutLocation


class DisjointSubcircuitsState:
    """Represent search-space states when cutting circuits to construct disjoint subcircuits.

    Each wire cut introduces a new wire. A mapping from qubit IDs
    in QASM-like statements to wire IDs is therefore created
    and maintained. Groups of wires form subcircuits. The mapping
    from wires to subcircuits is represented using an up-tree data
    structure over wires. The number of wires (width) in each
    subcircuit is also tracked to ensure subcircuits will fit on
    target quantum devices.

    Member Variables:
    ``wiremap``: an int Numpy array that provides the mapping from qubit IDs
    to wire IDs.

    ``num_wires``: an int which is the number of wires in the cut circuit.

    ``uptree``: an int Numpy array that contains the uptree data structure that
    defines groups of wires that form subcircuits. The uptree array
    map wire IDs to parent wire IDs in a subcircuit. If a wire points
    to itself, then that wire is the root wire in the corresponding
    subcircuit. Otherwise, you need to follow the parent links to find
    the root wire that corresponds to that subcircuit.

    ``width``: an int Numpy array that contains the number of wires in each
    subcircuit. The values of width are valid only for root wire IDs.

    ``bell_pairs``: a list of pairs of subcircuits (wires) that
    define the virtual Bell pairs that would need to be constructed in
    order to implement optimal LOCC wire and gate cuts using ancillas.

    ``gamma_LB``: a float that is the cumulative lower-bound gamma for LOCC
    circuit cuts that cannot be constructed using Bell pairs.

    ``gamma_UB``: a float that is the cumulative upper-bound gamma for all
    circuit cuts assuming all cuts are LO.

    ``no_merge``: a list that contains a list of subcircuit merging constaints.
    Each constraint can either be a pair of wire IDs or a list of pairs
    of wire IDs. In the case of a pair of wire IDs, the constraint is
    that the subcircuits that contain those wire IDs cannot be merged
    by subsequent search actions. In the case of a list of pairs of
    wire IDs, the constraint is that at least one pair of corresponding
    subcircuits cannot be merged.

    ``actions``: a list of instances of :class:`Action`.

    ``level``: an int which specifies the level in the search tree at which this search
    state resides, with 0 being the root of the search tree.
    """

    def __init__(self, num_qubits: int | None = None, max_wire_cuts: int | None = None):
        """Initialize an instance of :class:`DisjointSubcircuitsState` with the specified configuration variables."""
        if not (
            num_qubits is None or (isinstance(num_qubits, int) and num_qubits >= 0)
        ):
            raise ValueError("num_qubits must be either be None or a positive integer.")

        if not (
            max_wire_cuts is None
            or (isinstance(max_wire_cuts, int) and max_wire_cuts >= 0)
        ):
            raise ValueError(
                "max_wire_cuts must be either be None or a positive integer."
            )

        if num_qubits is None or max_wire_cuts is None:
            self.wiremap: NDArray[np.int_] | None = None
            self.num_wires: int | None = None

            self.uptree: NDArray[np.int_] | None = None
            self.width: NDArray[np.int_] | None = None

            self.bell_pairs: list[tuple[int, int]] | None = None
            self.gamma_LB: float | None = None
            self.gamma_UB: float | None = None

            self.no_merge: list[tuple] | None = None
            self.actions: list[Action] | None = None
            self.cut_actions_list: list | None = None
            self.level: int | None = None

        else:
            max_wires = num_qubits + max_wire_cuts

            self.wiremap = np.arange(num_qubits)
            self.num_wires = num_qubits

            self.uptree = np.arange(max_wires)
            self.width = np.ones(max_wires, dtype=int)

            self.bell_pairs = list()
            self.gamma_LB = 1.0
            self.gamma_UB = 1.0

            self.no_merge = list()
            self.actions = list()
            self.cut_actions_list = list()
            self.level = 0

    @no_type_check
    def __copy__(self) -> DisjointSubcircuitsState:
        """Make shallow copy."""
        new_state = DisjointSubcircuitsState()

        new_state.wiremap = self.wiremap.copy()
        new_state.num_wires = self.num_wires

        new_state.uptree = self.uptree.copy()
        new_state.width = self.width.copy()

        new_state.bell_pairs = self.bell_pairs.copy()
        new_state.gamma_LB = self.gamma_LB
        new_state.gamma_UB = self.gamma_UB

        new_state.no_merge = self.no_merge.copy()
        new_state.actions = self.actions.copy()
        new_state.cut_actions_list = self.cut_actions_list.copy()
        new_state.level = None

        return new_state

    def copy(self) -> DisjointSubcircuitsState:
        """Make shallow copy."""
        return copy.copy(self)

    def cut_actions_sublist(self) -> list[NamedTuple]:
        """Create a formatted list containing the actions carried out on an instance of :class:`DisjointSubcircuitState`.

        Also include the locations of these actions which are specified in terms of the associated gates and wires.
        """
        self.actions = cast(list, self.actions)
        cut_actions = get_actions_list(self.actions)

        # Output formatting for LO gate and wire cuts
        self.cut_actions_list = cast(list, self.cut_actions_list)
        for i in range(len(cut_actions)):
            if cut_actions[i].action.get_name() in ("CutLeftWire", "CutRightWire"):
                self.cut_actions_list.append(
                    OneWireCutIdentifier(
                        cut_actions[i].action.get_name(),
                        WireCutLocation(
                            cut_actions[i].gate_spec.instruction_id,
                            cut_actions[i].gate_spec.gate.name,
                            cut_actions[i].gate_spec.gate.qubits,
                            cut_actions[i].args[0][0],
                        ),
                    )
                )
            elif cut_actions[i].action.get_name() in (
                "CutTwoQubitGate",
                "CutBothWires",
            ):
                # For CutBothWires both inputs are cut and so the inputs need not be specified.
                self.cut_actions_list.append(
                    CutIdentifier(
                        cut_actions[i].action.get_name(),
                        GateCutLocation(
                            cut_actions[i].gate_spec.instruction_id,
                            cut_actions[i].gate_spec.gate.name,
                            cut_actions[i].gate_spec.gate.qubits,
                        ),
                    )
                )
            if not self.cut_actions_list:  # pragma: no cover
                self.cut_actions_list = cut_actions

        return self.cut_actions_list

    def print(self, simple: bool = False) -> None:  # pragma: no cover
        """Print the various properties of a :class:`DisjointSubcircuitState`."""
        cut_actions_list = self.cut_actions_sublist()
        self.actions = cast(list, self.actions)
        if simple:
            print(cut_actions_list)
        else:
            print("wiremap", self.wiremap)
            print("num_wires", self.num_wires)
            print("uptree", self.uptree)
            print("width", self.width)
            print("bell_pairs", self.bell_pairs)
            print("gamma_LB", self.gamma_LB)
            print("lowerBound", self.lower_bound_gamma())
            print("gamma_UB", self.gamma_UB)
            print("no_merge", self.no_merge)
            print("actions", get_actions_list(self.actions))
            print("level", self.level)

    def get_num_qubits(self) -> int:
        """Return the number of qubits in the circuit."""
        self.wiremap = cast(NDArray[np.int_], self.wiremap)
        return self.wiremap.shape[0]

    def get_max_width(self) -> int:
        """Return the maximum width across subcircuits."""
        self.width = cast(NDArray[np.int_], self.width)
        return int(np.amax(self.width))

    def get_sub_circuit_indices(self) -> list[int]:
        """Return a list of root indices for the subcircuits in the current cut circuit."""
        self.uptree = cast(NDArray[np.int_], self.uptree)
        self.num_wires = cast(int, self.num_wires)
        return [i for i, j in enumerate(self.uptree[: self.num_wires]) if i == j]

    def get_wire_root_mapping(self) -> list[int]:
        """Return a list of root wires for each wire in the current state of the circuit."""
        self.num_wires = cast(int, self.num_wires)
        return [self.find_wire_root(i) for i in range(self.num_wires)]

    def find_root_bell_pair(self, bell_pair: tuple[int, int]) -> tuple[int, int]:
        """Find the root wires for a Bell pair (represented as a pair of wires).

        Additionally, return a sorted tuple representing the Bell pair.
        """
        r0 = self.find_wire_root(bell_pair[0])
        r1 = self.find_wire_root(bell_pair[1])
        return (r0, r1) if (r0 < r1) else (r1, r0)

    def lower_bound_gamma(self) -> float:
        """Return a lower bound for gamma using the current counts for the circuit cuts involving bell pairs."""
        self.bell_pairs = cast(list, self.bell_pairs)
        root_bell_pairs = map(lambda x: self.find_root_bell_pair(x), self.bell_pairs)

        self.gamma_LB = cast(float, self.gamma_LB)
        return self.gamma_LB * calc_root_bell_pairs_gamma(root_bell_pairs)

    def upper_bound_gamma(self) -> float:
        """Return an upper bound for gamma using the current counts for the different types of (LO) circuit cuts."""
        self.gamma_UB = cast(float, self.gamma_UB)
        return self.gamma_UB

    def can_add_wires(self, num_wires: int) -> bool:
        """Return ``True`` if an additional ``num_wires`` can be cut without exceeding the maximum allowed number of wire cuts."""
        self.num_wires = cast(int, self.num_wires)
        self.uptree = cast(NDArray[np.int_], self.uptree)
        return self.num_wires + num_wires <= self.uptree.shape[0]

    def can_expand_subcircuit(self, root: int, num_wires: int, max_width: int) -> bool:
        """Return ``True`` if ``num_wires`` can be added to subcircuit root without exceeding the maximum allowed number of qubits."""
        self.width = cast(NDArray[np.int_], self.width)
        return self.width[root] + num_wires <= max_width

    def new_wire(self, qubit: Hashable) -> int:
        """Cut the wire associated with ``qubit`` and return the ID of the new wire now associated with qubit."""
        self.num_wires = cast(int, self.num_wires)
        self.uptree = cast(NDArray[np.int_], self.uptree)
        assert self.num_wires < self.uptree.shape[0], (
            "Max new wires exceeded " + f"{self.num_wires}, {self.uptree.shape[0]}"
        )

        self.wiremap = cast(NDArray[np.int_], self.wiremap)
        self.wiremap[qubit] = self.num_wires
        self.num_wires += 1

        qubit = cast(int, qubit)
        return self.wiremap[qubit]

    def get_wire(self, qubit: Hashable) -> int:
        """Return the ID of the wire currently associated with ``qubit``."""
        self.wiremap = cast(NDArray[np.int_], self.wiremap)
        qubit = cast(int, qubit)
        return self.wiremap[qubit]

    def find_wire_root(self, wire: int) -> int:
        """Return the ID of the root wire in the subcircuit that contains wire.

        Additionally, collapse the path to the root.
        """
        # Find the root wire in the subcircuit
        root = wire
        self.uptree = cast(NDArray[np.int_], self.uptree)
        while root != self.uptree[root]:
            root = self.uptree[root]

        # Collapse the path to the root
        while wire != root:
            parent = self.uptree[wire]
            self.uptree[wire] = root
            wire = parent

        return root

    def find_qubit_root(self, qubit: Hashable) -> int:
        """Return the ID of the root wire in the subcircuit associated with ``qubit``.

        Additionally, collapse the path to the root.
        """
        self.wiremap = cast(NDArray[np.int_], self.wiremap)
        qubit = cast(int, qubit)
        return self.find_wire_root(self.wiremap[qubit])

    def check_donot_merge_roots(self, root_1: int, root_2: int) -> bool:
        """Return True if the subcircuits represented by root wire IDs ``root_1`` and ``root_2`` should not be merged."""
        self.uptree = cast(NDArray[np.int_], self.uptree)
        assert root_1 == self.uptree[root_1] and root_2 == self.uptree[root_2], (
            "Arguments must be roots: "
            + f"{root_1} != {self.uptree[root_1]} "
            + f"or {root_2} != {self.uptree[root_2]}"
        )

        self.no_merge = cast(list, self.no_merge)
        for clause in self.no_merge:
            r1 = self.find_wire_root(clause[0])
            r2 = self.find_wire_root(clause[1])

            assert r1 != r2, "Do-Not-Merge clauses must not be identical"

            if (r1 == root_1 and r2 == root_2) or (r1 == root_2 and r2 == root_1):
                return True

        return False

    def verify_merge_constraints(self) -> bool:
        """Return ``True`` if all merge constraints are satisfied."""
        self.no_merge = cast(list, self.no_merge)
        for clause in self.no_merge:
            r1 = self.find_wire_root(clause[0])
            r2 = self.find_wire_root(clause[1])
            if r1 == r2:
                return False

        return True

    def assert_donot_merge_roots(self, wire_1: int, wire_2: int) -> None:
        """Add a constraint that the subcircuits associated with IDs ``wire_1`` and ``wire_2`` should not be merged."""
        assert self.find_wire_root(wire_1) != self.find_wire_root(
            wire_2
        ), f"{wire_1} cannot be the same subcircuit as {wire_2}"

        assert isinstance(self.no_merge, list)
        self.no_merge.append((wire_1, wire_2))

    def merge_roots(self, root_1: int, root_2: int) -> None:
        """Merge the subcircuits associated with root wire IDs ``root_1`` and ``root_2``.

        Additionally, update the statistics (i.e., width) associated with the merged subcircuit.
        """
        self.uptree = cast(NDArray[np.int_], self.uptree)
        self.width = cast(NDArray[np.int_], self.width)
        assert root_1 == self.uptree[root_1] and root_2 == self.uptree[root_2], (
            "Arguments must be roots: "
            + f"{root_1} != {self.uptree[root_1]} "
            + f"or {root_2} != {self.uptree[root_2]}"
        )

        assert root_1 != root_2, f"Cannot merge root {root_1} with itself"

        merged_root = min(root_1, root_2)
        other_root = max(root_1, root_2)
        self.uptree[other_root] = merged_root
        self.width[merged_root] += self.width[other_root]

    def add_action(
        self,
        action_obj: DisjointSearchAction,
        gate_spec: GateSpec,
        args: tuple | None = None,
    ) -> None:
        """Append the specified action to the list of search-space actions that have been performed."""
        if action_obj.get_name() is not None:
            self.actions = cast(list, self.actions)
            self.actions.append(Action(action_obj, gate_spec, [args]))

    def get_search_level(self) -> int:
        """Return the search level."""
        self.level = cast(int, self.level)
        return self.level

    def set_next_level(self, state: DisjointSubcircuitsState) -> None:
        """Set the search level to one plus the search level of the input state."""
        self.level = cast(int, self.level)
        state.level = cast(int, state.level)
        self.level = state.level + 1

    def export_cuts(self, circuit_interface: SimpleGateList):
        """Export LO cuts into the input circuit_interface for each of the cutting decisions made."""
        # This wire map assumes no reuse of qubits
        assert self.num_wires is not None
        wire_map = np.arange(self.num_wires)

        assert self.actions is not None
        for action in self.actions:
            action.action.export_cuts(  # type: ignore
                circuit_interface,
                wire_map,
                action.gate_spec,
                action.args,
            )

        root_list = self.get_sub_circuit_indices()
        wires_to_roots = self.get_wire_root_mapping()

        subcircuits = [
            list({wire_map[w] for w, r in enumerate(wires_to_roots) if r == root})
            for root in root_list
        ]

        circuit_interface.define_subcircuits(subcircuits)


def calc_root_bell_pairs_gamma(root_bell_pairs: Iterable[Hashable]) -> float:
    """Calculate the minimum-achievable LOCC gamma for circuit cuts that utilize virtual Bell pairs.

    The input can be an iterable over hashable identifiers that represent Bell pairs across
    disconnected subcircuits in a cut circuit. There must be a one-to-one mapping between
    identifiers and pairs of subcircuits. Repeated identifiers are interpreted
    as mutiple Bell pairs across the same pair of subcircuits, and the counts
    of such repeats are used to calculate gamma.
    """
    gamma = 1.0
    for n in Counter(root_bell_pairs).values():
        gamma *= 2 ** (n + 1) - 1

    return gamma


def get_actions_list(
    action_list: list[Action],
) -> list[Action]:
    """Return a list of cutting actions that have been performed on a state."""
    return action_list
