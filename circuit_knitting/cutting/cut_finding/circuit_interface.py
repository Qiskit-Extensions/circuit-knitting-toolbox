# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum circuit representation compatible with cut-finding optimizer."""
from __future__ import annotations

import copy
import numpy as np
import string
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import NamedTuple, Hashable, Iterable, cast, Sequence


class CircuitElement(NamedTuple):
    """Named tuple for specifying a circuit element."""

    name: str
    params: Sequence[float | int]
    qubits: Sequence[int | tuple[str, int]]
    gamma: float | None


class GateSpec(NamedTuple):
    """Named tuple for gate specification.

    ``cut_constraints`` can be of the form
    None,[],[None], or  [<cut_type_1>, ..., <cut_type_n>]

    A cut constraint of None indicates that no constraints are placed
    on how or whether cuts can be performed. An empty list [] or the
    list [None] indicates that no cuts are to be performed and the gate
    is to be applied without cutting. A list of cut types of the form
    [<cut_type_1> ... <cut_type_n>] indicates precisely which types of
    cuts can be considered. In this case, the cut type None must be
    explicitly included to indicate the possibilty of not cutting, if
    not cutting is to be considered. In the current version of the code,
    the allowed cut types are 'None', 'GateCut' and 'WireCut'.
    """

    instruction_id: int
    gate: CircuitElement
    cut_constraints: list | None


class CircuitInterface(ABC):
    """Access attributes of input circuit and perform operations on the internal circuit representations."""

    @abstractmethod
    def get_num_qubits(self):
        """Return the number of qubits in the input circuit."""

    @abstractmethod
    def get_multiqubit_gates(self):
        """Return a list that specifies the multiqubit gates in the input circuit.

        The returned list is a list of instances of :class:`GateSpec`.
        """

    @abstractmethod
    def insert_gate_cut(self, gate_ID, cut_type):
        """Mark the specified gate as being cut. The cut types can only be "LO" in this release."""

    @abstractmethod
    def insert_wire_cut(self, gate_ID, input_ID, src_wire_ID, dest_wire_ID, cut_type):
        """Insert insert a wire cut into the output circuit.

        Wire cuts are inserted just prior to the specified
        gate on the wire connected to the specified input of that gate.

        Gate inputs are numbered starting from 1. The wire/qubit ID of the wire to be cut
        is also provided as input to allow the wire choice to be verified.
        The ID of the new wire/qubit is also provided, which can then be used
        internally in derived classes to create new wires/qubits as needed.
        The cut type can only be "LO" in this release.
        """

    @abstractmethod
    def define_subcircuits(self, list_of_list_of_wires):
        """Define subcircuits using as input a list of subcircuits.

        Each subcircuit is specified as a list of wire IDs.
        """


class SimpleGateList(CircuitInterface):
    """Convert a simple list of gates into the form needed by the optimizer.

    Elements of the input list must be instances of :class:`CircuitElement`.
    The only exception to this is a barrier when one is placed across
    all the qubits in a circuit. That is specified by the string: "barrier".

    Qubit names can be any hashable objects. Gate names can also be any
    hashable objects, but they must be consistent with the names used by the
    optimizer to look up cutting rules for the specified gates.

    The constructor can be supplied with a list of qubit names to force a
    preferred ordering in the assignment of numeric qubit IDs to each name.

    Member Variables:
    `qubit_names` (NametoIDMap): an instance of :class:`NametoIDMap` that maps
    qubit names to numerical qubit IDs.

    `num_qubits` (int): the number of qubits in the input circuit. Qubit IDs
    whose values are greater than or equal to num_qubits represent qubits
    that were introduced as the result of wire cutting. These qubits are
    assigned generated names of the form ('cut', <qubit_name>) in
    ``qubit_names``, where <qubit_name> is the name of the wire/qubit
    that was cut to create the new wire/qubit.

    `circuit` (list): the internal representation of the circuit, which is
    a list of the following form:

        [ ... [<gate_specification>, None] ...]

    where <gate_specification> can be a string to denote a "barrier" across
    the entire circuit, or an instance of :class:`CircuitElement`.
    Moreover the qubit names have been replaced with qubit IDs
    in the gate specification.

    `new_circuit` (list): a list that defines the cut circuit.
    the cut circuit. In the absence of wire cuts, it has
    the form [...<gate_specification>...] The form of <gate_specification>
    is as mentioned above. As with ``circuit``, qubit IDs are used to identify
    wires/qubits. After wire cuts ``new_circuit``has lists of the form
    ["move", <source_wire_id>, <destination_wire_id>] inserted into it.

    `cut_type` (list): a list that assigns cut-type annotations to gates
    in ``new_circuit``.

    `new_gate_ID`_map (list): a list that maps the positions of gates
    in circuit to their new positions in ``new_circuit``.

    `output_wires` (list): a list that maps qubit IDs in circuit to the corresponding
    output wires of new_circuit so that observables defined for circuit
    can be remapped to ``new_circuit``.

    `subcircuits` (list): a list of list of wire IDs, where each list of
    wire IDs defines a subcircuit.
    """

    circuit: list
    new_circuit: list
    cut_type: list[str | None]
    qubit_names: NameToIDMap
    num_qubits: int
    new_gate_ID_map: NDArray[np.int_]
    output_wires: NDArray[np.int_]

    def __init__(
        self,
        input_circuit: list[CircuitElement | str],
        init_qubit_names: list[Hashable] = list(),
    ):
        """Assign member variables."""
        self.qubit_names = NameToIDMap(init_qubit_names)

        self.circuit = list()
        self.new_circuit = list()
        self.cut_type = list()
        for gate in input_circuit:
            self.cut_type.append(None)
            if not isinstance(gate, CircuitElement):
                assert gate == "barrier"
                self.circuit.append([copy.deepcopy(gate), None])
                self.new_circuit.append(copy.deepcopy(gate))
            else:
                gate_spec = CircuitElement(
                    name=gate.name,
                    params=gate.params,
                    qubits=[self.qubit_names.get_id(x) for x in gate.qubits],
                    gamma=gate.gamma,
                )
                self.circuit.append([copy.deepcopy(gate_spec), None])
                self.new_circuit.append(copy.deepcopy(gate_spec))
        self.new_gate_id_map = np.arange(len(self.circuit), dtype=int)
        self.num_qubits = self.qubit_names.get_array_size_needed()
        self.output_wires = np.arange(self.num_qubits, dtype=int)

        # Initialize the list of subcircuits assuming no cutting
        self.subcircuits: Sequence[list[int] | int] = list(list(range(self.num_qubits)))

    def get_num_qubits(self) -> int:
        """Return the number of qubits in the input circuit."""
        return self.num_qubits

    def get_num_wires(self) -> int:
        """Return the number of wires/qubits in the cut circuit."""
        return self.qubit_names.get_num_items()

    def get_multiqubit_gates(
        self,
    ) -> list[GateSpec]:
        """Extract the multiqubit gates from the circuit and prepend the index of the gate in the circuits to the gate specification.

        The elements of the resulting list are instances of :class:`GateSpec`.
        """
        subcircuit: list[GateSpec] = list()
        for k, circ_element in enumerate(self.circuit):
            gate = circ_element[0]
            cut_constraints = circ_element[1]
            assert cut_constraints is None
            if gate != "barrier":
                if len(gate.qubits) > 1 and gate.name != "barrier":  # type: ignore
                    subcircuit.append(GateSpec(k, gate, cut_constraints))

        return subcircuit

    def insert_gate_cut(self, gate_id: int, cut_type: str) -> None:
        """Mark the specified gate as being cut. The cut type in this release can only be "LO"."""
        gate_pos = self.new_gate_id_map[gate_id]
        self.cut_type[gate_pos] = cut_type

    def insert_wire_cut(
        self,
        gate_id: int,
        input_id: int,
        src_wire_id: int,
        dest_wire_id: int,
        cut_type: str,
    ) -> None:
        """Insert a wire cut into the output circuit.

        Wire cuts are inserted just prior to the specified
        gate on the wire connected to the specified input of that gate.

        Gate inputs are numbered starting from 1.  The
        wire/qubit ID of the source wire to be cut is also provided as
        input to allow the wire choice to be verified.  The ID of the
        (new) destination wire/qubit must also be provided. The cut
        type in this release can only be "LO".
        """
        gate_pos = self.new_gate_id_map[gate_id]
        new_gate_spec = self.new_circuit[gate_pos]

        # Gate inputs are numbered starting from 1, so we must
        # decrement the index to match qubit numbering.
        assert src_wire_id == new_gate_spec.qubits[input_id - 1], (
            f"Input wire ID {src_wire_id} does not match "
            + f"new_circuit wire ID {new_gate_spec.qubits[input_id-1]}"
        )

        # If the new wire does not yet exist, then define it
        if self.qubit_names.get_name(dest_wire_id) is None:
            wire_name = self.qubit_names.get_name(src_wire_id)
            self.qubit_names.define_id(dest_wire_id, ("cut", wire_name))

        # Replace src_wire_id with dest_wire_id in the part of new_circuit that
        # follows the wire-cut insertion point
        wire_map = list(range(self.qubit_names.get_array_size_needed()))
        wire_map[src_wire_id] = dest_wire_id

        self.replace_wire_ids(self.new_circuit[gate_pos:], wire_map)

        # Insert a move operator
        self.new_circuit.insert(gate_pos, ["move", src_wire_id, dest_wire_id])
        self.cut_type.insert(gate_pos, cut_type)
        self.new_gate_id_map[gate_id:] += 1

        # Update the output wires
        op = self.circuit[gate_id][0]
        qubit = op.qubits[input_id - 1]
        self.output_wires[qubit] = dest_wire_id

    def define_subcircuits(self, list_of_list_of_wires: list[list[int]]) -> None:
        """Assign subcircuits where each subcircuit is specified as a list of wire IDs."""
        self.subcircuits = list_of_list_of_wires

    def get_wire_names(self) -> list[Hashable]:
        """Return a list of the internal wire names used in the circuit.

        This consists of the original qubit names together with additional
        names of form ("cut", <name>) introduced to represent cut wires.
        """
        return list(self.qubit_names.get_items())

    def export_cut_circuit(
        self,
        name_mapping: None | str = "default",
    ) -> list[CircuitElement]:
        """Return a list of gates representing the cut circuit.

        If None is provided as the name_mapping, then the original qubit names are
        used with additional names of form ("cut", <name>) introduced as
        needed to represent cut wires.  If "default" is used as the mapping
        then :meth:`default_wire_name_mapping` defines the name mapping.
        """
        wire_map = self.make_wire_mapping(name_mapping)
        out = copy.deepcopy(self.new_circuit)

        wire_map = cast(list, wire_map)
        self.replace_wire_ids(out, wire_map)

        return out

    def export_output_wires(
        self,
        name_mapping: None | str = "default",
    ) -> dict[Hashable, Hashable | tuple[str, Hashable]]:
        """Return a dictionary that maps output qubits in the input circuit to the corresponding output wires/qubits in the cut circuit.

        If None is provided as the name_mapping, then the original qubit names are
        used with additional names of form ("cut", <name>) introduced as
        needed to represent cut wires.  If "default" is used as the mapping
        then :meth:``SimpleGateList.default_wire_name_mapping`` defines the name mapping.
        """
        wire_map = self.make_wire_mapping(name_mapping)
        out = dict()
        for in_wire, out_wire in enumerate(self.output_wires):
            out[self.qubit_names.get_name(in_wire)] = wire_map[out_wire]
        return out

    def export_subcircuits_as_string(
        self,
        name_mapping: None | str = "default",
    ) -> str:
        """Return a string that maps qubits/wires in the output circuit to subcircuits.

        This mapping is done per the Circuit Knitting Toolbox convention. This
        method only works with mappings to numeric qubit/wire names.
        """
        wire_map = self.make_wire_mapping(name_mapping)

        out: Sequence[int | str] = list(range(self.get_num_wires()))
        out = cast(list, out)
        alphabet = string.ascii_uppercase + string.ascii_lowercase
        for k, subcircuit in enumerate(self.subcircuits):
            subcircuit = cast(list, subcircuit)
            for wire in subcircuit:
                wire_map = cast(list, wire_map)
                out[wire_map[wire]] = alphabet[k]
        return "".join(out)

    def make_wire_mapping(
        self, name_mapping: None | str | dict
    ) -> Sequence[int | tuple[str, int]]:
        """Return a wire-mapping list given an input specification of a name mapping.

        If ``None ``is provided as the input name_mapping, then the original qubit names
        are mapped to themselves. If "default" is used as the ``name_mapping``,
        then :meth:``default_wire_name_mapping`` is used to define the name mapping.
        """
        if name_mapping is None:
            name_mapping = dict()
            for name in self.get_wire_names():
                name_mapping[name] = name

        elif name_mapping == "default":
            name_mapping = self.default_wire_name_mapping()  # type: ignore

        wire_mapping: list[int | tuple[str, int]] = list()

        for k in self.qubit_names.get_ids():
            name_mapping = cast(dict, name_mapping)
            wire_mapping.append(name_mapping[self.qubit_names.get_name(k)])

        return wire_mapping

    def default_wire_name_mapping(self) -> dict[Hashable, int]:
        """Return dictionary that maps wire names to default numeric output qubit names when exporting a cut circuit.

        Cut wires are assigned numeric IDs that are adjacent to the numeric ID of the wire prior to cutting so that Move
        operators are then applied against adjacent qubits. This is ensured by :meth:`SimpleGateList.sort_order`.
        """
        name_pairs = [(name, self.sort_order(name)) for name in self.get_wire_names()]

        name_pairs.sort(key=lambda x: x[1])

        name_map: dict[Hashable, int] = dict()
        for k, pair in enumerate(name_pairs):
            name_map[pair[0]] = k

        return name_map

    def sort_order(self, name: Hashable) -> int | float:
        """Order numeric IDs of wires to enable :meth:`SimpleGateList.default_wire_name_mapping`."""
        if isinstance(name, tuple):
            if name[0] == "cut":
                x = self.sort_order(name[1])
                x_int = int(x)
                x_frac = x - x_int
                return x_int + 0.5 * x_frac + 0.5

        return self.qubit_names.get_id(name)

    def replace_wire_ids(
        self,
        gate_list: Sequence[CircuitElement | Sequence[str | int]],
        # wire_map: Sequence[int | tuple[str, int]],
        wire_map: list[int],
    ) -> None:
        """Iterate through a list of gates and replace wire IDs with the values defined by the ``wire_map``."""
        for inst in gate_list:
            if isinstance(inst, CircuitElement):
                for k in range(len(inst.qubits)):
                    inst.qubits[k] = wire_map[inst.qubits[k]]  # type: ignore
            elif isinstance(inst, list):
                for k in range(1, len(inst)):
                    inst[k] = wire_map[inst[k]]


class NameToIDMap:
    """Class used to construct maps between hashable items (e.g., qubit names) and natural numbers (e.g., qubit IDs)."""

    def __init__(self, init_names: list[Hashable]):
        """Allow the name dictionary to be initialized with the names in ``init_names`` in the order the names appear.

        This is done in order to force a preferred ordering in the assigment of item IDs to those names.
        """
        self.next_id: int = 0
        self.item_dict: dict[Hashable, int] = dict()
        self.id_dict: dict[int, Hashable] = dict()

        for name in init_names:
            self.get_id(name)

    def get_id(self, item_name: Hashable) -> int:
        """Return the numeric ID associated with the specified hashable item.

        If the hashable item does not yet appear in the item dictionary, a new
        item ID is assigned.
        """
        if item_name not in self.item_dict:
            while self.next_id in self.id_dict:  # pragma: no cover
                self.next_id += 1

            self.item_dict[item_name] = self.next_id
            self.id_dict[self.next_id] = item_name
            self.next_id += 1

        return self.item_dict[item_name]

    def define_id(self, item_id: int, item_name: Hashable) -> None:
        """Assign a specific ID number to an item name."""
        assert item_id not in self.id_dict, f"item ID {item_id} already assigned"
        assert (
            item_name not in self.item_dict
        ), f"item name {item_name} already assigned"

        self.item_dict[item_name] = item_id
        self.id_dict[item_id] = item_name

    def get_name(self, item_id: int) -> Hashable | None:
        """Return the name associated with the specified ``item_id``.

        None is returned if ``item_id`` does not (yet) exist.
        """
        if item_id not in self.id_dict:
            return None

        return self.id_dict[item_id]

    def get_num_items(self) -> int:
        """Return the number of hashable items loaded thus far."""
        return len(self.item_dict)

    def get_array_size_needed(self) -> int:
        """Return one plus the maximum item ID assigned thus far, or zero if no items have been assigned.

        The value returned is thus the minimum size needed for a Python/Numpy array that maps item IDs to other hashables.
        """
        if self.get_num_items() == 0:  # pragma: no cover
            return 0

        return 1 + max(self.id_dict.keys())

    def get_items(self) -> Iterable[Hashable]:
        """Return the keys of the dictionary of hashable items loaded thus far."""
        return self.item_dict.keys()

    def get_ids(self) -> Iterable[int]:
        """Return the keys of the dictionary of ID's assigned to hashable items loaded thus far."""
        return self.id_dict.keys()
