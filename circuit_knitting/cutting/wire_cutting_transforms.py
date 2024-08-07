# This code is a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function to transform a :class:`.CutWire` instruction to a :class:`.Move` instruction."""
from __future__ import annotations

from typing import Callable
from itertools import groupby

import numpy as np
from qiskit.circuit import Qubit, QuantumCircuit, Operation
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import PauliList

from .instructions import Move
from .qpd.instructions import TwoQubitQPDGate


def cut_wires(circuit: QuantumCircuit, /) -> QuantumCircuit:
    r"""Transform all :class:`.CutWire` instructions in a circuit to :class:`.Move` instructions marked for cutting.

    The returned circuit will have one newly allocated qubit for every :class:`.CutWire` instruction.

    See Sec. 3 and Appendix A of `2302.03366v1
    <https://arxiv.org/abs/2302.03366v1>`__ for more information about the two
    different representations of wire cuts: single-qubit (:class:`.CutWire`)
    vs. two-qubit (:class:`.Move`).

    Args:
        circuit: Original circuit with :class:`.CutWire` instructions

    Returns:
        circuit: New circuit with :class:`.CutWire` instructions replaced by :class:`.Move` instructions wrapped in :class:`TwoQubitQPDGate`\ s
    """
    return _transform_cut_wires(
        circuit, lambda: TwoQubitQPDGate.from_instruction(Move())
    )


def _transform_cuts_to_moves(circuit: QuantumCircuit, /) -> QuantumCircuit:
    """Transform all :class:`.CutWire` instructions in a circuit to :class:`.Move` instructions.

    Args:
        circuit: Original circuit with :class:`.CutWire` instructions

    Returns:
        circuit: New circuit with :class:`.CutWire` instructions replaced by :class`.Move` instructions
    """
    return _transform_cut_wires(circuit, Move)


def _transform_cut_wires(
    circuit: QuantumCircuit, factory: Callable[[], Operation], /
) -> QuantumCircuit:
    new_circuit, mapping = _circuit_structure_mapping(circuit)

    for instructions in circuit.data:
        gate_index = [circuit.find_bit(qubit).index for qubit in instructions.qubits]

        if instructions in circuit.get_instructions("cut_wire"):
            # Replace cut_wire with move instruction
            new_circuit.compose(
                other=factory(),
                qubits=[mapping[gate_index[0]], mapping[gate_index[0]] + 1],
                inplace=True,
            )
            mapping[gate_index[0]] += 1
        else:
            new_circuit.compose(
                other=instructions.operation,
                qubits=[mapping[index] for index in gate_index],
                inplace=True,
            )

    return new_circuit


def _circuit_structure_mapping(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[int]]:
    new_circuit = QuantumCircuit()
    mapping = list(range(len(circuit.qubits)))

    cut_wire_index = [
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.get_instructions("cut_wire")
    ]
    cut_wire_freq = {key: len(list(group)) for key, group in groupby(cut_wire_index)}

    # Get intermediate mapping and add quantum bits to new_circuit
    for qubit in circuit.qubits:
        index = circuit.find_bit(qubit).index
        if index in cut_wire_freq.keys():
            for _ in range(cut_wire_freq[index]):
                mapping[index + 1 :] = map(
                    lambda item: item + 1, iter(mapping[index + 1 :])
                )
                new_circuit.add_bits([Qubit()])
        new_circuit.add_bits([qubit])

    # Add quantum and classical registers
    for qreg in circuit.qregs:
        new_circuit.add_register(qreg)

    new_circuit.add_bits(circuit.clbits)
    for creg in circuit.cregs:
        new_circuit.add_register(creg)

    return new_circuit, mapping


def expand_observables(
    observables: PauliList,
    original_circuit: QuantumCircuit,
    final_circuit: QuantumCircuit,
    /,
) -> PauliList:
    r"""Expand observable(s) according to the qubit mapping between ``original_circuit`` and ``final_circuit``.

    The :class:`.Qubit`\ s on ``final_circuit`` must be a superset of those on
    ``original_circuit``.

    Given a :class:`.PauliList` of observables, this function returns new
    observables with identity operators placed on the qubits that did not
    exist in ``original_circuit``.  This way, observables on
    ``original_circuit`` can be mapped to appropriate observables on
    ``final_circuit``.

    This function is designed to be used after calling ``final_circuit =
    transform_cuts_to_moves(original_circuit)`` (see
    :func:`.transform_cuts_to_moves`).

    This function requires ``observables.num_qubits ==
    original_circuit.num_qubits`` and returns new observables such that
    ``retval.num_qubits == final_circuit.num_qubits``.

    Args:
        observables: Observables corresponding to ``original_circuit``
        original_circuit: Original circuit
        final_circuit: Final circuit, whose qubits the original ``observables`` should be expanded to

    Returns:
        New :math:`N`-qubit observables which are compatible with the :math:`N`-qubit ``final_circuit``

    Raises:
        ValueError: ``observables`` and ``original_circuit`` have different number of qubits.
        ValueError: Qubit from ``original_circuit`` cannot be found in ``final_circuit``.
    """
    if observables.num_qubits != original_circuit.num_qubits:
        raise ValueError(
            "The `observables` and `original_circuit` must have the same number "
            f"of qubits. ({observables.num_qubits} != {original_circuit.num_qubits})"
        )
    mapping: list[int] = []
    for i, qubit in enumerate(original_circuit.qubits):
        try:
            idx = final_circuit.find_bit(qubit)[0]
        except CircuitError as ex:
            raise ValueError(
                f"The {i}-th qubit of the `original_circuit` cannot be found "
                "in the `final_circuit`."
            ) from ex
        mapping.append(idx)
    dims = (len(observables), final_circuit.num_qubits)
    z = np.full(dims, False)
    x = np.full(dims, False)
    z[:, mapping] = observables.z
    x[:, mapping] = observables.x
    return PauliList.from_symplectic(z, x, observables.phase.copy())
