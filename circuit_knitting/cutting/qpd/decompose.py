# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function to replace all QPD instructions in the circuit with local Qiskit operations and measurements."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.circuit import (
    QuantumCircuit,
    ClassicalRegister,
    CircuitInstruction,
    Measure,
)

from .instructions import BaseQPDGate, TwoQubitQPDGate
from ...utils.equivalence import equivalence_libraries


def decompose_qpd_instructions(
    circuit: QuantumCircuit,
    instruction_ids: Sequence[Sequence[int]],
    map_ids: Sequence[int] | None = None,
    *,
    translate_to_qpu: str | None = None,
    inplace: bool = False,
) -> QuantumCircuit:
    r"""
    Replace all QPD instructions in the circuit with local Qiskit operations and measurements.

    Args:
        circuit: The circuit containing QPD instructions
        instruction_ids: A 2D sequence, such that each inner sequence corresponds to indices
            of instructions comprising one decomposition in the circuit. The elements within a
            common sequence belong to a common decomposition and should be sampled together.
        map_ids: Indices to a specific linear mapping to be applied to the decompositions
            in the circuit. If no map IDs are provided, the circuit will be decomposed randomly
            according to the decompositions' joint probability distribution.
        translate_to_qpu: A QPU architecture for which the sampled instructions should be
            translated. Supported inputs are: {"heron", "eagle", None}
        inplace: Whether to modify the input circuit directly

    Returns:
        Circuit which has had all its :class:`BaseQPDGate` instances decomposed into local operations.

        The circuit will contain a new, final classical register to contain the QPD measurement
        outcomes (accessible at ``retval.cregs[-1]``).

    Raises:
        ValueError: An index in ``instruction_ids`` corresponds to a gate which is not a
            :class:`BaseQPDGate` instance.
        ValueError: A list within instruction_ids is not length 1 or 2.
        ValueError: The total number of indices in ``instruction_ids`` does not equal the number
            of :class:`BaseQPDGate` instances in the circuit.
        ValueError: Gates within the same decomposition hold different QPD bases.
        ValueError: Length of ``map_ids`` does not equal the number of decompositions in the circuit.
    """
    _validate_qpd_instructions(circuit, instruction_ids)

    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    if map_ids is not None:
        if len(instruction_ids) != len(map_ids):
            raise ValueError(
                f"The number of map IDs ({len(map_ids)}) must equal the number of "
                f"decompositions in the circuit ({len(instruction_ids)})."
            )
        # If mapping is specified, set each gate's mapping
        for i, decomp_gate_ids in enumerate(instruction_ids):
            for gate_id in decomp_gate_ids:
                circuit.data[gate_id].operation.basis_id = map_ids[i]

    # Convert all instances of BaseQPDGate in the circuit to Qiskit instructions
    _decompose_qpd_instructions(
        circuit, instruction_ids, translate_to_qpu=translate_to_qpu
    )

    return circuit


def _validate_qpd_instructions(
    circuit: QuantumCircuit, instruction_ids: Sequence[Sequence[int]]
):
    """Ensure the indices in instruction_ids correctly describe all the decompositions in the circuit."""
    # Make sure all instruction_ids correspond to QPDGates, and make sure each QPDGate in a given decomposition has
    # an equivalent QPDBasis to its sibling QPDGates
    for decomp_ids in instruction_ids:
        if len(decomp_ids) not in [1, 2]:
            raise ValueError(
                "Each decomposition must contain either one or two elements. Found a "
                f"decomposition with ({len(decomp_ids)}) elements."
            )
        if not isinstance(circuit.data[decomp_ids[0]].operation, BaseQPDGate):
            raise ValueError(
                f"A circuit data index ({decomp_ids[0]}) corresponds to a non-QPDGate "
                f"({circuit.data[decomp_ids[0]].operation.name})."
            )
        compare_basis = circuit.data[decomp_ids[0]].operation.basis
        for gate_id in decomp_ids:
            if not isinstance(circuit.data[gate_id].operation, BaseQPDGate):
                raise ValueError(
                    f"A circuit data index ({gate_id}) corresponds to a non-QPDGate "
                    f"({circuit.data[gate_id].operation.name})."
                )
            tmp_basis = circuit.data[gate_id].operation.basis
            if compare_basis != tmp_basis:
                raise ValueError(
                    "Gates within the same decomposition must share an equivalent QPDBasis."
                )

    # Make sure the total number of QPD gate indices equals the number of QPDGates in the circuit
    num_qpd_gates = sum(len(x) for x in instruction_ids)
    qpd_gate_total = 0
    for inst in circuit.data:
        if isinstance(inst.operation, BaseQPDGate):
            qpd_gate_total += 1
    if qpd_gate_total != num_qpd_gates:
        raise ValueError(
            f"The total number of QPDGates specified in instruction_ids ({num_qpd_gates}) "
            f"does not equal the number of QPDGates in the circuit ({qpd_gate_total})."
        )


def _decompose_qpd_measurements(
    circuit: QuantumCircuit, inplace: bool = True
) -> QuantumCircuit:
    """
    Create mid-circuit measurements.

    Convert all QPDMeasure instances to Measure instructions. Add any newly created
    classical bits to a new "qpd_measurements" register.
    """
    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    # Loop through the decomposed circuit to find QPDMeasure markers so we can
    # replace them with measurement instructions.  We can't use `_ids`
    # here because it refers to old indices, before the decomposition.
    qpd_measure_ids = [
        i
        for i, instruction in enumerate(circuit.data)
        if instruction.operation.name.lower() == "qpd_measure"
    ]

    # Create a classical register for the qpd measurement results.  This is
    # partly for convenience, partly to work around
    # https://github.com/Qiskit/qiskit-aer/issues/1660.
    reg = ClassicalRegister(len(qpd_measure_ids), name="qpd_measurements")
    circuit.add_register(reg)

    # Place the measurement instructions
    for idx, i in enumerate(qpd_measure_ids):
        gate = circuit.data[i]
        inst = CircuitInstruction(
            operation=Measure(), qubits=[gate.qubits], clbits=[reg[idx]]
        )
        circuit.data[i] = inst

    # If the user wants to access the qpd register, it will be the final
    # classical register of the returned circuit.
    assert circuit.cregs[-1] is reg

    return circuit


def _decompose_qpd_instructions(
    circuit: QuantumCircuit,
    instruction_ids: Sequence[Sequence[int]],
    inplace: bool = True,
    translate_to_qpu: str | None = None,
) -> QuantumCircuit:
    """Decompose all BaseQPDGate instances, ignoring QPDMeasure()."""
    if not inplace:
        circuit = circuit.copy()  # pragma: no cover

    # Decompose any 2q QPDGates into single qubit QPDGates
    qpdgate_ids_2q = []
    for decomp in instruction_ids:
        if len(decomp) != 1:
            continue  # pragma: no cover
        if isinstance(circuit.data[decomp[0]].operation, TwoQubitQPDGate):
            qpdgate_ids_2q.append(decomp[0])

    qpdgate_ids_2q = sorted(qpdgate_ids_2q)
    data_id_offset = 0
    for i in qpdgate_ids_2q:
        inst = circuit.data[i + data_id_offset]
        qpdcirc_2q_decomp = inst.operation.definition
        inst1 = CircuitInstruction(
            qpdcirc_2q_decomp.data[0].operation, qubits=[inst.qubits[0]]
        )
        inst2 = CircuitInstruction(
            qpdcirc_2q_decomp.data[1].operation, qubits=[inst.qubits[1]]
        )
        circuit.data[i + data_id_offset] = inst1
        data_id_offset += 1
        circuit.data.insert(i + data_id_offset, inst2)

    # Get equivalence library
    if translate_to_qpu is not None:
        translate_to_qpu = translate_to_qpu.lower()
    else:
        translate_to_qpu = "standard"
    equivalence = equivalence_libraries[translate_to_qpu]

    # Decompose all the QPDGates (should all be single qubit now) into Qiskit operations
    new_instruction_ids = []
    for i, inst in enumerate(circuit.data):
        if isinstance(inst.operation, BaseQPDGate):
            new_instruction_ids.append(i)
    data_id_offset = 0
    for i in new_instruction_ids:
        inst = circuit.data[i + data_id_offset]
        qubits = inst.qubits
        # All gates in decomposition should be local
        assert len(qubits) == 1
        # Gather instructions with which we will replace the QPDGate
        tmp_data = []
        for data in inst.operation.definition.data:
            # Can ignore clbits here, as QPDGates don't use clbits directly
            assert data.clbits == ()
            if equivalence is None:
                tmp_data.append(CircuitInstruction(data.operation, qubits=[qubits[0]]))
            else:
                equiv_entry = equivalence.get_entry(data.operation)
                # CKT SELs currently only provide at most one translation
                assert len(equiv_entry) <= 1
                if equiv_entry == []:
                    tmp_data.append(
                        CircuitInstruction(data.operation, qubits=[qubits[0]])
                    )
                else:
                    new_insts = equiv_entry[0]
                    for d in new_insts.data:
                        tmp_data.append(
                            CircuitInstruction(d.operation, qubits=[qubits[0]])
                        )

        # Replace QPDGate with local operations
        if tmp_data:
            # Overwrite the QPDGate with first instruction
            circuit.data[i + data_id_offset] = tmp_data[0]
            # Append remaining instructions immediately after original QPDGate position
            for data in tmp_data[1:]:
                data_id_offset += 1
                circuit.data.insert(i + data_id_offset, data)

        # If QPDGate decomposes to an identity operation, just delete it
        else:
            del circuit.data[i + data_id_offset]
            data_id_offset -= 1

    _decompose_qpd_measurements(circuit)

    return circuit
