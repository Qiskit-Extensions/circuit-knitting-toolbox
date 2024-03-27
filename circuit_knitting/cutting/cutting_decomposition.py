# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for decomposing circuits and observables for cutting."""

from __future__ import annotations


from collections import defaultdict
from collections.abc import Sequence, Hashable
from typing import NamedTuple

from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    Barrier,
)
from qiskit.quantum_info import PauliList

from ..utils.observable_grouping import observables_restricted_to_subsystem
from ..utils.transforms import separate_circuit, _partition_labels_from_circuit
from .qpd.qpd_basis import QPDBasis
from .qpd.instructions import TwoQubitQPDGate


class PartitionedCuttingProblem(NamedTuple):
    """The result of decomposing and separating a circuit and observable(s)."""

    subcircuits: dict[Hashable, QuantumCircuit]
    bases: list[QPDBasis]
    subobservables: dict[Hashable, PauliList] | None = None


def partition_circuit_qubits(
    circuit: QuantumCircuit, partition_labels: Sequence[Hashable], inplace: bool = False
) -> QuantumCircuit:
    r"""
    Replace all nonlocal gates belonging to more than one partition with instances of :class:`.TwoQubitQPDGate`.

    :class:`.TwoQubitQPDGate`\ s belonging to a single partition will not be affected.

    Args:
        circuit: The circuit to partition
        partition_labels: A sequence containing a partition label for each qubit in the
            input circuit. Nonlocal gates belonging to more than one partition
            will be replaced with :class:`.TwoQubitQPDGate`\ s.
        inplace: Flag denoting whether to copy the input circuit before acting on it

    Returns:
        The output circuit with each nonlocal gate spanning two partitions replaced by a
        :class:`.TwoQubitQPDGate`

    Raises:
        ValueError: The length of partition_labels does not equal the number of qubits in the circuit.
        ValueError: Input circuit contains unsupported gate.
    """
    if len(partition_labels) != len(circuit.qubits):
        raise ValueError(
            f"Length of partition_labels ({len(partition_labels)}) does not equal the number of qubits "
            f"in the input circuit ({len(circuit.qubits)})."
        )

    if not inplace:
        circuit = circuit.copy()

    # Find 2-qubit gates spanning more than one partition and replace it with a QPDGate.
    for i, instruction in enumerate(circuit.data):
        if instruction.operation.name == "barrier":
            continue
        qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        partitions_spanned = {partition_labels[idx] for idx in qubit_indices}

        # Ignore local gates and gates that span only one partition
        if (
            len(qubit_indices) <= 1
            or len(partitions_spanned) == 1
            or isinstance(instruction.operation, Barrier)
        ):
            continue

        if len(qubit_indices) > 2:
            raise ValueError(
                "Decomposition is only supported for two-qubit gates. Cannot "
                f"decompose ({instruction.operation.name})."
            )

        # Nonlocal gate exists in two separate partitions
        if isinstance(instruction.operation, TwoQubitQPDGate):
            continue

        qpd_gate = TwoQubitQPDGate.from_instruction(instruction.operation)
        circuit.data[i] = CircuitInstruction(qpd_gate, qubits=qubit_indices)

    return circuit


def cut_gates(
    circuit: QuantumCircuit, gate_ids: Sequence[int], inplace: bool = False
) -> tuple[QuantumCircuit, list[QPDBasis]]:
    r"""
    Transform specified gates into :class:`.TwoQubitQPDGate`\ s.

    Args:
        circuit: The circuit containing gates to be decomposed
        gate_ids: The indices of the gates to decompose
        inplace: Flag denoting whether to copy the input circuit before acting on it

    Returns:
        A copy of the input circuit with the specified gates replaced with :class:`.TwoQubitQPDGate`\ s
        and a list of :class:`.QPDBasis` instances -- one for each decomposed gate.

    Raises:
        ValueError: The input circuit should contain no classical bits or registers.
    """
    if len(circuit.cregs) != 0 or circuit.num_clbits != 0:
        raise ValueError(
            "Circuits input to cut_gates should contain no classical registers or bits."
        )
    # Replace specified gates with TwoQubitQPDGates
    if not inplace:
        circuit = circuit.copy()

    bases = []
    for gate_id in gate_ids:
        gate = circuit.data[gate_id]
        qubit_indices = [circuit.find_bit(qubit).index for qubit in gate.qubits]
        qpd_gate = TwoQubitQPDGate.from_instruction(gate.operation)
        bases.append(qpd_gate.basis)
        circuit.data[gate_id] = CircuitInstruction(qpd_gate, qubits=qubit_indices)

    return circuit, bases


def partition_problem(
    circuit: QuantumCircuit,
    partition_labels: Sequence[Hashable] | None = None,
    observables: PauliList | None = None,
) -> PartitionedCuttingProblem:
    r"""
    Separate an input circuit and observable(s).

    If ``partition_labels`` is provided, then qubits with matching partition
    labels will be grouped together, and non-local gates spanning more than one
    partition will be cut.

    If ``partition_labels`` is not provided, then it will be determined
    automatically from the connectivity of the circuit.  This automatic
    determination ignores any :class:`.TwoQubitQPDGate`\ s in the ``circuit``,
    as these denote instructions that are explicitly destined for cutting.  The
    resulting partition labels, in the automatic case, will be consecutive
    integers starting with 0.

    All cut instructions will be replaced with :class:`.SingleQubitQPDGate`\ s.

    If provided, ``observables`` will be separated along the boundaries specified by
    the partition labels.

    Args:
        circuit: The circuit to partition and separate
        partition_labels: A sequence of labels, such that each label corresponds
            to the circuit qubit with the same index
        observables: The observables to separate

    Returns:
        A tuple containing a dictionary mapping a partition label to a subcircuit,
        a list of QPD bases (one for each circuit gate or wire which was decomposed),
        and, optionally, a dictionary mapping a partition label to a list of Pauli observables.


    Raises:
        ValueError: The number of partition labels does not equal the number of qubits in the circuit.
        ValueError: An input observable acts on a different number of qubits than the input circuit.
        ValueError: An input observable has a phase not equal to 1.
        ValueError: The input circuit should contain no classical bits or registers.
    """
    if partition_labels is not None and len(partition_labels) != circuit.num_qubits:
        raise ValueError(
            f"The number of partition labels ({len(partition_labels)}) must equal the number "
            f"of qubits in the circuit ({circuit.num_qubits})."
        )
    if observables is not None and any(
        len(obs) != circuit.num_qubits for obs in observables
    ):
        raise ValueError(
            "An input observable acts on a different number of qubits than the input circuit."
        )
    if observables is not None and any(obs.phase != 0 for obs in observables):
        raise ValueError("An input observable has a phase not equal to 1.")

    if len(circuit.cregs) != 0 or circuit.num_clbits != 0:
        raise ValueError(
            "Circuits input to partition_problem should contain no classical registers or bits."
        )

    # Determine partition labels from connectivity (ignoring TwoQubitQPDGates)
    # if partition_labels is not specified
    if partition_labels is None:
        partition_labels = _partition_labels_from_circuit(
            circuit, ignore=lambda inst: isinstance(inst.operation, TwoQubitQPDGate)
        )

    # Partition the circuit with TwoQubitQPDGates and assign the order via their labels
    qpd_circuit = partition_circuit_qubits(circuit, partition_labels)

    bases = []
    i = 0
    for inst in qpd_circuit.data:
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            inst.operation.label = f"{inst.operation.label}_{i}"
            i += 1

    # Separate the decomposed circuit into its subcircuits
    qpd_circuit_dx = qpd_circuit.decompose(TwoQubitQPDGate)
    separated_circs = separate_circuit(qpd_circuit_dx, partition_labels)

    # Decompose the observables, if provided
    subobservables_by_subsystem = None
    if observables:
        subobservables_by_subsystem = decompose_observables(
            observables, partition_labels
        )

    return PartitionedCuttingProblem(
        separated_circs.subcircuits,  # type: ignore
        bases,
        subobservables=subobservables_by_subsystem,
    )


def decompose_observables(
    observables: PauliList, partition_labels: Sequence[Hashable]
) -> dict[Hashable, PauliList]:
    """
    Decompose a list of observables with respect to some qubit partition labels.

    Args:
        observables: A list of observables to decompose
        partition_labels: A sequence of partition labels, such that each label
            corresponds to the qubit in the same index

    Returns:
        A dictionary mapping a partition to its associated sub-observables
    """
    qubits_by_subsystem = defaultdict(list)
    for i, label in enumerate(partition_labels):
        qubits_by_subsystem[label].append(i)
    qubits_by_subsystem = dict(qubits_by_subsystem)  # type: ignore

    subobservables_by_subsystem = {
        label: observables_restricted_to_subsystem(qubits, observables)
        for label, qubits in qubits_by_subsystem.items()
    }

    return subobservables_by_subsystem
