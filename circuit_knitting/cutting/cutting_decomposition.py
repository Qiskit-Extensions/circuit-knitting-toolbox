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

from qiskit import transpile
from qiskit.utils import deprecate_func
from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    Barrier,
)
from qiskit.circuit.library.standard_gates import XGate
from qiskit.quantum_info import PauliList

from ..utils.observable_grouping import observables_restricted_to_subsystem
from ..utils.transforms import separate_circuit
from .qpd import QPDBasis, TwoQubitQPDGate


class PartitionedCuttingProblem(NamedTuple):
    """The result of decomposing and separating a circuit and observable(s)."""

    subcircuits: dict[str | int, QuantumCircuit]
    bases: list[QPDBasis]
    subobservables: dict[str | int, PauliList] | None = None


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


@deprecate_func(
    since="0.3",
    package_name="circuit-knitting-toolbox",
    removal_timeline="no earlier than v0.4.0",
    additional_msg=(
        "Instead, use ``circuit_knitting.cutting.cut_gates`` "
        "to automatically transform specified gates into "
        "``TwoQubitQPDGate`` instances."
    ),
)
def decompose_gates(
    circuit: QuantumCircuit, gate_ids: Sequence[int], inplace: bool = False
) -> tuple[QuantumCircuit, list[QPDBasis]]:  # pragma: no cover
    r"""
    Transform specified gates into :class:`.TwoQubitQPDGate`\ s.

    Deprecated as of 0.3.0. Instead, use :func:`~circuit_knitting.cutting.cut_gates`.
    """
    return cut_gates(circuit, gate_ids, inplace)


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
            "Circuits input to execute_experiments should contain no classical registers or bits."
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
    partition_labels: Sequence[str | int] | None = None,
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
            "Circuits input to execute_experiments should contain no classical registers or bits."
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
    observables: PauliList, partition_labels: Sequence[str | int]
) -> dict[str | int, PauliList]:
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


def find_gate_cuts(
    circuit: QuantumCircuit, budget: float, transpilation_options: dict
) -> tuple[QuantumCircuit, list[QPDBasis], list[int]]:
    r"""
    Find an optimized set of gates to cut, given a transpilation context.

    This function seeks to reduce the depth of the transpiled
    sub-experiments by cutting gates which result in the highest swap overhead.

    Args:
        circuit: The circuit to cut
        budget: The max sampling overhead allowed in returned cutting problem
        transpilation_options: A dictionary of kwargs to be passed to the Qiskit
            ``transpile`` function.

    Returns:
        A tuple containing:
            - A copy of the input circuit with SWAP-costly gates replaced with :class:`TwoQubitGate`\ s
            - A list of :class:`QPDBasis` instances -- one for each QPD gate in the circuit
            - A list of indices where the new :class:`TwoQubitGate`\ s are located in the output circuit
    """
    circ_copy = circuit.copy()

    # Sweep the circuit num_cuts times. In each sweep, find the gate that results in the biggest
    # reduction in depth, and replace it with a local, placeholder gate.
    cut_indices = []
    total_cost = 1
    while total_cost < budget:
        best_cuts = _evaluate_cuts(circ_copy, transpilation_options)
        improved_best = False
        for cut in best_cuts:
            if cut[1] * total_cost > budget:
                continue
            best_idx = cut[0]
            best_cost = cut[1]
            best_score = cut[2]
            improved_best = True
            break
        if not improved_best:
            break
        total_cost *= best_cost
        cut_indices.append(best_idx)
        print(f"Adding cut gate with cost {best_cost} and depth_savings {best_score * best_cost}")
        # Put a single qubit placeholder in place of the optimal gate
        qubit0 = circ_copy.find_bit(circ_copy.data[best_idx].qubits[0]).index
        # Use XGate as a placeholder since it will transpile to all our backends.
        # IGate cannot transpile to Eagle backends.
        circ_copy.data[best_idx] = CircuitInstruction(XGate(), qubits=(qubit0,))

    # Cut the deepest gates found in each step
    qpd_circuit, bases = cut_gates(circuit, cut_indices)

    return qpd_circuit, bases, cut_indices


def _evaluate_cuts(
    circuit: QuantumCircuit, transpilation_options: dict
) -> list[tuple[int, int]]:
    """Return the index and overhead for each supported gate in the circuit."""
    input_depth = transpile(circuit, **transpilation_options).depth()

    # For each supported gate in the circuit, assign a score based on the gate's
    # average SWAP overhead across num_reps transpilation runs
    cut_scores = []
    for i, inst in enumerate(circuit.data):
        if len(inst.qubits) != 2:
            continue
        del circuit.data[i]
        depth_savings = input_depth - transpile(circuit, **transpilation_options).depth()
        overhead = QPDBasis.from_instruction(inst.operation).overhead
        cut_score = depth_savings / overhead
        cut_scores.append((i, overhead, cut_score))
        circuit.data.insert(i, inst)

    return sorted(cut_scores, key=lambda x: x[2], reverse=True)
