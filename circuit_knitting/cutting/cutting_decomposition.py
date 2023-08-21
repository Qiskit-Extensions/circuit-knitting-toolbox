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

import numpy as np
from qiskit.utils import deprecate_func
from qiskit.circuit import (
    QuantumCircuit,
    ClassicalRegister,
    CircuitInstruction,
    Barrier,
)
from qiskit.quantum_info import PauliList

from ..utils.iteration import strict_zip
from ..utils.observable_grouping import (
    observables_restricted_to_subsystem,
    ObservableCollection,
    CommutingObservableGroup,
)
from ..utils.transforms import separate_circuit, _partition_labels_from_circuit
from .qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_weights,
    decompose_qpd_instructions,
    WeightType,
)


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


def generate_cutting_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    observables: PauliList | dict[str | int, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[str | int, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    """
    Generate cutting subexperiments and their associated weights.

    If the input circuit and observables are not split into multiple partitions, the output
    subexperiments will be contained within a 1D array.

    If the input circuit and observables is split into multiple partitions, the output
    subexperiments will be returned as a dictionary which maps a partition label to to
    a 1D array containing the subexperiments associated with that partition.

    In both cases, the subexperiment lists are ordered as follows:
        :math:`[sample_{0}observable_{0}, sample_{0}observable_{1}, ..., sample_{0}observable_{N}, ..., sample_{M}observable_{N}]`

    The weights will always be returned as a 1D array -- one weight for each unique sample.

    Args:
        circuits: The circuit(s) to partition and separate
        observables: The observable(s) to evaluate for each unique sample
        num_samples: The number of samples to draw from the quasi-probability distribution. If set
            to infinity, the weights will be generated rigorously rather than by sampling from
            the distribution.
    Returns:
        A tuple containing the cutting experiments and their associated weights.
        If the input circuits is a :class:`QuantumCircuit` instance, the output subexperiments
        will be a sequence of circuits -- one for every unique sample and observable. If the
        input circuits are represented as a dictionary keyed by partition labels, the output
        subexperiments will also be a dictionary keyed by partition labels and containing
        the subexperiments for each partition.
        The weights are always a sequence of length-2 tuples, where each tuple contains the
        weight and the :class:`WeightType`. Each weight corresponds to one unique sample.
    Raises:
        ValueError: ``num_samples`` must either be an integer or infinity.
        ValueError: :class:`SingleQubitQPDGate` instances must have the cut number
            appended to the gate label.
        ValueError: :class:`SingleQubitQPDGate` instances are not allowed in unseparated circuits.
    """
    if isinstance(num_samples, float):
        if num_samples != np.inf:
            raise ValueError("num_samples must either be an integer or infinity.")

    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on the format of the execute_experiments input args, but the 2nd half of this function
    # can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_list = [circuits]
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        subsystem_observables = {
            label: ObservableCollection(subobservables)
            for label, subobservables in subobservables_by_subsystem.items()
        }
        # Gather the unique bases from the circuit
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids = [qpd_gate_ids]

    else:
        is_separated = True
        subcircuit_list = [circuits[key] for key in sorted(circuits.keys())]
        # Gather the unique bases across the subcircuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(
            subcircuit_list
        )
        bases = _get_bases_by_partition(subcircuit_list, subcirc_qpd_gate_ids)

        # Create the commuting observable groups
        subsystem_observables = {
            label: ObservableCollection(so) for label, so in observables.items()
        }

    # Sample the joint quasiprobability decomposition
    random_samples = generate_qpd_weights(bases, num_samples=num_samples)

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])  # type: ignore

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the sub-experiments and weights
    subexperiments_dict: dict[str | int, list[QuantumCircuit]] = defaultdict(list)
    weights: list[tuple[float, WeightType]] = []
    for i, (subcircuit, label) in enumerate(
        strict_zip(subcircuit_list, sorted(subsystem_observables.keys()))
    ):
        for z, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples):
            actual_coeff = np.prod(
                [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
            )
            sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
            if i == 0:
                weights.append((sampled_coeff, weight_type))
            map_ids_tmp = map_ids
            if is_separated:
                map_ids_tmp = tuple(map_ids[j] for j in subcirc_map_ids[i])
            decomp_qc = decompose_qpd_instructions(
                subcircuit, subcirc_qpd_gate_ids[i], map_ids_tmp
            )
            so = subsystem_observables[label]
            for j, cog in enumerate(so.groups):
                meas_qc = _append_measurement_circuit(decomp_qc, cog)
                subexperiments_dict[label].append(meas_qc)

    subexperiments_out: list[QuantumCircuit] | dict[
        str | int, list[QuantumCircuit]
    ] = subexperiments_dict
    if len(subexperiments_dict.keys()) == 1:
        subexperiments_out = subexperiments_dict[list(subexperiments_dict.keys())[0]]

    return subexperiments_out, weights


def _get_mapping_ids_by_partition(
    circuits: Sequence[QuantumCircuit],
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Get indices to the QPD gates in each subcircuit and relevant map ids."""
    # Collect QPDGate id's and relevant map id's for each subcircuit
    subcirc_qpd_gate_ids: list[list[list[int]]] = []
    subcirc_map_ids: list[list[int]] = []
    decomp_ids = set()
    for circ in circuits:
        subcirc_qpd_gate_ids.append([])
        subcirc_map_ids.append([])
        for i, inst in enumerate(circ.data):
            if isinstance(inst.operation, SingleQubitQPDGate):
                try:
                    decomp_id = int(inst.operation.label.split("_")[-1])
                except (AttributeError, ValueError):
                    _raise_bad_qpd_gate_labels()
                decomp_ids.add(decomp_id)
                subcirc_qpd_gate_ids[-1].append([i])
                subcirc_map_ids[-1].append(decomp_id)

    return subcirc_qpd_gate_ids, subcirc_map_ids


def _raise_bad_qpd_gate_labels() -> None:
    raise ValueError(
        "BaseQPDGate instances in input circuit(s) should have their "
        'labels suffixed with "_<cut_#>" so that sibling SingleQubitQPDGate '
        "instances may be grouped and sampled together."
    )


def _get_bases_by_partition(
    circuits: Sequence[QuantumCircuit], subcirc_qpd_gate_ids: list[list[list[int]]]
) -> list[QPDBasis]:
    """Get a list of each unique QPD basis across the subcircuits."""
    # Collect the bases corresponding to each decomposed operation
    bases_dict = {}
    for i, subcirc in enumerate(subcirc_qpd_gate_ids):
        for basis_id in subcirc:
            try:
                decomp_id = int(
                    circuits[i].data[basis_id[0]].operation.label.split("_")[-1]
                )
            except (AttributeError, ValueError):  # pragma: no cover
                _raise_bad_qpd_gate_labels()  # pragma: no cover

            bases_dict[decomp_id] = circuits[i].data[basis_id[0]].operation.basis
    bases = [bases_dict[key] for key in sorted(bases_dict.keys())]

    return bases


def _append_measurement_circuit(
    qc: QuantumCircuit,
    cog: CommutingObservableGroup,
    /,
    *,
    qubit_locations: Sequence[int] | None = None,
    inplace: bool = False,
) -> QuantumCircuit:
    """Append a new classical register and measurement instructions for the given ``CommutingObservableGroup``.

    The new register will be named ``"observable_measurements"`` and will be
    the final register in the returned circuit, i.e. ``retval.cregs[-1]``.

    Args:
        qc: The quantum circuit
        cog: The commuting observable set for
            which to construct measurements
        qubit_locations: A ``Sequence`` whose length is the number of qubits
            in the observables, where each element holds that qubit's corresponding
            index in the circuit.  By default, the circuit and observables are assumed
            to have the same number of qubits, and the identity map
            (i.e., ``range(qc.num_qubits)``) is used.
        inplace: Whether to operate on the circuit in place (default: ``False``)

    Returns:
        The modified circuit
    """
    if qubit_locations is None:
        # By default, the identity map.
        if qc.num_qubits != cog.general_observable.num_qubits:
            raise ValueError(
                f"Quantum circuit qubit count ({qc.num_qubits}) does not match qubit "
                f"count of observable(s) ({cog.general_observable.num_qubits}).  "
                f"Try providing `qubit_locations` explicitly."
            )
        qubit_locations = range(cog.general_observable.num_qubits)
    else:
        if (
            len(qubit_locations) != cog.general_observable.num_qubits
        ):  # pragma: no cover
            raise ValueError(  # pragma: no cover
                f"qubit_locations has {len(qubit_locations)} element(s) but the "
                f"observable(s) have {cog.general_observable.num_qubits} qubit(s)."
            )
    if not inplace:
        qc = qc.copy()

    # Append the appropriate measurements to qc
    obs_creg = ClassicalRegister(len(cog.pauli_indices), name="observable_measurements")
    qc.add_register(obs_creg)
    # Implement the necessary basis rotations and measurements, as
    # in BackendEstimator._measurement_circuit().
    genobs_x = cog.general_observable.x
    genobs_z = cog.general_observable.z
    for clbit, subqubit in enumerate(cog.pauli_indices):
        # subqubit is the index of the qubit in the subsystem.
        # actual_qubit is its index in the system of interest (if different).
        actual_qubit = qubit_locations[subqubit]
        if genobs_x[subqubit]:
            if genobs_z[subqubit]:
                qc.sdg(actual_qubit)
            qc.h(actual_qubit)
        qc.measure(actual_qubit, obs_creg[clbit])

    return qc


def _get_bases(circuit: QuantumCircuit) -> tuple[list[QPDBasis], list[list[int]]]:
    """Get a list of each unique QPD basis in the circuit and the QPDGate indices."""
    bases = []
    qpd_gate_ids = []
    for i, inst in enumerate(circuit):
        if isinstance(inst.operation, SingleQubitQPDGate):
            raise ValueError(
                "SingleQubitQPDGates are not supported in unseparated circuits."
            )
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            qpd_gate_ids.append([i])

    return bases, qpd_gate_ids
