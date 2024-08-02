# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for evaluating circuit cutting experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence, Hashable

import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import PauliList

from ..utils.iteration import strict_zip
from ..utils.observable_grouping import ObservableCollection, CommutingObservableGroup
from .qpd import (
    WeightType,
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_weights,
    decompose_qpd_instructions,
)
from .cutting_decomposition import decompose_observables


def generate_cutting_experiments(
    circuits: QuantumCircuit | dict[Hashable, QuantumCircuit],
    observables: PauliList | dict[Hashable, PauliList],
    num_samples: int | float,
) -> tuple[
    list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]],
    list[tuple[float, WeightType]],
]:
    r"""
    Generate cutting subexperiments and their associated coefficients.

    If the input, ``circuits``, is a :class:`QuantumCircuit` instance, the
    output subexperiments will be contained within a 1D array, and ``observables`` is
    expected to be a :class:`PauliList` instance.

    If the input circuit and observables are specified by dictionaries with partition labels
    as keys, the output subexperiments will be returned as a dictionary which maps each
    partition label to a 1D array containing the subexperiments associated with that partition.

    In both cases, the subexperiment lists are ordered as follows:

        :math:`[sample_{0}observable_{0}, \ldots, sample_{0}observable_{N-1}, sample_{1}observable_{0}, \ldots, sample_{M-1}observable_{N-1}]`

    The coefficients will always be returned as a 1D array -- one coefficient for each unique sample.

    Args:
        circuits: The circuit(s) to partition and separate
        observables: The observable(s) to evaluate for each unique sample
        num_samples: The number of samples to draw from the quasi-probability distribution. If set
            to infinity, the weights will be generated rigorously rather than by sampling from
            the distribution.
    Returns:
        A tuple containing the cutting experiments and their associated coefficients.
        If the input circuits is a :class:`QuantumCircuit` instance, the output subexperiments
        will be a sequence of circuits -- one for every unique sample and observable. If the
        input circuits are represented as a dictionary keyed by partition labels, the output
        subexperiments will also be a dictionary keyed by partition labels and containing
        the subexperiments for each partition.
        The coefficients are always a sequence of length-2 tuples, where each tuple contains the
        coefficient and the :class:`WeightType`. Each coefficient corresponds to one unique sample.

    Raises:
        ValueError: ``num_samples`` must be at least one.
        ValueError: ``circuits`` and ``observables`` are incompatible types
        ValueError: :class:`SingleQubitQPDGate` instances must have their cut ID
            appended to the gate label so they may be associated with other gates belonging
            to the same cut.
        ValueError: :class:`SingleQubitQPDGate` instances are not allowed in unseparated circuits.
    """
    if observables is None:
        # FIXME: ensure there's at least one measurement in at least one of the subsystems.
        # And it's kind of weird, because any subcircuit without measurements does not need to be run!!
        pass
    else:
        if isinstance(circuits, QuantumCircuit) and not isinstance(
            observables, PauliList
        ):
            raise ValueError(
                "If the input circuits is a QuantumCircuit, the observables must be a PauliList."
            )
        if isinstance(circuits, dict) and not isinstance(observables, dict):
            raise ValueError(
                "If the input circuits are contained in a dictionary keyed by partition labels, the input observables must also be represented by such a dictionary."
            )
    if not num_samples >= 1:
        raise ValueError("num_samples must be at least 1.")

    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on the format of the execute_experiments input args, but the 2nd half of this function
    # can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_dict: dict[Hashable, QuantumCircuit] = {"A": circuits}
        if observables is None:
            subsystem_observables = None
        else:
            subobservables_by_subsystem = decompose_observables(
                observables, "A" * len(observables[0])
            )
            subsystem_observables = {
                label: ObservableCollection(subobservables)
                for label, subobservables in subobservables_by_subsystem.items()
            }
        # Gather the unique bases from the circuit
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]] = {"A": qpd_gate_ids}

    else:
        is_separated = True
        subcircuit_dict = circuits
        # Gather the unique bases across the subcircuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(
            subcircuit_dict
        )
        bases = _get_bases_by_partition(subcircuit_dict, subcirc_qpd_gate_ids)

        # Create the commuting observable groups
        if observables is None:
            subsystem_observables = None
        else:
            subsystem_observables = {
                label: ObservableCollection(so) for label, so in observables.items()
            }
    # Sample the joint quasiprobability decomposition
    random_samples = generate_qpd_weights(bases, num_samples=num_samples)

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the output experiments and their respective coefficients
    subexperiments_dict: dict[Hashable, list[QuantumCircuit]] = defaultdict(list)
    coefficients: list[tuple[float, WeightType]] = []
    for z, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples):
        actual_coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
        coefficients.append((sampled_coeff, weight_type))
        map_ids_tmp = map_ids
        for label, subcircuit in subcircuit_dict.items():
            if is_separated:
                map_ids_tmp = tuple(map_ids[j] for j in subcirc_map_ids[label])
            if subsystem_observables is None:
                new_qc = decompose_qpd_instructions(
                    subcircuit, subcirc_qpd_gate_ids[label], map_ids_tmp, inplace=False
                )
                subexperiments_dict[label].append(new_qc)
            else:
                so = subsystem_observables[label]
                for j, cog in enumerate(so.groups):
                    new_qc = _append_measurement_register(subcircuit, cog)
                    decompose_qpd_instructions(
                        new_qc, subcirc_qpd_gate_ids[label], map_ids_tmp, inplace=True
                    )
                    _append_measurement_circuit(new_qc, cog, inplace=True)
                    subexperiments_dict[label].append(new_qc)

    # Remove initial and final resets from the subexperiments.  This will
    # enable the `Move` operation to work on backends that don't support
    # `Reset`, as long as qubits are not re-used.  See
    # https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/452.
    # While we are at it, we also consolidate each run of multiple resets
    # (which can arise when re-using qubits) into a single reset.
    for subexperiments in subexperiments_dict.values():
        for circ in subexperiments:
            _remove_resets_in_zero_state(circ)
            _remove_final_resets(circ)
            _consolidate_resets(circ)

    # If the input was a single quantum circuit, return the subexperiments as a list
    subexperiments_out: list[QuantumCircuit] | dict[Hashable, list[QuantumCircuit]] = (
        dict(subexperiments_dict)
    )
    assert isinstance(subexperiments_out, dict)
    if isinstance(circuits, QuantumCircuit):
        assert len(subexperiments_out.keys()) == 1
        subexperiments_out = list(subexperiments_dict.values())[0]

    return subexperiments_out, coefficients


def _get_mapping_ids_by_partition(
    circuits: dict[Hashable, QuantumCircuit],
) -> tuple[dict[Hashable, list[list[int]]], dict[Hashable, list[int]]]:
    """Get indices to the QPD gates in each subcircuit and relevant map ids."""
    # Collect QPDGate id's and relevant map id's for each subcircuit
    subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]] = {}
    subcirc_map_ids: dict[Hashable, list[int]] = {}
    for label, circ in circuits.items():
        subcirc_qpd_gate_ids[label] = []
        subcirc_map_ids[label] = []
        for i, inst in enumerate(circ.data):
            if isinstance(inst.operation, SingleQubitQPDGate):
                try:
                    decomp_id = int(inst.operation.label.split("_")[-1])
                except (AttributeError, ValueError) as ex:
                    raise ValueError(
                        "SingleQubitQPDGate instances in input circuit(s) must have their "
                        'labels suffixed with "_<id>", where <id> is the index of the cut '
                        "relative to the other cuts in the circuit. For example, all "
                        "SingleQubitQPDGates belonging to the same cut, N, should have labels "
                        ' formatted as "<your_label>_N". This allows SingleQubitQPDGates '
                        "belonging to the same cut to be sampled jointly."
                    ) from ex
                subcirc_qpd_gate_ids[label].append([i])
                subcirc_map_ids[label].append(decomp_id)

    return subcirc_qpd_gate_ids, subcirc_map_ids


def _get_bases_by_partition(
    circuits: dict[Hashable, QuantumCircuit],
    subcirc_qpd_gate_ids: dict[Hashable, list[list[int]]],
) -> list[QPDBasis]:
    """Get a list of each unique QPD basis across the subcircuits."""
    # Collect the bases corresponding to each decomposed operation
    bases_dict = {}
    for label, subcirc in subcirc_qpd_gate_ids.items():
        circuit = circuits[label]
        for basis_id in subcirc:
            decomp_id = int(circuit.data[basis_id[0]].operation.label.split("_")[-1])
            bases_dict[decomp_id] = circuit.data[basis_id[0]].operation.basis
    bases = [bases_dict[key] for key in sorted(bases_dict.keys())]

    return bases


def _get_bases(circuit: QuantumCircuit) -> tuple[list[QPDBasis], list[list[int]]]:
    """Get a list of each unique QPD basis in the circuit and the QPDGate indices."""
    bases = []
    qpd_gate_ids = []
    for i, inst in enumerate(circuit):
        if isinstance(inst.operation, SingleQubitQPDGate):
            raise ValueError(
                "SingleQubitQPDGates are not supported in unseparable circuits."
            )
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            qpd_gate_ids.append([i])

    return bases, qpd_gate_ids


def _append_measurement_register(
    qc: QuantumCircuit,
    cog: CommutingObservableGroup,
    /,
    *,
    inplace: bool = False,
):
    """Append a new classical register for the given ``CommutingObservableGroup``.

    The new register will be named ``"observable_measurements"`` and will be
    the final register in the returned circuit, i.e. ``retval.cregs[-1]``.

    Args:
        qc: The quantum circuit
        cog: The commuting observable set for which to construct measurements
        inplace: Whether to operate on the circuit in place (default: ``False``)

    Returns:
        The modified circuit
    """
    if not inplace:
        qc = qc.copy()

    pauli_indices = _get_pauli_indices(cog)

    obs_creg = ClassicalRegister(len(pauli_indices), name="observable_measurements")
    qc.add_register(obs_creg)

    return qc


def _append_measurement_circuit(
    qc: QuantumCircuit,
    cog: CommutingObservableGroup,
    /,
    *,
    qubit_locations: Sequence[int] | None = None,
    inplace: bool = False,
) -> QuantumCircuit:
    """Append measurement instructions for the given ``CommutingObservableGroup``.

    The measurement results will be placed in a register with the name
    ``"observable_measurements"``.  Such a register can be created by calling
    :func:`_append_measurement_register` before calling the current function.

    Args:
        qc: The quantum circuit
        cog: The commuting observable set for which to construct measurements
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
        if len(qubit_locations) != cog.general_observable.num_qubits:
            raise ValueError(
                f"qubit_locations has {len(qubit_locations)} element(s) but the "
                f"observable(s) have {cog.general_observable.num_qubits} qubit(s)."
            )

    # Find observable_measurements register
    for reg in qc.cregs:
        if reg.name == "observable_measurements":
            obs_creg = reg
            break
    else:
        raise ValueError('Cannot locate "observable_measurements" register')

    pauli_indices = _get_pauli_indices(cog)

    if obs_creg.size != len(pauli_indices):
        raise ValueError(
            '"observable_measurements" register is the wrong size '
            "for the given commuting observable group "
            f"({obs_creg.size} != {len(pauli_indices)})"
        )

    if not inplace:
        qc = qc.copy()

    # Append the appropriate measurements to qc
    #
    # Implement the necessary basis rotations and measurements, as
    # in BackendEstimator._measurement_circuit().
    genobs_x = cog.general_observable.x
    genobs_z = cog.general_observable.z
    for clbit, subqubit in enumerate(pauli_indices):
        # subqubit is the index of the qubit in the subsystem.
        # actual_qubit is its index in the system of interest (if different).
        actual_qubit = qubit_locations[subqubit]
        if genobs_x[subqubit]:
            if genobs_z[subqubit]:
                # Rotate Y basis to Z basis
                qc.sx(actual_qubit)
            else:
                # Rotate X basis to Z basis
                qc.h(actual_qubit)
        # Measure in Z basis
        qc.measure(actual_qubit, obs_creg[clbit])

    return qc


def _get_pauli_indices(cog: CommutingObservableGroup) -> list[int]:
    """Return the indices to qubits to be measured."""
    # If the circuit has no measurements, the Sampler will fail.  So, we
    # measure one qubit as a temporary workaround to
    # https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/422
    pauli_indices = cog.pauli_indices
    if not pauli_indices:
        pauli_indices = [0]
    return pauli_indices


def _consolidate_resets(
    circuit: QuantumCircuit, inplace: bool = True
) -> QuantumCircuit:
    """Consolidate redundant resets into a single reset."""
    if not inplace:  # pragma: no cover
        circuit = circuit.copy()

    # Keep up with whether the previous instruction on a given qubit was a reset
    resets = [False] * circuit.num_qubits

    # Remove resets which are immediately following other resets
    remove_ids = []
    for i, inst in enumerate(circuit.data):
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "reset":
            if resets[qargs[0]]:
                remove_ids.append(i)
            else:
                resets[qargs[0]] = True
        else:
            for q in qargs:
                resets[q] = False

    for i in sorted(remove_ids, reverse=True):
        del circuit.data[i]

    return circuit


def _remove_resets_in_zero_state(
    circuit: QuantumCircuit, inplace: bool = True
) -> QuantumCircuit:
    """Remove resets if they are the first instruction on a qubit."""
    if not inplace:  # pragma: no cover
        circuit = circuit.copy()

    # Keep up with which qubits have at least one non-reset instruction
    active_qubits: set[int] = set()
    remove_ids = []
    for i, inst in enumerate(circuit.data):
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "reset":
            if qargs[0] not in active_qubits:
                remove_ids.append(i)
        else:
            for q in qargs:
                active_qubits.add(q)
            # Early terminate once all qubits have become active
            if len(active_qubits) == circuit.num_qubits:
                break

    for i in sorted(remove_ids, reverse=True):
        del circuit.data[i]

    return circuit


def _remove_final_resets(
    circuit: QuantumCircuit, inplace: bool = True
) -> QuantumCircuit:
    """Remove resets if they are the final instruction on a qubit."""
    if not inplace:  # pragma: no cover
        circuit = circuit.copy()

    # Keep up with whether we are at the end of a qubit
    # We iterate in reverse, so all qubits begin in the "end" state
    qubit_ended = set(range(circuit.num_qubits))
    remove_ids = []
    num_inst = len(circuit.data)
    for i, inst in enumerate(reversed(circuit.data)):
        qargs = [circuit.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "reset":
            if qargs[0] in qubit_ended:
                remove_ids.append(num_inst - 1 - i)
        else:
            for q in qargs:
                qubit_ended.discard(q)
            # Early terminate once all qubits have been touched
            if not qubit_ended:
                break

    for i in sorted(remove_ids, reverse=True):
        del circuit.data[i]

    return circuit
