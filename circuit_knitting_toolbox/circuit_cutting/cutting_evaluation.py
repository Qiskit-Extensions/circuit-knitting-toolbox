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

from collections.abc import Sequence
from itertools import chain
from multiprocessing.pool import ThreadPool

import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import PauliList
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.iteration import strict_zip
from .qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_samples,
    decompose_qpd_instructions,
)
from .cutting_decomposition import decompose_observables


def execute_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    observables: PauliList | dict[str | int, PauliList],
    num_samples: int,
    backends: Backend | Sequence[Backend] | None = None,
    shots: int = 1000,
) -> tuple[list[list[list[tuple[dict[str, int], int]]]], Sequence[float]]:
    """
    Generate the sampled circuits, append the observables, and run the sub-experiments.

    :param circuits: The circuit(s) resulting from decomposing nonlocal gates
    :type circuits: QuantumCircuit | dict
    :param observables: The observable(s) corresponding to the circuit(s). If
        a :class:`~qiskit.QuantumCircuit` is submitted for the ``circuits`` argument,
        a :class:`~qiskit.quantum_info.PauliList` is expected; otherwise, a mapping
        from partition label to subobservables is expected.
    :type observables: PauliList | dict
    :param num_samples: The number of samples to draw from the quasiprobability distribution
    :type num_samples: int
    :param backends: Backend(s) on which to run the sub-experiments. If no backend is selected,
        the ``AerSimulator`` from ``Qiskit Aer`` will be used. Experiments corresponding
        to the same qubit partition will be run on the same backend.
    :type backends: Backend | Sequence[Backend]
    :param shots: The number of shots to run for each experiment
    :type shots: int

    :return: A tuple containing a 3D list of length-2 tuples holding the counts and QPD bit
        information for each sub-experiment as well as the coefficients corresponding to each unique sample
    :rtype: tuple[list, Sequence[int]]

    :raises ValueError: The number of requested samples must be positive.
    :raises ValueError: The types of ``circuits`` and ``observables`` arguments are incompatible
    :raises ValueError: SingleQubitQPDGates are not supported in unseparable circuits
    """
    if num_samples <= 0:
        raise ValueError("The number of requested samples must be positive.")

    if isinstance(circuits, dict) and not isinstance(observables, dict):
        raise ValueError(
            "If a partition mapping (dict[label, subcircuit]) is passed as the "
            "circuits argument, a partition mapping (dict[label, subobservables]) "
            "is expected as the observables argument."
        )

    if isinstance(circuits, QuantumCircuit) and isinstance(observables, dict):
        raise ValueError(
            "If a QuantumCircuit is passed as the circuits argument, a PauliList "
            "is expected as the observables argument."
        )

    # Generate the sub-experiments to run on backend
    (
        subexperiments,
        coefficients,
        sampled_frequencies,
    ) = _generate_cutting_experiments(
        circuits,
        observables,
        num_samples,
    )

    # Create a rotating list of the backends of length equal to the number of partitions
    num_partitions = len(subexperiments[0])
    backends_repeated = _get_rotating_backends_list(backends, num_partitions)

    # Run each partition's sub-experiments within its own thread
    with ThreadPool() as pool:
        args = [
            [
                [sample[i] for sample in subexperiments],
                backends_repeated[i],
                shots,
            ]
            for i in range(num_partitions)
        ]
        counts_by_partition = pool.starmap(_run_experiments_batch, args)

    # Reformat the counts to match the shape of the input before returning
    num_unique_samples = len(subexperiments)
    counts: list[list[list[tuple[dict[str, int], int]]]] = [
        [] for _ in range(num_unique_samples)
    ]
    for i in range(num_unique_samples):
        for partition in counts_by_partition:
            counts[i].append(partition[i])

    return counts, coefficients


def append_measurement_circuit(
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

    :param qc: The quantum circuit.
    :type qc: QuantumCircuit
    :param cog: The commuting observable set for
        which to construct measurements.
    :type cog: CommutingObservableGroup
    :param qubit_locations: A ``Sequence`` whose length is the
        number of qubits in the observables, where each element holds that
        qubit's corresponding index in the circuit.  By default, the
        circuit and observables are assumed to have the same number of
        qubits, and the idenity map (i.e., ``range(qc.num_qubits)``) is
        used.
    :type qubit_locations: Sequence[int] | None
    :param inplace: Whether to operate on the circuit in place (default: False).
    :type inplace: bool

    :return: The new or modified circuit.
    :rtype: QuantumCircuit
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


def _generate_cutting_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    observables: PauliList | dict[str | int, PauliList],
    num_samples: int,
) -> tuple[list[list[list[QuantumCircuit]]], list[float], list[int]]:
    """Generate all the experiments to run on the backend and their associated coefficients."""
    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on whether the decomposed circuit was separated into subcircuits before calling
    # execute_experiments, but the 2nd half of this function can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_list = [circuits]
        _, subsystem_observables = decompose_observables(
            observables, "A" * len(observables[0])
        )
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
    random_samples = generate_qpd_samples(bases, num_samples=num_samples)

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the outputs -- sub-experiments, coefficients, and frequencies
    subexperiments: list[list[list[QuantumCircuit]]] = []
    coefficients = []
    sampled_frequencies = []
    for z, (map_ids, sample_weight) in enumerate(sorted_samples):
        redundancy = sample_weight[0]
        subexperiments.append([])
        actual_coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
        coefficients.append(sampled_coeff)
        sampled_frequencies.append(redundancy)
        for i, (subcircuit, label) in enumerate(
            strict_zip(subcircuit_list, sorted(subsystem_observables.keys()))
        ):
            map_ids_tmp = map_ids
            if is_separated:
                map_ids_tmp = tuple(map(map_ids.__getitem__, subcirc_map_ids[i]))
            decomp_qc = decompose_qpd_instructions(
                subcircuit, subcirc_qpd_gate_ids[i], map_ids_tmp
            )
            subexperiments[-1].append([])
            so = subsystem_observables[label]
            for j, cog in enumerate(so.groups):
                meas_qc = append_measurement_circuit(decomp_qc, cog)
                # Should have strictly 2 classical registers, "qpd measurements" and "observable_measurements"
                subexperiments[-1][-1].append(meas_qc)

    return subexperiments, coefficients, sampled_frequencies


def _run_experiments_batch(
    subexperiments: Sequence[Sequence[QuantumCircuit]],
    backend: Backend,
    shots: int,
) -> list[list[tuple[dict[str, int], int]]]:
    """Run subexperiments on the backend."""
    counts: list[list[tuple[dict[str, int], int]]] = []
    num_qpd_bits_flat = []

    # Run all the experiments in one big batch
    experiments_flat = list(chain.from_iterable(subexperiments))
    for circ in experiments_flat:
        if len(circ.cregs) < 1 or circ.cregs[-1].name != "observable_measurements":
            raise ValueError(
                "Circuit's last register is expected to be named 'observable_measurements'."
            )
        if len(circ.cregs) > 1 and circ.cregs[-2].name != "qpd_measurements":
            raise ValueError(
                "Circuit's next-to-last register is expected to be named 'qpd_measurements'."
            )

        if len(circ.cregs) < 2:
            num_qpd_bits_flat.append(0)  # No gates decomposed to a measurement
        else:
            num_qpd_bits_flat.append(len(circ.cregs[-2]))

    # Run all of the batched experiments
    counts_flat = backend.run(experiments_flat, shots=shots).result().get_counts()

    # Reshape the output data to match the input
    counts_reshaped = np.reshape(counts_flat, np.shape(subexperiments))
    num_qpd_bits = np.reshape(num_qpd_bits_flat, np.shape(subexperiments))

    # Create the counts tuples, which include the number of QPD measurement bits
    counts = [[] for _ in range(len(subexperiments))]
    for i, sample in enumerate(counts_reshaped):
        for j, count_dict in enumerate(sample):
            counts[i].append((count_dict, num_qpd_bits[i][j]))

    return counts


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
                decomp_id = int(inst.operation.label.split("_")[-1])
                decomp_ids.add(decomp_id)
                subcirc_qpd_gate_ids[-1].append([i])
                subcirc_map_ids[-1].append(decomp_id)

    return subcirc_qpd_gate_ids, subcirc_map_ids


def _get_bases_by_partition(
    circuits: Sequence[QuantumCircuit], subcirc_qpd_gate_ids: list[list[list[int]]]
) -> list[QPDBasis]:
    """Get a list of each unique QPD basis across the subcircuits."""
    # Collect the bases corresponding to each decomposed operation
    bases_dict = {}
    for i, subcirc in enumerate(subcirc_qpd_gate_ids):
        for basis_id in subcirc:
            decomp_id = int(
                circuits[i].data[basis_id[0]].operation.label.split("_")[-1]
            )
            bases_dict[decomp_id] = circuits[i].data[basis_id[0]].operation.basis
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


def _get_rotating_backends_list(
    backends: Backend | Sequence[Backend] | None, num_partitions: int
) -> Sequence[Backend]:
    """Return a list of backends, one for each partition."""
    if isinstance(backends, Backend):
        backends = [backends]
    if backends is None:
        backends = [AerSimulator()]
    return [backends[i % len(backends)] for i in range(num_partitions)]
