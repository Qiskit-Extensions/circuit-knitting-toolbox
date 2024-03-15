# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains functions for executing subcircuits."""

from __future__ import annotations

import itertools
import copy
from typing import Sequence, Any
from multiprocessing.pool import ThreadPool

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from qiskit.primitives import BaseSampler, Sampler as TestSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options


def run_subcircuit_instances(
    subcircuits: Sequence[QuantumCircuit],
    subcircuit_instances: dict[int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]],
    service: QiskitRuntimeService | None = None,
    backend_names: Sequence[str] | None = None,
    options: Sequence[Options] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Execute all provided subcircuits.

    Using the backend(s) provided, this executes all the subcircuits to generate the
    resultant probability vectors.
    subcircuit_instance_probs[subcircuit_idx][subcircuit_instance_idx] = measured probability

    Args:
        subcircuits: The list of subcircuits to execute
        subcircuit_instances: Dictionary containing information about each of the
            subcircuit instances
        service: The runtime service
        backend_names: The backend(s) used to execute the subcircuits
        options: Options for the runtime execution of subcircuits

    Returns:
        The probability vectors from each of the subcircuit instances
    """
    if backend_names and options:
        if len(backend_names) != len(options):
            raise AttributeError(
                f"The list of backend names is length ({len(backend_names)}), "
                f"but the list of options is length ({len(options)}). It is ambiguous "
                "how these options should be applied."
            )
    if service:
        if backend_names:
            backend_names_repeated: list[str | None] = [
                backend_names[i % len(backend_names)] for i, _ in enumerate(subcircuits)
            ]
            if options is None:
                options_repeated: list[Options | None] = [None] * len(
                    backend_names_repeated
                )
            else:
                options_repeated = [
                    options[i % len(options)] for i, _ in enumerate(subcircuits)
                ]
        else:
            backend_names_repeated = ["ibmq_qasm_simulator"] * len(subcircuits)
            if options:
                options_repeated = [options[0]] * len(subcircuits)
            else:
                options_repeated = [None] * len(subcircuits)
    else:
        backend_names_repeated = [None] * len(subcircuits)
        options_repeated = [None] * len(subcircuits)

    subcircuit_instance_probs: dict[int, dict[int, np.ndarray]] = {}
    with ThreadPool() as pool:
        args = [
            [
                subcircuit_instances[subcircuit_idx],
                subcircuit,
                service,
                backend_names_repeated[subcircuit_idx],
                options_repeated[subcircuit_idx],
            ]
            for subcircuit_idx, subcircuit in enumerate(subcircuits)
        ]
        subcircuit_instance_probs_list = pool.starmap(_run_subcircuit_batch, args)

        for i, partition_batch in enumerate(subcircuit_instance_probs_list):
            subcircuit_instance_probs[i] = partition_batch

    return subcircuit_instance_probs


def mutate_measurement_basis(meas: tuple[str, ...]) -> list[tuple[Any, ...]]:
    """
    Change of basis for all identity measurements.

    For every identity measurement, it is split into an I and Z measurement.
    I and Z measurement basis correspond to the same logical circuit.

    Args:
        meas: The current measurement bases

    Returns:
        The update measurement bases
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas_out = list(itertools.product(*mutated_meas))

        return mutated_meas_out


def modify_subcircuit_instance(
    subcircuit: QuantumCircuit, init: tuple[str, ...], meas: tuple[str, ...]
) -> QuantumCircuit:
    """
    Modify the initialization and measurement bases for a given subcircuit.

    Args:
        subcircuit: The subcircuit to be modified
        init: The current initializations
        meas: The current measement bases

    Returns:
        The updated circuit, modified so the initialziation
        and measurement operators are all in the standard computational basis

    Raises:
        Exeption: One of the inits or meas's are not an acceptable string
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)

    return subcircuit_instance_circuit


def run_subcircuits_using_sampler(
    subcircuits: Sequence[QuantumCircuit],
    sampler: BaseSampler,
) -> list[np.ndarray]:
    """
    Execute the subcircuit(s).

    Args:
        subcircuit: The subcircuits to be executed
        sampler: The Sampler to use for executions

    Returns:
        The probability distributions
    """
    for subcircuit in subcircuits:
        if subcircuit.num_clbits == 0:
            subcircuit.measure_all()

    quasi_dists = sampler.run(circuits=subcircuits).result().quasi_dists

    all_probabilities_out = []
    for i, qd in enumerate(quasi_dists):
        probabilities = qd.nearest_probability_distribution()
        probabilities_out = np.zeros(2 ** subcircuits[i].num_qubits, dtype=float)

        for state in probabilities:
            probabilities_out[state] = probabilities[state]
        all_probabilities_out.append(probabilities_out)

    return all_probabilities_out


def run_subcircuits(
    subcircuits: Sequence[QuantumCircuit],
    service: QiskitRuntimeService | None = None,
    backend_name: str | None = None,
    options: Options | None = None,
) -> list[np.ndarray]:
    """
    Execute the subcircuit(s).

    Args:
        subcircuit: The subcircuits to be executed
        service: The runtime service
        backend_name: The backend used to execute the subcircuits
        options: Options for the runtime execution of subcircuits

    Returns:
        The probability distributions
    """
    if service is not None:
        session = Session(service=service, backend=backend_name)
        sampler = Sampler(session=session, options=options)
        # Transpile circuits
        backend = service.backend(session.backend())
        pass_manager = generate_preset_pass_manager(
            optimization_level=1, backend=backend
        )
        subcircuits = pass_manager.run(subcircuits)
    else:
        sampler = TestSampler(options=options)

    return run_subcircuits_using_sampler(subcircuits, sampler)


def measure_prob(unmeasured_prob: np.ndarray, meas: tuple[Any, ...]) -> np.ndarray:
    """
    Compute the effective probability distribution from the subcircuit distribution.

    Args:
        unmeasured_prob: The outputs of the subcircuit execution
        meas: The measurement bases

    Returns:
        The updated measured probability distribution
    """
    if meas.count("comp") == len(meas):
        return np.array(unmeasured_prob)
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            # TODO: Add states merging here. Change effective_state to merged_bin
            measured_prob[effective_state] += sigma * p

        return measured_prob


def measure_state(full_state: int, meas: tuple[Any, ...]) -> tuple[int, int]:
    """
    Compute the corresponding effective_state for the given full_state.

    Measured in basis `meas`. Returns sigma (int), effective_state (int) where sigma = +-1

    Args:
        full_state: The current state (in decimal form)
        meas: The measurement bases

    Returns:
        Sigma (defined by the parity of non computational basis 1 measurements) and
        the effective state (defined by the measurements in the computational basis)
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0

    return sigma, effective_state


def _run_subcircuit_batch(
    subcircuit_instance: dict[tuple[tuple[str, ...], tuple[Any, ...]], int],
    subcircuit: QuantumCircuit,
    service: QiskitRuntimeService | None = None,
    backend_name: str | None = None,
    options: Options | None = None,
):
    """
    Execute a circuit using qiskit runtime.

    Args:
        subcircuit_instances: Dictionary containing information about each of the
            subcircuit instances
        subcircuit: The subcircuit to execute
        service: The runtime service
        backend_name: The backends used to execute the subcircuit
        options: Options for the runtime execution of subcircuit

    Returns:
        The measurement probabilities for the subcircuit batch, as calculated from the
        runtime execution
    """
    subcircuit_instance_probs = {}
    circuits_to_run = []

    # For each circuit associated with a given subcircuit
    for init_meas in subcircuit_instance:
        subcircuit_instance_idx = subcircuit_instance[init_meas]

        # Collect all of the circuits we need to evaluate, ensuring we don't have duplicates
        if subcircuit_instance_idx not in subcircuit_instance_probs:
            modified_subcircuit_instance = modify_subcircuit_instance(
                subcircuit=subcircuit,
                init=init_meas[0],
                meas=tuple(init_meas[1]),
            )
            circuits_to_run.append(modified_subcircuit_instance)
            mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
            for meas in mutated_meas:
                mutated_subcircuit_instance_idx = subcircuit_instance[
                    (init_meas[0], meas)
                ]
                # Set a placeholder in the probability dict to prevent duplicate circuits to the Sampler
                subcircuit_instance_probs[mutated_subcircuit_instance_idx] = np.array(
                    [0.0]
                )

    # Run all of our circuits in one batch
    subcircuit_inst_probs = run_subcircuits(
        circuits_to_run,
        service=service,
        backend_name=backend_name,
        options=options,
    )

    # Calculate the measured probabilities
    unique_subcircuit_check = {}
    i = 0
    for init_meas in subcircuit_instance:
        subcircuit_instance_idx = subcircuit_instance[init_meas]
        if subcircuit_instance_idx not in unique_subcircuit_check:
            subcircuit_inst_prob = subcircuit_inst_probs[i]
            i = i + 1
            mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
            for meas in mutated_meas:
                measured_prob = measure_prob(
                    unmeasured_prob=subcircuit_inst_prob, meas=meas
                )
                mutated_subcircuit_instance_idx = subcircuit_instance[
                    (init_meas[0], meas)
                ]
                subcircuit_instance_probs[mutated_subcircuit_instance_idx] = (
                    measured_prob
                )
                unique_subcircuit_check[mutated_subcircuit_instance_idx] = True

    return subcircuit_instance_probs
