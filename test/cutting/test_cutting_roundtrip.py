# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from __future__ import annotations

import pytest

import logging

import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library.standard_gates import (
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    XXPlusYYGate,
    XXMinusYYGate,
    CHGate,
    CXGate,
    CYGate,
    CZGate,
    CSGate,
    CSdgGate,
    CSXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
    ECRGate,
    SwapGate,
    iSwapGate,
    DCXGate,
)
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import PauliList, random_unitary
from qiskit.primitives import Estimator

from circuit_knitting.utils.simulation import ExactSampler
from circuit_knitting.cutting import (
    partition_problem,
    cut_gates,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from circuit_knitting.cutting.cutting_experiments import (
    generate_distribution_cutting_experiments,
)
from circuit_knitting.cutting.cutting_reconstruction import reconstruct_distribution
from circuit_knitting.cutting.instructions import Move

logger = logging.getLogger(__name__)


def append_random_unitary(circuit: QuantumCircuit, qubits):
    circuit.append(UnitaryGate(random_unitary(2 ** len(qubits))), qubits)


@pytest.fixture(
    params=[
        [SwapGate()],
        [iSwapGate()],
        [DCXGate()],
        [CXGate()],
        [CYGate()],
        [CZGate()],
        [CHGate()],
        [ECRGate()],
        [CSXGate()],
        [CSXGate().inverse()],
        [CSGate()],
        [CSdgGate()],
        [RYYGate(0.0)],
        [RZZGate(np.pi)],
        [RXXGate(np.pi / 3)],
        [RYYGate(np.pi / 7)],
        [RZZGate(np.pi / 11)],
        [CRXGate(0.0)],
        [CRYGate(np.pi)],
        [CRZGate(np.pi / 2)],
        [CRXGate(np.pi / 3)],
        [CRYGate(np.pi / 7)],
        [CRZGate(np.pi / 11)],
        [RXXGate(np.pi / 3), CRYGate(np.pi / 7)],
        [CPhaseGate(np.pi / 3)],
        [RXXGate(np.pi / 3), CPhaseGate(np.pi / 7)],
        [UnitaryGate(random_unitary(2**2))],
        [RZXGate(np.pi / 5)],
        [XXPlusYYGate(7 * np.pi / 11)],
        [XXMinusYYGate(11 * np.pi / 17)],
        [Move()],
        [Move(), Move()],
    ]
)
def example_circuit(
    request,
) -> tuple[QuantumCircuit, QuantumCircuit, list[list[int]]]:
    """Fixture for an example circuit.

    Returns both the original and one with QPDGates as a tuple.

    Except for the parametrized gates, the system can be separated according to
    the partition labels "AAB".

    """
    qc = QuantumCircuit(3)
    cut_indices = []
    for instruction in request.param:
        if instruction.name == "move" and len(cut_indices) % 2 == 1:
            # We should not entangle qubit 1 with the remainder of the system.
            # In fact, we're also assuming that the previous operation here was
            # a move.
            append_random_unitary(qc, [0])
            append_random_unitary(qc, [1])
        else:
            append_random_unitary(qc, [0, 1])
        append_random_unitary(qc, [2])
        cut_indices.append(len(qc.data))
        qubits = [1, 2]
        if len(cut_indices) % 2 == 0:
            qubits.reverse()
        qc.append(instruction, qubits)
    qc.barrier()
    append_random_unitary(qc, [0, 1])
    qc.barrier()
    append_random_unitary(qc, [2])

    return qc


def test_cutting_exact_reconstruction(example_circuit):
    """Test gate-cut circuit vs original circuit on statevector simulator

    This test uses a statevector simulator to consider the expectation value of
    each of the :math:`2^N` different possible projection operators in the z
    basis at the end of the circuit (or in other words, the precise probability
    of each full-circuit measurement outcome in the limit of infinite shots).
    This test ensures that each such expectation value remains the same under
    the given QPD decomposed gates.
    """
    qc0 = example_circuit
    qc = qc0.copy()

    observables = PauliList(["III", "IIY", "XII", "XYZ", "iZZZ", "-XZI"])
    phases = np.array([(-1j) ** obs.phase for obs in observables])
    observables_nophase = PauliList(["III", "IIY", "XII", "XYZ", "ZZZ", "XZI"])

    estimator = Estimator()
    exact_expvals = (
        estimator.run([qc0] * len(observables), list(observables)).result().values
    )
    subcircuits, bases, subobservables = partition_problem(
        qc, "AAB", observables=observables_nophase
    )
    subexperiments, coefficients = generate_cutting_experiments(
        subcircuits, subobservables, num_samples=np.inf
    )
    if np.random.randint(2):
        # Re-use a single sample
        sampler = ExactSampler()
        samplers = {label: sampler for label in subcircuits.keys()}
    else:
        # One sampler per partition
        samplers = {label: ExactSampler() for label in subcircuits.keys()}
    results = {
        label: sampler.run(subexperiments[label]).result()
        for label, sampler in samplers.items()
    }
    for label in results:
        for i, subexperiment in enumerate(subexperiments[label]):
            results[label].metadata[i]["num_qpd_bits"] = len(subexperiment.cregs[0])
    simulated_expvals = reconstruct_expectation_values(
        results, coefficients, subobservables
    )
    simulated_expvals *= phases

    logger.info("Max error: %f", np.max(np.abs(exact_expvals - simulated_expvals)))

    assert np.allclose(exact_expvals, simulated_expvals, atol=1e-8)


@pytest.fixture(
    params=[
        (8, 4, 1, 0.4),
    ]
)
def example_sampler_circuit(request):
    num_qubits, num_measurements, reps, gate_param = request.param

    circuit0 = EfficientSU2(
        num_qubits=num_qubits, reps=reps, entanglement="circular"
    ).decompose()
    circuit0.assign_parameters([gate_param] * len(circuit0.parameters), inplace=True)

    circuit0.add_register(ClassicalRegister(num_measurements))
    for i in range(num_measurements):
        circuit0.measure(i, i)

    cut_indices = [
        i
        for i, instruction in enumerate(circuit0.data)
        if {circuit0.find_bit(q)[0] for q in instruction.qubits} == {0, num_qubits - 1}
    ]
    circuit1, bases = cut_gates(circuit0, cut_indices)

    return circuit0, circuit1, bases


def test_cutting_exact_distribution_reconstruction(example_sampler_circuit):
    circuit0, circuit1, bases = example_sampler_circuit
    subexperiments, coefficients = generate_distribution_cutting_experiments(
        circuit1, np.inf
    )
    subexperiment_results = ExactSampler().run(subexperiments).result()
    reconstructed = reconstruct_distribution(
        subexperiment_results, circuit1.num_clbits, coefficients
    )
    exact = ExactSampler().run(circuit0).result().quasi_dists[0]
    for k in range(2**circuit1.num_clbits):
        assert reconstructed.get(k, 0.0) == pytest.approx(exact.get(k, 0.0))
