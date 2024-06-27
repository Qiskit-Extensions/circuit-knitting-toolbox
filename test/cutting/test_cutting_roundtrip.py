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
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
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
from qiskit.quantum_info import PauliList, random_unitary
from qiskit.primitives import Estimator
from qiskit_aer.primitives import Sampler

from circuit_knitting.utils.simulation import ExactSampler
from circuit_knitting.cutting import (
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from circuit_knitting.cutting.instructions import Move

logger = logging.getLogger(__name__)


def append_random_unitary(circuit: QuantumCircuit, qubits):
    circuit.unitary(random_unitary(2 ** len(qubits)), qubits)


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
        # XXPlusYYGate, XXMinusYYGate, with some combinations:
        #     beta == 0 or not; and
        #     within |theta| < pi or not
        [XXPlusYYGate(7 * np.pi / 11)],
        [XXPlusYYGate(17 * np.pi / 11, beta=0.4)],
        [XXPlusYYGate(-19 * np.pi / 11, beta=0.3)],
        [XXMinusYYGate(11 * np.pi / 17)],
        [Move()],
        [Move(), Move()],
    ]
)
def example_circuit(request) -> QuantumCircuit:
    """Fixture for an example circuit.

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
    """Test gate-cut circuit vs original circuit on statevector simulator"""
    qc = example_circuit

    observables = PauliList(["III", "IIY", "XII", "XYZ", "iZZZ", "-XZI"])
    phases = np.array([(-1j) ** obs.phase for obs in observables])
    observables_nophase = PauliList(["III", "IIY", "XII", "XYZ", "ZZZ", "XZI"])

    estimator = Estimator()
    exact_expvals = (
        estimator.run([qc] * len(observables), list(observables)).result().values
    )
    subcircuits, bases, subobservables = partition_problem(
        qc, "AAB", observables=observables_nophase
    )
    basis_gate_set = [None, "eagle", "heron"][
        np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
    ]
    subexperiments, coefficients = generate_cutting_experiments(
        subcircuits,
        subobservables,
        num_samples=np.inf,
        basis_gate_set=basis_gate_set,
    )
    if np.random.randint(2):
        # Re-use a single sampler
        sampler = ExactSampler()
        samplers = {label: sampler for label in subcircuits.keys()}
    else:
        # One sampler per partition
        samplers = {label: ExactSampler() for label in subcircuits.keys()}
    results = {
        label: sampler.run(subexperiments[label]).result()
        for label, sampler in samplers.items()
    }
    reconstructed_expvals = reconstruct_expectation_values(
        results, coefficients, subobservables
    )
    reconstructed_expvals *= phases

    logger.info("Max error: %f", np.max(np.abs(exact_expvals - reconstructed_expvals)))

    assert np.allclose(exact_expvals, reconstructed_expvals, atol=1e-8)


@pytest.mark.parametrize(
    "sampler,is_exact_sampler", [(Sampler(), False), (ExactSampler(), True)]
)
def test_sampler_with_identity_subobservable(sampler, is_exact_sampler):
    """This test ensures that the sampler works for a subcircuit with no observable measurements.

    Specifically, that

    - ``Sampler`` does not blow up (Issue #422); and
    - ``ExactSampler`` returns correct results

    This is related to https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/422.
    """
    # Create a circuit to cut
    qc = QuantumCircuit(3)
    append_random_unitary(qc, [0, 1])
    append_random_unitary(qc, [2])
    qc.rxx(np.pi / 3, 1, 2)
    append_random_unitary(qc, [0, 1])
    append_random_unitary(qc, [2])

    # Determine expectation value using cutting
    observables = PauliList(
        ["IIZ"]
    )  # Without the workaround to Issue #422, this observable causes a Sampler error.
    subcircuits, bases, subobservables = partition_problem(
        qc, "AAB", observables=observables
    )
    subexperiments, coefficients = generate_cutting_experiments(
        subcircuits, subobservables, num_samples=np.inf
    )
    samplers = {label: sampler for label in subexperiments.keys()}
    results = {
        label: sampler.run(subexperiments[label]).result()
        for label, sampler in samplers.items()
    }
    reconstructed_expvals = reconstruct_expectation_values(
        results, coefficients, subobservables
    )

    if is_exact_sampler:
        # Determine exact expectation values
        estimator = Estimator()
        exact_expvals = (
            estimator.run([qc] * len(observables), list(observables)).result().values
        )

        logger.info(
            "Max error: %f", np.max(np.abs(exact_expvals - reconstructed_expvals))
        )

        # Ensure both methods yielded equivalent expectation values
        assert np.allclose(exact_expvals, reconstructed_expvals, atol=1e-8)
