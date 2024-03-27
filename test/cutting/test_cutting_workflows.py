# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of various cutting workflows, particularly with regard to flexibility in transpilation."""

import pytest
from copy import deepcopy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, CXGate
from qiskit.quantum_info import PauliList
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

try:
    from qiskit.primitives import BackendSamplerV2
except ImportError:
    backendsamplerv2_available = False
else:
    backendsamplerv2_available = True
from qiskit_aer.primitives import Sampler
from qiskit_aer import AerSimulator

from circuit_knitting.cutting.qpd.instructions import SingleQubitQPDGate
from circuit_knitting.cutting.qpd import QPDBasis
from circuit_knitting.cutting.instructions import CutWire, Move
from circuit_knitting.cutting import (
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
    cut_wires,
    expand_observables,
)


def test_transpile_before_realizing_basis_id():
    """Test a workflow where a :class:`.SingleQubitQPDGate` is passed through the transpiler."""
    num_qubits = 4
    circuit = EfficientSU2(num_qubits, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII"])
    subcircuits, bases, subobservables = partition_problem(
        circuit=circuit, partition_labels="AABB", observables=observables
    )

    # Create a fake backend, and modify the target gate set so it thinks a
    # SingleQubitQPDGate is allowed.
    backend = GenericBackendV2(num_qubits=num_qubits)
    target = deepcopy(backend.target)
    sample_qpd_instruction = SingleQubitQPDGate(QPDBasis.from_instruction(CXGate()), 1)
    target.add_instruction(
        sample_qpd_instruction,
        {(i,): None for i in range(target.num_qubits)},
    )
    pass_manager = generate_preset_pass_manager(3, target=target)

    # Pass each subcircuit through the pass manager.
    subcircuits = {
        label: pass_manager.run(subcircuits["A"])
        for label, circuit in subcircuits.items()
    }


@pytest.mark.parametrize(
    "label1,label2",
    [
        ("foo", frozenset({1, 2, 4})),
        (42, "bar"),
    ],
)
def test_exotic_labels(label1, label2):
    """Test workflow with labels of non-uniform type."""
    circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    subcircuits, bases, subobservables = partition_problem(
        circuit=circuit,
        partition_labels=[label1, label1, label2, label2],
        observables=observables,
    )
    assert set(subcircuits.keys()) == {label1, label2}

    subexperiments, coefficients = generate_cutting_experiments(
        subcircuits, subobservables, num_samples=1500
    )
    assert subexperiments.keys() == subcircuits.keys()

    samplers = {
        label1: Sampler(run_options={"shots": 10}),
        label2: Sampler(run_options={"shots": 10}),
    }
    results = {
        label: sampler.run(subexperiments[label]).result()
        for label, sampler in samplers.items()
    }

    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    assert len(reconstructed_expvals) == len(observables)


def test_workflow_with_unused_qubits():
    """Issue #218"""
    qc = QuantumCircuit(2)
    subcircuits, _, subobservables = partition_problem(
        circuit=qc, partition_labels="AB", observables=PauliList(["XX"])
    )
    generate_cutting_experiments(
        subcircuits,
        subobservables,
        num_samples=10,
    )


def test_wire_cut_workflow_without_reused_qubits():
    """Test no resets in subexperiments when wire cut workflow has no re-used qubits."""
    qc = QuantumCircuit(2)
    qc.h(range(2))
    qc.cx(0, 1)
    qc.append(CutWire(), [0])
    qc.cx(1, 0)

    observables = PauliList(["IZ", "ZI", "ZZ", "XX"])

    qc_1 = cut_wires(qc)
    assert qc_1.num_qubits == 3

    observables_1 = expand_observables(observables, qc, qc_1)

    partitioned_problem = partition_problem(circuit=qc_1, observables=observables_1)

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned_problem.subcircuits,
        observables=partitioned_problem.subobservables,
        num_samples=np.inf,
    )

    for subsystem_subexpts in subexperiments.values():
        for subexpt in subsystem_subexpts:
            assert "reset" not in subexpt.count_ops()


def test_wire_cut_workflow_with_reused_qubits():
    """Test at most a single reset in subexperiments when wire cut workflow has a single re-used qubit."""
    qc = QuantumCircuit(8)
    for i in [*range(4), *range(5, 8)]:
        qc.rx(np.pi / 4, i)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.append(Move(), [3, 4])
    qc.cx(4, 5)
    qc.cx(4, 6)
    qc.cx(4, 7)
    qc.append(Move(), [4, 3])
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(2, 3)

    observables = PauliList(["ZIIIIIII", "IIIIZIII", "IIIIIIIZ"])

    partitioned_problem = partition_problem(
        circuit=qc, partition_labels="AAAABBBB", observables=observables
    )

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned_problem.subcircuits,
        observables=partitioned_problem.subobservables,
        num_samples=np.inf,
    )

    # The initial circuit had a single instance of qubit re-use with a Move
    # instruction.  Each A subexperiment should have a single reset, and each B
    # subexperiment should be free of resets.
    for subexpt in subexperiments["A"]:
        assert subexpt.count_ops()["reset"] == 1
    for subexpt in subexperiments["B"]:
        assert "reset" not in subexpt.count_ops()


@pytest.mark.skipif(
    not backendsamplerv2_available,
    reason="BackendSamplerV2 is not available in Qiskit < 1.1",
)
def test_reconstruction_with_samplerv2():
    """Smoke test for reconstruction using samplerv2"""
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.ryy(0.1, 1, 2)
    qc.h(range(4))

    observables = PauliList(["ZIII", "IZII", "IIZI", "IIIZ"])

    partitioned_problem = partition_problem(
        circuit=qc, partition_labels=[0, 0, 1, 1], observables=observables
    )

    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned_problem.subcircuits,
        observables=partitioned_problem.subobservables,
        num_samples=100,
    )

    # Use BackendSamplerV2 with AerSimulator, following
    # https://github.com/Qiskit/qiskit-aer/issues/2078#issuecomment-1971498534
    samplers = {
        label: BackendSamplerV2(backend=AerSimulator())
        for label in subexperiments.keys()
    }
    results = {
        label: sampler.run(subexperiments[label], shots=128).result()
        for label, sampler in samplers.items()
    }

    # Smoke test of reconstruction
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        partitioned_problem.subobservables,
    )

    assert len(reconstructed_expvals) == len(observables)
