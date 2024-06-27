# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quasiprobability decomposition functions."""

import unittest
import math
from collections import Counter
import itertools

import pytest
import numpy as np
import numpy.typing as npt
from ddt import ddt, data, unpack
from qiskit.circuit import QuantumCircuit, ClassicalRegister, CircuitInstruction
from qiskit.circuit.library import (
    EfficientSU2,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CPhaseGate,
    CSGate,
    CSdgGate,
    CSXGate,
    ECRGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    XXPlusYYGate,
    XXMinusYYGate,
    SwapGate,
    iSwapGate,
    DCXGate,
)

from circuit_knitting.utils.iteration import unique_by_eq, strict_zip
from circuit_knitting.cutting.instructions import Move
from circuit_knitting.cutting.qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    WeightType,
    generate_qpd_weights,
    decompose_qpd_instructions,
    qpdbasis_from_instruction,
)
from circuit_knitting.cutting.qpd.weights import (
    _generate_qpd_weights,
    _generate_exact_weights_and_conditional_probabilities,
)
from circuit_knitting.cutting.qpd.decompositions import (
    _nonlocal_qpd_basis_from_u,
    _u_from_thetavec,
    _explicitly_supported_instructions,
)


@ddt
class TestQPDFunctions(unittest.TestCase):
    def setUp(self):
        # Use HWEA for simplicity and easy visualization
        qpd_circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()

        # We will instantiate 2 QPDBasis objects using from_instruction
        rxx_gate = RXXGate(np.pi / 3)
        rxx_decomp = QPDBasis.from_instruction(rxx_gate)

        # Create two QPDGates and specify each of their bases
        # Labels are only used for visualizations
        qpd_gate1 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate2 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate1.basis_id = 0
        qpd_gate2.basis_id = 0

        # Create the circuit instructions
        qpd_inst1 = CircuitInstruction(qpd_gate1, qubits=[1, 2])
        qpd_inst2 = CircuitInstruction(qpd_gate2, qubits=[1, 2])

        # Hard-coded overwrite of the two CNOTS with our decomposed RXX gates
        qpd_circuit.data[9] = qpd_inst1
        qpd_circuit.data[20] = qpd_inst2

        self.qpd_gate1 = qpd_gate1
        self.qpd_gate2 = qpd_gate2
        self.qpd_circuit = qpd_circuit

    def test_generate_qpd_weights(self):
        with self.subTest("Negative number of samples"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_weights([], -100)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("num_samples == NaN"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_weights([], math.nan)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("Zero samples requested"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_weights([], 0)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("Empty case"):
            empty_samples = {(): (1000, WeightType.EXACT)}
            samples = generate_qpd_weights([])
            self.assertEqual(samples, empty_samples)
        with self.subTest("HWEA 100 samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_weights(bases, num_samples=100)
            assert sum(w for w, t in samples.values()) == pytest.approx(100)
            for decomp_ids in samples.keys():
                self.assertTrue(0 <= decomp_ids[0] < len(self.qpd_gate1.basis.maps))
                self.assertTrue(0 <= decomp_ids[1] < len(self.qpd_gate2.basis.maps))
        with self.subTest("HWEA 100.5 samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_weights(bases, num_samples=100.5)
            assert sum(w for w, t in samples.values()) == pytest.approx(100.5)
        with self.subTest("HWEA exact weights"):
            # Do the same thing with num_samples above the threshold for exact weights
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_weights(bases, num_samples=1000)
            assert sum(w for w, t in samples.values()) == pytest.approx(1000)
            assert all(t == WeightType.EXACT for w, t in samples.values())
            for decomp_ids in samples.keys():
                self.assertTrue(0 <= decomp_ids[0] < len(self.qpd_gate1.basis.maps))
                self.assertTrue(0 <= decomp_ids[1] < len(self.qpd_gate2.basis.maps))
        with self.subTest("HWEA exact weights via 'infinite' num_samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_weights(bases, num_samples=math.inf)
            assert sum(w for w, t in samples.values()) == pytest.approx(1)
            assert all(t == WeightType.EXACT for w, t in samples.values())

    def test_decompose_qpd_instructions(self):
        with self.subTest("Empty circuit"):
            circ = QuantumCircuit()
            new_circ = decompose_qpd_instructions(QuantumCircuit(), [])
            circ.add_register(ClassicalRegister(1, name="qpd_measurements"))
            self.assertEqual(circ, new_circ)
        with self.subTest("No QPD circuit"):
            circ = QuantumCircuit(2, 1)
            circ.h(0)
            circ.cx(0, 1)
            circ.measure(1, 0)
            new_circ = decompose_qpd_instructions(circ, [])
            circ.add_register(ClassicalRegister(1, name="qpd_measurements"))
            self.assertEqual(circ, new_circ)
        with self.subTest("Single QPD gate"):
            circ = QuantumCircuit(2)
            circ_compare = circ.copy()
            qpd_basis = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(qpd_basis)
            circ.data.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            decomp_circ = decompose_qpd_instructions(circ, [[0]], map_ids=[0])
            circ_compare.add_register(ClassicalRegister(1, name="qpd_measurements"))
            self.assertEqual(decomp_circ, circ_compare)
        with self.subTest("Single QPD gate with translation"):
            eagle_basis_gate_set = {"id", "rz", "sx", "x", "measure"}
            circ = QuantumCircuit(2)
            qpd_basis = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(qpd_basis)
            circ.data.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            decomp_circ = decompose_qpd_instructions(
                circ, [[0]], map_ids=[1], basis_gate_set="eagle"
            )
            for inst in decomp_circ.data:
                assert inst.operation.name in eagle_basis_gate_set
        with self.subTest("Incorrect map index size"):
            with pytest.raises(ValueError) as e_info:
                decomp_circ = decompose_qpd_instructions(
                    self.qpd_circuit, [[9], [20]], map_ids=[0]
                )
            assert (
                e_info.value.args[0]
                == "The number of map IDs (1) must equal the number of decompositions in the circuit (2)."
            )
        with self.subTest("Test unordered indices"):
            decomp = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            qpd_gate1 = TwoQubitQPDGate(basis=decomp)
            qpd_gate2 = TwoQubitQPDGate(basis=decomp)

            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0, 1]))
            qc.x([0, 1])
            qc.y([0, 1])
            qc.append(CircuitInstruction(qpd_gate2, qubits=[0, 1]))
            decompose_qpd_instructions(qc, [[5], [0]], map_ids=[0, 0])
        with self.subTest("Test measurement"):
            qpd_circ = QuantumCircuit(2)
            qpd_inst = CircuitInstruction(self.qpd_gate1, qubits=[0, 1])
            qpd_circ.data.append(qpd_inst)
            dx_circ_truth = QuantumCircuit(2)
            creg = ClassicalRegister(1, name="qpd_measurements")
            dx_circ_truth.add_register(creg)
            dx_circ_truth.h(0)
            dx_circ_truth.sx(1)
            dx_circ_truth.measure(0, 0)
            dx_circ_truth.h(0)
            dx_circ = decompose_qpd_instructions(qpd_circ, [[0]], [2])
            self.assertEqual(dx_circ_truth, dx_circ)
        with self.subTest("test_invalid_map_ids"):
            qc = QuantumCircuit()
            qpd_map_ids = ((),)
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, qpd_map_ids)
            assert (
                e_info.value.args[0]
                == "Each decomposition must contain either one or two elements. Found a decomposition with (0) elements."
            )
        with self.subTest("test_mismatching_qpd_ids"):
            decomp = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(basis=decomp)
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0]])
            assert (
                e_info.value.args[0]
                == "A circuit data index (0) corresponds to a non-QPDGate (h)."
            )
            qpd_gate1 = SingleQubitQPDGate(basis=decomp, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp, qubit_id=1)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.h(1)
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[1], [2, 3]])
            assert (
                e_info.value.args[0]
                == "A circuit data index (3) corresponds to a non-QPDGate (h)."
            )
        with self.subTest("test_mismatching_qpd_bases"):
            decomp1 = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            decomp2 = QPDBasis.from_instruction(RXXGate(np.pi / 4))
            qpd_gate1 = SingleQubitQPDGate(basis=decomp1, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp2, qubit_id=1)
            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0, 1]])
            assert (
                e_info.value.args[0]
                == "Gates within the same decomposition must share an equivalent QPDBasis."
            )
        with self.subTest("test_unspecified_qpd_gates"):
            decomp = QPDBasis.from_instruction(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(basis=decomp)
            qpd_gate1 = SingleQubitQPDGate(basis=decomp, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp, qubit_id=1)

            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            qc.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0, 1]])
            assert (
                e_info.value.args[0]
                == "The total number of QPDGates specified in instruction_ids (2) does not equal the number of QPDGates in the circuit (3)."
            )

    # Optimal values from https://arxiv.org/abs/2205.00016v2 Corollary 4.4 (page 10)
    @data(
        (CXGate(), 3),
        (CYGate(), 3),
        (CZGate(), 3),
        (CHGate(), 3),
        (ECRGate(), 3),
        (CRXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (RXXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RYYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RZZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RZXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (CPhaseGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CSGate(), 1 + np.sqrt(2)),
        (CSdgGate(), 1 + np.sqrt(2)),
        (CSXGate(), 1 + np.sqrt(2)),
        (CSXGate().inverse(), 1 + np.sqrt(2)),
        (SwapGate(), 7),
        (iSwapGate(), 7),
        (DCXGate(), 7),
        # XXPlusYYGate, XXMinusYYGate, with some combinations:
        #     beta == 0 or not; and
        #     within |theta| < pi or not
        (XXPlusYYGate(0.1), 1 + 4 * np.sin(0.05) + 2 * np.sin(0.05) ** 2),
        (XXPlusYYGate(4), 1 + 4 * np.sin(2) + 2 * np.sin(2) ** 2),
        (XXMinusYYGate(0.2, beta=0.2), 1 + 4 * np.sin(0.1) + 2 * np.sin(0.1) ** 2),
        (Move(), 4),
    )
    @unpack
    def test_optimal_kappa_for_known_gates(self, instruction, gamma):
        assert np.isclose(qpdbasis_from_instruction(instruction).kappa, gamma)

    @data(
        (RXXGate(np.pi / 7), 5, 5),
        (RYYGate(np.pi / 7), 5, 5),
        (RZZGate(np.pi / 7), 5, 5),
        (CRXGate(np.pi / 7), 5, 5),
        (CRYGate(np.pi / 7), 5, 5),
        (CRZGate(np.pi / 7), 5, 5),
        (CPhaseGate(np.pi / 7), 5, 5),
        (ECRGate(), 5, 5),
        (CXGate(), 5, 5),
        (CZGate(), 5, 5),
        (RZZGate(0), 1, 1),
        (RXXGate(np.pi), 1, 1),
        (CRYGate(np.pi), 5, 5),
    )
    @unpack
    def test_qpdbasis_from_instruction_unique_maps(
        self, instruction, q0_num_unique, q1_num_unique
    ):
        """
        Count the number of unique maps with non-zero weight on each qubit.

        Make sure it is as expected based on the instruction provided.
        """
        basis = qpdbasis_from_instruction(instruction)
        # Consider only maps with non-zero weight
        relevant_maps = [
            m for m, w in zip(basis.maps, basis.coeffs) if not np.isclose(w, 0)
        ]
        assert len(unique_by_eq(a for (a, b) in relevant_maps)) == q0_num_unique
        assert len(unique_by_eq(b for (a, b) in relevant_maps)) == q1_num_unique

    @data(
        ([RZZGate(np.pi)], 1e4, 1, 0),
        ([RZZGate(np.pi + 0.02)], 200, 6, 0),
        ([RZZGate(np.pi + 0.02)], 100, 1),
        ([RXXGate(0.1), RXXGate(0.1)], 100, 9),
        ([CXGate(), CXGate()], 1e4, 36, 0),
        ([CXGate(), CXGate()], 30, 0),
        # The following two check either side of the exact/sampled threshold.
        ([CXGate()], 6, 6, 0),
        ([CXGate()], np.nextafter(6, -math.inf), 0),
        # The following makes sure memory does not blow up with many cuts.
        ([RXXGate(0.1)] * 16, 10000, 2001),
    )
    @unpack
    def test_generate_qpd_weights_from_instructions(
        self, gates, num_samples, expected_exact, expected_sampled=None
    ):
        bases = [QPDBasis.from_instruction(gate) for gate in gates]
        samples = generate_qpd_weights(bases, num_samples)

        counts = Counter(weight_type for _, weight_type in samples.values())
        assert counts[WeightType.EXACT] == expected_exact
        if expected_sampled is not None:
            assert counts[WeightType.SAMPLED] == expected_sampled

        total_weight = sum(weight for weight, _ in samples.values())
        assert total_weight == pytest.approx(
            num_samples if math.isfinite(num_samples) else 1
        )

        # Test that the dictionary is actually in the expected order: exact
        # weights first, and largest to smallest within each type of weight.
        last_type = WeightType.EXACT
        upper_bound = math.inf
        for weight, weight_type in samples.values():
            if last_type != weight_type:
                # We allow one transition, from exact weights to sampled.
                assert weight_type == WeightType.SAMPLED
                last_type = weight_type
                upper_bound = math.inf
            assert weight <= upper_bound
            upper_bound = weight

        # All tests that follow require time & memory that scales exponentially
        # with number of gates cut, so skip them when the number is too high.
        if len(gates) > 3:
            return

        # Test conditional probabilities from
        # _generate_exact_weights_and_conditional_probabilities
        independent_probabilities = [basis.probabilities for basis in bases]
        probs: dict[tuple[int, ...], float] = {}
        conditional_probabilities: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}
        for (
            map_ids,
            probability,
        ) in _generate_exact_weights_and_conditional_probabilities(
            independent_probabilities, 1 / num_samples
        ):
            if len(map_ids) == len(bases):
                assert map_ids not in probs
                probs[map_ids] = probability
            else:
                conditional_probabilities[map_ids] = probability
        stack = [(1.0, ())]
        while stack:
            running_prob, map_ids_partial = stack.pop()
            # If it's missing from `conditional_probabilities`, that just means
            # to use the corresponding entry in `independent_probabilities`.
            try:
                vec = conditional_probabilities[map_ids_partial]
            except KeyError:
                vec = independent_probabilities[len(map_ids_partial)]
            for i, prob in enumerate(vec):
                pp = running_prob * prob
                if pp == 0:
                    continue
                map_ids = map_ids_partial + (i,)
                if len(map_ids) == len(bases):
                    assert map_ids not in probs
                    probs[map_ids] = pp
                else:
                    stack.append((pp, map_ids))
        # Now, systematically generate each exact weight, and compare with what
        # we generated above.
        for map_ids in itertools.product(
            *[range(len(probs)) for probs in independent_probabilities]
        ):
            exact = np.prod(
                [
                    probs[i]
                    for i, probs in strict_zip(map_ids, independent_probabilities)
                ]
            )
            assert probs.get(map_ids, 0.0) == pytest.approx(exact, abs=1e-14)

    def test_statistics_of_generate_qpd_weights(self):
        # Values inspired by the R[XX,YY,ZZ]Gate rotations
        def from_theta(theta):
            v = np.array(
                [
                    np.cos(theta) ** 2,
                    np.sin(theta) ** 2,
                    4 * np.cos(theta) * np.sin(theta),
                ]
            )
            return v / np.sum(v)

        probs = [from_theta(0.1), from_theta(0.2)]
        num_samples = 200
        weights = _generate_qpd_weights(probs, num_samples, _samples_multiplier=10000)
        for map_ids in [(0, 0), (0, 2), (0, 1), (2, 0), (2, 2), (2, 1)]:
            assert weights[map_ids][0] / num_samples == pytest.approx(
                probs[0][map_ids[0]] * probs[1][map_ids[1]]
            )
            assert weights[map_ids][1] == WeightType.EXACT
        for map_ids in [(1, 2), (1, 0), (1, 1)]:
            assert weights[map_ids][0] / num_samples == pytest.approx(
                probs[0][map_ids[0]] * probs[1][map_ids[1]], rel=0.5
            )
            assert weights[map_ids][1] == WeightType.SAMPLED

    def test_explicitly_supported_gates(self):
        gates = _explicitly_supported_instructions()
        self.assertEqual(
            {
                "rxx",
                "ryy",
                "rzz",
                "crx",
                "cry",
                "crz",
                "cx",
                "cy",
                "cz",
                "ch",
                "csx",
                "csxdg",
                "cs",
                "csdg",
                "cp",
                "ecr",
                "swap",
                "iswap",
                "dcx",
                "move",
            },
            gates,
        )

    def test_nonlocal_qpd_basis_from_u(self):
        with self.subTest("Invalid shape"):
            with pytest.raises(ValueError) as e_info:
                _nonlocal_qpd_basis_from_u([1, 2, 3])
            assert (
                e_info.value.args[0]
                == "u vector has wrong shape: (3,) (1D vector of length 4 expected)"
            )

    @data(
        ([np.pi / 4] * 3, [(1 + 1j) / np.sqrt(8)] * 4),
        ([np.pi / 4, np.pi / 4, 0], [0.5, 0.5j, 0.5j, 0.5]),
    )
    @unpack
    def test_u_from_thetavec(self, theta, expected):
        assert _u_from_thetavec(theta) == pytest.approx(expected)

    def test_u_from_thetavec_exceptions(self):
        with self.subTest("Invalid shape"):
            with pytest.raises(ValueError) as e_info:
                _u_from_thetavec([0, 1, 2, 3])
            assert (
                e_info.value.args[0]
                == "theta vector has wrong shape: (4,) (1D vector of length 3 expected)"
            )
