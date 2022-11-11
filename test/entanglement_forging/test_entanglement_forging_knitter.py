# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for EntanglementForgingKnitter module."""

import os
import unittest
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver

from circuit_knitting_toolbox.entanglement_forging import (
    EntanglementForgingAnsatz,
    EntanglementForgingKnitter,
    cholesky_decomposition,
    convert_cholesky_operator,
)
from circuit_knitting_toolbox.utils import IntegralDriver


class TestEntanglementForgingKnitter(unittest.TestCase):
    def setUp(self):
        self.energy_shift_h2 = 0.7199689944489797

        self.energy_shift_h2o = -61.37433060587931

        self.hcore_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_one_body.npy"),
        )
        self.eri_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_two_body.npy"),
        )
        self.energy_shift_o2 = -99.83894101027317

        self.hcore_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_one_body.npy"),
        )
        self.eri_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_two_body.npy"),
        )
        self.energy_shift_ch3 = -31.90914780401554

        self.hcore_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_one_body.npy"),
        )
        self.eri_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_two_body.npy"),
        )
        self.energy_shift_cn = -67.18561556466743

    def create_mock_ansatz_circuit(self, num_qubits: int) -> QuantumCircuit:
        theta = Parameter("θ")
        mock_gate = QuantumCircuit(1, name="mock gate")
        mock_gate.rz(theta, 0)

        theta_vec = [Parameter("θ%d" % i) for i in range(1)]
        ansatz = QuantumCircuit(num_qubits)
        ansatz.append(mock_gate.to_gate({theta: theta_vec[0]}), [0])

        return ansatz

    def test_entanglement_forging_H2(self):
        """
        Test to apply Entanglement Forging to compute the energy of a H2 molecule,
        given the optimial ansatz parameters.
        """

        # Specify molecule
        molecule = MoleculeInfo(
            ["H", "H"],
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.735),
            ],
            charge=0,
            multiplicity=1,
        )

        # Set up the ELectrionicStructureProblem
        driver = PySCFDriver.from_molecule(molecule)
        problem = driver.run()

        # Specify the ansatz and bitstrings
        ansatz = EntanglementForgingAnsatz(
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
            bitstrings_u=[(1, 0), (0, 1), (1, 0)],
            bitstrings_v=[(1, 0), (0, 1), (0, 1)],
        )

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Hard-coded optimal ansatz parameters
        ansatz_params = [0, 1.57079633]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -1.121936544469326)

    def test_entanglement_forging_H2O(self):  # pylint: disable=too-many-locals
        """
        Test to apply Entanglement Forging to compute the energy of a H20 molecule,
        given optimal ansatz parameters.
        """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        h1_x = radius_1
        h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
        h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

        molecule = MoleculeInfo(
            ["O", "H", "H"],
            [
                (0.0, 0.0, 0.0),
                (h1_x, 0.0, 0.0),
                (h2_x, h2_y, 0.0),
            ],
            charge=0,
            multiplicity=1,
        )
        driver = PySCFDriver.from_molecule(molecule, basis="sto6g")
        problem = driver.run()

        # solution
        orbitals_to_reduce = [0, 3]
        bitstrings = [
            (1, 1, 1, 1, 1, 0, 0),
            (1, 0, 1, 1, 1, 0, 1),
            (1, 0, 1, 1, 1, 1, 0),
        ]
        reduced_bitstrings = [
            tuple(bs)
            for bs in np.delete(bitstrings, orbitals_to_reduce, axis=-1).tolist()
        ]

        theta = Parameter("θ")
        theta_1, theta_2, theta_3, theta_4 = (
            Parameter("θ1"),
            Parameter("θ2"),
            Parameter("θ3"),
            Parameter("θ4"),
        )

        hop_gate = QuantumCircuit(2, name="Hop gate")
        hop_gate.h(0)
        hop_gate.cx(1, 0)
        hop_gate.cx(0, 1)
        hop_gate.ry(-theta, 0)
        hop_gate.ry(-theta, 1)
        hop_gate.cx(0, 1)
        hop_gate.h(0)

        circuit = QuantumCircuit(5)
        circuit.append(hop_gate.to_gate({theta: theta_1}), [0, 1])
        circuit.append(hop_gate.to_gate({theta: theta_2}), [3, 4])
        circuit.append(hop_gate.to_gate({theta: 0}), [1, 4])
        circuit.append(hop_gate.to_gate({theta: theta_3}), [0, 2])
        circuit.append(hop_gate.to_gate({theta: theta_4}), [3, 4])

        ansatz = EntanglementForgingAnsatz(
            circuit_u=circuit, bitstrings_u=reduced_bitstrings
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(
            problem, orbitals_to_reduce=orbitals_to_reduce
        )
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)

        ansatz_params = [0.0, 0.0, 0.0, 0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -75.68366174497027)

    def test_entanglement_forging_driver_H2(self):
        """Test for entanglement forging driver."""
        hcore = np.array([[-1.12421758, -0.9652574], [-0.9652574, -1.12421758]])
        mo_coeff = np.array([[0.54830202, 1.21832731], [0.54830202, -1.21832731]])
        eri = np.array(
            [
                [
                    [[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                ],
                [
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                    [[0.57187698, 0.44744572], [0.44744572, 0.77460594]],
                ],
            ]
        )

        driver = IntegralDriver(
            hcore=hcore,
            mo_coeff=mo_coeff,
            eri=eri,
            num_alpha=1,
            num_beta=1,
            nuclear_repulsion_energy=0.7199689944489797,
        )

        problem = driver.run()

        ansatz = EntanglementForgingAnsatz(
            bitstrings_u=[(1, 0), (0, 1)],
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0, np.pi / 2]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -1.1219365445030705)

    def test_asymmetric_bitstrings_O2(self):
        """Test for entanglement forging driver."""
        driver = IntegralDriver(
            hcore=self.hcore_o2,
            mo_coeff=np.eye(8, 8),
            eri=self.eri_o2,
            num_alpha=6,
            num_beta=6,
            nuclear_repulsion_energy=self.energy_shift_o2,
        )

        problem = driver.run()

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(8),
            bitstrings_u=[
                (1, 1, 1, 1, 1, 1, 0, 0),
                (1, 1, 1, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 0, 1, 1, 0),
                (1, 1, 0, 1, 1, 1, 1, 0),
            ],
            bitstrings_v=[
                (1, 1, 1, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 1, 1, 0, 0),
                (1, 1, 0, 1, 1, 1, 1, 0),
                (1, 1, 1, 1, 0, 1, 1, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -147.63645235088566)

    def test_asymmetric_bitstrings_CH3(self):
        """Test for entanglement forging driver."""
        driver = IntegralDriver(
            hcore=self.hcore_ch3,
            mo_coeff=np.eye(6, 6),
            eri=self.eri_ch3,
            num_alpha=3,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ch3,
        )

        problem = driver.run()

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(6),
            bitstrings_u=[
                (1, 1, 1, 0, 0, 0),
                (0, 1, 1, 0, 0, 1),
                (1, 0, 1, 0, 1, 0),
                (1, 0, 1, 1, 0, 0),
                (0, 1, 1, 1, 0, 0),
            ],
            bitstrings_v=[
                (1, 1, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 1),
                (1, 0, 0, 0, 1, 0),
                (1, 0, 0, 1, 0, 0),
                (0, 1, 0, 1, 0, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -39.09031477502881)

    def test_asymmetric_bitstrings_CN(self):
        """Test for asymmetric bitstrings with hybrid cross terms."""
        driver = IntegralDriver(
            hcore=self.hcore_cn,
            mo_coeff=np.eye(8, 8),
            eri=self.eri_cn,
            num_alpha=5,
            num_beta=4,
            nuclear_repulsion_energy=self.energy_shift_cn,
        )

        problem = driver.run()

        ansatz = EntanglementForgingAnsatz(
            circuit_u=self.create_mock_ansatz_circuit(8),
            bitstrings_u=[
                (1, 1, 1, 1, 1, 0, 0, 0),
                (1, 1, 1, 0, 1, 0, 1, 0),
                (1, 1, 0, 1, 1, 1, 0, 0),
                (1, 1, 1, 0, 1, 1, 0, 0),
                (1, 1, 0, 1, 1, 0, 1, 0),
                (1, 1, 1, 1, 1, 0, 0, 0),
            ],
            bitstrings_v=[
                (1, 1, 1, 1, 0, 0, 0, 0),
                (1, 1, 1, 0, 0, 0, 1, 0),
                (1, 1, 0, 1, 0, 1, 0, 0),
                (1, 1, 1, 0, 0, 1, 0, 0),
                (1, 1, 0, 1, 0, 0, 1, 0),
                (1, 0, 1, 1, 1, 0, 0, 0),
            ],
        )

        # Specify the decomposition method and get the forged operator
        hamiltonian_terms, energy_shift = cholesky_decomposition(problem)
        forged_hamiltonian = convert_cholesky_operator(hamiltonian_terms, ansatz)

        # Set up the forging knitter object
        forging_knitter = EntanglementForgingKnitter(ansatz)
        ansatz_params = [0.0]

        energy, _, _ = forging_knitter(ansatz_params, forged_hamiltonian)

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(energy + energy_shift, -91.06680541685226)
