# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for EntanglementForgingVQE module."""

import unittest
import numpy as np
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver

from circuit_knitting_toolbox.entanglement_forging import (
    EntanglementForgingAnsatz,
    EntanglementForgingGroundStateSolver,
)


class TestEntanglementForgingGroundStateSolver(unittest.TestCase):
    def setUp(self):
        # Hard-code some ansatz params/lambdas
        self.optimizer = SPSA(maxiter=0)

    def test_entanglement_forging_vqe_hydrogen(self):
        """Test of applying Entanglement Forged Solver to to compute the energy of a H2 molecule."""

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

        # Set up the ElectronicStructureProblem
        driver = PySCFDriver.from_molecule(molecule)
        problem = driver.run()

        # Specify the ansatz and bitstrings
        ansatz = EntanglementForgingAnsatz(
            circuit_u=TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1),
            bitstrings_u=[(1, 0), (0, 1)],
        )

        # Set up the entanglement forging vqe object
        solver = EntanglementForgingGroundStateSolver(
            ansatz=ansatz,
            optimizer=self.optimizer,
            initial_point=[0.0, np.pi / 2],
        )

        # Solve for the ground state energy
        results = solver.solve(problem)
        ground_state_energy = results.groundenergy + results.energy_shift

        # Ensure ground state energy output is within tolerance
        self.assertAlmostEqual(ground_state_energy, -1.121936544469326)
