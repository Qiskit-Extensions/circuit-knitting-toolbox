# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import unittest

import pytest
import numpy as np
from qiskit.quantum_info import PauliList, Pauli
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library.standard_gates import CXGate

from circuit_knitting.cutting.qpd import (
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDBasis,
)
from circuit_knitting.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting.cutting import generate_cutting_experiments
from circuit_knitting.cutting.qpd import WeightType
from circuit_knitting.cutting import partition_problem
from circuit_knitting.cutting.cutting_experiments import (
    generate_distribution_cutting_experiments,
    _append_measurement_circuit,
)


class TestCuttingExperiments(unittest.TestCase):
    def test_generate_cutting_experiments(self):
        with self.subTest("simple circuit and observable"):
            qc = QuantumCircuit(2)
            qc.append(
                TwoQubitQPDGate(QPDBasis.from_instruction(CXGate()), label="cut_cx"),
                qargs=[0, 1],
            )
            comp_coeffs = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, coeffs = generate_cutting_experiments(
                qc, PauliList(["ZZ"]), np.inf
            )
            assert coeffs == comp_coeffs
            assert len(coeffs) == len(subexperiments)
            for exp in subexperiments:
                assert isinstance(exp, QuantumCircuit)

        with self.subTest("simple circuit and observable as dict"):
            qc = QuantumCircuit(2)
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), label="cut_cx_0", qubit_id=0
                ),
                qargs=[0],
            )
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), label="cut_cx_0", qubit_id=1
                ),
                qargs=[1],
            )
            comp_coeffs = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, coeffs = generate_cutting_experiments(
                {"A": qc}, {"A": PauliList(["ZY"])}, np.inf
            )
            assert coeffs == comp_coeffs
            assert len(coeffs) == len(subexperiments["A"])
            for circ in subexperiments["A"]:
                assert isinstance(circ, QuantumCircuit)

        with self.subTest("test bad num_samples"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZZZ"]), 0)
            assert e_info.value.args[0] == "num_samples must be at least 1."
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZZZ"]), np.nan)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("test incompatible inputs"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, {"A": PauliList(["ZZZZ"])}, 4.5)
            assert (
                e_info.value.args[0]
                == "If the input circuits is a QuantumCircuit, the observables must be a PauliList."
            )
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments({"A": qc}, PauliList(["ZZZZ"]), 4.5)
            assert (
                e_info.value.args[0]
                == "If the input circuits are contained in a dictionary keyed by partition labels, the input observables must also be represented by such a dictionary."
            )
        with self.subTest("test bad label"):
            qc = QuantumCircuit(2)
            qc.append(
                TwoQubitQPDGate(QPDBasis.from_instruction(CXGate()), label="cut_cx"),
                qargs=[0, 1],
            )
            partitioned_problem = partition_problem(
                qc, "AB", observables=PauliList(["ZZ"])
            )
            partitioned_problem.subcircuits["A"].data[0].operation.label = "newlabel"

            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(
                    partitioned_problem.subcircuits,
                    partitioned_problem.subobservables,
                    np.inf,
                )
            assert e_info.value.args[0] == (
                "SingleQubitQPDGate instances in input circuit(s) must have their "
                'labels suffixed with "_<id>", where <id> is the index of the cut '
                "relative to the other cuts in the circuit. For example, all "
                "SingleQubitQPDGates belonging to the same cut, N, should have labels "
                ' formatted as "<your_label>_N". This allows SingleQubitQPDGates '
                "belonging to the same cut to be sampled jointly."
            )
        with self.subTest("test bad observable size"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZ"]), np.inf)
            assert e_info.value.args[0] == (
                "Quantum circuit qubit count (4) does not match qubit count of observable(s) (2)."
                "  Try providing `qubit_locations` explicitly."
            )
        with self.subTest("test single qubit qpd gate in unseparated circuit"):
            qc = QuantumCircuit(2)
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), 0, label="cut_cx_0"
                ),
                qargs=[0],
            )
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZ"]), np.inf)
            assert (
                e_info.value.args[0]
                == "SingleQubitQPDGates are not supported in unseparable circuits."
            )

    def test_append_measurement_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.add_register(ClassicalRegister(1, name="qpd_measurements"))
        cog = CommutingObservableGroup(Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"])))
        qc2 = qc.copy()
        qc2.add_register(ClassicalRegister(2, name="observable_measurements"))
        with self.subTest("In place"):
            qcx = qc.copy()
            assert _append_measurement_circuit(qcx, cog, inplace=True) is qcx
        with self.subTest("Out of place"):
            assert _append_measurement_circuit(qc, cog) is not qc
        with self.subTest("Correct measurement circuit"):
            qc2.measure(0, 1)
            qc2.h(1)
            qc2.measure(1, 2)
            assert _append_measurement_circuit(qc, cog) == qc2
        with self.subTest("Mismatch between qubit_locations and number of qubits"):
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog, qubit_locations=[0])
            assert (
                e_info.value.args[0]
                == "qubit_locations has 1 element(s) but the observable(s) have 2 qubit(s)."
            )
        with self.subTest("Mismatched qubits, no qubit_locations provided"):
            cog = CommutingObservableGroup(Pauli("X"), [Pauli("X")])
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog)
            assert (
                e_info.value.args[0]
                == "Quantum circuit qubit count (2) does not match qubit count of observable(s) (1).  Try providing `qubit_locations` explicitly."
            )

    def test_generate_distribution_cutting_experiments(self):
        with self.subTest("test bad num_samples"):
            qc = QuantumCircuit(4, 1)
            qc.measure(0, 0)
            with pytest.raises(ValueError) as e_info:
                generate_distribution_cutting_experiments(qc, 0)
            assert e_info.value.args[0] == "num_samples must be at least 1."
            with pytest.raises(ValueError) as e_info:
                generate_distribution_cutting_experiments(qc, np.nan)
            assert e_info.value.args[0] == "num_samples must be at least 1."
