# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pytest
import unittest
from ddt import ddt, data, unpack

import numpy as np
from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from circuit_knitting.utils.observable_grouping import *


@ddt
class TestEstimatorUtilities(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        self.qc0 = qc.copy()
        qc.add_register(ClassicalRegister(1, name="qpd_measurements"))
        self.qc1 = qc.copy()
        qc.add_register(ClassicalRegister(2, name="observable_measurements"))
        self.qc2 = qc

        self.cog = CommutingObservableGroup(
            Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"]))
        )

    def test_most_general_observable(self):
        with self.subTest("No elements"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable([])
            assert (
                e_info.value.args[0]
                == "Empty input sequence: consider performing no experiments rather than an experiment over the identity Pauli."
            )
        with self.subTest("Observable with wrong qubit count"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable([Pauli("X"), Pauli("ZZ")])
            assert (
                e_info.value.args[0]
                == "Observable 1 has incorrect qubit count (2 rather than 1)."
            )
        with self.subTest("Pass strings instead of Paulis"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable(["X", "ZZ"])
            assert (
                e_info.value.args[0]
                == "Input sequence includes something other than a Pauli."
            )
        with self.subTest("Incompatible observables"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable(PauliList(["X", "Z"]))
            assert (
                e_info.value.args[0]
                == "Observables are incompatible; cannot construct a single general observable."
            )
        with self.subTest("Redundancy"):
            assert most_general_observable(PauliList(["ZI", "IZ", "ZZ"])) == Pauli("ZZ")

    @data(
        ((0, 2), PauliList(["IXIX"]), PauliList(["XX"])),
        ((1, 3), PauliList(["ZZII", "iIZIZ"]), PauliList(["ZI", "II"])),
        ((1,), PauliList(["XYZ", "-XXX", "-iZZZ"]), PauliList(["Y", "X", "Z"])),
        (
            (),
            PauliList(["XX", "YY", "ZZ"]),
            PauliList.from_symplectic(np.zeros((3, 0)), np.zeros((3, 0))),
        ),
        (
            (2, 1),
            [Pauli("XYZ"), Pauli("iZYX"), Pauli("-ZXZ")],
            [Pauli("YX"), Pauli("YZ"), Pauli("XZ")],
        ),
    )
    @unpack
    def test_observables_restricted_to_subsystem(self, qubits, observables, expected):
        assert (
            observables_restricted_to_subsystem(qubits, PauliList(observables))
            == expected
        )
        assert observables_restricted_to_subsystem(qubits, list(observables)) == list(
            expected
        )

    def test_cog_with_nontrivial_phase(self):
        with pytest.raises(ValueError) as e_info:
            CommutingObservableGroup(Pauli("XXI"), [Pauli("iXXI")])
        assert (
            e_info.value.args[0]
            == "CommutingObservableGroup only supports Paulis with phase == 0. (Value provided: 3)"
        )

    def test_observable_collection(self):
        with self.subTest("Initialize with List[Pauli]"):
            oc = ObservableCollection([Pauli("XX"), Pauli("ZZ"), Pauli("XX")])
            assert [len(group.commuting_observables) for group in oc.groups] == [1, 1]

        with self.subTest("Qubit count doesn't match"):
            with pytest.raises(ValueError):
                ObservableCollection([Pauli("XX"), Pauli("ZZZ")])
