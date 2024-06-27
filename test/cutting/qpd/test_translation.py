# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPD gate translation."""

import unittest

import pytest
import numpy as np
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import SdgGate, U3Gate, PhaseGate

from circuit_knitting.cutting.qpd import translate_qpd_gate


class TestQPDGateTranslation(unittest.TestCase):
    def test_equivalence_heron(self):
        equiv = translate_qpd_gate(SdgGate(), "heron")
        assert equiv.data[0].operation.name == "rz"
        assert equiv.data[0].operation.params == [-np.pi / 2]

    def test_equivalence_eagle(self):
        equiv = translate_qpd_gate(PhaseGate(1.0), "eagle")
        assert equiv.data[0].operation.name == "rz"
        assert equiv.data[0].operation.params == [1.0]

    def test_unassigned_param(self):
        param = Parameter("θ")
        equiv = translate_qpd_gate(PhaseGate(param), "eagle")
        assert equiv.data[0].operation.name == "rz"
        assert equiv.data[0].operation.params == [param]

    def test_equivalence_unsupported_basis(self):
        with pytest.raises(ValueError) as e_info:
            translate_qpd_gate(SdgGate(), "falcon")
        assert e_info.value.args[0] == "Unknown basis gate set: falcon"

    def test_equivalence_unsupported_gate(self):
        with pytest.raises(ValueError) as e_info:
            translate_qpd_gate(U3Gate(1.0, 1.0, 1.0), "eagle")
        assert e_info.value.args[0] == "Cannot translate gate: u3"
