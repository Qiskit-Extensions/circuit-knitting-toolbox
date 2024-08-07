# This code is a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quasiprobability decomposition measure marker instruction."""
from __future__ import annotations

from qiskit.circuit import Instruction


class QPDMeasure(Instruction):
    """An instruction for denoting a QPD measurement location."""

    def __init__(self, label: str | None = None):
        """Create an instruction for marking probabilistic mid-circuit measurements."""
        super().__init__("qpd_measure", 1, 0, [], label=label)
