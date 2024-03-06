# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class used for specifying characteristics of the target QPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceConstraints:
    """Specify the characteristics (qubits per QPU and number of QPUs) of the target quantum device that must be respected."""

    qubits_per_QPU: int
    num_QPUs: int

    def __post_init__(self):
        """Post-init method for data class."""
        if self.qubits_per_QPU < 1 or self.num_QPUs < 1:
            raise ValueError(
                "qubits_per_QPU and num_QPUs must be positive definite integers."
            )

    def get_qpu_width(self) -> int:
        """Return the number of qubits supported on each individual QPU."""
        return self.qubits_per_QPU

    @classmethod
    def from_dict(cls, options: dict[str, int]) -> DeviceConstraints:
        """Return an instance of :class:`DeviceConstraints` initialized with the parameters passed in."""
        return cls(**options)
