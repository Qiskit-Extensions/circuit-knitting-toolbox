# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class containing the basis in which to decompose an operation."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.circuit import Instruction


class QPDBasis:
    """Basis in which to decompose an operation.

    This class defines a basis in which a quantum operation will be decomposed. The
    ideal (noise-free) quantum operation will be decomposed into a quasiprobabilistic
    mixture of noisy circuits.
    """

    def __init__(
        self,
        maps: Sequence[tuple[Sequence[Instruction], ...]],
        coeffs: Sequence[float],
    ):
        """
        Assign member variables.

        Args:
            maps: A sequence of tuples describing the noisy operations probabilistically
                used to simulate an ideal quantum operation.
            coeffs: Coefficients for quasiprobability representation. Each coefficient
                can be any real number.

        Returns:
            None
        """
        self._set_maps(maps)
        self.coeffs = coeffs  # Note: probabilities and kappa calculated through coeffs

    @property
    def maps(
        self,
    ) -> Sequence[tuple[Sequence[Instruction], ...]]:
        """Get mappings for each qubit in the decomposition."""
        return self._maps

    def _set_maps(
        self,
        maps: Sequence[tuple[Sequence[Instruction], ...]],
    ) -> None:
        if len(maps) == 0:
            raise ValueError("Number of maps passed to QPDBasis must be nonzero.")
        num_qubits = len(maps[0])
        if num_qubits > 2:
            raise ValueError("QPDBasis supports at most two qubits.")
        for i in range(1, len(maps)):
            if len(maps[i]) != num_qubits:
                raise ValueError(
                    f"All maps passed to QPDBasis must act on the same number of "
                    f"qubits. (Index {i} contains a {len(maps[i])}-tuple but should "
                    f"contain a {num_qubits}-tuple.)"
                )
        self._maps = maps

    @property
    def num_qubits(self) -> int:
        """Get number of qubits that this decomposition acts on."""
        return len(self._maps[0])

    @property
    def coeffs(self) -> Sequence[float]:
        """Quasiprobability decomposition coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: Sequence[float]) -> None:
        if len(coeffs) != len(self.maps):  # Note: cross-validation
            raise ValueError("Coefficients must be same length as maps.")
        weights = np.abs(coeffs)
        self._kappa = sum(weights)
        self._probabilities = weights / self._kappa
        self._coeffs = coeffs

    @property
    def probabilities(self) -> Sequence[float]:
        """Get the probabilities on which the maps will be sampled."""
        return self._probabilities

    @property
    def kappa(self) -> float:
        """
        Get the square root of the sampling overhead.

        This quantity is the sum of the magnitude of the coefficients.
        """
        return self._kappa

    @property
    def overhead(self) -> float:
        """
        Get the sampling overhead.

        The sampling overhead is the square of the sum of the magnitude of the coefficients.
        """
        return self._kappa**2

    @staticmethod
    def from_instruction(gate: Instruction, /) -> QPDBasis:
        """
        Generate a :class:`.QPDBasis` object, given a supported operation.

        This static method is provided for convenience; it simply
        calls :func:`~qpd.decompositions.qpdbasis_from_instruction` under the hood.

        Args:
            gate: The instruction from which to instantiate a decomposition

        Returns:
            The newly-instantiated :class:`QPDBasis` object
        """
        # pylint: disable=cyclic-import
        from .decompositions import qpdbasis_from_instruction

        return qpdbasis_from_instruction(gate)

    def __eq__(self, other):
        """Check equivalence for QPDBasis class."""
        if other.__class__ is not self.__class__:
            return False
        if len(self.maps) != len(other.maps) or len(self.coeffs) != len(other.coeffs):
            return False
        if self.maps != other.maps:
            return False
        if self.coeffs != other.coeffs:
            return False
        return True
