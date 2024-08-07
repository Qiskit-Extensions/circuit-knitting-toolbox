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

"""Quasiprobability decomposition gates."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Instruction, CircuitInstruction

from ..qpd_basis import QPDBasis


class BaseQPDGate(Instruction):
    """Base class for a gate to be decomposed using quasiprobability decomposition."""

    def __init__(
        self,
        name: str,
        basis: QPDBasis,
        num_qubits: int,
        *,
        basis_id: int | None = None,
        label: str | None = None,
    ):
        """
        Initialize the instruction, and assign member variables.

        Args:
            name: Name of the QPD gate.
            basis: A probabilistic basis to which the gate should be decomposed
            num_qubits: The number of qubits on which the QPD gate acts
            basis_id: An index to the basis to which the gate should be decomposed.
                This index is to basis.maps.
            label: An optional label for the gate
        """
        super().__init__(name, num_qubits, num_clbits=0, params=[], label=label)

        # Set class fields shared by all QPDGates
        self._set_basis(basis)
        # Checking performed in setter to ensure idx in range
        self.basis_id = basis_id

    @property
    def basis(self) -> QPDBasis:
        """
        Quasiprobability decomposition basis.

        Returns:
            The basis to which the gate should be decomposed
        """
        return self._basis

    def _set_basis(self, basis: QPDBasis) -> None:
        self._basis = basis

    @property
    def basis_id(self) -> int | None:
        """
        Index to basis used to decompose this gate.

        If set to None, a random basis will be chosen during decomposition.

        Returns:
            The basis index
        """
        return self._basis_id

    @basis_id.setter
    def basis_id(self, basis_id: int | None) -> None:
        """
        Set the index to the basis to which this gate should decompose.

        The index corresponds to self.basis.maps.

        Raises:
            ValueError: basis_id is out of range.
        """
        if basis_id is not None and basis_id not in range(0, len(self._basis.maps)):
            raise ValueError("Basis ID out of range")
        self._basis_id = basis_id

    def __eq__(self, other):
        """Check equivalence for QPDGate class."""
        return (
            type(other) is type(self)
            and self.basis == other.basis
            and self.basis_id == other.basis_id
            and self.num_qubits == other.num_qubits
            and self.name == other.name
            and self.label == other.label
        )


class TwoQubitQPDGate(BaseQPDGate):
    """Two qubit gate to be decomposed using quasiprobability decomposition."""

    def __init__(
        self,
        basis: QPDBasis,
        *,
        basis_id: int | None = None,
        label: str | None = None,
    ):
        """
        Initialize the two qubit QPD gate.

        Raises:
            ValueError: The :class:`QPDBasis` acts on a number of qubits not equal to 2.
        """
        if basis.num_qubits != 2:
            raise ValueError(
                "TwoQubitQPDGate only supports QPDBasis which act on two qubits."
            )
        super().__init__("qpd_2q", basis, 2, basis_id=basis_id, label=label)

    def _define(self) -> None:
        qc = QuantumCircuit(2)

        qpd_gate1 = SingleQubitQPDGate(
            basis=self.basis, qubit_id=0, basis_id=self.basis_id, label=self.label
        )
        qpd_gate2 = SingleQubitQPDGate(
            basis=self.basis, qubit_id=1, basis_id=self.basis_id, label=self.label
        )

        qc.append(CircuitInstruction(qpd_gate1, [qc.qubits[0]], []))
        qc.append(CircuitInstruction(qpd_gate2, [qc.qubits[1]], []))

        self.definition = qc

    @classmethod
    def from_instruction(cls, instruction: Instruction, /):
        """Create a :class:`TwoQubitQPDGate` which represents a cut version of the given ``instruction``."""
        decomposition = QPDBasis.from_instruction(instruction)
        return TwoQubitQPDGate(decomposition, label=f"cut_{instruction.name}")


class SingleQubitQPDGate(BaseQPDGate):
    """
    Single qubit gate to be decomposed using quasiprobability decomposition.

    This gate could be part of a larger decomposition on many qubits, or it
    could be a standalone single gate decomposition.
    """

    def __init__(
        self,
        basis: QPDBasis,
        qubit_id: int,
        *,
        basis_id: int | None = None,
        label: str | None = None,
    ):
        """
        Initialize the single qubit QPD gate, and assign member variables.

        Args:
            qubit_id: This gate's relative index to the decomposition which it belongs.
                Single qubit QPDGates should have qubit_id 0 if they describe a local
                decomposition, such as a wire cut.

        Raises:
            ValueError: qubit_id is out of range
        """
        super().__init__(
            name="qpd_1q", basis=basis, num_qubits=1, basis_id=basis_id, label=label
        )
        self._set_qubit_id(qubit_id)

    @property
    def qubit_id(self) -> int:
        """Relative qubit index of this gate in the overall decomposition."""
        return self._qubit_id

    def _set_qubit_id(self, qubit_id: int) -> None:
        if qubit_id >= self.basis.num_qubits:
            raise ValueError(
                f"'qubit_id' out of range. 'basis' acts on {self.basis.num_qubits} qubits, "
                f"but 'qubit_id' is {qubit_id}."
            )
        self._qubit_id = qubit_id

    def _define(self) -> None:
        if self.basis_id is None:
            # With basis_id is not set, it does not make sense to define this
            # operation in terms of more fundamental instructions, so we have
            # self.definition remain as None.
            return
        qc = QuantumCircuit(1)
        base = self.basis.maps[self.basis_id]
        for op in base[self.qubit_id]:
            qc.append(CircuitInstruction(op, [qc.qubits[0]], []))
        self.definition = qc

    @property
    def _directive(self):
        """``True`` if the ``basis_id`` is unassigned, which implies this instruction cannot be decomposed."""
        return self.basis_id is None

    def __eq__(self, other):
        """Check equivalence for SingleQubitQPDGate class."""
        return super().__eq__(other) and self.qubit_id == other.qubit_id
