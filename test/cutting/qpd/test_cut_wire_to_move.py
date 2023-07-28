# This code is part of Qiskit.
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

"""Test for the transform_to_move function."""
from __future__ import annotations

from pytest import fixture, mark
from qiskit.circuit import QuantumCircuit
from circuit_knitting.cutting.qpd.instructions.move import Move
from circuit_knitting.cutting.qpd.instructions.cut_wire import CutWire
from circuit_knitting.cutting.qpd.cut_wire_to_move import transform_to_move


@fixture
def circuit1() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.cx(1, 2)
    circuit.append(CutWire(), [1])
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)

    return circuit


@fixture
def resulting_circuit1() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit(5)
    circuit.cx(1, 4)
    circuit.append(Move(), (1, 2))
    circuit.cx(0, 2)
    circuit.append(Move(), (2, 3))
    circuit.cx(3, 4)

    mapping = [0, 3, 4]

    return circuit, mapping


@fixture
def circuit2() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.append(CutWire(), [2])
    circuit.cx(2, 3)

    return circuit


@fixture
def resulting_circuit2() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit(6)
    circuit.cx(0, 1)
    circuit.append(Move(), [1, 2])
    circuit.cx(2, 3)
    circuit.append(Move(), [3, 4])
    circuit.cx(4, 5)

    mapping = [0, 2, 4, 5]

    return circuit, mapping


@fixture
def circuit3() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.append(CutWire(), [2])
    circuit.cx(2, 3)
    circuit.append(CutWire(), [2])
    circuit.cx(0, 2)
    circuit.append(CutWire(), [0])
    circuit.cx(0, 1)

    return circuit


@fixture
def resulting_circuit3() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit(8)
    circuit.cx(0, 2)
    circuit.append(Move(), [2, 3])
    circuit.cx(3, 4)
    circuit.append(Move(), [4, 5])
    circuit.cx(5, 7)
    circuit.append(Move(), [5, 6])
    circuit.cx(0, 6)
    circuit.append(Move(), [0, 1])
    circuit.cx(1, 3)

    mapping = [1, 3, 6, 7]

    return circuit, mapping


@mark.parametrize(
    "sample_circuit, resulting_circuit",
    [
        ("circuit1", "resulting_circuit1"),
        ("circuit2", "resulting_circuit2"),
        ("circuit3", "resulting_circuit3"),
    ],
)
def test_transform_to_move(request, sample_circuit, resulting_circuit):
    """Tests the transformation of CutWire to Move instruction."""
    assert request.getfixturevalue(resulting_circuit)[0] == transform_to_move(
        request.getfixturevalue(sample_circuit)
    )


@mark.parametrize(
    "sample_circuit, resulting_circuit",
    [
        ("circuit1", "resulting_circuit1"),
        ("circuit2", "resulting_circuit2"),
        ("circuit3", "resulting_circuit3"),
    ],
)
def test_circuit_registers(request, sample_circuit, resulting_circuit):
    """Tests the mapping of original and new circuit registers."""
    initial_mapping = list(range(len(request.getfixturevalue(sample_circuit).qubits)))
    final_circuit = transform_to_move(request.getfixturevalue(sample_circuit))
    final_mapping = request.getfixturevalue(resulting_circuit)[1]

    assert all(
        final_circuit.qubits[final_index]
        == request.getfixturevalue(sample_circuit).qubits[index]
        for index, final_index in zip(initial_mapping, final_mapping)
    )