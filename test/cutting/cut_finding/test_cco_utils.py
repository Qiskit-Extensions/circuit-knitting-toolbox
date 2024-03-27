from __future__ import annotations

import pytest
from pytest import fixture
from qiskit.circuit.library import EfficientSU2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Instruction, CircuitInstruction
from circuit_knitting.cutting.cut_finding.cco_utils import (
    qc_to_cco_circuit,
    cco_to_qc_circuit,
)
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
)

# test circuit 1.
tc_1 = QuantumCircuit(2)
tc_1.h(1)
tc_1.barrier(1)
tc_1.s(0)
tc_1.barrier()
tc_1.cx(1, 0)

# test circuit 2
tc_2 = EfficientSU2(2, entanglement="linear", reps=2).decompose()
tc_2.assign_parameters([0.4] * len(tc_2.parameters), inplace=True)


# test circuit 3
@fixture
def internal_test_circuit():
    circuit = [
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="h", params=[], qubits=[0], gamma=None),
        CircuitElement(name="rx", params=[0.4], qubits=[0], gamma=None),
    ]
    interface = SimpleGateList(circuit)
    interface.insert_gate_cut(2, "LO")
    interface.define_subcircuits([[0, 1], [2, 3]])
    return interface


@pytest.mark.parametrize(
    "test_circuit, known_output",
    [
        (
            tc_1,
            [
                CircuitElement("h", [], [1], None),
                CircuitElement("barrier", [], [1], None),
                CircuitElement("s", [], [0], None),
                "barrier",
                CircuitElement("cx", [], [1, 0], 3),
            ],
        ),
        (
            tc_2,
            [
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [1], None),
                CircuitElement("cx", [], [0, 1], 3),
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [1], None),
                CircuitElement("cx", [], [0, 1], 3),
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [1], None),
            ],
        ),
    ],
)
def test_qc_to_cco_circuit(
    test_circuit: QuantumCircuit, known_output: list[CircuitElement, str]
):
    test_circuit_internal = qc_to_cco_circuit(test_circuit)
    assert test_circuit_internal == known_output


def test_cco_to_qc_circuit(internal_test_circuit: SimpleGateList):
    qc_cut = cco_to_qc_circuit(internal_test_circuit)
    assert qc_cut.data == [
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 0),
                Qubit(QuantumRegister(4, "q"), 1),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 2),
                Qubit(QuantumRegister(4, "q"), 3),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 0),
                Qubit(QuantumRegister(4, "q"), 1),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 2),
                Qubit(QuantumRegister(4, "q"), 3),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="h", num_qubits=1, num_clbits=0, params=[]),
            qubits=(Qubit(QuantumRegister(4, "q"), 0),),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="rx", num_qubits=1, num_clbits=0, params=[0.4]),
            qubits=(Qubit(QuantumRegister(4, "q"), 0),),
            clbits=(),
        ),
    ]
