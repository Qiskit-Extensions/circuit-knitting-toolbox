from __future__ import annotations

from circuit_knitting.cutting.cut_finding.circuit_interface import (
    CircuitElement,
    SimpleGateList,
    GateSpec,
)

from circuit_knitting.cutting.cut_finding.cut_optimization import (
    max_wire_cuts_circuit,
    max_wire_cuts_gamma,
)


class TestCircuitInterface:
    def test_circuit_conversion(self):
        """Test conversion of circuits to the internal representation
        used by the circuit-cutting optimizer.
        """

        # Assign gamma=None to single qubit gates.
        trial_circuit = [
            CircuitElement(name="h", params=[], qubits=["q1"], gamma=None),
            CircuitElement(name="barrier", params=[], qubits=["q1"], gamma=None),
            CircuitElement(name="s", params=[], qubits=["q0"], gamma=None),
            "barrier",
            CircuitElement(name="cx", params=[], qubits=["q1", "q0"], gamma=3),
        ]
        circuit_converted = SimpleGateList(
            trial_circuit
        )  # When init_qubit_names is initialized to [], the first qubit that
        # appears in the first gate in the list that specifies the circuit
        # is assigned ID 0.

        assert circuit_converted.get_num_qubits() == 2
        assert circuit_converted.get_num_wires() == 2
        assert circuit_converted.qubit_names.item_dict == {"q1": 0, "q0": 1}
        assert circuit_converted.get_multiqubit_gates() == [
            GateSpec(
                instruction_id=4,
                gate=CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
                cut_constraints=None,
            )
        ]

        assert circuit_converted.circuit == [
            [CircuitElement(name="h", params=[], qubits=[0], gamma=None), None],
            [CircuitElement(name="barrier", params=[], qubits=[0], gamma=None), None],
            [CircuitElement(name="s", params=[], qubits=[1], gamma=None), None],
            ["barrier", None],
            [CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3), None],
        ]

        assert max_wire_cuts_circuit(circuit_converted) == 2
        assert max_wire_cuts_gamma(7) == 2

        # Assign by hand a different qubit mapping by specifiying init_qubit_names.
        circuit_converted = SimpleGateList(trial_circuit, ["q0", "q1"])
        assert circuit_converted.qubit_names.item_dict == {"q0": 0, "q1": 1}
        assert circuit_converted.circuit == [
            [CircuitElement(name="h", params=[], qubits=[1], gamma=None), None],
            [CircuitElement(name="barrier", params=[], qubits=[1], gamma=None), None],
            [CircuitElement(name="s", params=[], qubits=[0], gamma=None), None],
            ["barrier", None],
            [CircuitElement(name="cx", params=[], qubits=[1, 0], gamma=3), None],
        ]

    def test_gate_cut_interface(self):
        """Test the internal representation of LO gate cuts."""

        trial_circuit = [
            CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        ]
        circuit_converted = SimpleGateList(trial_circuit)
        circuit_converted.insert_gate_cut(2, "LO")
        circuit_converted.define_subcircuits([[0, 1], [2, 3]])

        assert list(circuit_converted.new_gate_id_map) == [0, 1, 2, 3, 4]
        assert circuit_converted.cut_type == [None, None, "LO", None, None]
        assert (
            circuit_converted.export_subcircuits_as_string(name_mapping="default")
            == "AABB"
        )
        assert circuit_converted.export_cut_circuit(name_mapping="default") == [
            trial_circuit[0],
            trial_circuit[1],
            trial_circuit[2],
            trial_circuit[3],
            trial_circuit[4],
        ]

        # the following two methods are the same in the absence of wire cuts.
        assert (
            circuit_converted.export_output_wires(name_mapping="default")
            == circuit_converted.export_output_wires(name_mapping=None)
            == {0: 0, 1: 1, 2: 2, 3: 3}
        )

    def test_wire_cut_interface(self):
        """Test the internal representation of LO wire cuts."""

        trial_circuit = [
            CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        ]
        circuit_converted = SimpleGateList(trial_circuit)

        # cut first input wire of trial_circuit[2] and map it to wire id 4.
        circuit_converted.insert_wire_cut(2, 1, 1, 4, "LO")
        assert list(circuit_converted.output_wires) == [0, 4, 2, 3]

        assert circuit_converted.cut_type[2] == "LO"

        # the missing gate 2 corresponds to a move operation
        assert list(circuit_converted.new_gate_id_map) == [0, 1, 3, 4, 5]

        assert circuit_converted.export_cut_circuit(name_mapping=None) == [
            trial_circuit[0],
            trial_circuit[1],
            ["move", 1, ("cut", 1)],
            CircuitElement(name="cx", params=[], qubits=[("cut", 1), 2], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[0, ("cut", 1)], gamma=3),
            trial_circuit[4],
        ]

        # relabel wires after wire cuts according to 'None' name_mapping.
        assert circuit_converted.export_output_wires(name_mapping=None) == {
            0: 0,
            1: ("cut", 1),
            2: 2,
            3: 3,
        }

        # relabel wires after wire cuts according to 'default' name_mapping.
        assert circuit_converted.export_output_wires(name_mapping="default") == {
            0: 0,
            1: 2,
            2: 3,
            3: 4,
        }

        assert circuit_converted.export_cut_circuit(name_mapping="default") == [
            CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3),
            ["move", 1, 2],
            CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[0, 2], gamma=3),
            CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3),
        ]
