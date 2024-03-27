from __future__ import annotations

from pytest import mark, raises, fixture
from typing import Callable
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    DisjointSubcircuitsState,
)
from circuit_knitting.cutting.cut_finding.cut_optimization import (
    disjoint_subcircuit_actions,
)
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
    GateSpec,
)


@mark.parametrize("num_qubits, max_wire_cuts", [(2.1, 1.2), (None, -1), (-1, None)])
def test_StateInitialization(num_qubits, max_wire_cuts):
    """Test device constraints for being valid data types."""

    with raises(ValueError):
        _ = DisjointSubcircuitsState(num_qubits, max_wire_cuts)


@fixture
def test_circuit():
    circuit = [
        CircuitElement(name="h", params=[], qubits=["q1"], gamma=None),
        CircuitElement(name="barrier", params=[], qubits=["q1"], gamma=None),
        CircuitElement(name="s", params=[], qubits=["q0"], gamma=None),
        "barrier",
        CircuitElement(name="cx", params=[], qubits=["q1", "q0"], gamma=3),
    ]

    interface = SimpleGateList(circuit)

    # initialize DisjointSubcircuitsState object.
    state = DisjointSubcircuitsState(interface.get_num_qubits(), 2)

    two_qubit_gate = interface.get_multiqubit_gates()[0]

    return state, two_qubit_gate


def test_state_uncut(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):

    state, _ = test_circuit

    assert list(state.wiremap) == [0, 1]

    assert state.num_wires == 2

    assert state.get_num_qubits() == 2

    assert list(state.uptree) == [0, 1, 2, 3]

    assert list(state.width) == [1, 1, 1, 1]

    assert list(state.no_merge) == []

    assert state.get_search_level() == 0


def test_apply_gate(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):
    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action(None).next_state(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.num_wires == 2

    assert next_state.find_qubit_root(1) == 0

    assert next_state.get_wire_root_mapping() == [0, 0]

    assert list(next_state.uptree) == [0, 0, 2, 3]

    assert list(next_state.width) == [2, 1, 1, 1]

    assert list(next_state.no_merge) == []

    assert next_state.get_search_level() == 1


def test_cut_gate(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):
    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action("CutTwoQubitGate").next_state(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.check_donot_merge_roots(0, 1) is True

    assert next_state.num_wires == 2

    assert state.get_num_qubits() == 2

    assert next_state.get_wire_root_mapping() == [0, 1]

    assert list(next_state.uptree) == [0, 1, 2, 3]

    assert list(next_state.width) == [1, 1, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.get_search_level() == 1

    assert next_state.lower_bound_gamma() == 3  # one CNOT cut.

    assert (
        next_state.upper_bound_gamma() == 3
    )  # equal to lower_bound_gamma for single gate cuts.


def test_cut_left_wire(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):

    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action("CutLeftWire").next_state(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [
        2,
        1,
    ]  # qubit 0 is mapped onto wire ID 2 after cut.

    assert next_state.num_wires == 3

    assert state.get_num_qubits() == 2

    assert not next_state.can_expand_subcircuit(1, 1, 2)  # False

    assert next_state.can_expand_subcircuit(1, 1, 3)  # True

    assert next_state.can_add_wires(2) is False

    assert next_state.get_wire_root_mapping() == [0, 1, 1]

    assert next_state.check_donot_merge_roots(0, 1) is True

    assert list(next_state.uptree) == [0, 1, 1, 3]

    assert list(next_state.width) == [1, 2, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.get_max_width() == 2

    assert next_state.find_qubit_root(0) == 1

    assert next_state.get_search_level() == 1

    assert next_state.lower_bound_gamma() == 3

    assert next_state.upper_bound_gamma() == 4


def test_cut_right_wire(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):
    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action("CutRightWire").next_state(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [
        0,
        2,
    ]  # qubit 1 is mapped onto wire ID 2 after cut.

    assert next_state.num_wires == 3

    assert state.get_num_qubits() == 2

    assert next_state.can_add_wires(1) is True

    assert next_state.get_wire_root_mapping() == [0, 1, 0]

    assert next_state.check_donot_merge_roots(0, 1) is True

    assert list(next_state.uptree) == [0, 1, 0, 3]

    assert list(next_state.width) == [2, 1, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.find_qubit_root(1) == 0

    assert next_state.get_search_level() == 1


def test_cut_both_wires(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):
    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action("CutBothWires").next_state(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [2, 3]

    assert next_state.can_add_wires(1) is False

    assert next_state.num_wires == 4

    assert state.get_num_qubits() == 2

    assert next_state.get_wire_root_mapping() == [0, 1, 2, 2]

    assert (
        next_state.check_donot_merge_roots(0, 2)
        == next_state.check_donot_merge_roots(1, 2)
        is True
    )

    assert list(next_state.uptree) == [0, 1, 2, 2]

    assert list(next_state.width) == [1, 1, 2, 1]

    assert list(next_state.no_merge) == [(0, 2), (1, 3)]

    assert (
        next_state.find_qubit_root(0) == 2
    )  # maps to third wire initialized after cut.

    assert (
        next_state.find_qubit_root(1) == 2
    )  # maps to third wire because of the entangling gate.

    assert next_state.get_search_level() == 1

    assert next_state.lower_bound_gamma() == 9  # 3^n scaling.

    assert next_state.upper_bound_gamma() == 16  # The 4^n scaling that comes with LO.

    assert next_state.verify_merge_constraints() is True

    next_state.no_merge = [
        (0, 2),
        (1, 3),
        (2, 3),
    ]  # Enforce an incorrect set of no-merge constraints
    # and verify that verify_merge_constraints is False.
    assert next_state.verify_merge_constraints() is False


def test_no_wire_cuts(
    test_circuit: Callable[[], tuple[DisjointSubcircuitsState, GateSpec]]
):
    state, two_qubit_gate = test_circuit

    next_state = disjoint_subcircuit_actions.get_action("CutBothWires").next_state(
        state, two_qubit_gate, 1
    )  # Imposing a max_width < 2 means no wire cuts.

    assert next_state == []
