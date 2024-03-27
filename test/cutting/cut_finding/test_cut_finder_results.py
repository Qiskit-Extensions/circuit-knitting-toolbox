from __future__ import annotations

import numpy as np
from numpy import array
from pytest import fixture, raises
from qiskit import QuantumCircuit
from typing import Callable
from qiskit.circuit.library import EfficientSU2
from circuit_knitting.cutting.cut_finding.cco_utils import qc_to_cco_circuit
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    SimpleGateList,
)
from circuit_knitting.cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import (
    DeviceConstraints,
)
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    get_actions_list,
    OneWireCutIdentifier,
    WireCutLocation,
    CutIdentifier,
    GateCutLocation,
)
from circuit_knitting.cutting.cut_finding.lo_cuts_optimizer import (
    LOCutsOptimizer,
)
from circuit_knitting.cutting.cut_finding.cut_optimization import CutOptimization


@fixture
def empty_circuit():
    qc = QuantumCircuit(3)
    qc.barrier([0])
    qc.barrier([1])
    qc.barrier([2])


@fixture
def four_qubit_test_setup():
    qc = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    qc.assign_parameters([0.4] * len(qc.parameters), inplace=True)
    circuit_internal = qc_to_cco_circuit(qc)
    interface = SimpleGateList(circuit_internal)
    settings = OptimizationSettings(seed=12345)
    settings.set_engine_selection("CutOptimization", "BestFirst")
    return interface, settings


@fixture
def seven_qubit_test_setup():
    qc = QuantumCircuit(7)
    for i in range(7):
        qc.rx(np.pi / 4, i)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(3, 6)
    circuit_internal = qc_to_cco_circuit(qc)
    interface = SimpleGateList(circuit_internal)
    settings = OptimizationSettings(seed=12345)
    settings.set_engine_selection("CutOptimization", "BestFirst")
    return interface, settings


@fixture
def multiqubit_gate_test_setup():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    circuit_internal = qc_to_cco_circuit(qc)
    interface = SimpleGateList(circuit_internal)
    settings = OptimizationSettings(seed=12345)
    settings.set_engine_selection("CutOptimization", "BestFirst")
    return interface, settings


def test_no_cuts(
    four_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    # QPU with 4 qubits for a 4 qubit circuit results in no cutting.
    qubits_per_subcircuit = 4

    interface, settings = four_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize(interface, settings, constraint_obj)

    assert get_actions_list(output.actions) == []  # no cutting.

    assert interface.export_subcircuits_as_string(name_mapping="default") == "AAAA"


def test_four_qubit_circuit_three_qubit_qpu(
    four_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    # QPU with 3 qubits for a 4 qubit circuit enforces cutting.
    qubits_per_subcircuit = 3

    interface, settings = four_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.cut_actions_sublist()

    assert cut_actions_list == [
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=17, gate_name="cx", qubits=[2, 3]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=25, gate_name="cx", qubits=[2, 3]
            ),
        ),
    ]
    best_result = optimization_pass.get_results()

    assert output.upper_bound_gamma() == best_result.gamma_UB == 9  # 2 LO cnot cuts.

    assert optimization_pass.minimum_reached() is True  # matches optimal solution.

    assert (
        interface.export_subcircuits_as_string(name_mapping="default") == "AAAB"
    )  # circuit separated into 2 subcircuits.


def test_four_qubit_circuit_two_qubit_qpu(
    four_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    # QPU with 2 qubits enforces cutting.
    qubits_per_subcircuit = 2

    interface, settings = four_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.cut_actions_sublist()

    assert cut_actions_list == [
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=9, gate_name="cx", qubits=[1, 2]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=20, gate_name="cx", qubits=[1, 2]
            ),
        ),
    ]

    best_result = optimization_pass.get_results()

    assert output.upper_bound_gamma() == best_result.gamma_UB == 9  # 2 LO cnot cuts.

    assert optimization_pass.minimum_reached() is True  # matches optimal solution.

    assert (
        interface.export_subcircuits_as_string(name_mapping="default") == "AABB"
    )  # circuit separated into 2 subcircuits.

    assert (
        optimization_pass.get_stats()["CutOptimization"] == array([15, 46, 15, 6])
    ).all()  # matches known stats.


def test_seven_qubit_circuit_two_qubit_qpu(
    seven_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    # QPU with 2 qubits enforces cutting.
    qubits_per_subcircuit = 2

    interface, settings = seven_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.cut_actions_sublist()

    assert cut_actions_list == [
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=7, gate_name="cx", qubits=[0, 3]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=8, gate_name="cx", qubits=[1, 3]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=9, gate_name="cx", qubits=[2, 3]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=11, gate_name="cx", qubits=[3, 5]
            ),
        ),
        CutIdentifier(
            cut_action="CutTwoQubitGate",
            gate_cut_location=GateCutLocation(
                instruction_id=12, gate_name="cx", qubits=[3, 6]
            ),
        ),
    ]

    best_result = optimization_pass.get_results()

    assert output.upper_bound_gamma() == best_result.gamma_UB == 243  # 5 LO cnot cuts.

    assert optimization_pass.minimum_reached() is True  # matches optimal solution.

    assert (
        interface.export_subcircuits_as_string(name_mapping="default") == "ABCDDEF"
    )  # circuit separated into 2 subcircuits.


def test_one_wire_cut(
    seven_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    qubits_per_subcircuit = 4

    interface, settings = seven_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.cut_actions_sublist()

    assert cut_actions_list == [
        OneWireCutIdentifier(
            cut_action="CutLeftWire",
            wire_cut_location=WireCutLocation(
                instruction_id=10, gate_name="cx", qubits=[3, 4], input=1
            ),
        )
    ]

    assert (
        interface.export_subcircuits_as_string(name_mapping="default") == "AAAABBBB"
    )  # extra wires because of wire cuts
    # and not qubit reuse.

    best_result = optimization_pass.get_results()

    assert output.upper_bound_gamma() == best_result.gamma_UB == 4  # One LO wire cut.

    assert optimization_pass.minimum_reached() is True  # matches optimal solution


def test_two_wire_cuts(
    seven_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    qubits_per_subcircuit = 3

    interface, settings = seven_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.cut_actions_sublist()

    assert cut_actions_list == [
        OneWireCutIdentifier(
            cut_action="CutRightWire",
            wire_cut_location=WireCutLocation(
                instruction_id=9, gate_name="cx", qubits=[2, 3], input=2
            ),
        ),
        OneWireCutIdentifier(
            cut_action="CutLeftWire",
            wire_cut_location=WireCutLocation(
                instruction_id=11, gate_name="cx", qubits=[3, 5], input=1
            ),
        ),
    ]

    assert (
        interface.export_subcircuits_as_string(name_mapping="default") == "AABABCBCC"
    )  # extra wires because of wire cuts
    # and no qubit reuse. In the string above,
    # {A: wire 0, A:wire 1, B:wire 2, A: wire 3,
    # B: first cut on wire 3, C: second cut on wire 3,
    # B: wire 4, C: wire 5, C: wire 6}.

    best_result = optimization_pass.get_results()

    assert output.upper_bound_gamma() == best_result.gamma_UB == 16  # Two LO wire cuts.

    assert optimization_pass.minimum_reached() is True  # matches optimal solution


# check if unsupported search engine is flagged.
def test_supported_search_engine(
    four_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    qubits_per_subcircuit = 4

    interface, settings = four_qubit_test_setup

    settings.set_engine_selection("CutOptimization", "BeamSearch")

    search_engine = settings.get_engine_selection("CutOptimization")

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    with raises(ValueError) as e_info:
        _ = optimization_pass.optimize()
    assert e_info.value.args[0] == f"Search engine {search_engine} is not supported."


# The cutting of multiqubit gates is not supported at present.
def test_multiqubit_cuts(
    multiqubit_gate_test_setup: Callable[
        [], tuple[SimpleGateList, OptimizationSettings]
    ]
):
    # QPU with 2 qubits requires cutting.
    qubits_per_subcircuit = 2

    interface, settings = multiqubit_gate_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    with raises(ValueError) as e_info:
        _ = optimization_pass.optimize()
    assert e_info.value.args[0] == (
        "The input circuit must contain only single and two-qubits gates. "
        "Found 3-qubit gate: (ccx)."
    )


# Even if the input cost bounds are too stringent, greedy_cut_optimization
# is able to return a solution.
def test_greedy_search(
    four_qubit_test_setup: Callable[[], tuple[SimpleGateList, OptimizationSettings]]
):
    qubits_per_subcircuit = 3

    interface, settings = four_qubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_subcircuit)

    # Impose a stringent cost upper bound, insist gamma <=2.
    cut_opt = CutOptimization(interface, settings, constraint_obj)
    cut_opt.update_upperbound_cost((2, 4))
    state, cost = cut_opt.optimization_pass()

    # 2 cnot cuts are still found
    assert state is not None
    assert cost[0] == 9
