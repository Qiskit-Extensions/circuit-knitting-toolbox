# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function for automatically finding locations for gate and wire cuts."""

from __future__ import annotations

from typing import cast, Any
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit, CircuitInstruction

from .instructions import CutWire
from .cutting_decomposition import cut_gates
from .cut_finding.optimization_settings import OptimizationSettings
from .cut_finding.disjoint_subcircuits_state import DisjointSubcircuitsState
from .cut_finding.circuit_interface import SimpleGateList
from .cut_finding.lo_cuts_optimizer import LOCutsOptimizer
from .cut_finding.cco_utils import qc_to_cco_circuit


def find_cuts(
    circuit: QuantumCircuit,
    optimization: OptimizationParameters,
    constraints: DeviceConstraints,
) -> tuple[QuantumCircuit, dict[str, float]]:
    """Find cut locations in a circuit, given optimization parameters and cutting constraints.

    Args:
        circuit: The circuit to cut. The input circuit may not contain gates acting
            on more than two qubits.
        optimization: Options for controlling optimizer behavior. Currently, the optimal
            cuts are chosen using Dijkstra's best-first search algorithm.
        constraints: Constraints on how the circuit may be partitioned
    Returns:
        A circuit containing :class:`.BaseQPDGate` instances. The subcircuits
        resulting from cutting these gates will be runnable on the devices meeting
        the ``constraints``.

        A metadata dictionary:
            - cuts: A list of length-2 tuples describing each cut in the output circuit.
              The tuples are formatted as ``(cut_type: str, cut_id: int)``. The
              cut ID is the index of the cut gate or wire in the output circuit's
              ``data`` field.
            - sampling_overhead: The sampling overhead incurred from cutting the specified
              gates and wires.
            - minimum_reached: A bool indicating whether or not the search conclusively found
              the minimum of cost function. ``minimum_reached = False`` could also mean that the
              cost returned was actually the lowest possible cost but that the search was
              not allowed to run long enough to prove that this was the case.

    Raises:
        ValueError: The input circuit contains a gate acting on more than 2 qubits.
    """
    circuit_cco = qc_to_cco_circuit(circuit)
    interface = SimpleGateList(circuit_cco)

    opt_settings = OptimizationSettings(
        seed=optimization.seed,
        max_gamma=optimization.max_gamma,
        max_backjumps=optimization.max_backjumps,
        gate_lo=optimization.gate_lo,
        wire_lo=optimization.wire_lo,
    )

    # Hard-code the optimizer to an LO-only optimizer
    optimizer = LOCutsOptimizer(interface, opt_settings, constraints)

    # Find cut locations
    opt_out = optimizer.optimize()

    wire_cut_actions = []
    gate_ids = []

    opt_out = cast(DisjointSubcircuitsState, opt_out)
    opt_out.actions = cast(list, opt_out.actions)
    for action in opt_out.actions:
        if action.action.get_name() == "CutTwoQubitGate":
            gate_ids.append(action.gate_spec.instruction_id)
        else:
            # The cut-finding optimizer currently only supports 4 cutting
            # actions: {CutTwoQubitGate + these 3 wire cut types}
            assert action.action.get_name() in (
                "CutLeftWire",
                "CutRightWire",
                "CutBothWires",
            )
            wire_cut_actions.append(action)

    # First, replace all gates to cut with BaseQPDGate instances.
    # This assumes each gate to cut is replaced 1-to-1 with a QPD gate.
    # This may not hold in the future as we stop treating gate cuts individually.
    circ_out = cut_gates(circuit, gate_ids)[0]

    # Insert all the wire cuts
    counter = 0
    for action in sorted(wire_cut_actions, key=lambda a: a[1][0]):
        inst_id = action.gate_spec.instruction_id
        # action.args[0][0] will be either 1 (control) or 2 (target)
        qubit_id = action.args[0][0] - 1
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [circuit.data[inst_id].qubits[qubit_id]], []),
        )
        counter += 1

        if action.action.get_name() == "CutBothWires":
            # There should be two wires specified in the action in this case
            assert len(action.args) == 2
            qubit_id2 = action.args[1][0] - 1
            circ_out.data.insert(
                inst_id + counter,
                CircuitInstruction(
                    CutWire(), [circuit.data[inst_id].qubits[qubit_id2]], []
                ),
            )
            counter += 1

    # Return metadata describing the cut scheme
    metadata: dict[str, Any] = {"cuts": []}
    for i, inst in enumerate(circ_out.data):
        if inst.operation.name == "qpd_2q":
            metadata["cuts"].append(("Gate Cut", i))
        elif inst.operation.name == "cut_wire":
            metadata["cuts"].append(("Wire Cut", i))
    metadata["sampling_overhead"] = opt_out.upper_bound_gamma() ** 2
    metadata["minimum_reached"] = optimizer.minimum_reached()
    if opt_out.cut_actions_sublist is not None:
        metadata["actions"] = [(x[0], x[1]) for x in opt_out.cut_actions_sublist()]
    else:
        metadata["actions"] = "No cuts found"

    return circ_out, metadata


@dataclass
class OptimizationParameters:
    """Specify parameters that control the optimization.

    If either of the constraints specified by ``max_backjumps`` or ``max_gamma`` are exceeded, the search terminates but
    nevertheless returns the result of a greedy best first search, which gives an *upper-bound* on gamma.
    """

    #: The seed to use when initializing Numpy random number generators in the best first search priority queue.
    seed: int | None = OptimizationSettings().seed

    #: Maximum allowed value of gamma which, if exceeded, forces the search to terminate.
    max_gamma: float = OptimizationSettings().max_gamma

    #: Maximum number of backjumps that can be performed before the search is forced to terminate; setting it to ``None`` implies that no such restriction is placed.
    max_backjumps: None | int = OptimizationSettings().max_backjumps

    #: Bool indicating whether or not to allow LO gate cuts while finding cuts.
    gate_lo: bool = OptimizationSettings().gate_lo

    #: Bool indicating whether or not to allow LO wire cuts while finding cuts.
    wire_lo: bool = OptimizationSettings().wire_lo


@dataclass
class DeviceConstraints:
    """Specify the constraints (qubits per subcircuit) that must be respected."""

    qubits_per_subcircuit: int

    def __post_init__(self):
        """Post-init method for data class."""
        if self.qubits_per_subcircuit < 1:
            raise ValueError(
                "qubits_per_subcircuit must be a positive definite integer."
            )

    def get_qpu_width(self) -> int:
        """Return the number of qubits per subcircuit."""
        return self.qubits_per_subcircuit
