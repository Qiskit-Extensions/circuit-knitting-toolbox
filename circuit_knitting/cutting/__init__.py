# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Circuit Cutting (:mod:`circuit_knitting.cutting`).

.. currentmodule:: circuit_knitting.cutting

Circuit Cutting
===============

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    find_cuts
    cut_wires
    expand_observables
    partition_circuit_qubits
    partition_problem
    cut_gates
    generate_cutting_experiments
    reconstruct_expectation_values

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    PartitionedCuttingProblem
    instructions.CutWire
    instructions.Move

Quasi-Probability Decomposition (QPD)
=====================================

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    qpd.QPDBasis
    qpd.BaseQPDGate
    qpd.SingleQubitQPDGate
    qpd.TwoQubitQPDGate
    qpd.WeightType

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    qpd.generate_qpd_weights
    qpd.decompose_qpd_instructions
    qpd.qpdbasis_from_instruction

Automatic Cut Finding
=====================

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    cut_finding.OptimizationParameters
    cut_finding.DeviceConstraints

CutQC
=====

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    cutqc.run_subcircuit_instances
    cutqc.generate_summation_terms
    cutqc.build
    cutqc.verify
    cutqc.cut_circuit_wires
    cutqc.evaluate_subcircuits
    cutqc.reconstruct_full_distribution
"""

from .cutting_decomposition import (
    partition_circuit_qubits,
    partition_problem,
    cut_gates,
    PartitionedCuttingProblem,
)
from .find_cuts import find_cuts
from .cutting_experiments import generate_cutting_experiments
from .cutting_reconstruction import reconstruct_expectation_values
from .wire_cutting_transforms import cut_wires, expand_observables
from .cut_finding.quantum_device_constraints import DeviceConstraints
from .cut_finding.optimization_settings import OptimizationParameters

__all__ = [
    "find_cuts",
    "partition_circuit_qubits",
    "partition_problem",
    "cut_gates",
    "generate_cutting_experiments",
    "reconstruct_expectation_values",
    "PartitionedCuttingProblem",
    "cut_wires",
    "expand_observables",
    "DeviceConstraints",
    "OptimizationParameters",
]
