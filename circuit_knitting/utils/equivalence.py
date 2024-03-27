# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Equivalence utilities.

.. currentmodule:: circuit_knitting.utils.equivalence

.. autosummary::
   :toctree: ../stubs/

"""
from collections import defaultdict

import numpy as np
from qiskit.circuit import (
    EquivalenceLibrary,
    QuantumCircuit,
    QuantumRegister,
    Parameter,
)
from qiskit.circuit.library.standard_gates import (
    RZGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    IGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    PhaseGate,
)

_eagle_sel = HeronEquivalenceLibrary = EagleEquivalenceLibrary = EquivalenceLibrary()
equivalence_libraries = defaultdict(
    lambda: None, {"heron": EagleEquivalenceLibrary, "eagle": EagleEquivalenceLibrary}
)

########## Single-qubit Eagle native gate set: x, sx, rz, i ##########
# XGate
q = QuantumRegister(1, "q")
def_x = QuantumCircuit(q)
def_x.append(XGate(), [0], [])
_eagle_sel.add_equivalence(XGate(), def_x)

# SXGate
q = QuantumRegister(1, "q")
def_sx = QuantumCircuit(q)
def_sx.append(SXGate(), [0], [])
_eagle_sel.add_equivalence(SXGate(), def_sx)

# RZGate
q = QuantumRegister(1, "q")
def_rz = QuantumCircuit(q)
theta = Parameter("theta")
def_rz.append(RZGate(theta), [0], [])
_eagle_sel.add_equivalence(RZGate(theta), def_rz)

# IGate
q = QuantumRegister(1, "q")
def_i = QuantumCircuit(q)
def_i.append(IGate(), [0], [])
_eagle_sel.add_equivalence(IGate(), def_i)

######################################################################

# YGate
q = QuantumRegister(1, "q")
def_y = QuantumCircuit(q)
for inst in [RZGate(np.pi), XGate()]:
    def_y.append(inst, [0], [])
_eagle_sel.add_equivalence(YGate(), def_y)

# ZGate
q = QuantumRegister(1, "q")
def_z = QuantumCircuit(q)
def_z.append(RZGate(np.pi), [0], [])
_eagle_sel.add_equivalence(ZGate(), def_z)

# HGate
q = QuantumRegister(1, "q")
def_h = QuantumCircuit(q)
for inst in [RZGate(np.pi / 2), SXGate(), RZGate(np.pi / 2)]:
    def_h.append(inst, [0], [])
_eagle_sel.add_equivalence(HGate(), def_h)

# SGate
q = QuantumRegister(1, "q")
def_s = QuantumCircuit(q)
def_s.append(RZGate(np.pi / 2), [0], [])
_eagle_sel.add_equivalence(SGate(), def_s)

# SdgGate
q = QuantumRegister(1, "q")
def_sdg = QuantumCircuit(q)
def_sdg.append(RZGate(-np.pi / 2), [0], [])
_eagle_sel.add_equivalence(SdgGate(), def_sdg)

# SXdgGate
q = QuantumRegister(1, "q")
def_sxdg = QuantumCircuit(q)
for inst in [
    RZGate(np.pi / 2),
    RZGate(np.pi / 2),
    SXGate(),
    RZGate(np.pi / 2),
    RZGate(np.pi / 2),
]:
    def_sxdg.append(inst, [0], [])
_eagle_sel.add_equivalence(SXdgGate(), def_sxdg)

# TGate
q = QuantumRegister(1, "q")
def_t = QuantumCircuit(q)
def_t.append(RZGate(np.pi / 4), [0], [])
_eagle_sel.add_equivalence(TGate(), def_t)

# TdgGate
q = QuantumRegister(1, "q")
def_tdg = QuantumCircuit(q)
def_tdg.append(RZGate(-np.pi / 4), [0], [])
_eagle_sel.add_equivalence(TdgGate(), def_tdg)

# RXGate
q = QuantumRegister(1, "q")
def_rx = QuantumCircuit(q)
theta = Parameter("theta")
for inst in [RZGate(np.pi / 2), SXGate(), RZGate(theta + np.pi), RZGate(5 * np.pi / 2)]:
    def_rx.append(inst, [0], [])
_eagle_sel.add_equivalence(RXGate(theta), def_rx)

# RYGate
q = QuantumRegister(1, "q")
def_ry = QuantumCircuit(q)
theta = Parameter("theta")
for inst in [SXGate(), RZGate(theta + np.pi), SXGate(), RZGate(3 * np.pi)]:
    def_ry.append(inst, [0], [])
_eagle_sel.add_equivalence(RYGate(theta), def_ry)

# PhaseGate
q = QuantumRegister(1, "q")
def_p = QuantumCircuit(q)
theta = Parameter("theta")
def_p.append(RZGate(theta), [0], [])
_eagle_sel.add_equivalence(PhaseGate(theta), def_p)
