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

.. currentmodule:: circuit_knitting.cutting.qpd.equivalence

.. autosummary::
   :toctree: ../stubs/

"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit.library.standard_gates import (
    RZGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    PhaseGate,
)


_equivalence_from_gate_funcs: dict[str, Callable[[Gate], QuantumCircuit]] = {}


def _register_gate(*args):
    def g(f):
        for name in args:
            _equivalence_from_gate_funcs[name] = f
        return f

    return g


def translate_qpd_gate(gate: Gate, basis_gate_set: str, /) -> QuantumCircuit:
    """
    Translate a ``gate`` into a given basis gate set.

    This function is designed to handle only the gates to which a :class:`.QPDBasis` can
    decompose; therefore, not every Qiskit gate is supported by this function.

    Args:
        gate: The gate to translate

    Returns:
        A :class:`qiskit.QuantumCircuit` implementing the gate in the given basis gate set.

    Raises:
        ValueError: Unsupported basis gate set
        ValueError: Unsupported gate
    """
    # We otherwise ignore this arg for now since our only two equivalences are equivalent :)
    if basis_gate_set not in {"heron", "eagle"}:
        raise ValueError(f"Unknown basis gate set: {basis_gate_set}")
    try:
        f = _equivalence_from_gate_funcs[gate.name]
    except KeyError as exc:
        raise ValueError(f"Cannot translate gate: {gate.name}") from exc
    else:
        return f(gate)


# XGate
@_register_gate("x")
def _(gate: XGate):
    q = QuantumRegister(1, "q")
    def_x = QuantumCircuit(q)
    def_x.append(gate, [0], [])
    return def_x


# SXGate
@_register_gate("sx")
def _(gate: SXGate):
    q = QuantumRegister(1, "q")
    def_sx = QuantumCircuit(q)
    def_sx.append(gate, [0], [])
    return def_sx


# RZGate
@_register_gate("rz")
def _(gate: RZGate):
    q = QuantumRegister(1, "q")
    def_rz = QuantumCircuit(q)
    def_rz.append(gate, [0], [])
    return def_rz


# YGate
@_register_gate("y")
def _(_: YGate):
    q = QuantumRegister(1, "q")
    def_y = QuantumCircuit(q)
    for inst in [RZGate(np.pi), XGate()]:
        def_y.append(inst, [0], [])
    return def_y


# ZGate
@_register_gate("z")
def _(_: ZGate):
    q = QuantumRegister(1, "q")
    def_z = QuantumCircuit(q)
    def_z.append(RZGate(np.pi), [0], [])
    return def_z


# HGate
@_register_gate("h")
def _(_: HGate):
    q = QuantumRegister(1, "q")
    def_h = QuantumCircuit(q)
    for inst in [RZGate(np.pi / 2), SXGate(), RZGate(np.pi / 2)]:
        def_h.append(inst, [0], [])
    return def_h


# SGate
@_register_gate("s")
def _(_: SGate):
    q = QuantumRegister(1, "q")
    def_s = QuantumCircuit(q)
    def_s.append(RZGate(np.pi / 2), [0], [])
    return def_s


# SdgGate
@_register_gate("sdg")
def _(_: SdgGate):
    q = QuantumRegister(1, "q")
    def_sdg = QuantumCircuit(q)
    def_sdg.append(RZGate(-np.pi / 2), [0], [])
    return def_sdg


# SXdgGate
@_register_gate("sxdg")
def _(_: SXdgGate):
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
    return def_sxdg


# TGate
@_register_gate("t")
def _(_: TGate):
    q = QuantumRegister(1, "q")
    def_t = QuantumCircuit(q)
    def_t.append(RZGate(np.pi / 4), [0], [])
    return def_t


# TdgGate
@_register_gate("tdg")
def _(_: TdgGate):
    q = QuantumRegister(1, "q")
    def_tdg = QuantumCircuit(q)
    def_tdg.append(RZGate(-np.pi / 4), [0], [])
    return def_tdg


# RXGate
@_register_gate("rx")
def _(gate: RXGate):
    q = QuantumRegister(1, "q")
    def_rx = QuantumCircuit(q)
    param = gate.params[0]
    for inst in [
        RZGate(np.pi / 2),
        SXGate(),
        RZGate(param + np.pi),
        SXGate(),
        RZGate(5 * np.pi / 2),
    ]:
        def_rx.append(inst, [0], [])
    return def_rx


# RYGate
@_register_gate("ry")
def _(gate: RYGate):
    q = QuantumRegister(1, "q")
    def_ry = QuantumCircuit(q)
    param = gate.params[0]
    for inst in [SXGate(), RZGate(param + np.pi), SXGate(), RZGate(3 * np.pi)]:
        def_ry.append(inst, [0], [])


# PhaseGate
@_register_gate("p")
def _(gate: PhaseGate):
    q = QuantumRegister(1, "q")
    def_p = QuantumCircuit(q)
    param = gate.params[0]
    def_p.append(RZGate(param), [0], [])
    return def_p
