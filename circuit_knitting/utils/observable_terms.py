# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""Utilities for working with the unique terms of a collection of :class:`~qiskit.quantum_info.SparsePauliOp`\ s."""

from __future__ import annotations

from typing import Sequence, Iterable, Mapping

from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from .iteration import strict_zip


def gather_unique_observable_terms(
    observables: Sequence[Pauli | SparsePauliOp] | PauliList,
) -> PauliList:
    """Inspect the contents of each observable to find and return the unique Pauli terms."""
    if not observables:
        if isinstance(observables, PauliList):
            return observables.copy()
        raise ValueError("observables list cannot be empty")

    pauli_list: list[Pauli] = []
    pauli_set: set[Pauli] = set()
    for observable in observables:
        if isinstance(observable, Pauli):
            observable = SparsePauliOp(observable)
        for pauli, coeff in strict_zip(observable.paulis, observable.coeffs):
            assert pauli.phase == 0  # SparsePauliOp should do this for us
            if coeff == 0:
                continue
            if pauli not in pauli_set:
                pauli_list.append(pauli)
                pauli_set.add(pauli)

    if not pauli_list:
        # Handle this special case, which can happen if all coeffs are zero.
        # We create an empty PauliList with the correct number of qubits.  See
        # https://github.com/Qiskit/qiskit/pull/9770 for a proposal to make
        # this simpler in Qiskit.
        return PauliList(["I" * observables[0].num_qubits])[:0]

    try:
        return PauliList(pauli_list)
    except ValueError as ex:
        raise ValueError(
            "Cannot construct PauliList.  Do provided observables all have "
            "the same number of qubits?"
        ) from ex


def _reconstruct_observable_expval_from_terms(
    observable: Pauli | SparsePauliOp,
    term_expvals: Mapping[Pauli, float | complex],
) -> complex:
    if isinstance(observable, Pauli):
        observable = SparsePauliOp(observable)
    rv = 0.0j
    for pauli, coeff in strict_zip(observable.paulis, observable.coeffs):
        assert pauli.phase == 0  # SparsePauliOp should do this for us
        if coeff == 0:
            continue
        try:
            term_expval = term_expvals[pauli]
        except KeyError as ex:
            raise ValueError(
                "An observable contains a Pauli term whose expectation value "
                "was not provided."
            ) from ex
        rv += coeff * term_expval
    return rv


def reconstruct_observable_expvals_from_terms(
    observables: Iterable[Pauli | SparsePauliOp] | PauliList,
    term_expvals: Mapping[Pauli, float | complex],
) -> list[complex]:
    """Reconstruct the expectation values given the expectation value of each unique term."""
    return [
        _reconstruct_observable_expval_from_terms(observable, term_expvals)
        for observable in observables
    ]
