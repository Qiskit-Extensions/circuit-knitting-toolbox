# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Module for conducting Pauli observable grouping.

.. currentmodule:: circuit_knitting.utils.observable_grouping

.. autosummary::
   :toctree: ../stubs

   observables_restricted_to_subsystem
   CommutingObservableGroup
   ObservableCollection
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import Pauli, PauliList

from .iteration import strict_zip


_I = Pauli("I")


def observables_restricted_to_subsystem(
    qubits: Sequence[int], global_observables: Sequence[Pauli] | PauliList, /
) -> list[Pauli] | PauliList:
    """Restrict each observable to its support on a given subsystem.

    A :class:`~qiskit.quantum_info.PauliList` will be returned if a :class:`~qiskit.quantum_info.PauliList` is provided; otherwise,
    a ``list[Pauli]`` will be returned.

    Any phase information will be discarded, consistent with the standard behavior when slicing a Pauli.

    Args:
        qubits: The qubits in a subsystem
        global_observables: The list of observables

    Returns:
        Each :class:`~qiskit.quantum_info.Pauli` restricted to the subsystem.

    >>> observables_restricted_to_subsystem([1, 3], PauliList(["IXYZ", "iZZXX"]))
    PauliList(['IY', 'ZX'])

    """
    if isinstance(global_observables, PauliList):
        # Optimized path if we are given a PauliList
        o = global_observables
        return PauliList.from_symplectic(o.z[:, qubits], o.x[:, qubits])
    # As a fallback, deal with each Pauli individually and return a list
    return [observable[(qubits,)] for observable in global_observables]


def most_general_observable(
    commuting_observables: PauliList | Sequence[Pauli], /, num_qubits: int | None = None
) -> Pauli:
    """Return the most general observable from a sequence of commuting observables.

    Given a list of operators over qubits claimed to be mutually qubit-wise
    commuting, return the Pauli string we can measure to determine everything
    of interest.

    Args:
        commuting_observables: Input sequence of mutually qubit-wise commuting observables
        num_qubits: Number of qubits.  If ``None``, it is inferred from
            ``commuting_observables`` (default: ``None``).

    Raises:
        ValueError: The input sequence is empty (in which case, no experiment is even needed
            to measure the observables)
        ValueError: The input sequence is _not_ mutually qubit-wise commuting
        ValueError: An observable has an unexpected ``num_qubits``

    >>> most_general_observable(PauliList(["IIIZ", "IIZZ", "XIII"]))
    Pauli('XIZZ')

    """
    if len(commuting_observables) == 0:
        raise ValueError(
            "Empty input sequence: consider performing no experiments rather than an "
            "experiment over the identity Pauli."
        )
    if num_qubits is None:
        num_qubits = len(commuting_observables[0])
    rv = Pauli((np.zeros(num_qubits), np.zeros(num_qubits)))
    # TODO(perf): This is a slow Python loop within a loop, surely slower than
    # it _could_ be, *especially* if given a PauliList.
    #
    # Indeed, this can be made better by using pauli.x and pauli.z arrays
    # https://github.com/Qiskit/qiskit-terra/blob/061aee2685676271fd0860d0a2d699e36941ae5e/qiskit/primitives/backend_estimator.py#L403-L404
    for j, obs in enumerate(commuting_observables):
        if not isinstance(obs, Pauli):
            raise ValueError("Input sequence includes something other than a Pauli.")
        if len(obs) != num_qubits:
            raise ValueError(
                f"Observable {j} has incorrect qubit count ({len(obs)} rather than "
                f"{num_qubits})."
            )
        for i, o in enumerate(obs):
            if o == _I:
                continue
            rv_i = rv[i]
            if rv_i == o:
                continue
            if rv_i != _I:
                raise ValueError(
                    "Observables are incompatible; cannot construct a single general observable."
                )
            rv[i] = o
    return rv


@dataclass(frozen=True)
class CommutingObservableGroup:
    r"""Set of mutually qubit-wise commuting observables."""

    #: A single Pauli string that contains all qubit-wise measurements needed
    #: to measure everything in ``commuting_observables``.
    general_observable: Pauli

    #: Observables that can be measured simultaneously.
    commuting_observables: list[Pauli]

    #: The indices of non-identity :class:`~qiskit.quantum_info.Pauli`\ s in ``general_observable``.
    pauli_indices: list[int] = field(init=False)

    #: A bitmask for each observable in ``commuting_observables``; given an element, each bit corresponds to whether
    #: the corresponding entry in ``pauli_indices`` is relevant to that observable.
    pauli_bitmasks: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Post-init method for the data class."""
        # TODO(perf): These loops could be faster; see e.g. https://github.com/Qiskit/qiskit-terra/blob/061aee2685676271fd0860d0a2d699e36941ae5e/qiskit/primitives/backend_estimator.py#L398-L413
        pauli_indices: list[int] = [
            i for i, pauli in enumerate(self.general_observable) if pauli != _I
        ]
        pauli_bitmasks: list[int] = []
        for pauli in self.commuting_observables:
            if pauli.phase != 0:
                raise ValueError(
                    "CommutingObservableGroup only supports Paulis with phase == 0. "
                    f"(Value provided: {pauli.phase})"
                )
            v = 0
            for i, j in enumerate(pauli_indices):
                if pauli[j] != _I:
                    v |= 1 << i
            pauli_bitmasks.append(v)

        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        # says "when using frozen=True: __init__() cannot use simple assignment
        # to initialize fields, and must use object.__setattr__()."
        object.__setattr__(self, "pauli_indices", pauli_indices)
        object.__setattr__(self, "pauli_bitmasks", pauli_bitmasks)


class ObservableCollection:
    """Collection of observables organized for efficient taking of measurements.

    The observables are automatically organized into sets of mutually
    qubit-wise commuting observables, each represented by a
    :class:`.CommutingObservableGroup`.
    """

    def __init__(self, observables: PauliList | Iterable[Pauli], /):
        """Assign member variables.

        Args:
            observables: Observables of interest
        """
        # From here forward, we only need to process *unique* observables.
        if isinstance(observables, PauliList):
            unique_observables = observables.unique()
        else:
            unique_observables = PauliList(set(observables))

        # Group the desired observables into sets that mutually commute.  We
        # make it a list[list[Pauli]] rather than a list[PauliList] so that we
        # can "fix it up" later, in case we have *multiple* general observables
        # that can yield measurements of some of the observables.  That way, we
        # are maximizing information obtained from each shot.
        commuting_groups: list[list[Pauli]] = [
            list(group) for group in unique_observables.group_commuting(qubit_wise=True)
        ]

        # Construct the most general observable from each set of mutually
        # qubit-wise commuting observables.
        general_observables = self.construct_general_observables(commuting_groups)

        # TODO: Fix up commuting_groups here.  Currently, Terra's
        # `group_commuting` will put each observable into at most a single set.
        # However, a given observable might actually be mutually commuting with
        # more than one set.  Taking advantage of this will allow us to collect
        # more statistics for the observable without doing any additional
        # quantum experiments.  To be specific, we should handle the particular
        # special case of this where multiple `general_observables` actually
        # _contain_ the observable.  This covers what is probably the most
        # important case, evaluating the "expectation value" of the identity of
        # the current subsystem (which really just involves keeping track of
        # QPD measurements only).
        # https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/155

        # For each mutually commuting group of observables, we put together a
        # CommutingObservableGroup data structure.
        groups: list[CommutingObservableGroup] = [
            CommutingObservableGroup(
                general_observable,
                commuting_observables,
            )
            for (general_observable, commuting_observables) in strict_zip(
                general_observables, commuting_groups
            )
        ]

        # Now we construct a dict that maps each observable to location(s)
        # in the groups of commuting_observables.
        lookup: dict[Pauli, list[tuple[int, int]]] = defaultdict(list)
        for i, group in enumerate(groups):
            for j, obs in enumerate(group.commuting_observables):
                lookup[obs].append((i, j))
        lookup = dict(lookup)

        self._groups = groups
        self._lookup = lookup

    @staticmethod
    def construct_general_observables(
        commuting_subobservables: list[list[Pauli]], /
    ) -> list[Pauli]:
        """Construct the most general observable from each set of mutually commuting observables.

        In special cases, advanced users may want to subclass and override this
        ``staticmethod`` in order to measure additional qubits than the default
        for each general observable.

        """
        # NOTE: We _could_ make this return a PauliList instead of a
        # list[Pauli], but currently there is no clear benefit.
        return [most_general_observable(group) for group in commuting_subobservables]

    @property
    def groups(self) -> list[CommutingObservableGroup]:
        r"""List of :class:`.CommutingObservableGroup`\ s which, together, contain all desired observables."""
        return self._groups

    @property
    def lookup(self) -> dict[Pauli, list[tuple[int, int]]]:
        r"""Get dict which maps each :class:`~qiskit.quantum_info.Pauli` observable to a list of indices, ``(i, j)``, to commuting observables in ``groups``.

        For each element of the list, it means that the :class:`~qiskit.quantum_info.Pauli` is given by
        the ``j``th commuting observable in the ``i``th group.

        This list will be of length 1 at minimum, but may potentially be longer
        if multiple :class:`.CommutingObservableGroup`\ s are compatible with the given
        :class:`~qiskit.quantum_info.Pauli`.

        """
        return self._lookup
