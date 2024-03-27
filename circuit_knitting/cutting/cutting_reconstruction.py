# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for reconstructing the results of circuit cutting experiments."""

from __future__ import annotations

from collections.abc import Sequence, Hashable

import numpy as np
from qiskit.quantum_info import PauliList
from qiskit.primitives import (
    SamplerResult,  # for SamplerV1
    PrimitiveResult,  # for SamplerV2
)

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.bitwise import bit_count
from .cutting_decomposition import decompose_observables
from .cutting_experiments import _get_pauli_indices
from .qpd import WeightType


def reconstruct_expectation_values(
    results: (
        SamplerResult
        | PrimitiveResult
        | dict[Hashable, SamplerResult | PrimitiveResult]
    ),
    coefficients: Sequence[tuple[float, WeightType]],
    observables: PauliList | dict[Hashable, PauliList],
) -> list[float]:
    r"""
    Reconstruct an expectation value from the results of the sub-experiments.

    Args:
        results: The results from running the cutting subexperiments. If the cut circuit
            was not partitioned between qubits and run separately, this argument should be
            a :class:`~qiskit.primitives.SamplerResult` instance or a dictionary mapping
            a single partition to the results. If the circuit was partitioned and its
            pieces were run separately, this argument should be a dictionary mapping partition labels
            to the results from each partition's subexperiments.

            The subexperiment results are expected to be ordered in the same way the subexperiments
            are ordered in the output of :func:`.generate_cutting_experiments` -- one result for every
            sample and observable, as shown below. The Qiskit Sampler primitive will return the results
            in the same order the experiments are submitted, so users who do not use :func:`.generate_cutting_experiments`
            to generate their experiments should take care to order their subexperiments as follows before submitting them
            to the sampler primitive:

            :math:`[sample_{0}observable_{0}, \ldots, sample_{0}observable_{N-1}, sample_{1}observable_{0}, \ldots, sample_{M-1}observable_{N-1}]`

        coefficients: A sequence containing the coefficient associated with each unique subexperiment. Each element is a tuple
            containing the coefficient (a ``float``) together with its :class:`.WeightType`, which denotes
            how the value was generated. The contribution from each subexperiment will be multiplied by
            its corresponding coefficient, and the resulting terms will be summed to obtain the reconstructed expectation value.
        observables: The observable(s) for which the expectation values will be calculated.
            This should be a :class:`~qiskit.quantum_info.PauliList` if ``results`` is a
            :class:`~qiskit.primitives.SamplerResult` instance. Otherwise, it should be a
            dictionary mapping partition labels to the observables associated with that partition.

    Returns:
        A ``list`` of ``float``\ s, such that each float is an expectation
        value corresponding to the input observable in the same position

    Raises:
        ValueError: ``observables`` and ``results`` are of incompatible types.
        ValueError: An input observable has a phase not equal to 1.
    """
    if isinstance(observables, PauliList) and not isinstance(
        results, (SamplerResult, PrimitiveResult)
    ):
        raise ValueError(
            "If observables is a PauliList, results must be a SamplerResult or PrimitiveResult instance."
        )
    if isinstance(observables, dict) and not isinstance(results, dict):
        raise ValueError(
            "If observables is a dictionary, results must also be a dictionary."
        )

    # If circuit was not separated, transform input data structures to dictionary format
    if isinstance(observables, PauliList):
        if any(obs.phase != 0 for obs in observables):
            raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        results_dict: dict[Hashable, SamplerResult | PrimitiveResult] = {"A": results}
        expvals = np.zeros(len(observables))

    else:
        results_dict = results
        for label, subobservable in observables.items():
            if any(obs.phase != 0 for obs in subobservable):
                raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = observables
        expvals = np.zeros(len(list(observables.values())[0]))

    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }

    # Reconstruct the expectation values
    for i, coeff in enumerate(coefficients):
        current_expvals = np.ones((len(expvals),))
        for label, so in subsystem_observables.items():
            subsystem_expvals = [
                np.zeros(len(cog.commuting_observables)) for cog in so.groups
            ]
            current_result = results_dict[label]
            for k, cog in enumerate(so.groups):
                idx = i * len(so.groups) + k
                if isinstance(current_result, SamplerResult):
                    # SamplerV1 provides a SamplerResult
                    quasi_probs = current_result.quasi_dists[idx]
                    for outcome, quasi_prob in quasi_probs.items():
                        subsystem_expvals[k] += quasi_prob * _process_outcome(
                            cog, outcome
                        )
                else:
                    # SamplerV2 provides a PrimitiveResult
                    data_pub = current_result[idx].data
                    obs_array = data_pub.observable_measurements.array
                    qpd_array = data_pub.qpd_measurements.array
                    shots = qpd_array.shape[0]
                    for j in range(shots):
                        obs_outcomes = int.from_bytes(obs_array[j], "big")
                        qpd_outcomes = int.from_bytes(qpd_array[j], "big")
                        subsystem_expvals[k] += (1 / shots) * _process_outcome_v2(
                            cog, obs_outcomes, qpd_outcomes
                        )

            for k, subobservable in enumerate(subobservables_by_subsystem[label]):
                current_expvals[k] *= np.mean(
                    [subsystem_expvals[m][n] for m, n in so.lookup[subobservable]]
                )

        expvals += coeff[0] * current_expvals

    return list(expvals)


def _process_outcome(
    cog: CommutingObservableGroup, outcome: int | str, /
) -> np.typing.NDArray[np.float64]:
    """
    Process a single outcome of a QPD experiment with observables.

    Args:
        cog: The observable set being measured by the current experiment
        outcome: The outcome of the classical bits

    Returns:
        A 1D array of the observable measurements.  The elements of
        this vector correspond to the elements of ``cog.commuting_observables``,
        and each result will be either +1 or -1.
    """
    num_meas_bits = len(_get_pauli_indices(cog))

    outcome = _outcome_to_int(outcome)
    obs_outcomes = outcome & ((1 << num_meas_bits) - 1)
    qpd_outcomes = outcome >> num_meas_bits

    return _process_outcome_v2(cog, obs_outcomes, qpd_outcomes)


def _process_outcome_v2(
    cog: CommutingObservableGroup, obs_outcomes: int, qpd_outcomes: int, /
) -> np.typing.NDArray[np.float64]:
    """
    Process a single outcome of a QPD experiment with observables.

    Args:
        cog: The observable set being measured by the current experiment
        obs_outcomes: An integer containing the outcome bits of the ``observable_measurements`` register
        qpd_outcomes: An integer containing the outcome bits of the ``qpd_measurements`` register

    Returns:
        A 1D array of the observable measurements.  The elements of
        this vector correspond to the elements of ``cog.commuting_observables``,
        and each result will be either +1 or -1.
    """
    # qpd_factor will be -1 or +1, depending on the overall parity of qpd
    # measurements.
    qpd_factor = 1 - 2 * (bit_count(qpd_outcomes) & 1)

    rv = np.zeros(len(cog.pauli_bitmasks))
    for i, mask in enumerate(cog.pauli_bitmasks):
        # obs will be -1 or +1, depending on the measurement
        # of the current operator.
        obs = 1 - 2 * (bit_count(obs_outcomes & mask) & 1)
        rv[i] = qpd_factor * obs

    return rv


def _outcome_to_int(outcome: int | str) -> int:
    if isinstance(outcome, int):
        return outcome
    outcome = outcome.replace(" ", "")
    if len(outcome) < 2 or outcome[1] in ("0", "1"):
        outcome = outcome.replace(" ", "")
        return int(f"0b{outcome}", 0)
    return int(outcome, 0)
