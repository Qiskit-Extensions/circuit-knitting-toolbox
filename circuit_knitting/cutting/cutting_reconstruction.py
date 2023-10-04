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
from qiskit.primitives import SamplerResult

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.bitwise import bit_count
from ..utils.iteration import strict_zip
from .cutting_decomposition import decompose_observables
from .qpd import WeightType


def reconstruct_distribution(
    results: SamplerResult,
    original_circuit_num_clbits: int,
    coefficients: Sequence[tuple[float, WeightType]],
):
    """Reconstruct probability distribution."""
    num_meas_bits = original_circuit_num_clbits
    quasi_dists_out: dict[str | int, float] = {}
    for quasi_dist, (coeff, _) in strict_zip(results.quasi_dists, coefficients):
        for outcome, weight in quasi_dist.items():
            # NOTE: The registers are in the opposite order compared to
            # cutting_reconstruction._process_outcome.
            meas_outcomes = outcome & ((1 << num_meas_bits) - 1)
            qpd_outcomes = outcome >> num_meas_bits
            qpd_factor = 1 - 2 * (bit_count(qpd_outcomes) & 1)
            quasi_dists_out[meas_outcomes] = (
                quasi_dists_out.get(meas_outcomes, 0.0) + coeff * qpd_factor * weight
            )
    return quasi_dists_out


def reconstruct_expectation_values(
    results: SamplerResult | dict[Hashable, SamplerResult],
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
        ValueError: ``num_qpd_bits`` must be set for all result metadata dictionaries.
        TypeError: ``num_qpd_bits`` must be an integer.
    """
    if isinstance(observables, PauliList) and not isinstance(results, SamplerResult):
        raise ValueError(
            "If observables is a PauliList, results must be a SamplerResult instance."
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
        results_dict: dict[Hashable, SamplerResult] = {"A": results}
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
            for k, cog in enumerate(so.groups):
                quasi_probs = results_dict[label].quasi_dists[i * len(so.groups) + k]
                for outcome, quasi_prob in quasi_probs.items():
                    try:
                        num_qpd_bits = results_dict[label].metadata[
                            i * len(so.groups) + k
                        ]["num_qpd_bits"]
                    except KeyError as ex:
                        raise ValueError(
                            "The num_qpd_bits field must be set in each subexperiment "
                            "result metadata dictionary."
                        ) from ex
                    else:
                        subsystem_expvals[k] += quasi_prob * _process_outcome(
                            num_qpd_bits,
                            cog,
                            outcome,
                        )

            for k, subobservable in enumerate(subobservables_by_subsystem[label]):
                current_expvals[k] *= np.mean(
                    [subsystem_expvals[m][n] for m, n in so.lookup[subobservable]]
                )

        expvals += coeff[0] * current_expvals

    return list(expvals)


def _process_outcome(
    num_qpd_bits: int, cog: CommutingObservableGroup, outcome: int | str, /
) -> np.typing.NDArray[np.float64]:
    """
    Process a single outcome of a QPD experiment with observables.

    Args:
        num_qpd_bits: The number of QPD measurements in the circuit. It is
            assumed that the second to last creg in the generating circuit
            is the creg  containing the QPD measurements, and the last
            creg is associated with the observable measurements.
        cog: The observable set being measured by the current experiment
        outcome: The outcome of the classical bits

    Returns:
        A 1D array of the observable measurements.  The elements of
        this vector correspond to the elements of ``cog.commuting_observables``,
        and each result will be either +1 or -1.
    """
    outcome = _outcome_to_int(outcome)
    try:
        qpd_outcomes = outcome & ((1 << num_qpd_bits) - 1)
    except TypeError as ex:
        raise TypeError(
            f"num_qpd_bits must be an integer, but a {type(num_qpd_bits)} was passed."
        ) from ex

    meas_outcomes = outcome >> num_qpd_bits

    # qpd_factor will be -1 or +1, depending on the overall parity of qpd
    # measurements.
    qpd_factor = 1 - 2 * (bit_count(qpd_outcomes) & 1)

    rv = np.zeros(len(cog.pauli_bitmasks))
    for i, mask in enumerate(cog.pauli_bitmasks):
        # meas will be -1 or +1, depending on the measurement
        # of the current operator.
        meas = 1 - 2 * (bit_count(meas_outcomes & mask) & 1)
        rv[i] = qpd_factor * meas

    return rv


def _outcome_to_int(outcome: int | str) -> int:
    if isinstance(outcome, int):
        return outcome
    outcome = outcome.replace(" ", "")
    if len(outcome) < 2 or outcome[1] in ("0", "1"):
        outcome = outcome.replace(" ", "")
        return int(f"0b{outcome}", 0)
    return int(outcome, 0)
