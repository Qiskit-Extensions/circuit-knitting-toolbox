# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code for generating quasiprobability weights."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from enum import Enum
import itertools
import logging
import math

import numpy as np
import numpy.typing as npt

from .qpd_basis import QPDBasis
from ...utils.iteration import strict_zip


logger = logging.getLogger(__name__)


# math.sin(math.pi) is just above 1e-16, so at 1e-14, we are well above that.
# Numbers like this can often come up in the QPD coefficients.
_NONZERO_ATOL = 1e-14


class WeightType(Enum):
    """Type of weight associated with a QPD sample."""

    #: A weight given in proportion to its exact weight
    EXACT = 1

    #: A weight that was determined through some sampling procedure
    SAMPLED = 2


def _min_filter_nonzero(vals: npt.NDArray[np.float64], *, atol=_NONZERO_ATOL):
    return np.min(vals[np.logical_not(np.isclose(vals, 0, atol=atol))])


def __update_running_product_after_increment(
    running_product: list[float],
    state: Sequence[int],
    coeff_probabilities: Sequence[npt.NDArray[np.float64]],
):
    """
    Update the ``running_product`` list after the ``state`` has been incremented.

    This snippet is used twice in
    :func:`_generate_exact_weights_and_conditional_probabilities_assume_sorted`;
    hence, we write it only once here.
    """
    try:
        prev = running_product[-2]
    except IndexError:
        prev = 1.0
    running_product[-1] = prev * coeff_probabilities[len(state) - 1][state[-1]]


def _generate_exact_weights_and_conditional_probabilities_assume_sorted(
    coeff_probabilities: Sequence[npt.NDArray[np.float64]], threshold: float
):
    r"""
    Determine all exact weights above ``threshold`` and the conditional probabilities necessary to sample efficiently from all other weights.

    Each yielded element will be a 2-tuple, the first element of which will be
    a ``state``, represented by a tuple of ``int``\ s.

    If ``state`` contains the same number of elements as
    ``coeff_probabilities``, then the second element will be a ``float``,
    greater than or equal to ``threshold``, that contains the exact probability
    of this ``state``.

    If, on the other hand, ``state`` contains fewer elements than
    ``coeff_probabilities``, then the second element is a 1D ``np.ndarray`` of
    the conditional probabilities that can be used to sample the remaining
    weights that have not been evaluated exactly (i.e., are below
    ``threshold``).  These will all be normalized *except* the
    top-level one, which is given by ``state == ()``.

    If there are no exact weights below a given level, then the conditional
    probabilities at that level are equivalent to the corresponding element of
    `coeff_probabilities``, and this generator does not yield them.  Skipping
    redundant conditional probabilities, like this, allows for efficient
    sampling without traversing the entire tree, which is desirable since the
    tree grows exponentially with the number of cut gates.  By contrast, the
    performance with the provided strategy is no worse than linear in
    ``threshold``, in both time and memory.

    This function assumes each element of ``coeff_probabilities`` contains
    non-negative numbers, ordered largest to smallest.  For a generalization
    that allows ``coeff_probabilities`` to be provided in any order, see
    :func:`_generate_exact_weights_and_conditional_probabilities`.
    """
    assert len(coeff_probabilities) > 0

    next_pop = False  # Stores whether the next move is a pop or not
    state = [0]
    running_product = [coeff_probabilities[0][0]]
    running_conditional_probabilities: list[npt.NDArray[np.float64]] = []
    while True:
        if next_pop:
            # Pop
            state.pop()
            if len(state) + 1 == len(running_conditional_probabilities):
                # There were some exact weights found below us, so we need to
                # yield the conditional probabilities, unless all probabilities
                # at this level are zero *and* it's not the top level.
                current_condprobs = running_conditional_probabilities.pop()
                current_condprobs[
                    np.isclose(current_condprobs, 0, atol=_NONZERO_ATOL)
                ] = 0.0
                if not state:
                    # Don't normalize the top-level conditional probabilities.
                    yield (), current_condprobs
                else:
                    norm = np.sum(current_condprobs)
                    if norm != 0:
                        # Some, but not all, weights below us are exact.  If,
                        # instead, _all_ weights had been exact, we could have
                        # skipped this yield, as there is zero probability of
                        # reaching this partial state when sampling.
                        yield tuple(state), current_condprobs / norm
                    # Update the factor one level up from the popped one.
                    running_conditional_probabilities[-1][state[-1]] *= norm
            if not state:
                # We're all done
                return
            running_product.pop()
            # Increment the state counter.
            state[-1] += 1
            if state[-1] != len(coeff_probabilities[len(state) - 1]):
                # Increment successful (no overflow).  We don't need to pop again.
                next_pop = False
                __update_running_product_after_increment(
                    running_product, state, coeff_probabilities
                )
        else:
            if running_product[-1] < threshold:
                next_pop = True
            elif len(state) < len(coeff_probabilities):
                # Append 0 to work toward a "full" `state`
                running_product.append(
                    running_product[-1] * coeff_probabilities[len(state)][0]
                )
                state.append(0)
            else:
                # `state` is full.  Yield first.
                yield tuple(state), running_product[-1]
                # Since we found something exact, we need to update running_conditional_probabilities.
                while len(running_conditional_probabilities) < len(coeff_probabilities):
                    running_conditional_probabilities.append(
                        np.array(
                            coeff_probabilities[len(running_conditional_probabilities)],
                            dtype=float,
                        )
                    )
                # It's exact, so we want no probability of sampling it going forward.
                running_conditional_probabilities[-1][state[-1]] = 0
                # Increment the state counter.
                state[-1] += 1
                if state[-1] == len(coeff_probabilities[-1]):
                    # The state counter has overflowed, so our next move should be a
                    # pop.
                    next_pop = True
                else:
                    __update_running_product_after_increment(
                        running_product, state, coeff_probabilities
                    )


def _invert_permutation(p):
    # https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def _generate_exact_weights_and_conditional_probabilities(
    coeff_probabilities: Sequence[npt.NDArray[np.float64]], threshold: float
):
    """
    Generate exact weights and conditional probabilities.

    This is identical in behavior to
    :func:`_generate_exact_weights_and_conditional_probabilities_assume_sorted`,
    except it makes no assumption on the order of the coefficients.
    """
    permutations = [np.argsort(cp)[::-1] for cp in coeff_probabilities]
    sorted_coeff_probabilities = [
        cp[permutation] for cp, permutation in zip(coeff_probabilities, permutations)
    ]
    ipermutations = [_invert_permutation(p) for p in permutations]
    for (
        coeff_indices,
        probability,
    ) in _generate_exact_weights_and_conditional_probabilities_assume_sorted(
        sorted_coeff_probabilities, threshold
    ):
        orig_coeff_indices = tuple(
            perm[idx] for perm, idx in zip(permutations, coeff_indices)
        )
        if len(coeff_indices) != len(sorted_coeff_probabilities):
            # In this case, `probability` is actually a vector of conditional
            # probabilities, so apply the inverse permutation.
            probability = probability[ipermutations[len(coeff_indices)]]
        yield orig_coeff_indices, probability


def generate_qpd_weights(
    qpd_bases: Sequence[QPDBasis], num_samples: float = 1000
) -> dict[tuple[int, ...], tuple[float, WeightType]]:
    """
    Generate weights from the joint quasiprobability distribution.

    Each weight whose absolute value is above a threshold of ``1 /
    num_samples`` will be evaluated exactly. The remaining weights -- those in
    the tail of the distribution -- will then be sampled from, resulting in at
    most ``num_samples`` unique weights.

    Args:
        qpd_bases: The :class:`QPDBasis` objects from which to generate weights
        num_samples: Controls the number of weights to generate

    Returns:
        A mapping from a given decomposition to its weight.
        Keys are tuples of indices -- one index per input :class:`QPDBasis`. The indices
        correspond to a specific decomposition mapping in the basis.

        Values are tuples.  The first element is a number corresponding to the
        weight of the contribution.  The second element is the :class:`WeightType`,
        either ``EXACT`` or ``SAMPLED``.
    """
    independent_probabilities = [np.asarray(basis.probabilities) for basis in qpd_bases]
    # In Python 3.7 and higher, dicts are guaranteed to remember their
    # insertion order.  For user convenience, we sort by exact weights first,
    # then sampled weights.  Within each, values are sorted largest to
    # smallest.
    lst = sorted(
        _generate_qpd_weights(independent_probabilities, num_samples).items(),
        key=lambda x: ((v := x[1])[1].value, -v[0]),
    )
    return dict(lst)


def _generate_qpd_weights(
    independent_probabilities: Sequence[npt.NDArray[np.float64]],
    num_samples: float,
    *,
    _samples_multiplier: int = 1,
) -> dict[tuple[int, ...], tuple[float, WeightType]]:
    if not num_samples >= 1:
        raise ValueError("num_samples must be at least 1.")

    retval: dict[tuple[int, ...], tuple[float, WeightType]] = {}

    threshold = 1 / num_samples

    # Determine if the smallest *nonzero* probability is above the threshold implied by
    # num_samples.  If so, then we can evaluate all weights exactly.
    smallest_probability = np.prod(
        [_min_filter_nonzero(probs) for probs in independent_probabilities]
    )
    if smallest_probability >= threshold:
        # All weights exactly
        logger.info("All exact weights")
        multiplier = num_samples if math.isfinite(num_samples) else 1.0
        for map_ids in itertools.product(
            *[range(len(probs)) for probs in independent_probabilities]
        ):
            probability = np.prod(
                [
                    probs[i]
                    for i, probs in strict_zip(map_ids, independent_probabilities)
                ]
            )
            if probability < _NONZERO_ATOL:
                continue
            retval[map_ids] = (multiplier * probability, WeightType.EXACT)
        return retval

    conditional_probabilities: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}
    weight_to_sample = 1.0

    largest_probability = np.prod(
        [np.max(probs) for probs in independent_probabilities]
    )
    if not largest_probability >= threshold:
        logger.info("No exact weights")
    else:
        logger.info("Some exact weights")
        for (
            map_ids,
            probability,
        ) in _generate_exact_weights_and_conditional_probabilities(
            independent_probabilities, threshold
        ):
            # As described in the docstring to
            # _generate_exact_weights_and_conditional_probabilities_assume_sorted,
            # there are two possible pieces of information that might be
            # yielded by the generator.  The following branch distinguishes
            # between them.
            if len(map_ids) == len(independent_probabilities):
                # The generator produced a state together with its exact probability.
                weight = probability * num_samples
                retval[map_ids] = (weight, WeightType.EXACT)
            else:
                # The generator produced a partial state along with a vector of
                # conditional probabilities.
                conditional_probabilities[map_ids] = probability
                if map_ids == ():
                    weight_to_sample = np.sum(probability)
                    conditional_probabilities[map_ids] /= weight_to_sample

    # Loop through each gate and sample from the remainder of the distribution.

    # Start by rescaling.
    weight_to_sample *= num_samples
    # The following variable, `samples_needed`, must be integer and at least 1.
    # `_samples_multiplier` will typically be 1, but we set it higher in
    # testing to collect additional statistics, faster.
    samples_needed = math.ceil(weight_to_sample) * _samples_multiplier
    # At the time of writing, the below assert should never fail.  But if
    # future code changes result in inputs where it _may_ fail, then the only
    # thing that should be needed if this is reached is to return `retval` in
    # this case, since presumably it must contain all weights as exact weights.
    assert samples_needed >= 1
    single_sample_weight = weight_to_sample / samples_needed

    # Figure out if we've reached the special case where everything except
    # *one* weight has been calculated exactly, so there's only one thing left
    # to sample.  If that's the case, then we can calculate that one remaining
    # weight exactly as well and skip sampling.
    if conditional_probabilities:
        running_state: tuple[int, ...] = ()
        while len(running_state) < len(independent_probabilities):
            # If it's missing from `conditional_probabilities`, that just means
            # to use the corresponding entry in `independent_probabilities`.
            try:
                probs = conditional_probabilities[running_state]
            except KeyError:
                probs = independent_probabilities[len(running_state)]
            x = np.flatnonzero(probs)
            assert len(x) != 0
            if len(x) > 1:
                break
            running_state += (x[0],)
        else:
            assert running_state not in retval
            retval[running_state] = (weight_to_sample, WeightType.EXACT)
            return retval

    # Form the joint samples, collecting them into a dict with counts for each.
    random_samples: dict[tuple[int, ...], int] = {}
    _populate_samples(
        random_samples,
        samples_needed,
        independent_probabilities,
        conditional_probabilities,
    )
    # Insert the samples into the dict we are about to return.
    for outcome, count in random_samples.items():
        assert outcome not in retval
        retval[outcome] = (count * single_sample_weight, WeightType.SAMPLED)

    return retval


def _populate_samples(
    random_samples: dict[tuple[int, ...], int],
    num_desired: int,
    independent_probabilities: Sequence,
    conditional_probabilities: dict[tuple[int, ...], npt.NDArray[np.float64]],
    running_state: tuple[int, ...] = (),
) -> None:
    """
    Generate random samples from the conditional probabilitity distributions.

    Items get populated into the ``random_samples`` dict, rather than returned.

    This function is designed to call itself recursively.  The number of
    elements in ``running_state`` will match the recursion depth.
    """
    if running_state not in conditional_probabilities:
        # Everything below us is sampled, so we can sample directly from the
        # remaining independent probability distributions.
        samples_by_decomp = []
        for probs in independent_probabilities[len(running_state) :]:
            samples_by_decomp.append(
                np.random.choice(range(len(probs)), num_desired, p=probs)
            )
        for outcome, count in Counter(zip(*samples_by_decomp)).items():
            assert (running_state + outcome) not in random_samples
            random_samples[running_state + outcome] = count
        return

    # There are some exact weight(s) below us, so we must consider the
    # conditional probabilities at the current level.
    probs = conditional_probabilities[running_state]
    current_outcomes = np.random.choice(range(len(probs)), num_desired, p=probs)
    for current_outcome, count in Counter(current_outcomes).items():
        outcome = running_state + (current_outcome,)
        if len(outcome) == len(independent_probabilities):
            # It's a full one
            assert outcome not in random_samples
            random_samples[outcome] = count
        else:
            # Recurse
            _populate_samples(
                random_samples,
                count,
                independent_probabilities,
                conditional_probabilities,
                outcome,
            )
