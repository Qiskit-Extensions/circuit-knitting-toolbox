# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes required to implement Dijkstra's (best-first) search algorithm."""

from __future__ import annotations

import heapq
import numpy as np
from typing import TYPE_CHECKING, Callable, cast
from itertools import count

from .optimization_settings import OptimizationSettings
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .search_space_generator import SearchFunctions

if TYPE_CHECKING:  # pragma: no cover
    from .cut_optimization import CutOptimizationFuncArgs


class BestFirstPriorityQueue:
    """Class that implements priority queues for best-first search.

    The tuples that are pushed onto the priority queues have the form:

    (<cost>, <neg_search_depth>, <random_num>, <seq_count>, <search_state>),

    where:

    <cost> (numeric or tuple) is a numeric cost or tuple of numeric
    lexically-ordered costs that are to be minimized.

    <neg_search_depth> (int) is the negative of the search depth of the
    search state represented by the tuple. Thus, if several search states
    have identical costs, priority is given to the deepest states to
    encourage depth-first behavior.

    <random_num> is a pseudo-random number that randomly break ties in a
    stable manner if several search states have identical costs at identical
    search depths.

    <seq_count> is a sequential count of push operations that is used to
    break further ties just in case two states with the same costs and
    depths are somehow assigned the same pseudo-random numbers.

    <search_state> is a state object generated by the optimization process.
    Because of the design of the tuple entries that precede it, state objects
    never get evaluated in the heap-managment comparisons that are performed
    internally by the priority-queue implementation.
    """

    def __init__(self, seed: int | None):
        """Assign member variables."""
        self.random_gen: np.random.Generator = np.random.default_rng(seed)
        self.unique: count[int] = count()
        self.pqueue: list[tuple] = []

    def put(
        self,
        state: DisjointSubcircuitsState,
        depth: int,
        cost: float | tuple[float, float],
    ) -> None:
        """Push state onto the priority queue.

        The search depth and cost of the state must also be provided as input.
        """
        heapq.heappush(
            self.pqueue,
            (cost, (-depth), self.random_gen.random(), next(self.unique), state),
        )

    def get(
        self,
    ) -> tuple:
        """Return the lowest cost state currently on the queue, along with its depth and cost.

        None, None, None is returned if the priority queue is empty.
        """
        if self.qsize() == 0:  # pragma: no cover
            return None, None, None

        best: tuple = heapq.heappop(self.pqueue)

        return best[-1], (-best[1]), best[0]

    def qsize(self) -> int:
        """Return the size of the priority queue."""
        return len(self.pqueue)

    def clear(self) -> None:
        """Clear all entries in the priority queue."""
        self.pqueue.clear()


class BestFirstSearch:
    """Implement Dijkstra's best-first search algorithm.

    The search proceeds by choosing the deepest, lowest-cost state
    in the search frontier and generating next states. Successive calls to
    :meth:`BestFirstSearch.optimization_pass` will resume the search at
    the next deepest, lowest-cost state in the search frontier. The costs
    of goal states that are returned are used to constrain subsequent searches.
    None is returned if no (additional) feasible solutions can be found, or
    when no (additional) solutions can be found without exceeding the lowest
    upper-bound cost across the goal states previously returned.

    Member Variables:

    ``seed`` (int) is the seed to use when initializing Numpy random number
    generators in :class:`BestFirstPriorityQueue` instances.

    ``cost_func`` is a function that computes cost values from search states.
    Input arguments to :meth:`BestFirstSearch.optimization_pass` are also passed
    to the ``cost_func``. The cost returned can be numeric or tuples of numerics.
    In the latter case, lexicographical comparisons are performed per Python semantics.

    ``next_state_func`` is a function that returns a list
    of next states generated from the input state. Input arguments to
    to :meth:`BestFirstSearch.optimization_pass` are also passed to
    the ``next_state_func``.

    ``goal_state_func`` is a function that returns True if
    the input state is a solution state of the search. Input arguments to
    :meth:`BestFirstSearch.optimization_pass` are also passed to the ``goal_state_func``.

     ``upperbound_cost_func`` can either be None or a function that returns
    an upper bound to the optimal cost given a goal_state as input.
    The upper bound is used to prune next-states from the search in
    subsequent calls :meth:`BestFirstSearch.optimization_pass`.
    If ``upperbound_cost_func`` is None, the cost of the ``goal_state`` as
    determined by cost_func is used asan upper bound to the optimal cost.
    Input arguments to :meth:`BestFirstSearch.optimization_pass`
    are also passed to the ``upperbound_cost_func``.

    ``mincost_bound_func`` can either be None or a function that
    returns a cost bound that is compared to the minimum cost across all
    vertices in a search frontier. If the minimum cost exceeds the min-cost
    bound, the search is terminated even if a goal state has not yet been found.
    A ``mincost_bound_func`` that is None is equivalent to an infinite min-cost bound.

    ``stop_at_first_min`` (Boolean) is a flag that indicates whether or not to
    stop the search after the first minimum-cost goal state has been reached.
    In the absence of any QPD assignments, it always makes sense to stop once
    the first minimum has been reached and therefore, we set this bool to True.

    ``max_backjumps`` (int or None) is the maximum number of backjump operations that
    can be performed before the search is forced to terminate. None indicates
    that no restriction is placed on the number of backjump operations.

    ``pqueue`` is an instance of :class:`BestFirstPriorityQueue`.

    ``upperbound_cost`` (float or tuple) is the cost bound obtained by applying
    the upperbound_cost_func to the goal states that are encountered.

    ``mincost_bound`` (float or tuple) is the cost bound imposed on the minimum
    cost across all vertices in the search frontier. The search is forced to
    terminate when the minimum cost exceeds this cost bound.

    ``min_reached`` (Boolean) is a flag that indicates whether or not the
    first minimum-cost goal state has been reached.

    ``num_states_visited`` (int) is the number of states that have been dequeued
    and processed in the search.

    ``num_next_states`` (int) is the number of next-states generated from the
    states visited.

    ``num_enqueues`` (int) is the number of next-states pushed onto the search
    priority queue after cost pruning.

    ``num_backjumps`` (int) is the number of times a backjump operation is
    performed. In the case of (Dijkstra's) best-first search, a backjump
    occurs when the depth of the lowest-cost state in the search frontier
    is less than or equal to the depth of the previous lowest-cost state.
    """

    def __init__(
        self,
        optimization_settings: OptimizationSettings,
        search_functions: SearchFunctions,
        stop_at_first_min: bool = True,
    ):
        """Initialize an instance of :class:`BestFirstSearch`.

        In addition to specifying the optimization settings
        and the functions used to perform the search, an optional Boolean flag
        can be provided to indicate whether to stop the search after
        the first minimum-cost goal state has been reached (True), or whether
        subsequent calls to :meth:`BestFirstSearch.optimization_pass` should
        return any additional minimum-cost goal states that might exist
        (False).
        """
        self.seed = optimization_settings.get_seed
        self.cost_func = search_functions.cost_func
        self.next_state_func = search_functions.next_state_func
        self.goal_state_func = search_functions.goal_state_func
        self.upperbound_cost_func = search_functions.upperbound_cost_func
        self.mincost_bound_func = search_functions.mincost_bound_func
        self.stop_at_first_min = stop_at_first_min
        self.max_backjumps = optimization_settings.get_max_backjumps
        self.pqueue = BestFirstPriorityQueue(self.seed)
        self.upperbound_cost = None
        self.mincost_bound = None
        self.min_reached = False
        self.num_states_visited = 0
        self.num_next_states = 0
        self.num_enqueues = 0
        self.num_backjumps = 0
        self.penultimate_stats: np.typing.NDArray | None = None

    def initialize(
        self,
        initial_state_list: list[DisjointSubcircuitsState],
        *args: CutOptimizationFuncArgs,
    ) -> None:
        """Clear the priority queue and push an initial list of states into it."""
        self.pqueue.clear()
        self.upperbound_cost = None
        self.mincost_bound = None
        self.min_reached = False
        self.num_states_visited = 0
        self.num_next_states = 0
        self.num_enqueues = 0
        self.num_backjumps = 0
        self.penultimate_stats = self.get_stats()
        self.put(initial_state_list, 0, args)

    def optimization_pass(
        self,
        *args: CutOptimizationFuncArgs,
    ) -> (
        tuple[None, None]
        | tuple[
            DisjointSubcircuitsState | None,
            float | tuple[float, float],
        ]
    ):
        """Perform best-first search.

        Run until either a goal state is reached,
        or cost-bounds are reached or no further
        goal states can be found.

        If no further goal states can be found,
        None is returned. The cost of the returned
        state is also returned. Any input arguments to
        :meth:`optimization_pass` are passed along to
        the search-space functions employed.
        """
        if self.mincost_bound_func is not None:
            self.mincost_bound = self.mincost_bound_func(*args)  # type: ignore

        prev_depth = None
        while (
            self.pqueue.qsize() > 0
            and (not self.stop_at_first_min or not self.min_reached)
            and (self.max_backjumps is None or self.num_backjumps < self.max_backjumps)
        ):
            state, depth, cost = self.pqueue.get()

            self.update_minimum_reached(cost)

            if cost is None or self.cost_bounds_exceeded(cost):
                return None, None

            self.num_states_visited += 1

            if prev_depth is not None and depth <= prev_depth:
                self.num_backjumps += 1

            prev_depth = depth
            self.goal_state_func = cast(Callable, self.goal_state_func)
            if self.goal_state_func(state, *args):
                self.penultimate_stats = self.get_stats()
                self.update_upperbound_goal_state(state, *args)
                self.update_minimum_reached(cost)

                return state, cost

            self.next_state_func = cast(Callable, self.next_state_func)
            next_state_list = self.next_state_func(state, *args)
            self.put(next_state_list, depth + 1, args)

        # If all states have been explored, then the minimum has been reached
        if self.pqueue.qsize() == 0:
            self.min_reached = True

        return None, None

    def minimum_reached(self) -> bool:
        """Return True if the optimization reached a global minimum."""
        return self.min_reached

    def get_stats(self, penultimate: bool = False) -> np.typing.NDArray[np.int_] | None:
        """Return statistics of the search that was performed.

        This is a Numpy array containing the number of states visited
        (dequeued), the number of next-states generated, the number of
        next-states that are enqueued after cost pruning, and the number
        of backjumps performed. Return None if no search is performed.
        If the bool penultimate is set to True, return the stats that
        correspond to the penultimate step in the search.
        """
        if penultimate:
            return self.penultimate_stats

        return np.array(
            (
                self.num_states_visited,
                self.num_next_states,
                self.num_enqueues,
                self.num_backjumps,
            ),
            dtype=int,
        )

    def get_upperbound_cost(
        self,
    ) -> float | tuple[float, float] | None:
        """Return the current upperbound cost."""
        return self.upperbound_cost

    def update_upperbound_cost(self, cost_bound: float | tuple[float, float]) -> None:
        """Update the cost upper bound based on an input cost bound."""
        if cost_bound is not None and (
            self.upperbound_cost is None or cost_bound < self.upperbound_cost
        ):
            self.upperbound_cost = cost_bound  # type: ignore

    def update_upperbound_goal_state(
        self, goal_state: DisjointSubcircuitsState, *args: CutOptimizationFuncArgs
    ) -> None:
        """Update the cost upper bound based on a goal state reached in the search."""
        if self.upperbound_cost_func is not None:
            bound = self.upperbound_cost_func(goal_state, *args)
        else:  # pragma: no cover
            assert self.cost_func is not None
            bound = self.cost_func(goal_state, *args)  # type: ignore

        if self.upperbound_cost is None or bound < self.upperbound_cost:
            self.upperbound_cost = bound  # type: ignore

    def put(
        self,
        state_list: list[DisjointSubcircuitsState],
        depth: int,
        args: tuple[CutOptimizationFuncArgs, ...],
    ) -> None:
        """Push a list of (next) states onto the best-first priority queue."""
        self.num_next_states += len(state_list)

        for state in state_list:
            assert self.cost_func is not None
            cost = self.cost_func(state, *args)

            if self.upperbound_cost is None or cost <= self.upperbound_cost:
                self.pqueue.put(state, depth, cost)
                self.num_enqueues += 1

    def update_minimum_reached(
        self, min_cost: None | float | tuple[float, float]
    ) -> bool:
        """Update the min_reached flag indicating that a global optimum has been reached."""
        if min_cost is None or (
            self.upperbound_cost is not None and self.upperbound_cost <= min_cost
        ):
            self.min_reached = True

        return self.min_reached

    def cost_bounds_exceeded(self, cost: None | float | tuple[float, float]) -> bool:
        """Return True if any cost bounds have been exceeded."""
        return cost is not None and (
            (self.mincost_bound is not None and cost > self.mincost_bound)
            or (self.upperbound_cost is not None and cost > self.upperbound_cost)
        )
