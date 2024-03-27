# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes needed to generate and explore a search space."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Callable, TYPE_CHECKING

from .disjoint_subcircuits_state import DisjointSubcircuitsState

if TYPE_CHECKING:  # pragma: no cover
    from .cut_optimization import CutOptimizationFuncArgs
    from .cutting_actions import DisjointSearchAction


class ActionNames:
    """Map action names to individual action objects and group names to lists of action objects.

    The action objects are used to generate a search space.

    Member Variables:

    ``action_dict``: maps action names to action objects.

    ``group_dict``: maps group names to lists of action objects.
    """

    action_dict: dict[str, DisjointSearchAction]
    group_dict: dict[str, list[DisjointSearchAction]]

    def __init__(self):
        """Initialize :class:`ActionNames` with the specified configuration variables."""
        self.action_dict = dict()
        self.group_dict = dict()

    def copy(
        self, list_of_groups: list[DisjointSearchAction | None] | None = None
    ) -> ActionNames:
        """Return copy of :class:`ActionNames` with actions whose group affiliations intersect with ``list_of_groups``.

        The default is to return a copy containing all actions.
        """
        action_list = get_action_subset(list(self.action_dict.values()), list_of_groups)
        new_container = ActionNames()
        assert action_list is not None
        for action in action_list:
            new_container.define_action(action)

        return new_container

    def define_action(self, action_object: DisjointSearchAction) -> None:
        """Insert specified ``action_object`` into look-up dictionaries using associated name of action and group names."""
        assert (
            action_object.get_name() not in self.action_dict
        ), f"Action {action_object.get_name()} is already defined"

        self.action_dict[action_object.get_name()] = action_object

        group_name = action_object.get_group_names()

        if isinstance(group_name, list) or isinstance(group_name, tuple):
            for name in group_name:
                if name not in self.group_dict:
                    self.group_dict[name] = list()
                self.group_dict[name].append(action_object)
        else:  # pragma: no cover
            if group_name not in self.group_dict:
                self.group_dict[group_name] = list()
            self.group_dict[group_name].append(action_object)

    def get_action(self, action_name: str) -> DisjointSearchAction | None:
        """Return the action object associated with the specified ``action_name``.

        ``None`` is returned if there is no associated action object.
        """
        if action_name in self.action_dict:
            return self.action_dict[action_name]
        return None

    def get_group(self, group_name: str) -> list[DisjointSearchAction] | None:
        """Return the list of action objects associated with ``group_name``.

        ``None`` is returned if there are no associated action objects.
        """
        if group_name in self.group_dict:
            return self.group_dict[group_name]
        return None


def get_action_subset(
    action_list: list[DisjointSearchAction] | None,
    action_groups: list[DisjointSearchAction | None] | None,
) -> list[DisjointSearchAction] | None:
    """Return the subset of actions in ``action_list`` whose group affiliations intersect with ``action_groups``."""
    if action_groups is None:
        return action_list

    if len(action_groups) == 0:  # pragma: no cover
        action_groups = [None]

    groups = set(action_groups)

    assert action_list is not None
    return [
        a for a in action_list if len(groups.intersection(set(a.get_group_names()))) > 0
    ]


@dataclass
class SearchFunctions:
    """Contain functions needed to generate and explore a search space.

    In addition to the required input arguments, the function
    signatures are assumed to also allow additional input arguments
    that are needed to perform the corresponding computations.

    Member Variables:

    ``cost_func``: a function that computes cost values
    from search states. The cost returned can be numeric or tuples of
    numerics. In the latter case, lexicographical comparisons are performed
    per Python semantics.

    ``next_state_func``: a function that returns a list
    of next states generated from the input state. A :class:`ActionNames`
    instance should be incorporated into the additional input arguments
    in order to generate next-states.

    ``goal_state_func``: a function that returns ``True`` if
    the input state is a solution state of the search.

    ``upperbound_cost_func`` can either be ``None`` or a
    function that returns an upper bound to the optimal cost given a ``goal_state``
    as input.  The upper bound is used to prune next-states from the search in
    subsequent calls to the :meth:`optimization_pass` method of the search algorithm.
    If upperbound_cost_func is ``None``, the cost of the ``goal_state`` as determined
    by ``cost_func`` is used as an upper bound to the optimal cost. If
    ``upperbound_cost_func`` returns ``None``, the effect is equivalent to returning
    an infinite upper bound (i.e., no cost pruning is performed on subsequent
    optimization calls).

    ``mincost_bound_func`` can either be ``None`` or a function that
    returns a cost bound that is compared to the minimum cost across all
    vertices in a search frontier. If the minimum cost exceeds the min-cost
    bound, the search is terminated even if a goal state has not yet been found.
    Returning ``None`` is equivalent to returning an infinite min-cost bound (i.e.,
    min-cost checking is effectively not performed). A ``mincost_bound_func`` that
    is ``None`` is likewise equivalent to an infinite min-cost bound.
    """

    cost_func: (
        Callable[
            [DisjointSubcircuitsState, CutOptimizationFuncArgs],
            float | tuple[float, float],
        ]
        | None
    ) = None

    next_state_func: (
        Callable[
            [DisjointSubcircuitsState, CutOptimizationFuncArgs],
            list[DisjointSubcircuitsState],
        ]
        | None
    ) = None

    goal_state_func: (
        Callable[[DisjointSubcircuitsState, CutOptimizationFuncArgs], bool] | None
    ) = None

    upperbound_cost_func: (
        Callable[
            [DisjointSubcircuitsState, CutOptimizationFuncArgs],
            tuple[float, float],
        ]
        | None
    ) = None

    mincost_bound_func: (
        Callable[[CutOptimizationFuncArgs], None | tuple[float, float]] | None
    ) = None


@dataclass
class SearchSpaceGenerator:
    """Contain both the functions and the associated actions needed to generate and explore a search space.

    Member Variables:
    functions: a data class that holds the functions needed to generate
    and explore a search space.

    actions: a container class that holds the search
    action objects needed to generate and explore a search space.
    The actions are expected to be passed as arguments to the search
    functions by a search engine.
    """

    functions: SearchFunctions | None = None
    actions: ActionNames | None = None
