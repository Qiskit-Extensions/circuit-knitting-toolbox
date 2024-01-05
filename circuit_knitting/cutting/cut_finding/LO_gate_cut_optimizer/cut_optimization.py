""" File containing the classes required to search for optimal cut locations."""
import numpy as np
from .cutting_actions import disjoint_subcircuit_actions
from .utils import selectSearchEngine, greedyBestFirstSearch
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .search_space_generator import (
    getActionSubset,
    SearchFunctions,
    SearchSpaceGenerator,
)


class CutOptimizationFuncArgs:

    """Class for passing relevant arguments to the CutOptimization
    search-space generating functions.
    """

    def __init__(self):
        self.entangling_gates = None
        self.search_actions = None
        self.max_gamma = None
        self.qpu_width = None
        self.greedy_multiplier = None


def CutOptimizationCostFunc(state, func_args):
    """Return the cost function whose goal is to minimize the gamma 
    lower bound while giving preference to circuit partitions 
    that balance the width of the resulting partitions.
    """
    return (state.lowerBoundGamma(), state.getMaxWidth())


def CutOptimizationUpperBoundCostFunc(goal_state, func_args):
    """Return the gamma upper bound."""
    return (goal_state.upperBoundGamma(), np.inf)


def CutOptimizationMinCostBoundFunc(func_args):
    """Return an a priori min-cost bound defined in the optimization settings."""
    if func_args.max_gamma is None:
        return None

    return (func_args.max_gamma, np.inf)


def CutOptimizationNextStateFunc(state, func_args):
    """Generate a list of next states from the input state."""

    # Get the entangling gate spec that is to be processed next based
    # on the search level of the input state
    gate_spec = func_args.entangling_gates[state.getSearchLevel()]

    # Determine which search actions can be performed, taking into
    # account any user-specified constraints that might have been
    # placed on how the current entangling gate is to be handled
    # in the search
    if len(gate_spec[1]) == 3:
        action_list = func_args.search_actions.getGroup("TwoQubitGates")
    else:
        action_list = func_args.search_actions.getGroup("MultiqubitGates")

    action_list = getActionSubset(action_list, gate_spec[2])

    # Apply the search actions to generate a list of next states
    next_state_list = []
    for action in action_list:
        next_state_list.extend(action.nextState(state, gate_spec, func_args.qpu_width))

    return next_state_list


def CutOptimizationGoalStateFunc(state, func_args):
    """Return True if the input state is a goal state (i.e., the cutting
    decisions made satisfy the device constraints and the optimization settings).
    """
    return state.getSearchLevel() >= len(func_args.entangling_gates)


# Object that holds the search-space functions for generating
# the cut optimization search space.
cut_optimization_search_funcs = SearchFunctions(
    cost_func=CutOptimizationCostFunc,
    upperbound_cost_func=CutOptimizationUpperBoundCostFunc,
    next_state_func=CutOptimizationNextStateFunc,
    goal_state_func=CutOptimizationGoalStateFunc,
    mincost_bound_func=CutOptimizationMinCostBoundFunc,
)


def greedyCutOptimization(
    circuit_interface,
    optimization_settings,
    device_constraints,
    search_space_funcs=cut_optimization_search_funcs,
    search_actions=disjoint_subcircuit_actions,
):
    func_args = CutOptimizationFuncArgs()
    func_args.entangling_gates = circuit_interface.getMultiQubitGates()
    func_args.search_actions = search_actions
    func_args.max_gamma = optimization_settings.getMaxGamma()
    func_args.qpu_width = device_constraints.getQPUWidth()

    start_state = DisjointSubcircuitsState(circuit_interface.getNumQubits())

    return greedyBestFirstSearch(start_state, search_space_funcs, func_args)


################################################################################


class CutOptimization:

    """Class that implements the action of circuit cutting to create disjoint
    subcircuits. It uses upper and lower bounds on the resulting gamma in order 
    to decide where and how to cut while deferring the exact
    choices of quasiprobability decompositions.

    Member Variables:

    circuit (CircuitInterface) is the interface object for the circuit
    to be cut.

    settings (OptimizationSettings) is an object that contains the settings
    that control the optimization process.

    constraints (DeviceConstraints) is an object that contains the device
    constraints that solutions must obey.

    search_funcs (SearchFunctions) is an object that holds the functions
    needed to generate and explore the cut optimization search space.

    func_args (CutOptimizationFuncArgs) is an object that contains the
    necessary device constraints and optimization settings parameters that
    aree needed by the cut optimization search-space function.

    search_actions (ActionNames) is an object that contains the allowed
    actions that are used to generate the search space.

    search_engine (BestFirstSearch) is an object that implements the
    search algorithm.
    """

    def __init__(
        self,
        circuit_interface,
        optimization_settings,
        device_constraints,
        search_engine_config={
            "CutOptimization": SearchSpaceGenerator(
                functions=cut_optimization_search_funcs,
                actions=disjoint_subcircuit_actions,
            )
        },
    ):
        """A CutOptimization object must be initialized with
        a specification of all of the parameters of the optimization to be
        performed: i.e., the circuit to be cut, the optimization settings,
        the target-device constraints, the functions for generating the
        search space, and the allowed search actions."""

        generator = search_engine_config["CutOptimization"]
        search_space_funcs = generator.functions
        search_space_actions = generator.actions

        # Extract the subset of allowed actions as defined in the settings object
        cut_groups = optimization_settings.getCutSearchGroups()
        cut_actions = search_space_actions.copy(cut_groups)

        self.circuit = circuit_interface
        self.settings = optimization_settings
        self.constraints = device_constraints
        self.search_funcs = search_space_funcs
        self.search_actions = cut_actions

        self.func_args = CutOptimizationFuncArgs()
        self.func_args.entangling_gates = self.circuit.getMultiQubitGates()
        self.func_args.search_actions = self.search_actions
        self.func_args.max_gamma = self.settings.getMaxGamma()
        self.func_args.qpu_width = self.constraints.getQPUWidth()

        # Perform an initial greedy best-first search to determine an upper
        # bound for the optimal gamma
        self.greedy_goal_state = greedyCutOptimization(
            self.circuit,
            self.settings,
            self.constraints,
            search_space_funcs=self.search_funcs,
            search_actions=self.search_actions,
        )
        ################################################################################

        # Push the start state onto the search_engine
        start_state = DisjointSubcircuitsState(self.circuit.getNumQubits())

        sq = selectSearchEngine(
            "CutOptimization",
            self.settings,
            self.search_funcs,
            stop_at_first_min=False,
        )

        sq.initialize([start_state], self.func_args)

        # Use the upper bound for the optimal gamma to constrain the search
        if self.greedy_goal_state is not None:
            sq.updateUpperBoundGoalState(self.greedy_goal_state, self.func_args)

        self.search_engine = sq
        self.goal_state_returned = False

    def optimizationPass(self):
        """Produce, in each call, a goal state representing a distinct set of cutting decisions.
        The first goal state returned corresponds to cutting decisions that minimize the
        lower bound on the resulting gamma.  None is returned once no additional choices of cuts
        can be made without exceeding the minimum upper bound across all cutting decisions
        previously returned, subject to the optimization settings.
        """
        state, cost = self.search_engine.optimizationPass(self.func_args)

        if state is None and not self.goal_state_returned:
            state = self.greedy_goal_state
            cost = self.search_funcs.cost_func(state, self.func_args)

        self.goal_state_returned = True

        return state, cost

    def minimumReached(self):
        """Return True if the optimization reached a global minimum."""
        return self.search_engine.minimumReached()

    def getStats(self, penultimate=False):
        """Return the search-engine statistics."""
        return self.search_engine.getStats(penultimate=penultimate)

    def getUpperBoundCost(self):
        """Return the current upperbound cost."""
        return self.search_engine.getUpperBoundCost()

    def updateUpperBoundCost(self, cost_bound):
        """Update the cost upper bound based on an
        input cost bound.
        """
        self.search_engine.updateUpperBoundCost(cost_bound)
