# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the entanglement forging ground state solver class and associated support methods."""

from dataclasses import dataclass, field
from time import time
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Tuple,
)

import scipy
import numpy as np
from nptyping import NDArray

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.algorithms.optimizers import SPSA, Optimizer, OptimizerResult
from qiskit.opflow import PauliSumOp, OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit_nature import ListOrDictType
from qiskit_nature.algorithms import GroundStateSolver
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import (
    BaseProblem,
    ElectronicStructureProblem,
)
from qiskit_nature.results import EigenstateResult
from qiskit_ibm_runtime import QiskitRuntimeService, Options

from .entanglement_forging_ansatz import EntanglementForgingAnsatz
from .entanglement_forging_knitter import EntanglementForgingKnitter
from .entanglement_forging_operator import EntanglementForgingOperator
from .cholesky_decomposition import cholesky_decomposition, convert_cholesky_operator
from .entanglement_forging_ansatz import EntanglementForgingAnsatz

RESULT = Union[scipy.optimize.OptimizeResult, OptimizerResult]
OBJECTIVE = Callable[[NDArray], float]
MINIMIZER = Callable[
    [
        OBJECTIVE,  # the objective function to minimize
        NDArray,  # the initial point for the optimization
    ],
    RESULT,  # a result object (either SciPy's or Qiskit's)
]
_T = TypeVar("_T")  # Pylint does not allow single character class names.

ListOrDict = Union[List[Optional[_T]], Dict[str, _T]]


@dataclass
class EntanglementForgingEvaluation:
    """Entanglement Forging Evaluation."""

    parameters: Sequence[float]
    eigenvalue: float
    eigenstate: NDArray


@dataclass
class EntanglementForgingHistory:
    """Entanglement Forging History."""

    evaluations: List[EntanglementForgingEvaluation] = field(default_factory=list)
    optimal_evaluation: Optional[EntanglementForgingEvaluation] = None

    def store_evaluation(self, evaluation: EntanglementForgingEvaluation):
        """Store an evaluation iteration and update optimal values, as necessary."""
        self.evaluations.append(evaluation)
        if (
            self.optimal_evaluation is None
            or evaluation.eigenvalue < self.optimal_evaluation.eigenvalue
        ):
            self.optimal_evaluation = evaluation


class EntanglementForgingResult(EigenstateResult):
    """Entanglement Forging Result."""

    def __init__(self) -> None:
        """Initialize `EigenstateResult` parent class and set class fields."""
        super().__init__()
        self._energy_shift: float = 0.0
        self._history: List[EntanglementForgingEvaluation] = []

    @property
    def groundenergy(self) -> Optional[float]:
        """Return ground energy."""
        return self._groundenergy

    @groundenergy.setter
    def groundenergy(self, energy: Optional[float]) -> None:
        """Set ground energy."""
        self._groundenergy = energy

    @property
    def history(self) -> List[EntanglementForgingEvaluation]:
        """Return optimizer history."""
        return self._history

    @history.setter
    def history(self, history: List[EntanglementForgingEvaluation]) -> None:
        """Set optimizer history."""
        self._history = history

    @property
    def groundstate(
        self,
    ) -> Optional[
        Union[
            str,
            dict,
            Result,
            list,
            np.ndarray,
            Statevector,
            QuantumCircuit,
            Instruction,
            OperatorBase,
        ]
    ]:
        """Return ground state."""
        return self._groundstate

    @groundstate.setter
    def groundstate(
        self,
        groundstate: Optional[
            Union[
                str,
                dict,
                Result,
                list,
                np.ndarray,
                Statevector,
                QuantumCircuit,
                Instruction,
                OperatorBase,
            ]
        ],
    ) -> None:
        """Set ground state."""
        self._groundstate = groundstate

    @property
    def energy_shift(self) -> float:
        """Return the energy shift."""
        return self._energy_shift

    @energy_shift.setter
    def energy_shift(self, value: float) -> None:
        """Set the energy shift."""
        self._energy_shift = value

    @property
    def elapsed_time(self) -> float:
        """Return the elapsed time."""
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, value: float):
        """Set the elapsed time."""
        self._elapsed_time = value


class EntanglementForgingGroundStateSolver(GroundStateSolver):
    """A class which estimates the ground state energy of a molecule."""

    def __init__(
        self,
        ansatz: Optional[EntanglementForgingAnsatz] = None,
        service: Optional[QiskitRuntimeService] = None,
        optimizer: Optional[Union[Optimizer, MINIMIZER]] = None,
        initial_point: Optional[NDArray] = None,
        fix_first_bitstring: bool = False,
        hf_energy: Optional[float] = None,
        orbitals_to_reduce: Optional[Sequence[int]] = None,
        backend_names: Optional[Union[str, List[str]]] = None,
        options: Optional[Union[Options, List[Options]]] = None,
    ):
        """
        Assign the necessary class variables and initialize any defaults.

        Args:
            - ansatz: Class which holes the ansatz circuit and bitstrings
            - service: The service used to spawn Qiskit primitives
            - optimizer: Optimizer to use to optimize the ansatz circuit parameters
            - initial_point: Initial values for ansatz parameters
            - fix_first_bitstring: Flag for setting the HF bitstring values
            - hf_energy: Value to assign to the HF bitstring circuit experiments
            - orbitals_to_reduce: List of orbital indices to remove from the problem before
                decomposition.
            - backend_names: Backend name or list of backend names to use during parallel computation
            - options: Options or list of options to be applied to the backends

        Returns:
            - None
        """
        # These fields will be used when solve is called
        self._knitter: Optional[EntanglementForgingKnitter] = None
        self._history: EntanglementForgingHistory = EntanglementForgingHistory()
        self._energy_shift = 0.0

        # Set class fields
        self._ansatz: Optional[EntanglementForgingAnsatz] = ansatz
        self._service: Optional[QiskitRuntimeService] = service
        self._initial_point: Optional[NDArray] = initial_point
        self._fix_first_bitstring: bool = fix_first_bitstring
        self._hf_energy: Optional[float] = hf_energy
        self._orbitals_to_reduce = orbitals_to_reduce
        self.backend_names = backend_names  # type: ignore
        self.options = options

        self._optimizer: Union[Optimizer, MINIMIZER] = optimizer or SPSA()

    @property
    def ansatz(self) -> Optional[EntanglementForgingAnsatz]:
        """Return ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[EntanglementForgingAnsatz]) -> None:
        """Set the ansatz."""
        self._ansatz = ansatz

    @property
    def service(self) -> Optional[QiskitRuntimeService]:
        """Return service."""
        return self._service

    @service.setter
    def service(self, service: Optional[QiskitRuntimeService]) -> None:
        """Set the service."""
        self._service = service

    @property
    def orbitals_to_reduce(self) -> Optional[Sequence[int]]:
        """Return the orbitals to reduce."""
        return self._orbitals_to_reduce

    @orbitals_to_reduce.setter
    def orbitals_to_reduce(self, orbitals_to_reduce: Optional[Sequence[int]]) -> None:
        """Set the orbitals to reduce."""
        self._orbitals_to_reduce = orbitals_to_reduce

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        self._optimizer = optimizer

    @property
    def initial_point(self) -> Optional[NDArray]:
        """Return the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[NDArray]) -> None:
        """Set the initial point."""
        self._initial_point = initial_point

    @property
    def fix_first_bitstring(self) -> bool:
        """Return the value of fix_first_bitstring flag."""
        return self._fix_first_bitstring

    @fix_first_bitstring.setter
    def fix_first_bitstring(self, fix_first_bitstring: bool) -> None:
        """Set the fix_first_bitstring flag."""
        self._fix_first_bitstring = fix_first_bitstring

    @property
    def hf_energy(self) -> Optional[float]:
        """Return the value of the Hartree-Fock energy."""
        return self._hf_energy

    @hf_energy.setter
    def hf_energy(self, hf_energy: Optional[float]) -> None:
        """Set the Hartree-Fock energy field."""
        self._hf_energy = hf_energy

    @property
    def backend_names(self) -> Optional[List[str]]:
        """Return the backend names."""
        return self._backend_names

    @backend_names.setter
    def backend_names(self, backend_names: Union[str, List[str]]) -> None:
        """Set the backend names."""
        if isinstance(backend_names, str):
            self._backend_names = [backend_names]
        else:
            self._backend_names = backend_names

    @property
    def options(self) -> Optional[List[Options]]:
        """Return the options."""
        return self._options

    @options.setter
    def options(self, options: Union[Options, List[Options]]) -> None:
        """Set the options."""
        if isinstance(options, Options):
            self._options = [options]
        else:
            self._options = options

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[
            ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]
        ] = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            - problem: a class encoding a problem to be solved.
            - aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            - An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        if self._fix_first_bitstring and self._hf_energy is None:
            raise AttributeError(
                "If the fix_first_bitstring flag is set, the hf_energy field must also be set."
            )
        if not isinstance(problem, ElectronicStructureProblem):
            raise AttributeError(
                "EntanglementForgingGroundStateSolver only accepts ElectronicStructureProblem as input to its solve method."
            )
        if self._backend_names and self._options:
            if len(self._backend_names) != len(self._options):
                if len(self._options) == 1:
                    self._options = [self._options[0]] * len(self._backend_names)
                else:
                    raise AttributeError(
                        f"The list of backend names is length ({len(self._backend_names)}), but the list of options is length ({len(self._options)}). It is ambiguous how to combine the options with the backends."
                    )
        if self._ansatz is None:
            raise AttributeError("Ansatz must be set before calling solve.")
        if self._initial_point is None:
            self._initial_point = np.array(
                [0.0 for i in range(len(self._ansatz.circuit_u.parameters))]
            )

        hamiltonian_terms = self.get_qubit_operators(problem)
        ef_operator = convert_cholesky_operator(hamiltonian_terms, self._ansatz)

        if self._service:
            backend_names = self._backend_names or ["ibmq_qasm_simulator"]
            self._knitter = EntanglementForgingKnitter(
                self._ansatz,
                fix_first_bitstring=self._fix_first_bitstring,
                hf_energy=self._hf_energy,
                service=self._service,
                backend_names=backend_names,
                options=self._options,
            )
        else:
            self._knitter = EntanglementForgingKnitter(
                self._ansatz,
                fix_first_bitstring=self._fix_first_bitstring,
                hf_energy=self._hf_energy,
                options=self._options,
            )
        self._history = EntanglementForgingHistory()
        self._eval_count = 0

        evaluate_eigenvalue = self.get_eigenvalue_evaluation(ef_operator)

        start_time = time()

        if callable(self._optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_eigenvalue, x0=self._initial_point
            )
        else:
            optimizer_result = self.optimizer.minimize(
                fun=evaluate_eigenvalue, x0=self._initial_point
            )

        elapsed_time = time() - start_time

        optimal_evaluation = self._history.optimal_evaluation
        if optimal_evaluation is None:
            raise RuntimeError("Unable to retrieve optimal evaluation.")

        result = EntanglementForgingResult()
        result.eigenenergies = np.array(
            [e.eigenvalue for e in self._history.evaluations]
        )
        result.eigenstates = np.array([e.eigenstate for e in self._history.evaluations])  # type: ignore
        if self._history.optimal_evaluation is None:
            raise ValueError(
                "Something unexpected happened, and the solver was not able to find an optimal parametrization."
            )
        result.groundenergy = self._history.optimal_evaluation.eigenvalue
        result.groundstate = self._history.optimal_evaluation.eigenstate
        result.energy_shift = self._energy_shift
        result.history = self._history.evaluations
        result.elapsed_time = elapsed_time

        # Close any runtime sessions
        self._knitter.close_sessions()

        return result

    def get_eigenvalue_evaluation(
        self, operator: EntanglementForgingOperator
    ) -> Callable[[Sequence[float]], Union[float, Iterable[float]]]:
        """Produce a callable function which provides an estimation of the minimum eigenvalue of the input operator.

        Args:
          - operator (EntanglementForgingOperator): The decomposed Hamiltonian in entanglement forging format
        Returns:
          - callable (Callable):  Callable function which provides an estimation of the mihnimum eigenvalue
              of the input operator given some ansatz circuit parameters.
        """

        def evaluate_eigenvalue(parameters: Sequence[float]) -> float:
            if self._knitter is None:
                raise AttributeError(
                    "Knitter must be set before evaluating eigenvalue."
                )

            eigenvalue, schmidt_coeffs, _ = self._knitter(
                ansatz_parameters=parameters, forged_operator=operator
            )
            self._history.store_evaluation(
                EntanglementForgingEvaluation(parameters, eigenvalue, schmidt_coeffs)
            )

            return eigenvalue

        return evaluate_eigenvalue

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[
            ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]
        ] = None,
    ) -> Tuple[PauliSumOp, Optional[ListOrDictType[PauliSumOp]]]:
        """Construct decomposed qubit operators from an ``ElectronicStructureProblem``.

        Args:
          - problem (BaseProblem): A class encoding a problem to be solved.
          - aux_operators (ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]): Additional auxiliary operators to evaluate.

        Returns:
          - hamiltonian_ops: qubit operator representing the decomposed Hamiltonian.
        """
        if not isinstance(problem, ElectronicStructureProblem):
            raise AttributeError(
                "EntanglementForgingGroundStateSolver only supports ElectronicStructureProblem."
            )
        decomposed_operator = cholesky_decomposition(problem, self._orbitals_to_reduce)
        hamiltonian_ops = decomposed_operator[0]
        self._energy_shift = decomposed_operator[1]

        return hamiltonian_ops

    def returns_groundstate(self) -> bool:
        """Whether this class returns only the ground state energy or also the ground state itself.

        Returns:
            - True, if this class also returns the ground state in the results object.
            False otherwise.
        """
        return True

    @property
    def qubit_converter(self):
        """Not implemented."""

    @property
    def solver(self):
        """Not implemented."""

    def evaluate_operators(
        self,
        state: Union[
            str,
            dict,
            Result,
            list,
            np.ndarray,
            Statevector,
            QuantumCircuit,
            Instruction,
            OperatorBase,
        ],
        operators: Union[PauliSumOp, OperatorBase, list, dict],
    ) -> Union[float, List[float], Dict[str, List[float]]]:
        """Not necessary to implement."""
        raise NotImplementedError(
            "This class method is not used by EntanglementForgingGroundStateSolver."
        )
