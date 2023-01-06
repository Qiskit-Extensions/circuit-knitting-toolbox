# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the knitter class and associated functions."""
from typing import List, Optional, Sequence, Tuple, Union, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from nptyping import Float, Int, NDArray, Shape

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator as TestEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Estimator

from .entanglement_forging_ansatz import Bitstring, EntanglementForgingAnsatz
from .entanglement_forging_operator import EntanglementForgingOperator


class EntanglementForgingKnitter:
    """Container for Knitter class functions and attributes.

    A class which performs entanglement forging and returns the
    ground state energy and Schmidt coefficients found for given
    ansatz parameters and Schmidt coefficients.
    """

    # Attributes:
    #     - _ansatz (EntanglementForgingAnsatz): the ansatz containing the
    #         information for the circuit structure and bitstrings to be used
    #     - _backend_names (List[str]): the names of the backends to use
    #     - _service (QiskitRuntimeService): the service used to access backends
    #     - _tensor_circuits_u (List[QuantumCircuit]): the set of circuits used for the first
    #         operator that have the same Schmidt values
    #     - _superposition_circuits_u (List[QuantumCircuit]): the set of circuits used for
    #         the first operator that have different Schmidt values
    #     - _tensor_circuits_v (List[QuantumCircuit]): the set of circuits used for the second
    #         operator that have the same Schmidt values
    #     - _superposition_circuits_v (List[QuantumCircuit]): the set of circuits used for
    #         the second operator that have different Schmidt values

    def __init__(
        self,
        ansatz: EntanglementForgingAnsatz,
        service: Optional[QiskitRuntimeService] = None,
        backend_names: Optional[Union[str, List[str]]] = None,
        options: Optional[Union[Options, List[Options]]] = None,
    ):
        """
        Assign the necessary member variables.

        Args:
            - ansatz (EntanglementForgingAnsatz): The container for the circuit structure and bitstrings
                to be used (and generate the stateprep circuits)
            - service (QiskitRuntimeService): The service used to spawn Qiskit primitives and runtime jobs
            - backend_names (List[str]): Names of the backends to use for calculating expectation values
            - options (List[Options]): Options to use with the backends

        Returns:
            - None
        """
        # Call backend_names setter to update the session_ids hidden class field
        self._session_ids: Optional[List[Union[str, None]]] = None
        self._backend_names: Optional[List[str]] = None
        self._options: Optional[List[Options]] = None
        # Call constructors
        self.backend_names = backend_names  # type: ignore
        self.options = options

        # The service hidden class field is a json representing the QiskitRuntimeService object
        self._service = service.active_account() if service is not None else service

        # Save the parameterized ansatz and bitstrings
        self._ansatz: EntanglementForgingAnsatz = EntanglementForgingAnsatz(
            circuit_u=ansatz.circuit_u,
            bitstrings_u=ansatz.bitstrings_u,
            bitstrings_v=ansatz.bitstrings_v or ansatz.bitstrings_u,
        )

        # self._tensor_circuits   = [|b1⟩,|b2⟩,...,|b2^N⟩]
        # self._superpos_circuits = [
        #           |𝜙^0_𝑏2𝑏1⟩,|𝜙^2_𝑏2𝑏1⟩,
        #           |𝜙^0_𝑏3𝑏1⟩,|𝜙^2_𝑏3𝑏1⟩,|𝜙^0_𝑏3𝑏2⟩,|𝜙^2_𝑏3𝑏2⟩
        #           |𝜙^0_𝑏4𝑏1⟩,|𝜙^2_𝑏4𝑏1⟩,|𝜙^0_𝑏4𝑏2⟩,|𝜙^2_𝑏4𝑏2⟩,|𝜙^0_𝑏4𝑏3⟩,|𝜙^2_𝑏4𝑏3⟩,
        #           ...
        #           ...,|𝜙^0_𝑏2^N𝑏(2^N-2)⟩,|𝜙^2_𝑏2^N𝑏(2^N-2)⟩,|𝜙^0_𝑏2^N𝑏(2^N-1)⟩,|𝜙^2_𝑏2^N𝑏(2^N-1)⟩]
        #
        (
            self._tensor_circuits_u,
            self._superposition_circuits_u,
        ) = _construct_stateprep_circuits(self._ansatz.bitstrings_u)
        if self._ansatz.bitstrings_are_symmetric:
            self._tensor_circuits_v, self._superposition_circuits_v = (
                self._tensor_circuits_u,
                self._superposition_circuits_u,
            )
        else:
            (
                self._tensor_circuits_v,
                self._superposition_circuits_v,
            ) = _construct_stateprep_circuits(
                self._ansatz.bitstrings_v  # type: ignore
            )

    @property
    def ansatz(self) -> EntanglementForgingAnsatz:
        """
        Property function for the ansatz.

        Returns:
            - (EntanglementForgingAnsatz): the ansatz member variable
        """
        return self._ansatz

    @property
    def backend_names(self) -> Optional[List[str]]:
        """
        List of backend names to be used.

        Returns:
            - (List[str]): the backend_names member variable
        """
        return self._backend_names

    @backend_names.setter
    def backend_names(self, backend_names: Optional[Union[str, List[str]]]) -> None:
        """
        Change the backend_names class field.

        Args:
            - backend_names (List[str]): the list of backends to use

        Returns:
            - None
        """
        if isinstance(backend_names, str):
            self._backend_names = [backend_names]
        else:
            self._backend_names = backend_names
        if backend_names:
            self._session_ids = [None] * len(backend_names)

    @property
    def options(self) -> Optional[List[Options]]:
        """
        List of options to be used.

        Returns:
            - (List[Options]): the options member variable
        """
        return self._options

    @options.setter
    def options(self, options: Optional[Union[Options, List[Options]]]) -> None:
        """
        Change the options class field.

        Args:
            - options (List[Options]): the list of options to use

        Returns:
            - None
        """
        if isinstance(options, Options):
            self._options = [options]
        else:
            self._options = options

    @property
    def service(self) -> Optional[QiskitRuntimeService]:
        """
        Property function for service class field.

        Returns:
            - (QiskitRuntimeService): the service member variable
        """
        return QiskitRuntimeService(**self._service)

    @service.setter
    def service(self, service: Optional[QiskitRuntimeService]) -> None:
        """
        Change the service class field.

        Args:
            - service (QiskitRuntimeService): the service used to spawn Qiskit primitives

        Returns:
            - None
        """
        self._service = service.active_account() if service is not None else service

    def __call__(
        self,
        ansatz_parameters: Sequence[float],
        forged_operator: EntanglementForgingOperator,
    ) -> Tuple[
        float, NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]
    ]:  # noqa: D301, D202
        r"""Calculate the energy.

        Computes ⟨H⟩ - the energy value and the Schmidt matrix, $h_{n, m}$, given
        some ansatz parameter values.

        $h_{n, n} = \sum_{a, b} w_{a, b} \left [ \lambda_n^2 \langle b_n | U^t P_a U | b_n \rangle
            \langle b_n | V^t P_b V | b_n \rangle \right ]$

        $h_{n, m} = \sum_{a, b} w_{a, b} \left [ \lambda_n \lambda_m \sum_{p \in 4} -1^p \langle \phi^p_{b_n, b_m}
            | U^t P_a U | \phi^p_{b_n, b_m} \rangle \langle  \phi^p_{b_n, b_m} | V^t P_b V |  \phi^p_{b_n, b_m} \rangle \right ]$

        Energy = $ \sum_{n=1}^{2^N} \left ( h_{n, n} + \sum_{m=1}^{n-1} h_{n, m} \right ) $

        For now, we are only using $p \in \{0, 2 \} $ as opposed to $ p \in \{ 0, 1, 2, 3 \} $.

        Additionally, U = V is currently required, but may change in future versions.

        Args:
            - self
            - ansatz_parameters (Sequence[float]): the parameters to be used by the ansatz circuit,
                must be the same length as the circuit's parameters
            - forged_operator (EntanglementForgingOperator): the operator to forge the expectation
                value from

        Returns:
            - (Tuple[float, NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]): a tuple
                containing the energy (i.e. forged expectation value), the Schmidt coefficients,
                and the full Schmidt decomposition matrix
        """
        if self._backend_names and self._options:
            if len(self._backend_names) != len(self._options):
                if len(self._options) == 1:
                    self._options = [self._options[0]] * len(self._backend_names)
                else:
                    raise AttributeError(
                        f"The list of backend names is length ({len(self._backend_names)}), but the list of options is length ({len(self._options)}). It is ambiguous how to combine the options with the backends."
                    )
        # For now, we only assign the parameters to a copy of the ansatz
        circuit_u = self._ansatz.circuit_u.bind_parameters(ansatz_parameters)

        # Create the tensor and superposition stateprep circuits
        # tensor_ansatze   = [U|bi⟩      for |bi⟩       in  tensor_circuits]
        # superposition_ansatze = [U|𝜙^𝑝_𝑏𝑛𝑏𝑚⟩ for |𝜙^𝑝_𝑏𝑛𝑏𝑚⟩ in superposition_circuits]
        tensor_ansatze_u = [
            prep_circ.compose(circuit_u) for prep_circ in self._tensor_circuits_u
        ]
        superposition_ansatze_u = [
            prep_circ.compose(circuit_u) for prep_circ in self._superposition_circuits_u
        ]

        tensor_ansatze_v = []
        superposition_ansatze_v = []
        if not self._ansatz.bitstrings_are_symmetric:
            tensor_ansatze_v = [
                prep_circ.compose(circuit_u) for prep_circ in self._tensor_circuits_v
            ]
            superposition_ansatze_v = [
                prep_circ.compose(circuit_u)
                for prep_circ in self._superposition_circuits_v
            ]

        # Partition the expectation values for parallel calculation
        if self._backend_names:
            num_partitions = len(self._backend_names)
        else:
            num_partitions = 1

        tensor_ansatze = tensor_ansatze_u + tensor_ansatze_v
        superposition_ansatze = superposition_ansatze_u + superposition_ansatze_v

        partitioned_tensor_ansatze = _partition(tensor_ansatze, num_partitions)
        partitioned_superposition_ansatze = _partition(
            superposition_ansatze, num_partitions
        )

        # Get the RuntimeService as a hashable dictionary
        service_args = None
        if self._service:
            service_args = self._service

        session_ids: Optional[List[Union[str, None]]] = None
        if self._session_ids is None:
            session_ids = [None] * num_partitions
        else:
            session_ids = self._session_ids

        partitioned_expval_futures = []
        with ThreadPoolExecutor() as executor:
            for partition_index, (
                tensor_ansatze_partition,
                superposition_ansatze_partition,
            ) in enumerate(
                zip(partitioned_tensor_ansatze, partitioned_superposition_ansatze)
            ):
                backend_name = (
                    None
                    if self._backend_names is None
                    else self._backend_names[partition_index]
                )
                options = (
                    None if self._options is None else self._options[partition_index]
                )
                tensor_pauli_list = list(forged_operator.tensor_paulis)
                superposition_pauli_list = list(forged_operator.superposition_paulis)
                partitioned_expval_futures.append(
                    executor.submit(
                        _estimate_expvals,
                        tensor_ansatze_partition,
                        tensor_pauli_list,
                        superposition_ansatze_partition,
                        superposition_pauli_list,
                        service_args,
                        backend_name,
                        options,
                        session_ids[partition_index],
                    )
                )

        tensor_expvals = []
        superposition_expvals = []
        for i, partitioned_expval_future in enumerate(partitioned_expval_futures):
            (
                partition_tensor_expvals,
                partition_superposition_expvals,
                job_id,
            ) = partitioned_expval_future.result()
            tensor_expvals.extend(partition_tensor_expvals)
            superposition_expvals.extend(partition_superposition_expvals)
            # Start a session for each thread if this is the first run
            if job_id and (session_ids[i] is None):
                if self._session_ids is None:
                    raise ValueError(
                        "Something unexpected happened. The session_ids field must be set when a job_id is present."
                    )
                self._session_ids[i] = job_id

        # Compute the Schmidt matrix
        h_schmidt = self._compute_h_schmidt(
            forged_operator, np.array(tensor_expvals), np.array(superposition_expvals)
        )
        evals, evecs = np.linalg.eigh(h_schmidt)
        schmidt_coeffs = evecs[:, 0]
        energy = evals[0]

        return energy, schmidt_coeffs, h_schmidt

    def _compute_h_schmidt(
        self,
        forged_operator: EntanglementForgingOperator,
        tensor_expvals: NDArray[Shape["*, *"], Float],
        superpos_expvals: NDArray[Shape["*, *"], Float],
    ) -> NDArray[Shape["*, *"], Float]:  # noqa: D202
        """
        Compute the Schmidt decomposition of the Hamiltonian.

        Args:
            - forged_operator (EntanglementForgingOperator): the operator that the
                forged expectation values are computed with
            - tensor_expvals (NDArray[Shape["*, *"], Float]): the expectation values
                for the tensor circuits (i.e. same Schmidt coefficients)
            - superpos_expvals (NDArray[Shape["*, *"], Float]): the expectation values
                for the superposition circuits (i.e. different Schmidt coefficients)

        Returns:
           - (NDArray[Shape["*, *"], Float]): the Schmidt matrix
        """

        # Calculate the diagonal entries of the Schmidt matrix by
        # summing the expectation values associated with the tensor terms
        # h𝑛𝑛 = Σ_ab 𝑤𝑎𝑏•[ 𝜆𝑛^2•⟨b𝑛|U^t•P𝑎•U|b𝑛⟩⟨b𝑛|V^t•P𝑏•V|b𝑛⟩ ]
        if self._ansatz.bitstrings_are_symmetric:
            h_schmidt_diagonal = np.einsum(
                "ij, xi, xj->x",
                forged_operator.w_ij,  # type: ignore
                tensor_expvals,
                tensor_expvals,
            )
        else:
            num_tensor_terms = int(np.shape(tensor_expvals)[0] / 2)
            h_schmidt_diagonal = np.einsum(
                "ij, xi, xj->x",
                forged_operator.w_ij,  # type: ignore
                tensor_expvals[:num_tensor_terms, :],
                tensor_expvals[num_tensor_terms:, :],
            )
        h_schmidt = np.diag(h_schmidt_diagonal)

        # Including the +/-Y superpositions would increase this to 4
        num_lin_combos = 2

        # superpos_ansatze[2i]   = U|𝜙^0_𝑏𝑛𝑏𝑚⟩
        # superpos_expvals[2i]   = [⟨𝜙^0_𝑏𝑛𝑏𝑚|U^t•𝑃𝑎•U|𝜙^0_𝑏𝑛𝑏𝑚⟩ for 𝑃𝑎 in superpos_paulis]
        # superpos_expvals[2i+1] = [⟨𝜙^1_𝑏𝑛𝑏𝑚|U^t•𝑃𝑎•U|𝜙^1_𝑏𝑛𝑏𝑚⟩ for 𝑃𝑎 in superpos_paulis]
        superpos_expvals = np.array(superpos_expvals)

        if self._ansatz.bitstrings_are_symmetric:
            p_plus_x = superpos_expvals[0::num_lin_combos, :]
            p_minus_x = superpos_expvals[1::num_lin_combos, :]
            p_delta_x_u = p_plus_x - p_minus_x
            p_delta_x_v = p_delta_x_u
        else:
            num_superpos_terms = int(np.shape(superpos_expvals)[0] / 2)
            pvss_u = superpos_expvals[:num_superpos_terms, :]
            pvss_v = superpos_expvals[num_superpos_terms:, :]

            p_plus_x_u = pvss_u[0::num_lin_combos, :]
            p_minus_x_u = pvss_u[1::num_lin_combos, :]
            p_delta_x_u = p_plus_x_u - p_minus_x_u

            p_plus_x_v = pvss_v[0::num_lin_combos, :]
            p_minus_x_v = pvss_v[1::num_lin_combos, :]
            p_delta_x_v = p_plus_x_v - p_minus_x_v

        # Calculate and assign the off-diagonal values of the Schmidt matrix by
        # summing the expectation values associated with the superpos terms
        h_schmidt_off_diagonals = np.einsum(
            "ab,xa,xb->x", forged_operator.w_ab, p_delta_x_u, p_delta_x_v  # type: ignore
        )
        # Create off diagonal index list
        superpos_indices = []
        for x in range(self._ansatz.subspace_dimension):
            for y in range(self._ansatz.subspace_dimension):
                if x == y:
                    continue
                superpos_indices += [(x, y)]

        # h𝑛𝑚 = Σ_ab 𝑤𝑎𝑏 •[ 𝜆𝑛𝜆𝑚•Σ_𝑝∈ℤ4 -1^𝑝•⟨𝜙^𝑝_𝑏𝑛𝑏𝑚|U^t•𝑃𝑎•U|𝜙^𝑝_𝑏𝑛𝑏𝑚⟩•
        #                                   ⟨𝜙^𝑝_𝑏𝑛𝑏𝑚|V^t•𝑃𝑏•V|𝜙^𝑝_𝑏𝑛𝑏𝑚⟩ ]
        for element, indices in zip(h_schmidt_off_diagonals, superpos_indices):
            h_schmidt[indices] = element

        return h_schmidt

    def close_sessions(self) -> None:
        """
        Close all the sessions opened by this object.

        Args:
            - None

        Returns:
            - None
        """
        if self._session_ids is None:
            return
        else:
            for i, session_id in enumerate(self._session_ids):
                if self._backend_names is None:
                    raise ValueError(
                        f"There was a problem closing session id ({session_id}). No backend to associate with session."
                    )
                session = Session(service=self.service, backend=self._backend_names[i])
                session._session_id = session_id
                session.close()

        return


def _construct_stateprep_circuits(
    bitstrings: List[Bitstring],
    subsystem_id: Optional[str] = None,
) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:  # noqa: D301
    r"""Prepare all circuits.

    Function to make the state preparation circuits. This constructs a set
    of circuits $ | b_n \rangle $ and $ | \phi^{p}_{n, m} \rangle $.

    The circuits $ | b_n \rangle $ are computational basis states specified by
    bitstrings $ b_n $, while the circuits $ | \phi^{p}_{n, m} \rangle $ are
    superpositions over pairs of bitstrings:

    $ | \phi^{p}_{n, m} \rangle = (| b_n \rangle + i^p | b_m \rangle) / \sqrt{2} $,
    as defined in <https://arxiv.org/abs/2104.10220>. Note that the output
    scaling (for the square root) is done in the estimator function.

    Example:
    _construct_stateprep_circuits([[0, 1], [1, 0]]) yields:

    bs0
    q_0: ─────
         ┌───┐
    q_1: ┤ X ├
         └───┘
    bs1
         ┌───┐
    q_0: ┤ X ├
         └───┘
    q_1: ─────

    bs0bs1xplus
         ┌───┐
    q_0: ┤ H ├──■──
         ├───┤┌─┴─┐
    q_1: ┤ X ├┤ X ├
         └───┘└───┘
    bs0bs1xmin
         ┌───┐┌───┐
    q_0: ┤ H ├┤ Z ├──■──
         ├───┤└───┘┌─┴─┐
    q_1: ┤ X ├─────┤ X ├
         └───┘     └───┘
    bs1bs0xplus
         ┌───┐
    q_0: ┤ H ├──■──
         ├───┤┌─┴─┐
    q_1: ┤ X ├┤ X ├
         └───┘└───┘
    bs1bs0xmin
         ┌───┐┌───┐
    q_0: ┤ H ├┤ Z ├──■──
         ├───┤└───┘┌─┴─┐
    q_1: ┤ X ├─────┤ X ├
         └───┘     └───┘

    Args:
        - bitstrings (List[Bitstring]): the input list of bitstrings used to generate the state preparation circuits
        - subsystem_id (Optional[str]): the subsystem the bitstring reflects ("u" or "v")

    Returns:
        - (Tuple[List[QuantumCircuit], List[QuantumCircuit]]): A tuple containing the tensor (i.e., non-superposition
            or bitstring) circuits in the first index (length = len(bitstrings)) and the super-position circuits
            as the second element
    """
    # If empty, just return
    if not bitstrings:
        return [], []

    if subsystem_id is None:
        subsystem_id = "u"
    # If the spin-up and spin-down spin orbitals are together a 2*N qubit system,
    # the bitstring should be N bits long.
    bitstring_array = np.asarray(bitstrings)
    tensor_prep_circuits = [
        _prepare_bitstring(bs, name=f"bs{subsystem_id}{str(bs_idx)}")
        for bs_idx, bs in enumerate(bitstring_array)
    ]

    superpos_prep_circuits = []
    # Create superposition circuits for each bitstring pair
    for bs1_idx, bs1 in enumerate(bitstring_array):
        for bs2_idx, bs2 in enumerate(bitstring_array):
            if bs1_idx == bs2_idx:
                continue
            diffs = np.where(bs1 != bs2)[0]
            if len(diffs) > 0:
                i = diffs[0]
                if bs1[i]:
                    x = bs2
                else:
                    x = bs1

                # Find the first position the bitstrings differ and place a
                # hadamard in that position
                S = np.delete(diffs, 0)
                qcirc = _prepare_bitstring(np.concatenate((x[:i], [0], x[i + 1 :])))
                qcirc.h(i)

                # Create a superposition circuit for each psi value in {0, 2}
                psi_xplus, psi_xmin = [
                    qcirc.copy(
                        name=f"bs{subsystem_id}{bs1_idx}bs{subsystem_id}{bs2_idx}{name}"
                    )
                    for name in ["xplus", "xmin"]
                ]
                psi_xmin.z(i)
                for psi in [psi_xplus, psi_xmin]:
                    for target in S:
                        psi.cx(i, target)
                    superpos_prep_circuits.append(psi)

            # If the two bitstrings are equivalent (i.e. bn==bm)
            else:
                qcirc = _prepare_bitstring(
                    bs1,
                    name=f"bs{subsystem_id}{bs1_idx}bs{subsystem_id}{bs2_idx}_hybrid_",
                )
                psi_xplus, psi_xmin = [
                    qcirc.copy(name=f"{qcirc.name}{name}") for name in ["xplus", "xmin"]
                ]
                superpos_prep_circuits += [psi_xplus, psi_xmin]

    return tensor_prep_circuits, superpos_prep_circuits


def _prepare_bitstring(
    bitstring: Union[NDArray[Shape["*"], Int], Bitstring],
    name: Optional[str] = None,
) -> QuantumCircuit:
    """Prepare the bitstring circuits.

    Generate a computational basis state from the input bitstring by applying an X gate to
    every qubit that has a 1 in the bitstring.

    Args:
        - bitstring (Union[NDArray[Shape["*"], Int], Bitstring]): the container for the
            bitstring information. Must contain 0s and 1s and the 1s are used to determine
            where to put the X gates
        - name (str, optional): the name of the circuit

    Returns:
        - (QuantumCircuit): the prepared circuit
    """
    qcirc = QuantumCircuit(len(bitstring), name=name)
    for qb_idx, bit in enumerate(bitstring):
        if bit:
            qcirc.x(qb_idx)
    return qcirc


def _partition(a, n):
    """Partitions the input.

    Function that partitions the input, a, into a generator containing
    n sub-partitions of a (that are the same type as a).
    Example:
    _partition([1, 2, 3], 2) -> (i for i in [[1, 2], [3]])

    Args:
        - a (iterable): an object with length and indexing to be partitioned
        - n (int): the number of partitions
    Returns:
        - (generator): the generator containing the paritions
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def _estimate_expvals(
    tensor_ansatze: List[QuantumCircuit],
    tensor_paulis: List[Pauli],
    superposition_ansatze: List[QuantumCircuit],
    superposition_paulis: List[Pauli],
    service_args: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = None,
    options: Optional[Options] = None,
    session_id: Optional[str] = None,
) -> Tuple[List[NDArray], List[NDArray], Optional[str]]:
    """Run quantum circuits to generate the expectation values.

    Function to estimate the exepctation value of some observables on the
    tensor and superposition circuits used for reconstructing the full
    expectation value from the Schmidt decomposed circuit. The ray decorator
    indicates that this is an actor function (that runs its own python
    process).

    Args:
        - tensor_ansatze (List[QuantumCircuit]): the circuits that have the same
            Schmidt coefficient
        - tensor_paulis (List[Pauli]): the pauli operators to measure and calculate
            the expectation values from for the circuits with the same Schmidt coefficient
        - superposition_ansatze (List[QuantumCircuit]): the circuits with different
            Schmidt coefficients
        - superposition_paulis (List[Pauli]): the pauli operators to measure and calculate
            the expectation values from for the circuits with different Schmidt
            coefficients
        - service_args (Dict[str, Any]): The service account used to spawn Qiskit primitives
        - backend_name (str): The backend to use to evaluate the grouped experiments
        - options (Options): The options to use with the backend
        - session_id (str): The session id to use when calling primitive programs

    Returns:
        - (Tuple[List[NDArray], List[NDArray], Optional[str]]): the expectation values for the
            tensor circuits and superposition circuits
    """
    all_circuits = tensor_ansatze + superposition_ansatze
    all_observables = tensor_paulis + superposition_paulis

    ansatz_t: List[QuantumCircuit] = []
    observables_t: List[Pauli] = []
    for i, circuit in enumerate(tensor_ansatze):
        ansatz_t += [circuit] * len(tensor_paulis)
        observables_t += tensor_paulis

    ansatz_s: List[QuantumCircuit] = []
    observables_s: List[Pauli] = []
    for i, circuit in enumerate(superposition_ansatze):
        ansatz_s += [circuit] * len(superposition_paulis)
        observables_s += superposition_paulis

    all_ansatze_for_estimator = ansatz_t + ansatz_s
    all_observables_for_estimator = observables_t + observables_s

    # ID for this job. If it is the first job for the knitter, it will become the session ID
    job_id: Optional[str] = None
    if service_args is not None:
        # Set the backend. Default to runtime qasm simulator
        if backend_name is None:
            raise ValueError(
                "If passing a QiskitRuntimeService, a list of backend names must be specified."
            )
        service = QiskitRuntimeService(**service_args)
        session = Session(service=service, backend=backend_name)
        session._session_id = session_id
        estimator = Estimator(session=session, options=options)

        job = estimator.run(
            circuits=all_ansatze_for_estimator,
            observables=all_observables_for_estimator,
        )

        results = job.result().values

        job_id = job.job_id

    else:
        estimator = TestEstimator(
            circuits=all_circuits,
            observables=all_observables,
        )
        results = (
            estimator.run(
                circuits=all_ansatze_for_estimator,
                observables=all_observables_for_estimator,
            )
            .result()
            .values
        )

    # Post-process the results to get our expectation values in the right format
    num_tensor_expvals = len(tensor_ansatze) * len(tensor_paulis)
    estimator_results_t = results[:num_tensor_expvals]
    estimator_results_s = results[num_tensor_expvals:]

    tensor_expval_list = list(
        estimator_results_t.reshape((len(tensor_ansatze), len(tensor_paulis)))
    )
    superposition_expval_list = list(
        estimator_results_s.reshape(
            (len(superposition_ansatze), len(superposition_paulis))
        )
    )

    # Scale the superposition terms
    for i, ansatz in enumerate(superposition_ansatze):
        # Scale the expectation values to account for 1/sqrt(2) coefficients
        if "hybrid_xmin" in ansatz.name:
            superposition_expval_list[i] *= 0.0
        elif "hybrid_xplus" in ansatz.name:
            pass
        else:
            superposition_expval_list[i] *= 0.5

    return tensor_expval_list, superposition_expval_list, job_id
