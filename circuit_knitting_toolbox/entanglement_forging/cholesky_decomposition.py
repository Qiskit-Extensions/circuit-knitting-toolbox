# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the EntanglementForgingGroundStateSolver class and associated functions."""

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nptyping import Float, Int, NDArray, Shape
from qiskit.opflow import ListOp, PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.drivers import ElectronicStructureDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)

from .entanglement_forging_ansatz import EntanglementForgingAnsatz
from .entanglement_forging_operator import EntanglementForgingOperator
from .entanglement_forging_ansatz import EntanglementForgingAnsatz


SingleBodyIntegrals = NDArray[Shape["N, N"], Float]
Matrix = NDArray[Shape["N, N"], Float]
TwoBodyIntegrals = NDArray[Shape["N, N, N, N"], Float]


def get_cholesky_op(
    l_op: NDArray, g: int, converter: QubitConverter, opname: str
) -> PauliSumOp:
    """
    Convert a two-body term into a cholesky operator.

    Args:
        - l_op: Two body integrals
        - g: integral index
        - converter: Qubit converter to be used
        - opname: Prefix for output cholesky operator name

    Returns:
        - cholesky_operator: The converted operator
    """
    cholesky_int = OneBodyElectronicIntegrals(
        basis=ElectronicBasis.SO, matrices=l_op[:, :, g]
    )
    cholesky_property = IntegralProperty("cholesky_op", [cholesky_int])
    if isinstance(cholesky_property.second_q_ops(), dict):
        cholesky_op = converter.convert(cholesky_property.second_q_ops()["cholesky_op"])
    else:
        cholesky_op = converter.convert(cholesky_property.second_q_ops()[0])
    cholesky_op._name = opname + "_chol" + str(g)
    return cholesky_op


def cholesky_decomposition(
    problem: ElectronicStructureProblem,
    orbitals_to_reduce: Optional[Sequence[int]] = None,
) -> Tuple[ListOp, float]:
    """
    Construct the decomposed Hamiltonian from an input ``ElectronicStructureProblem``.

    Args:
        - problem (ElectronicStructureProblem): An ``ElectronicStructureProblem`` from which the decomposed Hamiltonian will be
            calculated.
        - orbitals_to_reduce (Optional[Sequence[int]]): A list of orbital indices to remove from the problem before decomposition.

    Returns:
        - Tuple containing
            - cholesky_operator (ListOp): A list of operators representing the decomposed Hamiltonian.
              shape: [single-body hamiltonian, cholesky_0, ..., cholesky_N]
            - freeze_shift (float): An energy shift resulting from the decomposition. This shift should be re-applied after
              calculating properties of the decomposed operator (i.e. ground state energy).
    """
    if problem.grouped_property_transformed is None:
        raise AttributeError(
            "There was a problem retrieving the grouped properties from the ElectronicStructureProblem."
        )
    if problem.driver is None:
        raise AttributeError("The ElectronicStructureProblem has no driver.")

    if not isinstance(problem.driver, ElectronicStructureDriver):
        raise AttributeError(
            "The ElectronicStructureProblem's driver should be an instance of ElectronicStructureDriver."
        )

    electronic_basis_transform = problem.grouped_property_transformed.get_property(
        "ElectronicBasisTransform"
    )
    if electronic_basis_transform is None:
        raise AttributeError(
            "There was a problem retrieving the ElectronicBasisTransform property from the ElectronicStructureProblem."
        )

    electronic_energy = problem.grouped_property_transformed.get_property(
        "ElectronicEnergy"
    )
    if electronic_energy is None:
        raise AttributeError(
            "There was a problem retrieving the ElectronicEnergy property from the ElectronicStructureProblem."
        )

    particle_number = problem.grouped_property_transformed.get_property(
        "ParticleNumber"
    )
    if particle_number is None:
        raise AttributeError(
            "There was a problem retrieving the ParticleNumber property from the ElectronicStructureProblem."
        )

    # Get data for generating the cholesky decomposition
    mo_coeff: Matrix = electronic_basis_transform.coeff_alpha
    hcore: SingleBodyIntegrals = electronic_energy.get_electronic_integral(
        ElectronicBasis.AO, 1
    )._matrices[0]
    eri: TwoBodyIntegrals = electronic_energy.get_electronic_integral(
        ElectronicBasis.AO, 2
    )._matrices[0]
    num_alpha = particle_number.num_alpha

    # Store the reduced orbitals as virtual and occupied lists
    if orbitals_to_reduce is None:
        orbitals_to_reduce = []
    orbitals_to_reduce_dict: Dict[
        str, NDArray[Shape["*"], Int]
    ] = _get_orbitals_to_reduce(orbitals_to_reduce, num_alpha)

    # Hold fields used to calculate the final energy shift
    # Freeze shift will be calculated during decomposition
    freeze_shift = 0.0
    nuclear_repulsion_energy = electronic_energy.nuclear_repulsion_energy

    h_1_op, h_chol_ops, freeze_shift, _, _ = _get_fermionic_ops_with_cholesky(
        mo_coeff,
        hcore,
        eri,
        opname="H",
        halve_transformed_h2=True,
        occupied_orbitals_to_reduce=orbitals_to_reduce_dict["occupied"],
        virtual_orbitals_to_reduce=orbitals_to_reduce_dict["virtual"],
    )

    op_list = [h_1_op] + h_chol_ops
    operator = ListOp(op_list)

    energy_shift = freeze_shift + nuclear_repulsion_energy

    return operator, energy_shift


def convert_cholesky_operator(
    operator: ListOp,
    ansatz: EntanglementForgingAnsatz,
) -> EntanglementForgingOperator:
    """
    Convert the Cholesky operator (ListOp) into the entanglement forging format.

    Args:
        - operator: A `ListOp` containing the single-body Hamiltonian followed
            by the Cholesky operators.
            shape: [single-body hamiltonian, cholesky_0, ..., cholesky_N]
        - ansatz:
            The ansatz for which to compute expectation values of operator. The
            `EntanglementForgingAnsatz` also contains the bitstrings for each subsystem..

    Returns:
        - forged_operator: An `EntanglementForgingOperator` object describing the
            decomposed operator.
    """
    calculate_hybrid_cross_terms = len(set(ansatz.bitstrings_u)) < len(
        ansatz.bitstrings_u
    ) or len(set(ansatz.bitstrings_v)) < len(ansatz.bitstrings_v)

    op1 = operator[0]
    cholesky_ops = operator[1:]

    # The block below calculate the Pauli-pair prefactors w_ij and returns
    # them as a dictionary
    tensor_paulis = set()
    superpos_paulis = set()
    paulis_each_op = [
        {
            label: weight
            for label, weight in op.primitive.to_list()
            if np.abs(weight) > 0
        }
        for op in [op1] + list(cholesky_ops)
    ]

    # Gather the elements in the Pauli basis for tensor and superpos terms
    paulis_each_op = [paulis_each_op[0]] + [p for p in paulis_each_op[1:] if p]
    for op_idx, paulis_this_op in enumerate(paulis_each_op):
        pnames = list(paulis_this_op.keys())
        tensor_paulis.update(pnames)

        # If hybrid terms are needed, the superposition basis includes
        # terms from the single body Hamiltonian.
        if calculate_hybrid_cross_terms or op_idx > 0:
            superpos_paulis.update(pnames)

    # ensure Identity string is represented since we will need it
    identity_string = "I" * len(pnames[0])
    tensor_paulis.add(identity_string)
    superpos_paulis.add(identity_string)

    # Sort the Pauli bases
    tensor_pauli_names = list(sorted(tensor_paulis))
    superpos_pauli_names = list(sorted(superpos_paulis))

    # Map the tensor Pauli terms to their place in the tensor index
    pauli_ordering_for_tensor_states = {
        pname: idx for idx, pname in enumerate(tensor_pauli_names)
    }
    # Map the superpos Pauli basis terms to their place in the superpos index
    pauli_ordering_for_superpos_states = {
        pname: idx for idx, pname in enumerate(superpos_pauli_names)
    }

    # Create arrays for the tensor and superpos weights, respectively
    w_ij = np.zeros((len(tensor_pauli_names), len(tensor_pauli_names)))
    w_ab = np.zeros((len(superpos_pauli_names), len(superpos_pauli_names)))

    # Processes the non-Cholesky operator
    identity_idx = pauli_ordering_for_tensor_states[identity_string]
    identity_idx_superpos = pauli_ordering_for_tensor_states[identity_string]
    for pname_i, w_i in paulis_each_op[0].items():
        i = pauli_ordering_for_tensor_states[pname_i]
        w_ij[i, identity_idx] += np.real(w_i)  # H_spin-up
        w_ij[identity_idx, i] += np.real(w_i)  # H_spin-down

        # In the special case where bn=bm, we need terms from the
        # single body system represented in the cross terms
        if calculate_hybrid_cross_terms:
            w_ab[i, identity_idx_superpos] += np.real(w_i)
            w_ab[identity_idx_superpos, i] += np.real(w_i)

    # Processes the Cholesky operators (indexed by gamma)
    for paulis_this_gamma in paulis_each_op[1:]:
        for pname_1, w_1 in paulis_this_gamma.items():
            i = pauli_ordering_for_tensor_states[pname_1]
            superpos_ordering1 = pauli_ordering_for_superpos_states[pname_1]
            for pname_2, w_2 in paulis_this_gamma.items():
                j = pauli_ordering_for_tensor_states[pname_2]
                superpos_ordering2 = pauli_ordering_for_superpos_states[pname_2]
                w_ij[i, j] += np.real(w_1 * w_2)
                w_ab[superpos_ordering1, superpos_ordering2] += np.real(w_1 * w_2)

    # Convert from string representation to Pauli objects
    tensor_pauli_list = [Pauli(name) for name in tensor_pauli_names]
    superpos_pauli_list = [Pauli(name) for name in superpos_pauli_names]

    forged_operator = EntanglementForgingOperator(
        tensor_paulis=tensor_pauli_list,
        superposition_paulis=superpos_pauli_list,
        w_ij=w_ij,
        w_ab=w_ab,
    )

    return forged_operator


def _get_fermionic_ops_with_cholesky(
    mo_coeff: Matrix,
    h1: SingleBodyIntegrals,
    h2: TwoBodyIntegrals,
    opname: str,
    halve_transformed_h2: bool = False,
    occupied_orbitals_to_reduce: Optional[NDArray[Shape["*"], Int]] = None,
    virtual_orbitals_to_reduce: Optional[NDArray[Shape["*"], Int]] = None,
    epsilon_cholesky: float = 1e-10,
) -> Tuple[PauliSumOp, List[PauliSumOp], float, SingleBodyIntegrals, TwoBodyIntegrals,]:
    r"""
    Decompose the Hamiltonian operators into a form appropriate for entanglement forging.

    Args:
        - mo_coeff (NDArray[Shape["N, N"], Float]): 2D array representing coefficients for converting from AO to MO basis.
        - h1 (NDArray[Shape["N, N"], Float]): 2D array representing operator
            coefficients of one-body integrals in the AO basis.
        - h2 (NDArray[Shape["N, N, N, N"], Float]): 4D array representing operator coefficients
            of two-body integrals in the AO basis.
        - halve_transformed_h2 (Optional[bool]): Should be set to True for Hamiltonian
            operator to agree with Qiskit conventions.
        - occupied_orbitals_to_reduce (Optional[NDArray[Shape["*"], Int]]): Optional; A list of occupied orbitals that will be removed.
        - virtual_orbitals_to_reduce (Optional[NDArray[Shape["*"], Int]]):Optional; A list of virtual orbitals that will be removed.
        - epsilon_cholesky (Optional[float]): The threshold for the decomposition (typically a number close to 0).

    Returns:
        - qubit_op (PauliSumOp): H_1 in the Cholesky decomposition.
        - cholesky_ops (List[PauliSumOp]): L_\\gamma in the Cholesky decomposition
        - freeze_shift (float): Energy shift due to freezing.
        - h1 (NDArray[Shape["N, N"], Float]): 2D array representing operator coefficients of one-body
            integrals in the MO basis.
        - h2 (NDArray[Shape["N, N, N, N"], Float]): 4D array representing operator coefficients of
            two-body integrals in the MO basis.
    """
    if virtual_orbitals_to_reduce is None:
        virtual_orbitals_to_reduce = np.array([])
    if occupied_orbitals_to_reduce is None:
        occupied_orbitals_to_reduce = np.array([])

    coeff_mo = copy.copy(mo_coeff)

    h1 = np.einsum("pi,pr->ir", coeff_mo, h1)
    h1 = np.einsum("rj,ir->ij", coeff_mo, h1)  # h_{pq} in MO basis

    # Do the cholesky decomposition
    if h2 is not None:
        num_gammas, l_op = _get_modified_cholesky(h2, epsilon_cholesky)

        # Obtain L_{pr,g} in the MO basis
        l_op = np.einsum("prg,pi,rj->ijg", l_op, coeff_mo, coeff_mo)
    else:
        size = len(h1)
        num_gammas, l_op = 0, np.zeros(shape=(size, size, 0))

    if len(occupied_orbitals_to_reduce) > 0:
        orbitals_not_to_reduce_array = np.array(
            sorted(set(range(len(h1))) - set(occupied_orbitals_to_reduce))
        )

        h1_frozenpart = h1[
            np.ix_(occupied_orbitals_to_reduce, occupied_orbitals_to_reduce)
        ]
        h1_activepart = h1[
            np.ix_(orbitals_not_to_reduce_array, orbitals_not_to_reduce_array)
        ]
        l_frozenpart = l_op[
            np.ix_(occupied_orbitals_to_reduce, occupied_orbitals_to_reduce)
        ]
        l_activepart = l_op[
            np.ix_(orbitals_not_to_reduce_array, orbitals_not_to_reduce_array)
        ]

        freeze_shift = (
            2 * np.einsum("pp", h1_frozenpart)
            + 2 * np.einsum("ppg,qqg", l_frozenpart, l_frozenpart)
            - np.einsum("pqg,qpg", l_frozenpart, l_frozenpart)
        )

        h1 = (
            h1_activepart
            + 2 * np.einsum("ppg,qsg->qs", l_frozenpart, l_activepart)
            - np.einsum(
                "psg,qpg->qs",
                l_op[np.ix_(occupied_orbitals_to_reduce, orbitals_not_to_reduce_array)],
                l_op[np.ix_(orbitals_not_to_reduce_array, occupied_orbitals_to_reduce)],
            )
        )
        l_op = l_activepart

    else:
        freeze_shift = 0

    if virtual_orbitals_to_reduce.shape[0]:
        virtual_orbitals_to_reduce -= len(occupied_orbitals_to_reduce)  # type: ignore
        orbitals_not_to_reduce = list(
            sorted(set(range(len(h1))) - set(virtual_orbitals_to_reduce))  # type: ignore
        )
        h1 = h1[np.ix_(orbitals_not_to_reduce, orbitals_not_to_reduce)]
        l_op = l_op[np.ix_(orbitals_not_to_reduce, orbitals_not_to_reduce)]
    else:
        pass

    h2 = np.einsum("prg,qsg->prqs", l_op, l_op)

    if halve_transformed_h2:
        h2 /= 2  # type: ignore
    h1_int = OneBodyElectronicIntegrals(basis=ElectronicBasis.SO, matrices=h1)
    h2_int = TwoBodyElectronicIntegrals(basis=ElectronicBasis.SO, matrices=h2)
    int_property = IntegralProperty("fer_op", [h1_int, h2_int])

    if isinstance(int_property.second_q_ops(), dict):
        fer_op = int_property.second_q_ops()["fer_op"]
    else:
        fer_op = int_property.second_q_ops()[0]

    converter = QubitConverter(JordanWignerMapper())
    qubit_op = converter.convert(fer_op)

    qubit_op._name = opname + "_onebodyop"

    cholesky_ops = [
        get_cholesky_op(l_op, g, converter, opname) for g in range(l_op.shape[2])
    ]

    return qubit_op, cholesky_ops, freeze_shift, h1, h2


def _get_modified_cholesky(
    two_body_overlap_integrals: NDArray[Shape["*, *, *"], Float], eps: float
):
    """Perform modified Cholesky decomposition on the two-body integrals given an epsilon value."""
    n_basis_states = two_body_overlap_integrals.shape[0]  # number of basis states
    # Max (chmax) and current (n_gammas) number of Cholesky vectors
    ch_max, n_gammas = 10 * n_basis_states, 0

    w_op = two_body_overlap_integrals.reshape(n_basis_states**2, n_basis_states**2)
    l_op = np.zeros((n_basis_states**2, ch_max))
    d_max = np.diagonal(w_op).copy()
    nu_max = np.argmax(d_max)
    v_max = d_max[nu_max]

    while v_max > eps:
        l_op[:, n_gammas] = w_op[:, nu_max]
        if n_gammas > 0:
            l_op[:, n_gammas] -= np.dot(l_op[:, 0:n_gammas], l_op.T[0:n_gammas, nu_max])
        l_op[:, n_gammas] /= np.sqrt(v_max)
        d_max[: n_basis_states**2] -= l_op[: n_basis_states**2, n_gammas] ** 2
        n_gammas += 1
        nu_max = np.argmax(d_max)
        v_max = d_max[nu_max]

    l_op = l_op[:, :n_gammas].reshape((n_basis_states, n_basis_states, n_gammas))

    return n_gammas, l_op


def _get_orbitals_to_reduce(
    orbitals_to_reduce: Iterable[int],
    num_alpha: int,
) -> Dict[str, NDArray[Shape["*"], Int]]:
    orb_to_reduce_dict = {
        "occupied": np.asarray(orbitals_to_reduce),
        "virtual": np.asarray(orbitals_to_reduce),
        "all": np.asarray(orbitals_to_reduce),
    }

    # Populate the occupied list within the dict
    orb_to_reduce_dict["occupied"] = orb_to_reduce_dict["occupied"][
        orb_to_reduce_dict["occupied"] < num_alpha
    ]

    # Populate the virtual list within the dict
    orb_to_reduce_dict["virtual"] = orb_to_reduce_dict["virtual"][
        orb_to_reduce_dict["virtual"] >= num_alpha
    ]

    return orb_to_reduce_dict
