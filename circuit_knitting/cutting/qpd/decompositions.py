# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instruction to :class:`.QPDBasis` decompositions."""

from __future__ import annotations

from collections.abc import Sequence, Callable

import numpy as np
from qiskit.circuit import (
    Gate,
    ControlledGate,
    Instruction,
    Reset,
)
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import (
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CSGate,
    CSdgGate,
    CSXGate,
    RXXGate,
    RYYGate,
    RZZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    ECRGate,
    CPhaseGate,
    SwapGate,
    iSwapGate,
    DCXGate,
)
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

from .qpd_basis import QPDBasis
from .instructions import QPDMeasure
from ..instructions import Move
from ...utils.iteration import unique_by_id


_qpdbasis_from_instruction_funcs: dict[str, Callable[[Instruction], QPDBasis]] = {}


def _register_qpdbasis_from_instruction(*args):
    def g(f):
        for name in args:
            _qpdbasis_from_instruction_funcs[name] = f
        return f

    return g


def qpdbasis_from_instruction(gate: Instruction, /) -> QPDBasis:
    """
    Generate a :class:`.QPDBasis` object, given a supported operation.

    All two-qubit gates which implement the :meth:`~qiskit.circuit.Gate.to_matrix` method are
    supported.  This should include the vast majority of gates with no unbound
    parameters, but there are some special cases (see, e.g., `qiskit issue #10396
    <https://github.com/Qiskit/qiskit-terra/issues/10396>`__).

    The :class:`.Move` operation, which can be used to specify a wire cut,
    is also supported.

    Returns:
        The newly-instantiated :class:`QPDBasis` object

    Raises:
        ValueError: Instruction not supported.
        ValueError: Cannot decompose instruction with unbound parameters.
        ValueError: ``to_matrix`` conversion of two-qubit gate failed.
    """
    try:
        f = _qpdbasis_from_instruction_funcs[gate.name]
    except KeyError:
        pass
    else:
        return f(gate)

    if isinstance(gate, Gate) and gate.num_qubits == 2:
        try:
            mat = gate.to_matrix()
        except Exception as ex:
            raise ValueError(
                f"`to_matrix` conversion of two-qubit gate ({gate.name}) failed. "
                "Often, this can be caused by unbound parameters."
            ) from ex
        d = TwoQubitWeylDecomposition(mat)
        u = _u_from_thetavec([d.a, d.b, d.c])
        retval = _nonlocal_qpd_basis_from_u(u)
        for operations in unique_by_id(m[0] for m in retval.maps):
            operations.insert(0, UnitaryGate(d.K2r))
            operations.append(UnitaryGate(d.K1r))
        for operations in unique_by_id(m[1] for m in retval.maps):
            operations.insert(0, UnitaryGate(d.K2l))
            operations.append(UnitaryGate(d.K1l))
        return retval

    raise ValueError(f"Instruction not supported: {gate.name}")


def _explicitly_supported_instructions() -> set[str]:
    """
    Return a set of instruction names with explicit support for automatic decomposition.

    These instructions are *explicitly* supported by :func:`qpdbasis_from_instruction`.
    Other instructions may be supported too, via a KAK decomposition.

    Returns:
        Set of gate names supported for automatic decomposition.
    """
    return set(_qpdbasis_from_instruction_funcs)


def _copy_unique_sublists(lsts: tuple[list, ...], /) -> tuple[list, ...]:
    """
    Copy each list in a sequence of lists while preserving uniqueness.

    This is useful to ensure that the two sets of ``maps`` in a
    :class:`QPDBasis` will be independent of each other.  This enables one to
    subsequently edit the ``maps`` independently of each other (e.g., to apply
    single-qubit pre- or post-rotations.
    """
    copy_by_id: dict[int, list] = {}
    for lst in lsts:
        if id(lst) not in copy_by_id:
            copy_by_id[id(lst)] = lst.copy()
    return tuple(copy_by_id[id(lst)] for lst in lsts)


def _u_from_thetavec(
    theta: np.typing.NDArray[np.float64] | Sequence[float], /
) -> np.typing.NDArray[np.complex128]:
    r"""
    Exponentiate the non-local portion of a KAK decomposition.

    This implements Eq. (6) of https://arxiv.org/abs/2006.11174v2:

    .. math::

       \exp [ i ( \sum_\alpha^3 \theta_\alpha \, \sigma_\alpha \otimes \sigma_\alpha ) ]
       =
       \sum_{\alpha=0}^3 u_\alpha \, \sigma_\alpha \otimes \sigma_\alpha

    where each :math:`\theta_\alpha` is assumed to be real, and
    :math:`u_\alpha` is complex in general.
    """
    theta = np.asarray(theta)
    if theta.shape != (3,):
        raise ValueError(
            f"theta vector has wrong shape: {theta.shape} (1D vector of length 3 expected)"
        )
    # First, we note that if we choose the basis vectors II, XX, YY, and ZZ,
    # then the following matrix represents one application of the summation in
    # the exponential:
    #
    #   0   θx  θy  θz
    #   θx  0  -θz -θy
    #   θy -θz  0  -θx
    #   θz -θy -θx  0
    #
    # This matrix is symmetric and can be exponentiated by diagonalizing it.
    # Its eigendecomposition is given by:
    eigvals = np.array(
        [
            -np.sum(theta),
            -theta[0] + theta[1] + theta[2],
            -theta[1] + theta[2] + theta[0],
            -theta[2] + theta[0] + theta[1],
        ]
    )
    eigvecs = np.ones([1, 1]) / 2 - np.eye(4)
    # Finally, we exponentiate the eigenvalues of the matrix in diagonal form.
    # We also project to the vector [1,0,0,0] on the right, since the
    # multiplicative identity is given by II.
    return np.transpose(eigvecs) @ (np.exp(1j * eigvals) * eigvecs[:, 0])


def _nonlocal_qpd_basis_from_u(
    u: np.typing.NDArray[np.complex128] | Sequence[complex], /
) -> QPDBasis:
    u = np.asarray(u)
    if u.shape != (4,):
        raise ValueError(
            f"u vector has wrong shape: {u.shape} (1D vector of length 4 expected)"
        )
    # The following operations are described in Sec. 2.3 of
    # https://quantum-journal.org/papers/q-2021-01-28-388/
    #
    # Projective measurements in each basis
    A0x = [HGate(), QPDMeasure(), HGate()]
    A0y = [SXGate(), QPDMeasure(), SXdgGate()]
    A0z = [QPDMeasure()]
    # Single qubit rotations that swap two axes.  There are "plus" and "minus"
    # versions of these rotations.  The "minus" rotations also flip the sign
    # along that axis.
    Axyp = [SGate(), YGate()]
    Axym = [ZGate()] + Axyp
    Ayzp = [SXGate(), ZGate()]
    Ayzm = [XGate()] + Ayzp
    Azxp = [HGate()]
    Azxm = [YGate()] + Azxp
    # Single qubit rotations by ±pi/4 about each axis.
    B0xp = [SXGate()]
    B0xm = [SXdgGate()]
    B0yp = [RYGate(0.5 * np.pi)]
    B0ym = [RYGate(-0.5 * np.pi)]
    B0zp = [SGate()]
    B0zm = [SdgGate()]
    # Projective measurements, each followed by the proper flip.
    Bxy = A0z + [XGate()]
    Byz = A0x + [YGate()]
    Bzx = A0y + [ZGate()]
    # The following values occur repeatedly in the coefficients
    uu01 = u[0] * np.conj(u[1])
    uu02 = u[0] * np.conj(u[2])
    uu03 = u[0] * np.conj(u[3])
    uu12 = u[1] * np.conj(u[2])
    uu23 = u[2] * np.conj(u[3])
    uu31 = u[3] * np.conj(u[1])
    coeffs, maps1, maps2 = zip(
        # First line of Eq. (19) in
        # https://quantum-journal.org/papers/q-2021-01-28-388/
        (np.abs(u[0]) ** 2, [], []),  # Identity
        (np.abs(u[1]) ** 2, [XGate()], [XGate()]),
        (np.abs(u[2]) ** 2, [YGate()], [YGate()]),
        (np.abs(u[3]) ** 2, [ZGate()], [ZGate()]),
        # Second line
        (2 * np.real(uu01), A0x, A0x),
        (2 * np.real(uu02), A0y, A0y),
        (2 * np.real(uu03), A0z, A0z),
        (0.5 * np.real(uu12), Axyp, Axyp),
        (-0.5 * np.real(uu12), Axyp, Axym),
        (-0.5 * np.real(uu12), Axym, Axyp),
        (0.5 * np.real(uu12), Axym, Axym),
        (0.5 * np.real(uu23), Ayzp, Ayzp),
        (-0.5 * np.real(uu23), Ayzp, Ayzm),
        (-0.5 * np.real(uu23), Ayzm, Ayzp),
        (0.5 * np.real(uu23), Ayzm, Ayzm),
        (0.5 * np.real(uu31), Azxp, Azxp),
        (-0.5 * np.real(uu31), Azxp, Azxm),
        (-0.5 * np.real(uu31), Azxm, Azxp),
        (0.5 * np.real(uu31), Azxm, Azxm),
        (-0.5 * np.real(uu01), B0xp, B0xp),
        (0.5 * np.real(uu01), B0xp, B0xm),
        (0.5 * np.real(uu01), B0xm, B0xp),
        (-0.5 * np.real(uu01), B0xm, B0xm),
        (-0.5 * np.real(uu02), B0yp, B0yp),
        (0.5 * np.real(uu02), B0yp, B0ym),
        (0.5 * np.real(uu02), B0ym, B0yp),
        (-0.5 * np.real(uu02), B0ym, B0ym),
        (-0.5 * np.real(uu03), B0zp, B0zp),
        (0.5 * np.real(uu03), B0zp, B0zm),
        (0.5 * np.real(uu03), B0zm, B0zp),
        (-0.5 * np.real(uu03), B0zm, B0zm),
        (-2 * np.real(uu12), Bxy, Bxy),
        (-2 * np.real(uu23), Byz, Byz),
        (-2 * np.real(uu31), Bzx, Bzx),
        # Third line
        (np.imag(uu01), A0x, B0xp),
        (-np.imag(uu01), A0x, B0xm),
        (np.imag(uu01), B0xp, A0x),
        (-np.imag(uu01), B0xm, A0x),
        (np.imag(uu02), A0y, B0yp),
        (-np.imag(uu02), A0y, B0ym),
        (np.imag(uu02), B0yp, A0y),
        (-np.imag(uu02), B0ym, A0y),
        (np.imag(uu03), A0z, B0zp),
        (-np.imag(uu03), A0z, B0zm),
        (np.imag(uu03), B0zp, A0z),
        (-np.imag(uu03), B0zm, A0z),
        (np.imag(uu12), Axyp, Bxy),
        (-np.imag(uu12), Axym, Bxy),
        (np.imag(uu12), Bxy, Axyp),
        (-np.imag(uu12), Bxy, Axym),
        (np.imag(uu23), Ayzp, Byz),
        (-np.imag(uu23), Ayzm, Byz),
        (np.imag(uu23), Byz, Ayzp),
        (-np.imag(uu23), Byz, Ayzm),
        (np.imag(uu31), Azxp, Bzx),
        (-np.imag(uu31), Azxm, Bzx),
        (np.imag(uu31), Bzx, Azxp),
        (-np.imag(uu31), Bzx, Azxm),
    )
    maps = list(zip(maps1, _copy_unique_sublists(maps2)))
    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_instruction("swap")
def _(unused_gate: SwapGate):
    return _nonlocal_qpd_basis_from_u([(1 + 1j) / np.sqrt(8)] * 4)


@_register_qpdbasis_from_instruction("iswap")
def _(unused_gate: iSwapGate):
    return _nonlocal_qpd_basis_from_u([0.5, 0.5j, 0.5j, 0.5])


@_register_qpdbasis_from_instruction("dcx")
def _(unused_gate: DCXGate):
    retval = qpdbasis_from_instruction(iSwapGate())
    # Modify basis according to DCXGate definition in Qiskit circuit library
    # https://github.com/Qiskit/qiskit-terra/blob/e9f8b7c50968501e019d0cb426676ac606eb5a10/qiskit/circuit/library/standard_gates/equivalence_library.py#L938-L944
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, SdgGate())
        operations.insert(0, HGate())
    for operations in unique_by_id(m[1] for m in retval.maps):
        operations.insert(0, SdgGate())
        operations.append(HGate())
    return retval


@_register_qpdbasis_from_instruction("rxx", "ryy", "rzz", "crx", "cry", "crz")
def _(gate: RXXGate | RYYGate | RZZGate | CRXGate | CRYGate | CRZGate):
    # Constructing a virtual two-qubit gate by sampling single-qubit operations - Mitarai et al
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pdf
    if gate.name in ("rxx", "crx"):
        pauli = XGate()
        r_plus = SXGate()
        # x basis measurement (and back again)
        measurement_0 = [HGate(), QPDMeasure(), HGate()]
    elif gate.name in ("ryy", "cry"):
        pauli = YGate()
        r_plus = RYGate(0.5 * np.pi)
        # y basis measurement (and back again)
        measurement_0 = [SXGate(), QPDMeasure(), SXdgGate()]
    else:
        assert gate.name in ("rzz", "crz")
        pauli = ZGate()
        r_plus = SGate()
        # z basis measurement
        measurement_0 = [QPDMeasure()]

    r_minus = r_plus.inverse()
    measurement_1 = measurement_0.copy()

    # Specify the operations to be inserted for systems 1 and 2
    maps = [
        ([], []),  # Identity
        ([pauli], [pauli]),
        (measurement_0, [r_plus]),
        (measurement_0, [r_minus]),
        ([r_plus], measurement_1),
        ([r_minus], measurement_1),
    ]

    theta = _theta_from_instruction(gate)

    if gate.name[0] == "c":
        # Following Eq. (C.4) of https://arxiv.org/abs/2205.00016v2,
        # which implements controlled single qubit rotations in terms of a
        # parametric two-qubit rotation.
        theta = -theta / 2
        # Modify `maps`
        for operations in unique_by_id(m[0] for m in maps):
            if operations and gate.name != "crz":
                if gate.name == "cry":
                    operations.insert(0, SGate())
                    operations.append(SdgGate())
                operations.insert(0, HGate())
                operations.append(HGate())
        rot = {"x": RXGate, "y": RYGate, "z": RZGate}[gate.name[2]](-theta)
        for operations in unique_by_id(m[1] for m in maps):
            operations.append(rot)

    # Generate QPD coefficients for gates specified by Mitarai et al -
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pd
    theta_prime = -theta / 2  # theta as defined by Mitarai and Fujii
    cs_theta_prime = np.cos(theta_prime) * np.sin(theta_prime)
    coeffs = [
        np.cos(theta_prime) ** 2,
        np.sin(theta_prime) ** 2,
        # Qiskit defines RZ(theta) = exp(-i*Z*theta/2), i.e., with a minus
        # sign in the exponential; hence r_plus really represents
        # exp(-i*Z*pi/4), and similar for RX and RY.  Following Mitarai and
        # Fujii Eq. (6), the correct signs are therefore given by
        -cs_theta_prime,
        cs_theta_prime,
        -cs_theta_prime,
        cs_theta_prime,
    ]

    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_instruction("cs", "csdg")
def _(gate: CSGate | CSdgGate):
    theta = np.pi / 2
    rot_gate = TGate()
    if gate.name == "csdg":
        theta *= -1
        rot_gate = rot_gate.inverse()
    retval = qpdbasis_from_instruction(CRZGate(theta))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, rot_gate)
    return retval


@_register_qpdbasis_from_instruction("cp")
def _(gate: CPhaseGate):
    theta = _theta_from_instruction(gate)
    retval = qpdbasis_from_instruction(CRZGate(theta))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, PhaseGate(theta / 2))
    return retval


@_register_qpdbasis_from_instruction("csx")
def _(unused_gate: CSXGate):
    retval = qpdbasis_from_instruction(CRXGate(np.pi / 2))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, TGate())
    return retval


@_register_qpdbasis_from_instruction("csxdg")
def _(unused_gate: ControlledGate):
    retval = qpdbasis_from_instruction(CRXGate(-np.pi / 2))
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, TdgGate())
    return retval


@_register_qpdbasis_from_instruction("cx", "cy", "cz", "ch")
def _(gate: CXGate | CYGate | CZGate | CHGate):
    # Constructing a virtual two-qubit gate by sampling single-qubit operations - Mitarai et al
    # https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc/pdf
    measurement_0 = [SdgGate(), QPDMeasure()]
    measurement_1 = measurement_0.copy()

    # Specify the operations to be inserted for systems 1 and 2
    maps = [
        ([SdgGate()], [SdgGate()]),
        ([SGate()], [SGate()]),
        # R-R+ simplifies to I
        (measurement_0, []),
        (measurement_0, [ZGate()]),
        # R-R+ simplifies to I
        ([], measurement_1),
        ([ZGate()], measurement_1),
    ]

    if gate.name != "cz":
        # Modify `maps` to sandwich the target operations inside of basis rotations
        for operations in unique_by_id(m[1] for m in maps):
            if operations:
                if gate.name in ("cx", "cy"):
                    operations.insert(0, HGate())
                    operations.append(HGate())
                    if gate.name == "cy":
                        operations.insert(0, SdgGate())
                        operations.append(SGate())
                elif gate.name == "ch":
                    operations.insert(0, RYGate(-np.pi / 4))
                    operations.append(RYGate(np.pi / 4))

    coeffs = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5]

    return QPDBasis(maps, coeffs)


@_register_qpdbasis_from_instruction("ecr")
def _(unused_gate: ECRGate):
    retval = qpdbasis_from_instruction(CXGate())
    # Modify basis according to ECRGate definition in Qiskit circuit library
    # https://github.com/Qiskit/qiskit-terra/blob/d9763523d45a747fd882a7e79cc44c02b5058916/qiskit/circuit/library/standard_gates/equivalence_library.py#L656-L663
    for operations in unique_by_id(m[0] for m in retval.maps):
        operations.insert(0, SGate())
        operations.append(XGate())
    for operations in unique_by_id(m[1] for m in retval.maps):
        operations.insert(0, SXGate())
    return retval


def _theta_from_instruction(gate: Gate, /) -> float:
    param_gates = {"rxx", "ryy", "rzz", "crx", "cry", "crz", "cp"}

    # Internal function should only be called for supported gates
    assert gate.name in param_gates

    # If theta is a bound ParameterExpression, convert to float, else raise error.
    try:
        theta = float(gate.params[0])
    except TypeError as err:
        raise ValueError(
            f"Cannot decompose ({gate.name}) instruction with unbound parameters."
        ) from err

    return theta


@_register_qpdbasis_from_instruction("move")
def _(unused_gate: Move):
    i_measurement = [Reset()]
    x_measurement = [HGate(), QPDMeasure(), Reset()]
    y_measurement = [SXGate(), QPDMeasure(), Reset()]
    z_measurement = [QPDMeasure(), Reset()]

    prep_0 = [Reset()]
    prep_1 = [Reset(), XGate()]
    prep_plus = [Reset(), HGate()]
    prep_minus = [Reset(), XGate(), HGate()]
    prep_iplus = [Reset(), SXdgGate()]
    prep_iminus = [Reset(), XGate(), SXdgGate()]

    # https://arxiv.org/abs/1904.00102v2 Eqs. (12)-(19)
    maps1, maps2, coeffs = zip(
        (i_measurement, prep_0, 0.5),
        (i_measurement, prep_1, 0.5),
        (x_measurement, prep_plus, 0.5),
        (x_measurement, prep_minus, -0.5),
        (y_measurement, prep_iplus, 0.5),
        (y_measurement, prep_iminus, -0.5),
        (z_measurement, prep_0, 0.5),
        (z_measurement, prep_1, -0.5),
    )
    maps = list(zip(maps1, maps2))
    return QPDBasis(maps, coeffs)
