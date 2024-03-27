# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from ddt import ddt, data, unpack

import pytest
import numpy as np
from qiskit.result import QuasiDistribution
from qiskit.primitives import (
    SamplerResult,
    PrimitiveResult,
    PubResult,
    BitArray,
)
from qiskit.primitives.containers import make_data_bin
from qiskit.quantum_info import Pauli, PauliList

from circuit_knitting.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting.cutting.qpd import WeightType
from circuit_knitting.cutting.cutting_reconstruction import (
    _process_outcome,
    reconstruct_expectation_values,
)


@ddt
class TestCuttingReconstruction(unittest.TestCase):
    def setUp(self):
        self.cog = CommutingObservableGroup(
            Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"]))
        )

    def test_cutting_reconstruction(self):
        with self.subTest("Test PauliList observable"):
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            weights = [(1.0, WeightType.EXACT)]
            observables = PauliList(["ZZ"])
            expvals = reconstruct_expectation_values(results, weights, observables)
            self.assertEqual([1.0], expvals)
        with self.subTest("Test mismatching inputs"):
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            weights = [(0.5, WeightType.EXACT), (0.5, WeightType.EXACT)]
            observables = {"A": PauliList(["Z"]), "B": PauliList(["Z"])}
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "If observables is a dictionary, results must also be a dictionary."
            )
            results2 = {"A": results}
            observables = PauliList(["ZZ"])
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results2, weights, observables)
            assert (
                e_info.value.args[0]
                == "If observables is a PauliList, results must be a SamplerResult or PrimitiveResult instance."
            )
        with self.subTest("Test unsupported phase"):
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            weights = [(0.5, WeightType.EXACT)]
            observables = PauliList(["iZZ"])
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "An input observable has a phase not equal to 1."
            )
            results = {"A": results}
            observables = {"A": observables}
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "An input observable has a phase not equal to 1."
            )
        with self.subTest("Test SamplerV2 result"):
            data_bin_cls = make_data_bin(
                [("observable_measurements", BitArray), ("qpd_measurements", BitArray)],
                shape=(),
            )
            obs_data = BitArray(
                np.array([0, 0, 0, 1, 0, 0, 2, 3, 1, 0], dtype=np.uint8).reshape(-1, 1),
                2,
            )
            qpd_data = BitArray(
                np.array([0, 1, 1, 3, 2, 0, 1, 3, 0, 2], dtype=np.uint8).reshape(-1, 1),
                2,
            )
            data_bin = data_bin_cls(
                observable_measurements=obs_data, qpd_measurements=qpd_data
            )
            pub_result = PubResult(data_bin)
            results = PrimitiveResult([pub_result])
            weights = [(1.0, WeightType.EXACT)]
            observables = PauliList(["II", "IZ", "ZI", "ZZ"])
            expvals = reconstruct_expectation_values(results, weights, observables)
            assert expvals == pytest.approx([0.0, -0.6, 0.0, -0.2])

    @data(
        ("000", [1, 1, 1]),
        ("001", [-1, 1, -1]),
        ("010", [1, -1, -1]),
        ("011", [-1, -1, 1]),
        ("100", [-1, -1, -1]),
        ("101", [1, -1, 1]),
        ("110", [-1, 1, 1]),
        ("111", [1, 1, -1]),
    )
    @unpack
    def test_process_outcome(self, outcome, expected):
        for o in (
            outcome,
            f"0b{outcome}",
            int(f"0b{outcome}", 0),
            hex(int(f"0b{outcome}", 0)),
        ):
            assert np.all(_process_outcome(self.cog, o) == expected)
