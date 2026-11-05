# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for :mod:`qiskit_mitigation.utils.meas_qubits`."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from qiskit_mitigation.utils.meas_qubits import (
    _contiguous_runs,
    _expand_meas_data,
    _expand_meas_data_and_apply_flips,
    _get_active_qubits,
    _get_observable_qubit_indices,
    derive_meas_qubits,
)


class TestGetActiveQubits(unittest.TestCase):
    """Tests for :func:`_get_active_qubits`."""

    def test_gates_make_qubits_active(self):
        qc = QuantumCircuit(5)
        qc.x(0)
        qc.cx(3, 4)
        self.assertEqual(_get_active_qubits(qc), [0, 3, 4])

    def test_empty_circuit_has_no_active_qubits(self):
        self.assertEqual(_get_active_qubits(QuantumCircuit(3)), [])

    def test_barrier_and_delay_do_not_activate(self):
        qc = QuantumCircuit(3)
        qc.barrier()
        qc.delay(100, 1)
        qc.x(2)
        self.assertEqual(_get_active_qubits(qc), [2])

    def test_final_measurements_do_not_activate(self):
        qc = QuantumCircuit(3, 3)
        qc.x(0)
        qc.measure([0, 1, 2], [0, 1, 2])
        stripped = qc.remove_final_measurements(inplace=False)
        self.assertEqual(_get_active_qubits(stripped), [0])

    def test_mid_circuit_measurements_make_qubits_active(self):
        qc = QuantumCircuit(3, 3)
        qc.measure(1, 1)
        qc.x(1)
        self.assertEqual(_get_active_qubits(qc), [1])

    def test_input_circuit_is_not_modified(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        _get_active_qubits(qc)
        self.assertEqual(qc.count_ops().get("x"), 1)


class TestGetObservableQubitIndices(unittest.TestCase):
    """Tests for :func:`_get_observable_qubit_indices`."""

    def test_union_over_all_observables_and_terms(self):
        observables = [SparsePauliOp(["IIIZ", "IXII"]), SparsePauliOp("ZIII")]
        self.assertEqual(_get_observable_qubit_indices(observables), [0, 2, 3])

    def test_identity_has_no_support(self):
        self.assertEqual(_get_observable_qubit_indices([SparsePauliOp("III")]), [])

    def test_accepts_observables_array(self):
        observables = ObservablesArray.coerce([SparsePauliOp("IZI")])
        self.assertEqual(_get_observable_qubit_indices(observables), [1])


class TestDeriveMeasQubits(unittest.TestCase):
    """Tests for :func:`derive_meas_qubits`."""

    def setUp(self):
        self.circuit = QuantumCircuit(6)
        self.circuit.x(0)
        self.circuit.cx(3, 4)

    # Pauli labels are little-endian: the last character acts on qubit 0.
    def test_union_of_active_and_observed(self):
        self.assertEqual(derive_meas_qubits(self.circuit, [SparsePauliOp("ZIIIII")]), [0, 3, 4, 5])

    def test_observable_inside_the_active_set_adds_nothing(self):
        self.assertEqual(derive_meas_qubits(self.circuit, [SparsePauliOp("IIIIIZ")]), [0, 3, 4])

    def test_idle_observed_qubit_is_included(self):
        self.assertEqual(derive_meas_qubits(self.circuit, [SparsePauliOp("IIIIZI")]), [0, 1, 3, 4])

    def test_result_is_sorted_and_unique(self):
        result = derive_meas_qubits(self.circuit, [SparsePauliOp(["ZIIIII", "IIIIIZ"])])
        self.assertEqual(result, sorted(set(result)))

    def test_empty_set_raises(self):
        with self.assertRaisesRegex(ValueError, "no qubit is active or observed"):
            derive_meas_qubits(QuantumCircuit(3), [SparsePauliOp("III")])

    def test_twirling_strategy_all_warns(self):
        with self.assertWarns(UserWarning):
            derive_meas_qubits(
                self.circuit, [SparsePauliOp("IIIIIZ")], {"twirling_strategy": "all"}
            )

    def test_twirling_strategy_all_warn_message(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            derive_meas_qubits(
                self.circuit, [SparsePauliOp("IIIIIZ")], {"twirling_strategy": "all"}
            )
        self.assertIn("twirling_strategy", str(w[0].message))

    def test_other_boxing_options_are_accepted(self):
        options = {"twirling_strategy": "active", "inject_noise_targets": "all"}
        self.assertEqual(
            derive_meas_qubits(self.circuit, [SparsePauliOp("IIIIIZ")], options), [0, 3, 4]
        )

    def test_none_boxing_options_are_accepted(self):
        self.assertEqual(
            derive_meas_qubits(self.circuit, [SparsePauliOp("IIIIIZ")], None), [0, 3, 4]
        )


class TestExpandMeasData(unittest.TestCase):
    """Tests for :func:`_expand_meas_data` and :func:`_expand_meas_data_and_flips`."""

    def test_columns_land_on_their_circuit_qubit(self):
        data = np.array([[[True, False, True]]])  # columns of qubits 1, 3 and 4
        expanded = _expand_meas_data(data, [1, 3, 4], 6)
        np.testing.assert_array_equal(expanded, [[[False, True, False, False, True, False]]])

    def test_dtype_and_leading_axes_are_kept(self):
        data = np.ones((2, 3, 4, 2), dtype=bool)
        expanded = _expand_meas_data(data, [0, 5], 6)
        self.assertEqual(expanded.shape, (2, 3, 4, 6))
        self.assertEqual(expanded.dtype, bool)

    def test_none_returns_the_input_unchanged(self):
        data = np.ones((1, 1, 2, 3), dtype=bool)
        self.assertIs(_expand_meas_data(data, None, None), data)

    def test_only_one_argument_raises(self):
        data = np.ones((1, 2), dtype=bool)
        with self.assertRaises(ValueError):
            _expand_meas_data(data, [0, 1], None)
        with self.assertRaises(ValueError):
            _expand_meas_data(data, None, 4)

    def test_matches_fancy_index_reference(self):
        """Both the slice path and the many-runs path agree with plain fancy-index assignment."""
        rng = np.random.default_rng(1)
        num_qubits = 60
        layouts = {
            "single qubit": [7],
            "first qubit": [0],
            "last qubit": [num_qubits - 1],
            "one run": list(range(10, 40)),
            "all qubits": list(range(num_qubits)),
            "few runs": [0, 1, 2, 10, 11, 30, 45, 46],
            "many runs": list(range(0, num_qubits, 2)),
        }
        for name, meas_qubits in layouts.items():
            for shape in [(5, len(meas_qubits)), (2, 3, 4, len(meas_qubits))]:
                with self.subTest(layout=name, shape=shape):
                    data = rng.integers(0, 2, shape).astype(bool)
                    reference = np.zeros((*shape[:-1], num_qubits), dtype=bool)
                    reference[..., meas_qubits] = data
                    expanded = _expand_meas_data(data, meas_qubits, num_qubits)
                    np.testing.assert_array_equal(expanded, reference)
                    self.assertTrue(expanded.flags.c_contiguous)

    def test_non_contiguous_input_is_expanded(self):
        base = np.arange(24).reshape(2, 3, 4)[..., ::2]  # non-contiguous, last axis of length 2
        expanded = _expand_meas_data(base, [1, 4], 6)
        np.testing.assert_array_equal(expanded[..., [1, 4]], base)
        np.testing.assert_array_equal(expanded[..., [0, 2, 3, 5]], 0)

    def test_no_measured_qubits_gives_zeros(self):
        expanded = _expand_meas_data(np.ones((2, 0), dtype=bool), [], 3)
        np.testing.assert_array_equal(expanded, np.zeros((2, 3), dtype=bool))

    def test_contiguous_runs(self):
        self.assertEqual(_contiguous_runs([3]), [(3, 4)])
        self.assertEqual(_contiguous_runs([0, 1, 2]), [(0, 3)])
        self.assertEqual(_contiguous_runs([0, 1, 4, 6, 7]), [(0, 2), (4, 5), (6, 8)])

    def test_flips_are_applied_before_expanding(self):
        data = np.array([[True, True]])
        flips = np.array([[True, False]])
        expanded = _expand_meas_data_and_apply_flips(data, flips, [1, 2], 4)
        np.testing.assert_array_equal(expanded, [[False, False, True, False]])

    def test_flips_equal_expanding_both_and_xoring(self):
        rng = np.random.default_rng(2)
        meas_qubits = [0, 1, 5, 6, 7, 20]
        data = rng.integers(0, 2, (2, 3, 8, 6)).astype(bool)
        flips = rng.integers(0, 2, data.shape).astype(bool)
        expected = _expand_meas_data(data, meas_qubits, 25) ^ _expand_meas_data(
            flips, meas_qubits, 25
        )
        result = _expand_meas_data_and_apply_flips(data.copy(), flips, meas_qubits, 25)
        np.testing.assert_array_equal(result, expected)

    def test_missing_flips_only_expand(self):
        expanded = _expand_meas_data_and_apply_flips(np.ones((1, 2), dtype=bool), None, [0, 1], 4)
        np.testing.assert_array_equal(expanded, [[True, True, False, False]])

    def test_flips_without_trimming_are_applied_in_place(self):
        data = np.array([[True, False]])
        result = _expand_meas_data_and_apply_flips(data, np.array([[True, True]]), None, None)
        self.assertIs(result, data)
        np.testing.assert_array_equal(result, [[False, True]])

    def test_flips_helper_only_one_argument_raises(self):
        data = np.ones((1, 2), dtype=bool)
        with self.assertRaises(ValueError):
            _expand_meas_data_and_apply_flips(data, None, [0, 1], None)
        with self.assertRaises(ValueError):
            _expand_meas_data_and_apply_flips(data, None, None, 4)


if __name__ == "__main__":
    unittest.main()
