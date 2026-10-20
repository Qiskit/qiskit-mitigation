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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from qiskit_mitigation.utils.meas_qubits import (
    derive_meas_qubits,
    expand_meas_data,
    expand_meas_data_and_flips,
    get_active_qubits,
    get_observable_qubit_indices,
)


class TestGetActiveQubits(unittest.TestCase):
    """Tests for :func:`get_active_qubits`."""

    def test_gates_make_qubits_active(self):
        qc = QuantumCircuit(5)
        qc.x(0)
        qc.cx(3, 4)
        self.assertEqual(get_active_qubits(qc), [0, 3, 4])

    def test_empty_circuit_has_no_active_qubits(self):
        self.assertEqual(get_active_qubits(QuantumCircuit(3)), [])

    def test_barrier_and_delay_do_not_activate(self):
        qc = QuantumCircuit(3)
        qc.barrier()
        qc.delay(100, 1)
        qc.x(2)
        self.assertEqual(get_active_qubits(qc), [2])

    def test_final_measurements_do_not_activate(self):
        qc = QuantumCircuit(3, 3)
        qc.x(0)
        qc.measure([0, 1, 2], [0, 1, 2])
        self.assertEqual(get_active_qubits(qc), [0])

    def test_input_circuit_is_not_modified(self):
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        get_active_qubits(qc)
        self.assertEqual(qc.count_ops().get("measure"), 2)


class TestGetObservableQubitIndices(unittest.TestCase):
    """Tests for :func:`get_observable_qubit_indices`."""

    def test_union_over_all_observables_and_terms(self):
        observables = [SparsePauliOp(["IIIZ", "IXII"]), SparsePauliOp("ZIII")]
        self.assertEqual(get_observable_qubit_indices(observables), [0, 2, 3])

    def test_identity_has_no_support(self):
        self.assertEqual(get_observable_qubit_indices([SparsePauliOp("III")]), [])

    def test_accepts_observables_array(self):
        observables = ObservablesArray.coerce([SparsePauliOp("IZI")])
        self.assertEqual(get_observable_qubit_indices(observables), [1])


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

    def test_twirling_strategy_all_raises(self):
        with self.assertRaisesRegex(ValueError, "twirling_strategy"):
            derive_meas_qubits(
                self.circuit, [SparsePauliOp("IIIIIZ")], {"twirling_strategy": "all"}
            )

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
    """Tests for :func:`expand_meas_data` and :func:`expand_meas_data_and_flips`."""

    def test_columns_land_on_their_circuit_qubit(self):
        data = np.array([[[True, False, True]]])  # columns of qubits 1, 3 and 4
        expanded = expand_meas_data(data, [1, 3, 4], 6)
        np.testing.assert_array_equal(expanded, [[[False, True, False, False, True, False]]])

    def test_dtype_and_leading_axes_are_kept(self):
        data = np.ones((2, 3, 4, 2), dtype=bool)
        expanded = expand_meas_data(data, [0, 5], 6)
        self.assertEqual(expanded.shape, (2, 3, 4, 6))
        self.assertEqual(expanded.dtype, bool)

    def test_none_returns_the_input_unchanged(self):
        data = np.ones((1, 1, 2, 3), dtype=bool)
        self.assertIs(expand_meas_data(data, None, None), data)

    def test_only_one_argument_raises(self):
        data = np.ones((1, 2), dtype=bool)
        with self.assertRaises(ValueError):
            expand_meas_data(data, [0, 1], None)
        with self.assertRaises(ValueError):
            expand_meas_data(data, None, 4)

    def test_data_and_flips_are_both_expanded(self):
        data = np.ones((1, 2), dtype=bool)
        flips = np.zeros((1, 2), dtype=bool)
        new_data, new_flips = expand_meas_data_and_flips(data, flips, [1, 2], 4)
        np.testing.assert_array_equal(new_data, [[False, True, True, False]])
        np.testing.assert_array_equal(new_flips, [[False, False, False, False]])

    def test_missing_flips_stay_none(self):
        _, flips = expand_meas_data_and_flips(np.ones((1, 2), dtype=bool), None, [0, 1], 4)
        self.assertIsNone(flips)


if __name__ == "__main__":
    unittest.main()
