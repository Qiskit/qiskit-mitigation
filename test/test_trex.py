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

"""Tests for the ``TREX`` class."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, QubitSparsePauli, SparsePauliOp
from qiskit_mitigation.trex import TREX
from samplomatic.quantum_program import SamplexItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple non-parametric circuit."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _fake_trex_item_result(
    num_randomizations: int,
    shots: int,
    num_qubits: int,
    cal_data: np.ndarray | None = None,
    flips: np.ndarray | None = None,
) -> dict:
    """Return a minimal TREX calibration item-result dict.

    Shape of calibration data: (randomizations, shots, num_qubits).
    """
    if cal_data is None:
        cal_data = np.zeros((num_randomizations, shots, num_qubits), dtype=bool)
    if flips is None:
        flips = np.zeros((num_randomizations, shots, num_qubits), dtype=bool)
    return {
        "_trex_cal": cal_data,
        "measurement_flips._trex_cal": flips,
    }


def _mock_mitigation_task(num_qubits: int = 2) -> MagicMock:
    """Return a MagicMock that looks like a MitigationTask with a circuit."""
    task = MagicMock()
    task.circuit = _simple_circuit(num_qubits)
    return task


# ---------------------------------------------------------------------------
# TREX.__init__
# ---------------------------------------------------------------------------


class TestTREXInit(unittest.TestCase):
    """Tests for :meth:`TREX.__init__`."""

    def test_tasks_is_empty_list(self):
        """``tasks`` must be an empty list immediately after construction."""
        trex = TREX()
        self.assertEqual(trex.tasks, [])

    def test_noise_model_is_none(self):
        """``noise_model`` must be None immediately after construction."""
        trex = TREX()
        self.assertIsNone(trex.noise_model)

    def test_program_item_index_is_none(self):
        """``_program_item_index`` must be None immediately after construction."""
        trex = TREX()
        self.assertIsNone(trex._program_item_index)


# ---------------------------------------------------------------------------
# TREX._edit_boxing_options
# ---------------------------------------------------------------------------


class TestEditBoxingOptions(unittest.TestCase):
    """Tests for :meth:`TREX._edit_boxing_options`."""

    def setUp(self):
        self.trex = TREX()
        self.task = _mock_mitigation_task()

    def test_task_appended_to_tasks(self):
        """Calling ``_edit_boxing_options`` must append the task to ``self.tasks``."""
        self.trex._edit_boxing_options(self.task, None)
        self.assertIn(self.task, self.trex.tasks)

    def test_none_options_returns_defaults(self):
        """``None`` boxing_options must return default TREX options."""
        result = self.trex._edit_boxing_options(self.task, None)
        self.assertEqual(result, {"enable_measures": True, "measure_annotations": "all"})

    def test_missing_enable_measures_is_injected(self):
        """When ``enable_measures`` is absent, it is injected as ``True``."""
        options = {"some_key": "some_value"}
        result = self.trex._edit_boxing_options(self.task, options)
        self.assertTrue(result["enable_measures"])

    def test_missing_enable_measures_sets_measure_annotations_to_all(self):
        """When ``enable_measures`` is absent, ``measure_annotations`` is set to ``'all'``."""
        options = {}
        result = self.trex._edit_boxing_options(self.task, options)
        self.assertEqual(result["measure_annotations"], "all")

    def test_raises_when_enable_measures_is_false(self):
        """A ``False`` value for ``enable_measures`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.trex._edit_boxing_options(self.task, {"enable_measures": False})

    def test_measure_annotations_overridden_to_all(self):
        """Any non-``'all'`` value for ``measure_annotations`` is replaced with ``'all'``."""
        options = {"enable_measures": True, "measure_annotations": "change_basis"}
        result = self.trex._edit_boxing_options(self.task, options)
        self.assertEqual(result["measure_annotations"], "all")

    def test_valid_options_returned_unchanged(self):
        """Options already containing the correct values are returned as-is."""
        options = {"enable_measures": True, "measure_annotations": "all"}
        result = self.trex._edit_boxing_options(self.task, options)
        self.assertTrue(result["enable_measures"])
        self.assertEqual(result["measure_annotations"], "all")

    def test_multiple_tasks_accumulated(self):
        """Calling ``_edit_boxing_options`` multiple times appends all tasks."""
        task2 = _mock_mitigation_task()
        self.trex._edit_boxing_options(self.task, None)
        self.trex._edit_boxing_options(task2, None)
        self.assertEqual(len(self.trex.tasks), 2)
        self.assertIn(self.task, self.trex.tasks)
        self.assertIn(task2, self.trex.tasks)


# ---------------------------------------------------------------------------
# TREX.has_calibration_result
# ---------------------------------------------------------------------------


class TestHasCalibrationResult(unittest.TestCase):
    """Tests for :meth:`TREX.has_calibration_result`."""

    def test_false_before_prepare(self):
        """Must return ``False`` when ``_program_item_index`` is ``None``."""
        trex = TREX()
        self.assertFalse(trex.has_calibration_result())

    def test_true_after_index_is_set(self):
        """Must return ``True`` when ``_program_item_index`` is set to any integer."""
        trex = TREX()
        trex._program_item_index = 0
        self.assertTrue(trex.has_calibration_result())

    def test_true_for_nonzero_index(self):
        """Must return ``True`` for index > 0."""
        trex = TREX()
        trex._program_item_index = 5
        self.assertTrue(trex.has_calibration_result())


# ---------------------------------------------------------------------------
# TREX._prepare_calibration_circuit  (static method)
# ---------------------------------------------------------------------------


class TestPrepareCalibrationCircuit(unittest.TestCase):
    """Tests for :meth:`TREX._prepare_calibration_circuit`."""

    def test_returns_samplex_item(self):
        """Must return a ``SamplexItem``."""
        circuits = [_simple_circuit(2)]
        result = TREX._prepare_calibration_circuit(circuits, num_randomizations=10)
        self.assertIsInstance(result, SamplexItem)

    def test_samplex_item_has_correct_shape(self):
        """The ``SamplexItem`` shape must equal ``(num_randomizations,)``."""
        num_randomizations = 7
        circuits = [_simple_circuit(2)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=num_randomizations)
        self.assertEqual(item.shape, (num_randomizations,))

    def test_calibration_circuit_is_quantum_circuit(self):
        """The ``SamplexItem.circuit`` must be a ``QuantumCircuit``."""
        circuits = [_simple_circuit(2)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5)
        self.assertIsInstance(item.circuit, QuantumCircuit)

    def test_calibration_circuit_has_trex_cal_register(self):
        """The calibration circuit must include a classical register named ``'_trex_cal'``."""
        circuits = [_simple_circuit(2)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5)
        reg_names = [r.name for r in item.circuit.cregs]
        self.assertIn("_trex_cal", reg_names)

    def test_num_qubits_is_max_of_input_circuits(self):
        """Calibration circuit must span the largest qubit count across all input circuits."""
        circuits = [_simple_circuit(2), _simple_circuit(4)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5)
        self.assertEqual(item.circuit.num_qubits, 4)

    def test_single_qubit_circuit(self):
        """Single-qubit circuit must produce a valid 1-qubit calibration circuit."""
        circuits = [QuantumCircuit(1)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=3)
        self.assertIsInstance(item, SamplexItem)
        self.assertEqual(item.circuit.num_qubits, 1)


# ---------------------------------------------------------------------------
# TREX.prepare
# ---------------------------------------------------------------------------


class TestPrepare(unittest.TestCase):
    """Tests for :meth:`TREX.prepare`."""

    def test_raises_when_no_tasks(self):
        """``prepare`` must raise ``ValueError`` when no tasks have been registered."""
        trex = TREX()
        mock_qp = MagicMock()
        mock_qp.items = []
        with self.assertRaises(ValueError):
            trex.prepare(num_randomizations=10, quantum_program=mock_qp)

    def test_appends_calibration_item_to_quantum_program(self):
        """``prepare`` must append exactly one calibration item to ``quantum_program.items``."""
        trex = TREX()
        task = _mock_mitigation_task()
        trex.tasks.append(task)
        mock_qp = MagicMock()
        mock_qp.items = []
        trex.prepare(num_randomizations=5, quantum_program=mock_qp)
        self.assertEqual(len(mock_qp.items), 1)

    def test_program_item_index_is_set(self):
        """``_program_item_index`` must be set after ``prepare`` is called."""
        trex = TREX()
        task = _mock_mitigation_task()
        trex.tasks.append(task)
        mock_qp = MagicMock()
        mock_qp.items = []
        trex.prepare(num_randomizations=5, quantum_program=mock_qp)
        self.assertIsNotNone(trex._program_item_index)

    def test_program_item_index_equals_original_list_length(self):
        """``_program_item_index`` must equal the length of ``quantum_program.items`` before the call."""
        trex = TREX()
        task = _mock_mitigation_task()
        trex.tasks.append(task)
        mock_qp = MagicMock()
        mock_qp.items = [MagicMock(), MagicMock()]  # two existing items
        trex.prepare(num_randomizations=5, quantum_program=mock_qp)
        self.assertEqual(trex._program_item_index, 2)

    def test_has_calibration_result_true_after_prepare(self):
        """``has_calibration_result`` must return ``True`` after ``prepare``."""
        trex = TREX()
        task = _mock_mitigation_task()
        trex.tasks.append(task)
        mock_qp = MagicMock()
        mock_qp.items = []
        trex.prepare(num_randomizations=5, quantum_program=mock_qp)
        self.assertTrue(trex.has_calibration_result())

    def test_returns_the_given_quantum_program(self):
        """``prepare`` must return the ``quantum_program`` instance it was given.

        Regression test: it used to return the ``QuantumProgram`` class object.
        """
        trex = TREX()
        task = _mock_mitigation_task()
        trex.tasks.append(task)
        mock_qp = MagicMock()
        mock_qp.items = []
        returned = trex.prepare(num_randomizations=5, quantum_program=mock_qp)
        self.assertIs(returned, mock_qp)


# ---------------------------------------------------------------------------
# TREX.compute_noise_model
# ---------------------------------------------------------------------------


class TestComputeNoiseModel(unittest.TestCase):
    """Tests for :meth:`TREX.compute_noise_model`."""

    def setUp(self):
        self.trex = TREX()
        self.trex._program_item_index = 0

    def test_raises_when_trex_cal_key_missing(self):
        """Must raise ``ValueError`` when ``'_trex_cal'`` key is absent from results."""
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(return_value={"wrong_key": np.zeros((1, 4, 2))})
        with self.assertRaises(ValueError):
            self.trex.compute_noise_model(mock_results)

    def test_returns_pauli_lindblad_map(self):
        """Must return a ``PauliLindbladMap``."""
        cal_data = np.zeros((2, 4, 2), dtype=bool)
        flips = np.zeros((2, 4, 2), dtype=bool)
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(
            return_value=_fake_trex_item_result(2, 4, 2, cal_data, flips)
        )
        result = self.trex.compute_noise_model(mock_results)
        self.assertIsInstance(result, PauliLindbladMap)

    def test_noise_model_stored_on_instance(self):
        """The returned ``PauliLindbladMap`` must also be stored as ``self.noise_model``."""
        cal_data = np.zeros((2, 4, 2), dtype=bool)
        flips = np.zeros((2, 4, 2), dtype=bool)
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(
            return_value=_fake_trex_item_result(2, 4, 2, cal_data, flips)
        )
        noise_model = self.trex.compute_noise_model(mock_results)
        self.assertIs(self.trex.noise_model, noise_model)

    def test_all_zeros_gives_zero_flip_rate(self):
        """All-zero calibration data with zero flips → zero flip rates for all qubits."""
        num_qubits = 2
        cal_data = np.zeros((4, 8, num_qubits), dtype=bool)
        flips = np.zeros((4, 8, num_qubits), dtype=bool)
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(
            return_value=_fake_trex_item_result(4, 8, num_qubits, cal_data, flips)
        )
        noise_model = self.trex.compute_noise_model(mock_results)
        self.assertIsInstance(noise_model, PauliLindbladMap)
        self.assertEqual(noise_model.num_qubits, num_qubits)


# ---------------------------------------------------------------------------
# TREX.calculate_trex_factor  (static method)
# ---------------------------------------------------------------------------


class TestCalculateTrexFactor(unittest.TestCase):
    """Tests for :meth:`TREX.calculate_trex_factor`."""

    def test_identity_noise_gives_factor_one(self):
        """An identity (zero-rate) noise model must produce a TREX factor of 1.0."""
        noise_data = PauliLindbladMap.identity(2)
        factor = TREX.calculate_trex_factor(noise_data, Pauli("ZZ"))
        self.assertIsInstance(factor, float)
        self.assertAlmostEqual(factor, 1.0)

    def test_identity_observable_has_unit_factor_input_noise_model(self):
        """Test identity observable produces a factor of one."""
        error_prop = 0.3
        error_rate = -0.5 * np.log(1.0 - 2.0 * error_prop)
        noise_model = PauliLindbladMap.from_sparse_list([("X", [0], error_rate)], num_qubits=1)

        result = TREX.calculate_trex_factor(noise_model, "I")

        self.assertEqual(result, 1.0)

    def test_numpy_array_input_all_zeros(self):
        """With all-zero calibration data array, ⟨Z⟩ = 1 → TREX factor = 1."""
        # Shape (randomizations, shots, num_qubits) = (2, 8, 2)
        cal_data = np.zeros((2, 8, 2), dtype=bool)
        factor = TREX.calculate_trex_factor(cal_data, Pauli("ZI"))
        self.assertAlmostEqual(factor, 1.0)

    def test_numpy_array_input_all_ones(self):
        """With all-ones calibration data (all qubits excited), eigenvalue = -1 → factor = -1."""
        cal_data = np.ones((2, 8, 2), dtype=bool)
        factor = TREX.calculate_trex_factor(cal_data, Pauli("ZI"))
        self.assertAlmostEqual(factor, -1.0)

    def test_string_observable_accepted(self):
        """A string observable label must be accepted without raising."""
        noise_data = PauliLindbladMap.identity(2)
        factor = TREX.calculate_trex_factor(noise_data, "ZZ")
        self.assertIsInstance(factor, float)

    def test_qubit_sparse_pauli_observable_accepted(self):
        """A ``QubitSparsePauli`` observable must be accepted without conversion."""
        noise_data = PauliLindbladMap.identity(2)
        factor = TREX.calculate_trex_factor(noise_data, QubitSparsePauli("ZZ"))
        self.assertAlmostEqual(factor, 1.0)

    def test_calculates_factor_based_on_noise_model_for_string_observable(self):
        """Test TREX factor is inverse fidelity of the observable support."""
        error_prop = 0.1
        error_rate = -0.5 * np.log(1.0 - 2.0 * error_prop)
        noise_model = PauliLindbladMap.from_sparse_list([("X", [0], error_rate)], num_qubits=2)

        result = TREX.calculate_trex_factor(noise_model, "IZ")
        # in single qubit case, the result should be 1 / (1 - 2*error_prop)
        self.assertEqual(result, 1 / (1 - 2 * error_prop))

    def test_calculates_factor_based_on_noise_model_for_pauli_observable(self):
        """Test non-Z Paulis are converted to Z support on the same qubits."""
        error_prop = 0.1
        error_rate = -0.5 * np.log(1.0 - 2.0 * error_prop)
        error_prop2 = 0.2
        error_rate2 = -0.5 * np.log(1.0 - 2.0 * error_prop2)
        noise_model = PauliLindbladMap.from_sparse_list(
            [("X", [0], error_rate), ("X", [1], error_rate2)], num_qubits=2
        )

        result = TREX.calculate_trex_factor(noise_model, Pauli("XY"))
        # in the two qubit case, the result should be the fidelity minus the errors cause by each
        # qubit, plus the errors caused by even number of qubits - which is both qubits
        self.assertEqual(
            result, 1 / (1 - (2 * error_prop + 2 * error_prop2 - 2 * error_prop * 2 * error_prop2))
        )

    def test_calculates_factor_based_on_calibration_circ_for_string_observable(self):
        """Test TREX factor is inverse fidelity of the observable support."""
        cal_data_flipped = np.zeros((10, 100, 2), dtype=bool)
        for rand in range(10):
            cal_data_flipped[rand, rand * 10 : (rand + 1) * 10, :] = True

        result = TREX.calculate_trex_factor(cal_data_flipped, "IZ")

        # in single qubit case, the result should be 1 / (1 - 2*error_prop)
        self.assertEqual(result, 1 / (1 - 2 * 0.1))

    def test_calculates_factor_based_on_calibration_circ_for_pauli_observable(self):
        """Test non-Z Paulis are converted to Z support on the same qubits."""
        cal_data_flipped = np.zeros((10, 100, 2), dtype=bool)
        for rand in range(10):
            cal_data_flipped[rand, rand * 10 : (rand + 1) * 10, 0] = True
        # flip the second qubit with no overlap with the first one
        cal_data_flipped[0, 10:30, 1] = True
        for rand in range(1, 8):
            cal_data_flipped[rand, rand * 10 + 10 : (rand + 1) * 10 + 20, 1] = True
        cal_data_flipped[8, 90:100, 1] = True
        cal_data_flipped[8, 0:10, 1] = True
        cal_data_flipped[9, :20, 1] = True

        result = TREX.calculate_trex_factor(cal_data_flipped, Pauli("XY"))
        # in the two qubit case, the result should be the fidelity minus the errors cause by each
        # qubit, plus the errors caused by randomization with even number of qubit flips
        # the flips have no overlap so the result should be 1 / (1 - (2*error_prop1 + 2*error_prop2)
        self.assertAlmostEqual(result, 1 / (1 - (2 * 0.1 + 2 * 0.2)))


# ---------------------------------------------------------------------------
# TREX.trex_factors_each_term  (static method)
# ---------------------------------------------------------------------------


class TestTrexFactorsEachTerm(unittest.TestCase):
    """Tests for :meth:`TREX.trex_factors_each_term`."""

    def _identity_noise(self, num_qubits: int) -> PauliLindbladMap:
        return PauliLindbladMap.identity(num_qubits)

    def test_returns_dict(self):
        """Must return a dictionary."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        self.assertIsInstance(result, dict)

    def test_dict_keys_are_strings(self):
        """All keys in the returned dict must be strings."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        for key in result:
            self.assertIsInstance(key, str)

    def test_dict_values_are_floats(self):
        """All values in the returned dict must be floats."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        for value in result.values():
            self.assertIsInstance(value, float)

    def test_identity_noise_all_factors_are_one(self):
        """With identity noise, all TREX factors must equal 1.0."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        for value in result.values():
            self.assertAlmostEqual(value, 1.0)

    def test_accepts_observables_array(self):
        """Must accept an ``ObservablesArray`` as observables."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        self.assertIsInstance(result, dict)

    def test_multiple_terms_in_observable(self):
        """An observable with multiple terms (ZZ + ZI) produces entries for each term."""
        noise = self._identity_noise(2)
        obs = ObservablesArray([SparsePauliOp(["ZZ", "ZI"], [1.0, 1.0])])
        result = TREX.trex_factors_each_term(noise, obs)
        self.assertGreater(len(result), 0)

    def test_factors_kept_for_all_observables(self):
        """Terms of every observable must appear in the result, not just the last one's.

        Regression test: the factors dict used to be overwritten per observable, so only
        the final observable's terms survived.
        """
        noise = PauliLindbladMap.from_sparse_list(
            [("X", [0], 0.05), ("X", [1], 0.07)], num_qubits=2
        )
        obs = ObservablesArray([SparsePauliOp("ZI"), SparsePauliOp("IZ")])
        result = TREX.trex_factors_each_term(noise, obs)
        self.assertEqual(sorted(result), ["IZ", "ZI"])
        self.assertAlmostEqual(result["ZI"], TREX.calculate_trex_factor(noise, "ZI"))
        self.assertAlmostEqual(result["IZ"], TREX.calculate_trex_factor(noise, "IZ"))

    def test_accepts_sequence_of_sparse_pauli_ops(self):
        """A plain sequence of ``SparsePauliOp`` must give the same result as an array.

        Regression test: iterating a raw ``SparsePauliOp`` used to raise ``TypeError``.
        """
        noise = PauliLindbladMap.from_sparse_list(
            [("X", [0], 0.05), ("X", [1], 0.07)], num_qubits=2
        )
        observables = [SparsePauliOp("ZI"), SparsePauliOp("IZ")]
        from_sequence = TREX.trex_factors_each_term(noise, observables)
        from_array = TREX.trex_factors_each_term(noise, ObservablesArray(observables))
        self.assertEqual(from_sequence, from_array)
        self.assertEqual(sorted(from_sequence), ["IZ", "ZI"])


if __name__ == "__main__":
    unittest.main()
