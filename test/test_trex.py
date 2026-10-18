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
import warnings
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, QubitSparsePauli, SparsePauliOp
from qiskit_mitigation.mitigation_task import MitigationTask
from qiskit_mitigation.pec import PEC
from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.utils import find_combined_unique_layers, load_tasks_from_result
from qiskit_mitigation.zne.pea import PEA
from samplomatic import InjectNoise
from samplomatic.quantum_program import QuantumProgram, QuantumProgramResult, SamplexItem
from samplomatic.utils import get_annotation

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
        "_meas": cal_data,
        "measurement_flips._meas": flips,
    }


def _mock_mitigation_task(num_qubits: int = 2, meas_qubits: list[int] | None = None) -> MagicMock:
    """Return a MagicMock that looks like a MitigationTask with a circuit.

    ``meas_qubits`` defaults to all the qubits of the circuit, like in a prepared task.
    """
    task = MagicMock()
    task.circuit = _simple_circuit(num_qubits)
    task.meas_qubits = list(range(num_qubits)) if meas_qubits is None else meas_qubits
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

    def test_task_is_not_registered(self):
        """Editing the boxing options must not register a task, registration is a separate step."""
        self.trex._edit_boxing_options(None)
        self.assertEqual(self.trex.tasks, [])

    def test_none_options_returns_defaults(self):
        """``None`` boxing_options must return default TREX options."""
        result = self.trex._edit_boxing_options(None)
        self.assertEqual(result, {"enable_measures": True, "measure_annotations": "all"})

    def test_missing_enable_measures_is_injected(self):
        """When ``enable_measures`` is absent, it is injected as ``True``."""
        options = {"some_key": "some_value"}
        result = self.trex._edit_boxing_options(options)
        self.assertTrue(result["enable_measures"])

    def test_missing_enable_measures_sets_measure_annotations_to_all(self):
        """When ``enable_measures`` is absent, ``measure_annotations`` is set to ``'all'``."""
        options = {}
        result = self.trex._edit_boxing_options(options)
        self.assertEqual(result["measure_annotations"], "all")

    def test_raises_when_enable_measures_is_false(self):
        """A ``False`` value for ``enable_measures`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.trex._edit_boxing_options({"enable_measures": False})

    def test_measure_annotations_overridden_to_all(self):
        """Any non-``'all'`` value for ``measure_annotations`` is replaced with ``'all'``."""
        options = {"enable_measures": True, "measure_annotations": "change_basis"}
        result = self.trex._edit_boxing_options(options)
        self.assertEqual(result["measure_annotations"], "all")

    def test_valid_options_returned_unchanged(self):
        """Options already containing the correct values are returned as-is."""
        options = {"enable_measures": True, "measure_annotations": "all"}
        result = self.trex._edit_boxing_options(options)
        self.assertTrue(result["enable_measures"])
        self.assertEqual(result["measure_annotations"], "all")


class TestRegisterTask(unittest.TestCase):
    """Tests for :meth:`TREX._register_task`."""

    def test_task_appended_to_tasks(self):
        trex = TREX()
        task = _mock_mitigation_task()
        trex._register_task(task)
        self.assertEqual(trex.tasks, [task])

    def test_multiple_tasks_accumulated(self):
        trex = TREX()
        task1, task2 = _mock_mitigation_task(), _mock_mitigation_task()
        trex._register_task(task1)
        trex._register_task(task2)
        self.assertEqual(trex.tasks, [task1, task2])


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

    def test_calibration_circuit_has_meas_register(self):
        """The calibration circuit must include a classical register named ``'_meas'``."""
        circuits = [_simple_circuit(2)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5)
        reg_names = [r.name for r in item.circuit.cregs]
        self.assertIn("_meas", reg_names)

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

    def test_meas_register_spans_all_qubits_by_default(self):
        """Without ``meas_qubits`` the ``_meas`` register has a bit per qubit of the widest circuit."""
        circuits = [_simple_circuit(2), _simple_circuit(4)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5)
        meas_register = next(r for r in item.circuit.cregs if r.name == "_meas")
        self.assertEqual(len(meas_register), 4)

    def test_meas_register_spans_only_meas_qubits(self):
        """With ``meas_qubits`` only these qubits are measured, in a register of matching width."""
        circuits = [_simple_circuit(5)]
        item = TREX._prepare_calibration_circuit(circuits, num_randomizations=5, meas_qubits=[1, 3])
        meas_register = next(r for r in item.circuit.cregs if r.name == "_meas")
        self.assertEqual(len(meas_register), 2)
        # the circuit keeps its full width, so the qubit indices stay those of the mitigated circuits
        self.assertEqual(item.circuit.num_qubits, 5)
        measured = {
            item.circuit.find_bit(qubit).index
            for instruction in item.circuit.data
            for qubit in instruction.qubits
        }
        self.assertEqual(measured, {1, 3})

    def test_meas_qubits_order_matches_register_bits(self):
        """Register bit ``i`` holds the measurement of ``meas_qubits[i]``."""
        circuits = [_simple_circuit(4)]
        item = TREX._prepare_calibration_circuit(
            circuits, num_randomizations=5, meas_qubits=[0, 2, 3]
        )
        pairs = [
            (item.circuit.find_bit(qubit).index, item.circuit.find_bit(clbit).index)
            for instruction in item.circuit.data
            if instruction.operation.name == "measure"
            for qubit, clbit in zip(instruction.qubits, instruction.clbits, strict=True)
        ]
        self.assertEqual(sorted(pairs), [(0, 0), (2, 1), (3, 2)])


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

    def test_meas_qubits_is_union_of_the_tasks_meas_qubits(self):
        """The calibration measures the sorted union of the ``meas_qubits`` of all the tasks."""
        trex = TREX()
        trex.tasks.append(_mock_mitigation_task(num_qubits=3, meas_qubits=[0, 2]))
        trex.tasks.append(_mock_mitigation_task(num_qubits=5, meas_qubits=[2, 4]))
        qp = QuantumProgram(shots=10)
        trex.prepare(num_randomizations=2, quantum_program=qp)
        self.assertEqual(trex.meas_qubits, [0, 2, 4])
        self.assertEqual(trex.circuit_num_qubits, 5)

    def test_calibration_circuit_measures_only_the_union(self):
        """The calibration item has a ``_meas`` register of the size of the union."""
        trex = TREX()
        trex.tasks.append(_mock_mitigation_task(num_qubits=3, meas_qubits=[0, 2]))
        trex.tasks.append(_mock_mitigation_task(num_qubits=5, meas_qubits=[2, 4]))
        qp = QuantumProgram(shots=10)
        trex.prepare(num_randomizations=2, quantum_program=qp)
        meas_register = next(r for r in qp.items[-1].circuit.cregs if r.name == "_meas")
        self.assertEqual(len(meas_register), 3)

    def test_task_without_meas_qubits_falls_back_to_all_its_qubits(self):
        """A task whose ``meas_qubits`` is ``None`` contributes every qubit of its circuit."""
        trex = TREX()
        task = _mock_mitigation_task(num_qubits=3)
        task.meas_qubits = None
        trex.tasks.append(task)
        qp = QuantumProgram(shots=10)
        trex.prepare(num_randomizations=2, quantum_program=qp)
        self.assertEqual(trex.meas_qubits, [0, 1, 2])

    def test_passthrough_data_contains_meas_qubits(self):
        """The passthrough data stores what is needed to expand the calibration data."""
        trex = TREX()
        trex.tasks.append(_mock_mitigation_task(num_qubits=4, meas_qubits=[1, 3]))
        qp = QuantumProgram(shots=10)
        trex.prepare(num_randomizations=2, quantum_program=qp)
        passthrough = qp.passthrough_data["qiskit_mitigation"][-1]
        self.assertEqual(passthrough["mitigation"], "trex")
        self.assertEqual(passthrough["meas_qubits"], [1, 3])
        self.assertEqual(passthrough["circuit_num_qubits"], 4)


# ---------------------------------------------------------------------------
# TREX.compute_noise_model
# ---------------------------------------------------------------------------


class TestComputeNoiseModel(unittest.TestCase):
    """Tests for :meth:`TREX.compute_noise_model`."""

    def setUp(self):
        self.trex = TREX()
        self.trex._program_item_index = 0

    def test_raises_when_meas_key_missing(self):
        """Must raise ``ValueError`` when ``'_meas'`` key is absent from results."""
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

    def test_deprecated_trex_cal_register_emits_warning_and_returns_noise_model(self):
        """When ``'_trex_cal'`` is present, a ``DeprecationWarning`` must be emitted and
        the noise model must still be computed correctly from ``'_trex_cal'`` data."""
        num_qubits = 2
        cal_data = np.zeros((2, 4, num_qubits), dtype=bool)
        flips = np.zeros((2, 4, num_qubits), dtype=bool)
        item_result = {
            "_trex_cal": cal_data,
            "measurement_flips._trex_cal": flips,
        }
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(return_value=item_result)

        with self.assertWarns(DeprecationWarning):
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


class TestTrexFactorsEachTermPartialMeasurement(unittest.TestCase):
    """Tests for :meth:`TREX.trex_factors_each_term` with calibration arrays of partial measurements."""

    @staticmethod
    def _compressed_calibration_data() -> np.ndarray:
        """Calibration data of a 3-qubit circuit that measured qubits 1 and 2 only.

        Column 0 is qubit 1 and flips in 10% of the shots, column 1 is qubit 2 and flips in 20% of the shots.
        """
        cal_data = np.zeros((10, 100, 2), dtype=bool)
        cal_data[:, :10, 0] = True
        cal_data[:, :20, 1] = True
        return cal_data

    def test_factors_match_the_full_width_calibration_data(self):
        """Compressed calibration data gives the same factors as the equivalent full-width data."""
        compressed = self._compressed_calibration_data()
        full_width = np.zeros((10, 100, 3), dtype=bool)
        full_width[..., [1, 2]] = compressed
        observables = [SparsePauliOp("IZI"), SparsePauliOp("ZII"), SparsePauliOp("ZZI")]

        result = TREX.trex_factors_each_term(
            compressed, observables, meas_qubits=[1, 2], circuit_num_qubits=3
        )

        self.assertEqual(result, TREX.trex_factors_each_term(full_width, observables))

    def test_factor_is_read_from_the_column_of_the_circuit_qubit(self):
        """A term on qubit 1 reads qubit 1's column (10% flips), not the column at index 1."""
        result = TREX.trex_factors_each_term(
            self._compressed_calibration_data(),
            [SparsePauliOp("IZI"), SparsePauliOp("ZII")],
            meas_qubits=[1, 2],
            circuit_num_qubits=3,
        )

        self.assertAlmostEqual(result["IZI"], 1 / (1 - 2 * 0.1))
        self.assertAlmostEqual(result["ZII"], 1 / (1 - 2 * 0.2))

    def test_without_meas_qubits_the_array_is_used_as_is(self):
        """Without ``meas_qubits`` the columns are taken as circuit qubits, as before."""
        result = TREX.trex_factors_each_term(
            self._compressed_calibration_data(), [SparsePauliOp("IZ"), SparsePauliOp("ZI")]
        )

        self.assertAlmostEqual(result["IZ"], 1 / (1 - 2 * 0.1))
        self.assertAlmostEqual(result["ZI"], 1 / (1 - 2 * 0.2))

    def test_raises_when_a_term_acts_on_an_unmeasured_qubit(self):
        """A term on a qubit that is missing from the calibration data raises."""
        with self.assertRaisesRegex(ValueError, r"qubits \[0\].*not in ``meas_qubits``"):
            TREX.trex_factors_each_term(
                self._compressed_calibration_data(),
                [SparsePauliOp("IIZ")],
                meas_qubits=[1, 2],
                circuit_num_qubits=3,
            )

    def test_unmeasured_qubits_outside_the_observable_support_are_allowed(self):
        """Terms that do not touch the unmeasured qubits are fine."""
        result = TREX.trex_factors_each_term(
            self._compressed_calibration_data(),
            [SparsePauliOp("ZZI")],
            meas_qubits=[1, 2],
            circuit_num_qubits=3,
        )

        self.assertIn("ZZI", result)

    def test_raises_when_only_meas_qubits_is_given(self):
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            TREX.trex_factors_each_term(
                self._compressed_calibration_data(), [SparsePauliOp("IZI")], meas_qubits=[1, 2]
            )

    def test_raises_when_only_circuit_num_qubits_is_given(self):
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            TREX.trex_factors_each_term(
                self._compressed_calibration_data(), [SparsePauliOp("IZI")], circuit_num_qubits=3
            )

    def test_noise_map_is_not_expanded(self):
        """A ``PauliLindbladMap`` is indexed by circuit qubits and ignores ``meas_qubits``."""
        noise = PauliLindbladMap.from_sparse_list([("X", [2], 0.05)], num_qubits=3)
        observables = [SparsePauliOp("ZII")]

        with_meas_qubits = TREX.trex_factors_each_term(
            noise, observables, meas_qubits=[1, 2], circuit_num_qubits=3
        )

        self.assertEqual(with_meas_qubits, TREX.trex_factors_each_term(noise, observables))


# ---------------------------------------------------------------------------
# TREX._add_data_to_passthrough_data  (static method)
# ---------------------------------------------------------------------------


class TestAddDataToPassthroughData(unittest.TestCase):
    """Tests for :meth:`TREX._add_data_to_passthrough_data`."""

    def _mock_qp(self, passthrough_data):
        """Return a mock QuantumProgram with the given passthrough_data."""
        qp = MagicMock()
        qp.passthrough_data = passthrough_data
        return qp

    def test_creates_passthrough_data_when_none(self):
        """When ``passthrough_data`` is ``None``, it must be initialised with the data."""
        data = {"mitigation": "trex", "version": "0.1"}
        qp = self._mock_qp(None)
        TREX._add_data_to_passthrough_data(data, qp)
        self.assertEqual(qp.passthrough_data, {"qiskit_mitigation": [data]})

    def test_creates_qiskit_mitigation_key_when_absent(self):
        """When ``passthrough_data`` exists but lacks ``'qiskit_mitigation'``, the key must be added."""
        data = {"mitigation": "trex", "version": "0.1"}
        qp = self._mock_qp({"other_key": "value"})
        TREX._add_data_to_passthrough_data(data, qp)
        self.assertIn("qiskit_mitigation", qp.passthrough_data)
        self.assertEqual(qp.passthrough_data["qiskit_mitigation"], [data])

    def test_appends_when_qiskit_mitigation_key_exists(self):
        """When ``'qiskit_mitigation'`` already exists, new data must be appended."""
        existing = {"mitigation": "other", "version": "0.1"}
        data = {"mitigation": "trex", "version": "0.1"}
        qp = self._mock_qp({"qiskit_mitigation": [existing]})
        TREX._add_data_to_passthrough_data(data, qp)
        self.assertEqual(len(qp.passthrough_data["qiskit_mitigation"]), 2)
        self.assertIn(data, qp.passthrough_data["qiskit_mitigation"])

    def test_trex_calibration_flag_set_when_present(self):
        """Entries that already have a non-None ``'trex_calibration'`` key must have it set to ``True``."""
        existing = {"mitigation": "other", "trex_calibration": False}
        data = {"mitigation": "trex", "version": "0.1"}
        qp = self._mock_qp({"qiskit_mitigation": [existing]})
        TREX._add_data_to_passthrough_data(data, qp)
        # The pre-existing entry had trex_calibration set (non-None), so it should now be True
        self.assertTrue(existing["trex_calibration"])

    def test_trex_calibration_not_set_when_absent(self):
        """Entries without ``'trex_calibration'`` key must not have it added."""
        existing = {"mitigation": "other"}
        data = {"mitigation": "trex", "version": "0.1"}
        qp = self._mock_qp({"qiskit_mitigation": [existing]})
        TREX._add_data_to_passthrough_data(data, qp)
        self.assertNotIn("trex_calibration", existing)


# ---------------------------------------------------------------------------
# TREX._box_circuit
# ---------------------------------------------------------------------------


class TestTREXBoxCircuit(unittest.TestCase):
    """Tests for the TREX override of :meth:`_box_circuit`."""

    def setUp(self):
        self.trex = TREX()
        self.circuit = _simple_circuit()

    # --- happy path ---

    def test_returns_quantum_circuit(self):
        """The boxed circuit must return whatever the pass manager run produces."""
        sentinel = QuantumCircuit(2)
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = sentinel
            result = self.trex._box_circuit(self.circuit, {})
        self.assertIs(result, sentinel)

    def test_none_options_returns_quantum_circuit(self):
        """``None`` boxing_options must default gracefully and return a valid circuit."""
        sentinel = QuantumCircuit(2)
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = sentinel
            result = self.trex._box_circuit(self.circuit, None)
        self.assertIs(result, sentinel)

    # --- enable_measures default injection ---

    def test_enable_measures_injected_when_absent(self):
        """``enable_measures=True`` and ``measure_annotations='twirl'`` must be forwarded to
        ``generate_boxing_pass_manager`` even when absent from the caller's dict, and the
        original dict must be untouched."""
        options: dict = {}
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.trex._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertTrue(kwargs.get("enable_measures"))
        self.assertEqual(kwargs.get("measure_annotations"), "twirl")
        # Original dict must not have been mutated.
        self.assertNotIn("enable_measures", options)
        self.assertNotIn("measure_annotations", options)

    def test_none_options_forwards_twirl_defaults(self):
        """``None`` boxing_options must cause ``enable_measures=True`` and
        ``measure_annotations='twirl'`` to be forwarded to ``generate_boxing_pass_manager``."""
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.trex._box_circuit(self.circuit, None)
        _, kwargs = mock_gen.call_args
        self.assertTrue(kwargs.get("enable_measures"))
        self.assertEqual(kwargs.get("measure_annotations"), "twirl")

    # --- validation errors ---

    def test_raises_when_enable_measures_is_false(self):
        """A ``False`` value for ``enable_measures`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.trex._box_circuit(self.circuit, {"enable_measures": False})

    def test_raises_when_measure_annotations_is_not_twirl(self):
        """Any ``measure_annotations`` value other than ``'twirl'`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.trex._box_circuit(
                self.circuit, {"enable_measures": True, "measure_annotations": "change_basis"}
            )

    def test_raises_when_measure_annotations_is_all(self):
        """``measure_annotations='all'`` (used by TREX calibration, not _box_circuit) must raise."""
        with self.assertRaises(ValueError):
            self.trex._box_circuit(
                self.circuit, {"enable_measures": True, "measure_annotations": "all"}
            )

    def test_twirl_annotation_explicit_is_accepted(self):
        """Explicitly passing ``measure_annotations='twirl'`` must not raise."""
        sentinel = QuantumCircuit(2)
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = sentinel
            result = self.trex._box_circuit(
                self.circuit, {"enable_measures": True, "measure_annotations": "twirl"}
            )
        self.assertIs(result, sentinel)

    def test_raises_on_bad_boxing_options(self):
        """Options rejected by ``generate_boxing_pass_manager`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.trex._box_circuit(self.circuit, {"invalid_option_xyz": True})

    # --- original dict immutability for explicit enable_measures ---

    def test_original_dict_not_mutated_when_enable_measures_present(self):
        """When ``enable_measures`` is already in the input, the original dict must still
        be untouched after the call."""
        options = {"enable_measures": True, "measure_annotations": "twirl"}
        with patch("qiskit_mitigation.trex.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.trex._box_circuit(self.circuit, options)
        # Values must be unchanged.
        self.assertTrue(options["enable_measures"])
        self.assertEqual(options["measure_annotations"], "twirl")
        # No extra keys must have been added.
        self.assertEqual(set(options.keys()), {"enable_measures", "measure_annotations"})


# ---------------------------------------------------------------------------
# TREX.create_instance_from_passthrough_data
# ---------------------------------------------------------------------------


class TestTREXCreateInstanceFromPassthroughData(unittest.TestCase):
    """Tests for :meth:`TREX.create_instance_from_passthrough_data`."""

    @staticmethod
    def _minimal_passthrough(**overrides):
        data = {
            "mitigation": "trex",
            "version": "0.1",
            "program_item_index": 0,
        }
        data.update(overrides)
        return data

    def test_happy_path_returns_trex_instance(self):
        """A valid passthrough dict must return a ``TREX`` instance."""
        trex = TREX.create_instance_from_passthrough_data(self._minimal_passthrough())
        self.assertIsInstance(trex, TREX)

    def test_program_item_index_is_set(self):
        """``_program_item_index`` must match the value from the passthrough dict."""
        trex = TREX.create_instance_from_passthrough_data(
            self._minimal_passthrough(program_item_index=7)
        )
        self.assertEqual(trex._program_item_index, 7)

    def test_old_format_without_partial_measurement_keys_means_full_measurement(self):
        """Results saved before ``meas_qubits`` existed have neither key and are full-width."""
        passthrough = self._minimal_passthrough()
        self.assertNotIn("meas_qubits", passthrough)
        self.assertNotIn("circuit_num_qubits", passthrough)
        trex = TREX.create_instance_from_passthrough_data(passthrough)
        self.assertIsNone(trex.meas_qubits)
        self.assertIsNone(trex.circuit_num_qubits)

    def test_raises_when_version_key_missing(self):
        """Missing ``'version'`` key must raise ``ValueError``."""
        passthrough = self._minimal_passthrough()
        del passthrough["version"]
        with self.assertRaises(ValueError):
            TREX.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_mitigation_key_missing(self):
        """Missing ``'mitigation'`` key must raise ``ValueError``."""
        passthrough = self._minimal_passthrough()
        del passthrough["mitigation"]
        with self.assertRaises(ValueError):
            TREX.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_mitigation_is_not_trex(self):
        """A ``'mitigation'`` value other than ``'trex'`` must raise ``ValueError``."""
        passthrough = self._minimal_passthrough(mitigation="pec")
        with self.assertRaises(ValueError):
            TREX.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_mitigation_is_none(self):
        """``mitigation=None`` must raise ``ValueError``."""
        passthrough = self._minimal_passthrough(mitigation=None)
        with self.assertRaises(ValueError):
            TREX.create_instance_from_passthrough_data(passthrough)


class TestComputeNoiseModelPartialQubits(unittest.TestCase):
    """Tests for _compute_noise_model_0_1 when only a subset of qubits is measured."""

    def _make_trex(self, meas_qubits):
        trex = TREX()
        trex._program_item_index = 0
        trex.meas_qubits = meas_qubits
        return trex

    def _fake_result(self, cal_data, flips):
        """Wrap compressed cal data as a mock QuantumProgramResult."""
        item = {"_meas": cal_data, "measurement_flips._meas": flips}
        mock_results = MagicMock()
        mock_results.__getitem__ = MagicMock(return_value=item)
        return mock_results

    def test_noise_map_uses_circuit_qubit_index_not_compressed_index(self):
        """When meas_qubits=[2] on a 5-qubit circuit, the noise map must place the
        flip rate on qubit 2 (circuit index), not qubit 0 (compressed index).
        """
        # Compressed data shape (R, S, M=1) — column 0 corresponds to circuit qubit 2.
        # All-ones → every shot measured |1⟩ → flip rate = 1.0 on circuit qubit 2.
        cal_data = np.ones((2, 8, 1), dtype=bool)
        flips = np.zeros((2, 8, 1), dtype=bool)

        trex = self._make_trex(meas_qubits=[2])
        noise_model = trex.compute_noise_model(self._fake_result(cal_data, flips))

        # Noise map must span at least up to qubit 2 (num_qubits = max(meas_qubits)+1 = 3)
        self.assertGreaterEqual(noise_model.num_qubits, 3)

        # The flip rate on circuit qubit 2 should be ~1.0 (all-ones data).
        # With rate r, pauli_fidelity of Z on qubit 2 = exp(-2r).
        # We can verify indirectly: TREX.calculate_trex_factor with Z on qubit 2 should
        # give a factor != 1 (because flip rate is non-zero), while qubit 0 factor = 1.
        factor_q2 = TREX.calculate_trex_factor(noise_model, "ZII")  # Z on qubit 2 (3-qubit map)
        factor_q0 = TREX.calculate_trex_factor(noise_model, "IIZ")  # Z on qubit 0
        self.assertNotAlmostEqual(factor_q2, 1.0, places=3)
        self.assertAlmostEqual(factor_q0, 1.0, places=10)

    def test_all_zeros_partial_meas_gives_zero_flip_rate(self):
        """meas_qubits=[1, 3] with all-zero data → zero flip rates → noise map has no rates."""
        cal_data = np.zeros((2, 8, 2), dtype=bool)
        flips = np.zeros((2, 8, 2), dtype=bool)

        trex = self._make_trex(meas_qubits=[1, 3])
        noise_model = trex.compute_noise_model(self._fake_result(cal_data, flips))

        # num_qubits = max([1, 3]) + 1 = 4
        self.assertEqual(noise_model.num_qubits, 4)
        # Zero flip rates → identity noise → TREX factors all 1
        for obs in ["IZII", "IIIZ"]:  # Z on qubit 1, Z on qubit 3
            factor = TREX.calculate_trex_factor(noise_model, obs)
            self.assertAlmostEqual(factor, 1.0, places=10)

    def test_meas_qubits_none_falls_back_to_sequential(self):
        """When meas_qubits is None (legacy), column i maps to circuit qubit i."""
        cal_data = np.zeros((2, 8, 2), dtype=bool)
        flips = np.zeros((2, 8, 2), dtype=bool)

        trex = self._make_trex(meas_qubits=None)
        noise_model = trex.compute_noise_model(self._fake_result(cal_data, flips))

        # Falls back to meas_qubits = [0, 1] → num_qubits = 2
        self.assertEqual(noise_model.num_qubits, 2)

    def test_noise_map_uses_circuit_num_qubits_when_set(self):
        """When circuit_num_qubits is set, the noise map num_qubits matches it."""
        cal_data = np.zeros((2, 8, 2), dtype=bool)
        flips = np.zeros((2, 8, 2), dtype=bool)

        trex = self._make_trex(meas_qubits=[0, 1])
        trex.circuit_num_qubits = 156
        noise_model = trex.compute_noise_model(self._fake_result(cal_data, flips))

        self.assertEqual(noise_model.num_qubits, 156)

    def test_calculate_trex_factor_mismatched_qubits(self):
        """calculate_trex_factor handles mismatched qubit counts between noise map and observable."""
        error_rate = -0.5 * np.log(1.0 - 2.0 * 0.1)
        noise_2q = PauliLindbladMap.from_sparse_list([("X", [0], error_rate)], num_qubits=2)

        # 156-qubit identity observable
        factor_id = TREX.calculate_trex_factor(noise_2q, "I" * 156)
        self.assertAlmostEqual(factor_id, 1.0)

        # 156-qubit observable with Z on qubit 0 (last character in Pauli string)
        obs_156 = "I" * 155 + "Z"
        factor_z = TREX.calculate_trex_factor(noise_2q, obs_156)
        self.assertAlmostEqual(factor_z, 1 / (1 - 2 * 0.1))

    def test_passthrough_contains_meas_qubits_and_circuit_num_qubits(self):
        """TREX.prepare and create_instance_from_passthrough_data round-trip meas_qubits and circuit_num_qubits."""
        qc = QuantumCircuit(5)
        task = MagicMock()
        task.circuit = qc
        task.meas_qubits = [1, 3]

        trex = TREX()
        trex.tasks.append(task)

        qp = QuantumProgram(shots=100)
        trex.prepare(num_randomizations=2, quantum_program=qp)

        passthrough = qp.passthrough_data["qiskit_mitigation"][0]
        self.assertEqual(passthrough["meas_qubits"], [1, 3])
        self.assertEqual(passthrough["circuit_num_qubits"], 5)

        restored_trex = TREX.create_instance_from_passthrough_data(passthrough)
        self.assertEqual(restored_trex.meas_qubits, [1, 3])
        self.assertEqual(restored_trex.circuit_num_qubits, 5)


class TestTrexEndToEndWithPartialMeasurement(unittest.TestCase):
    """Prepare, build a result and postprocess with TREX when the tasks measure only some qubits.

    Every qubit gets its own calibration flip rate, so reading the wrong calibration column
    (for example because of a mix-up of the compressed and the circuit qubit indices) changes the
    result. The data of the tasks themselves is all zeros, so the expectation value of an observable is
    exactly its TREX factor, ``exp(2 * sum of the flip rates of the qubits of the term)``.
    """

    SHOTS = 10
    RANDOMIZATIONS = 2
    FLIP_RATES: ClassVar[dict[int, float]] = {0: 0.0, 1: 0.1, 2: 0.2, 4: 0.3}

    def _result(self, program, trex, with_pauli_signs=False):
        """Return the result of the program with all-zero data and the calibration of FLIP_RATES."""
        calibration = np.zeros((self.RANDOMIZATIONS, self.SHOTS, len(trex.meas_qubits)), dtype=bool)
        for column, qubit in enumerate(trex.meas_qubits):
            calibration[:, : round(self.FLIP_RATES[qubit] * self.SHOTS), column] = True
        item_results = []
        for item in program.items[:-1]:
            data = np.zeros((*item.shape, self.SHOTS, len(item.circuit.cregs[-1])), dtype=bool)
            item_result = {"_meas": data, "measurement_flips._meas": np.zeros_like(data)}
            if with_pauli_signs:
                item_result["pauli_signs"] = np.zeros((*item.shape, 1), dtype=bool)
            item_results.append(item_result)
        item_results.append(
            {"_meas": calibration, "measurement_flips._meas": np.zeros_like(calibration)}
        )
        return QuantumProgramResult(item_results, passthrough_data=program.passthrough_data)

    @staticmethod
    def _noise_maps(task, circuit, meas_qubits):
        layers = find_combined_unique_layers(
            [circuit], mitigation_types=[task], meas_qubits=meas_qubits, box_types="gates"
        )
        return {
            get_annotation(layer.operation, InjectNoise).ref: PauliLindbladMap.identity(
                len(layer.qubits)
            )
            for layer in layers
        }

    def test_tasks_of_different_widths_read_their_own_calibration_columns(self):
        """A 3-qubit and a 5-qubit task share one calibration measuring the union of their qubits."""
        narrow = QuantumCircuit(3)
        narrow.sx(0)
        narrow.sx(2)
        wide = QuantumCircuit(5)
        wide.sx(1)
        wide.sx(4)
        trex = TREX()
        program = MitigationTask().prepare(
            narrow,
            [SparsePauliOp("ZIZ")],
            None,
            shots_per_randomization=self.SHOTS,
            num_randomizations=self.RANDOMIZATIONS,
            trex=trex,
            meas_qubits=[0, 2],
        )
        program = MitigationTask().prepare(
            wide,
            [SparsePauliOp("ZIIZI")],
            None,
            shots_per_randomization=self.SHOTS,
            num_randomizations=self.RANDOMIZATIONS,
            trex=trex,
            meas_qubits=[1, 4],
            quantum_program=program,
        )
        trex.prepare(self.RANDOMIZATIONS, program)
        self.assertEqual(trex.meas_qubits, [0, 1, 2, 4])
        self.assertEqual(trex.circuit_num_qubits, 5)

        result = self._result(program, trex)
        narrow_task, wide_task = load_tasks_from_result(result)
        # the calibration columns are the union [0, 1, 2, 4]; each task must pick its own qubits
        np.testing.assert_allclose(
            narrow_task.postprocess(result).data.evs, [np.exp(2 * (0.0 + 0.2))]
        )
        np.testing.assert_allclose(
            wide_task.postprocess(result).data.evs, [np.exp(2 * (0.1 + 0.3))]
        )

    def test_pec_with_trex_and_partial_measurement(self):
        circuit = QuantumCircuit(5)
        circuit.sx(1)
        circuit.cz(1, 4)
        trex = TREX()
        pec = PEC()
        program = pec.prepare(
            circuit,
            [SparsePauliOp("ZIIZI")],
            None,
            noise_maps=self._noise_maps(pec, circuit, [1, 4]),
            shots_per_randomization=self.SHOTS,
            num_randomizations=self.RANDOMIZATIONS,
            scale_randomizations_by_gamma=False,
            trex=trex,
            meas_qubits=[1, 4],
        )
        trex.prepare(self.RANDOMIZATIONS, program)

        result = self._result(program, trex, with_pauli_signs=True)
        (restored,) = load_tasks_from_result(result)
        np.testing.assert_allclose(restored.postprocess(result).data.evs, [np.exp(2 * (0.1 + 0.3))])

    def test_pea_with_trex_and_partial_measurement(self):
        circuit = QuantumCircuit(5)
        circuit.sx(1)
        circuit.cz(1, 4)
        trex = TREX()
        pea = PEA()
        program = pea.prepare(
            circuit,
            [SparsePauliOp("ZIIZI")],
            None,
            noise_maps=self._noise_maps(pea, circuit, [1, 4]),
            noise_factors=[1, 3],
            shots_per_randomization=self.SHOTS,
            num_randomizations=self.RANDOMIZATIONS,
            trex=trex,
            meas_qubits=[1, 4],
        )
        trex.prepare(self.RANDOMIZATIONS, program)

        result = self._result(program, trex, with_pauli_signs=True)
        (restored,) = load_tasks_from_result(result)
        with warnings.catch_warnings():
            # the synthetic data has no variance, which makes the extrapolation fall back
            warnings.filterwarnings("ignore", message="No positive, finite standard errors")
            evs = restored.postprocess(result, extrapolator=["fallback"]).data.evs
        np.testing.assert_allclose(evs, [np.exp(2 * (0.1 + 0.3))])


if __name__ == "__main__":
    unittest.main()
