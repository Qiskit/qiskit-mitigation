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

"""Tests for qiskit_mitigation.utils.utils."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp, CircuitInstruction, Gate
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp
from qiskit_mitigation import PEC, MitigationTask
from qiskit_mitigation.trex import TREX
from qiskit_mitigation.utils.meas_qubits import get_active_qubits, validate_meas_qubits
from qiskit_mitigation.utils.utils import (
    _find_box_type,
    find_combined_unique_layers,
    get_measured_qubits,
    load_tasks_from_result,
)
from qiskit_mitigation.zne.gate_folding import GateFolding
from qiskit_mitigation.zne.pea import PEA
from samplomatic.quantum_program import QuantumProgramResult
from samplomatic.utils import find_unique_box_instructions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(passthrough_data):
    """Return a QuantumProgramResult with the given passthrough_data."""
    # shape: (num_randomizations, num_configs, shots, num_qubits)
    data = np.zeros((5, 4, 64, 2), dtype=bool)
    result = QuantumProgramResult(data)
    result.passthrough_data = passthrough_data
    return result


def _task_passthrough(mitigation=None, trex_calibration=False, **extra):
    """Return a minimal valid passthrough dict for a single MitigationTask (mitigation=None by default)."""
    data = {
        "mitigation": mitigation,
        "version": "0.1",
        "trex_calibration": trex_calibration,
        "observables": SparsePauliOp("ZZ"),
        "param_basis_pairs": None,
        "param_shape": None,
        "broadcast_obs_and_params": True,
        "meas_bases": None,
        "program_item_index": 0,
    }
    data.update(extra)
    return data


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple Bell-pair-like circuit with measurements."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# load_tasks_from_result — passthrough_data type validation
# ---------------------------------------------------------------------------


class TestLoadTasksFromResultPassthroughType(unittest.TestCase):
    def test_raises_when_passthrough_is_not_dict(self):
        """Non-dict passthrough_data must raise ValueError."""
        result = _make_result(passthrough_data="not-a-dict")
        with self.assertRaises(ValueError, msg="Expected ValueError for non-dict passthrough_data"):
            load_tasks_from_result(result)

    def test_raises_when_passthrough_is_none(self):
        """None passthrough_data must raise ValueError."""
        result = _make_result(passthrough_data=None)
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_raises_when_passthrough_is_list(self):
        """List passthrough_data must raise ValueError."""
        result = _make_result(passthrough_data=[{"mitigation": "gate_folding"}])
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)


# ---------------------------------------------------------------------------
# load_tasks_from_result — missing / empty 'mitigation' key
# ---------------------------------------------------------------------------


class TestLoadTasksFromResultMitigationKey(unittest.TestCase):
    def test_raises_when_mitigation_key_missing(self):
        """A dict with no 'qiskit_mitigation' key must raise ValueError."""
        result = _make_result(passthrough_data={"other": "value"})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_raises_when_mitigation_is_empty_list(self):
        """An empty 'qiskit_mitigation' list must raise ValueError."""
        result = _make_result(passthrough_data={"qiskit_mitigation": []})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_raises_when_mitigation_is_not_a_sequence(self):
        """A non-sequence value for 'qiskit_mitigation' must raise ValueError."""
        result = _make_result(passthrough_data={"qiskit_mitigation": 42})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_raises_when_version_missing_in_task(self):
        """Missing 'version' in a task passthrough entry must raise ValueError."""
        task_data = _task_passthrough()
        del task_data["version"]
        result = _make_result(passthrough_data={"qiskit_mitigation": [task_data]})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)


# ---------------------------------------------------------------------------
# load_tasks_from_result — normal (non-TREX) task loading
# ---------------------------------------------------------------------------


class TestLoadTasksFromResultNonTrex(unittest.TestCase):
    def test_returns_list(self):
        """load_tasks_from_result must return a list."""
        task_data = _task_passthrough()
        result = _make_result({"qiskit_mitigation": [task_data]})
        tasks = load_tasks_from_result(result)
        self.assertIsInstance(tasks, list)

    def test_single_non_trex_task_returns_one_mitigation_task(self):
        """A single non-TREX task produces exactly one MitigationTask."""
        task_data = _task_passthrough()
        result = _make_result({"qiskit_mitigation": [task_data]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], MitigationTask)

    def test_multiple_non_trex_tasks_all_returned(self):
        """Multiple non-TREX tasks each produce one MitigationTask."""
        passthrough = {
            "qiskit_mitigation": [
                _task_passthrough(program_item_index=0),
                _task_passthrough(program_item_index=1),
                _task_passthrough(program_item_index=2),
            ]
        }
        result = _make_result(passthrough)
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertIsInstance(t, MitigationTask)

    def test_non_trex_task_program_item_index_set(self):
        """The program_item_index on the returned task must match the passthrough value."""
        task_data = _task_passthrough(program_item_index=7)
        result = _make_result({"qiskit_mitigation": [task_data]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(tasks[0]._program_item_index, 7)


# ---------------------------------------------------------------------------
# load_tasks_from_result — TREX calibration task
# ---------------------------------------------------------------------------


class TestLoadTasksFromResultWithTrex(unittest.TestCase):
    def test_trex_only_task_skipped_when_return_trex_false(self):
        """A TREX entry with return_trex=False (default) is silently skipped — result is empty."""
        trex_data = _task_passthrough(
            mitigation="trex", trex_calibration=False, program_item_index=0
        )
        result = _make_result({"qiskit_mitigation": [trex_data]})
        tasks = load_tasks_from_result(result)
        # "trex" match arm does nothing when return_trex=False → zero items appended
        self.assertEqual(len(tasks), 0)

    def test_task_with_trex_calibration_and_trex_present_returns_task(self):
        """A task that requires TREX calibration succeeds when a TREX entry is present."""
        trex_cal_data = _task_passthrough(
            mitigation="trex", trex_calibration=False, program_item_index=0
        )
        user_task_data = _task_passthrough(trex_calibration=True, program_item_index=1)
        result = _make_result({"qiskit_mitigation": [trex_cal_data, user_task_data]})
        tasks = load_tasks_from_result(result)
        # trex_cal_data → skipped (return_trex=False)
        # user_task_data (trex_calibration=True, mitigation=None) → 1 MitigationTask
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], MitigationTask)

    def test_task_with_trex_calibration_but_no_trex_raises(self):
        """A task that requires TREX calibration must raise ValueError if no TREX entry exists."""
        user_task_data = _task_passthrough(trex_calibration=True, program_item_index=0)
        result = _make_result({"qiskit_mitigation": [user_task_data]})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_mixed_trex_and_non_trex_tasks(self):
        """TREX calibration entry + one TREX-calibrated task + one plain task.

        trex_cal (mitigation="trex") → skipped by second loop (return_trex=False)
        trex_task (trex_calibration=True, mitigation=None) → 1 MitigationTask
        plain_task (trex_calibration=False, mitigation=None) → 1 MitigationTask
        Total: 2 tasks
        """
        trex_cal = _task_passthrough(
            mitigation="trex", trex_calibration=False, program_item_index=0
        )
        trex_task = _task_passthrough(trex_calibration=True, program_item_index=1)
        plain_task = _task_passthrough(trex_calibration=False, program_item_index=2)
        result = _make_result({"qiskit_mitigation": [trex_cal, trex_task, plain_task]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            self.assertIsInstance(t, MitigationTask)


# ---------------------------------------------------------------------------
# load_tasks_from_result — PEC, GateFolding and PEA task types
# ---------------------------------------------------------------------------


class TestLoadTasksFromResultMitigationTypes(unittest.TestCase):
    """Tests that PEC, GateFolding and PEA passthrough types are dispatched correctly."""

    def _pec_passthrough(self, **extra):
        data = {
            "mitigation": "pec",
            "version": "0.1",
            "trex_calibration": False,
            "observables": SparsePauliOp("ZZ"),
            "param_basis_pairs": None,
            "param_shape": None,
            "broadcast_obs_and_params": True,
            "meas_bases": None,
            "program_item_index": 0,
            "pec_gamma": 1.0,
        }
        data.update(extra)
        return data

    def _gate_folding_passthrough(self, **extra):
        data = {
            "mitigation": "gate_folding",
            "version": "0.1",
            "trex_calibration": False,
            "observables": SparsePauliOp("ZZ"),
            "param_basis_pairs": None,
            "param_shape": None,
            "broadcast_obs_and_params": True,
            "meas_bases": None,
            "program_item_index": 0,
            "noise_factors": np.array([1.0, 3.0, 5.0]),
            "extrapolator": ["linear"],
            "extrapolated_noise_factors": None,
        }
        data.update(extra)
        return data

    def _pea_passthrough(self, **extra):
        data = {
            "mitigation": "pea",
            "version": "0.1",
            "trex_calibration": False,
            "observables": SparsePauliOp("ZZ"),
            "param_basis_pairs": None,
            "param_shape": None,
            "broadcast_obs_and_params": True,
            "meas_bases": None,
            "program_item_index": 0,
            "noise_factors": np.array([1.0, 3.0, 5.0]),
            "extrapolator": ["linear"],
            "extrapolated_noise_factors": None,
        }
        data.update(extra)
        return data

    def test_pec_task_loaded_correctly(self):
        """A 'pec' mitigation entry must produce a PEC instance."""
        result = _make_result({"qiskit_mitigation": [self._pec_passthrough()]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], PEC)

    def test_gate_folding_task_loaded_correctly(self):
        """A 'GateFolding' mitigation entry must produce a GateFolding instance."""
        result = _make_result({"qiskit_mitigation": [self._gate_folding_passthrough()]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], GateFolding)

    def test_pea_task_loaded_correctly(self):
        """A 'pea' mitigation entry must produce a PEA instance."""
        result = _make_result({"qiskit_mitigation": [self._pea_passthrough()]})
        tasks = load_tasks_from_result(result)
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], PEA)

    def test_unknown_mitigation_type_raises(self):
        """An unknown 'mitigation' type must raise ValueError."""
        bad_data = {
            "mitigation": "unknown_type_xyz",
            "version": "0.1",
            "trex_calibration": False,
            "observables": SparsePauliOp("ZZ"),
            "param_basis_pairs": None,
            "param_shape": None,
            "broadcast_obs_and_params": True,
            "meas_bases": None,
            "program_item_index": 0,
        }
        result = _make_result({"qiskit_mitigation": [bad_data]})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)

    def test_return_trex_true_includes_trex_task(self):
        """With ``return_trex=True`` a TREX calibration entry must be included."""
        trex_data = _task_passthrough(
            mitigation="trex", trex_calibration=False, program_item_index=0
        )
        result = _make_result({"qiskit_mitigation": [trex_data]})
        tasks = load_tasks_from_result(result, return_trex=True)
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], TREX)

    def test_trex_calibration_true_but_no_trex_in_program_raises(self):
        """trex_calibration=True with no TREX entry in program must raise ValueError."""
        bad_task = _task_passthrough(trex_calibration=True)
        result = _make_result({"qiskit_mitigation": [bad_task]})
        with self.assertRaises(ValueError):
            load_tasks_from_result(result)


# ---------------------------------------------------------------------------
# find_box_type — unit tests
# ---------------------------------------------------------------------------


class TestFindBoxType(unittest.TestCase):
    """Tests for find_box_type using real boxed circuits."""

    def setUp(self):
        self.circuit = _simple_circuit()

    def _get_all_boxes(self, circuit: QuantumCircuit) -> list[CircuitInstruction]:
        """Box a circuit with MitigationTask and return all CircuitInstructions."""
        boxed = MitigationTask._box_circuit(circuit, None)
        return [instr for instr in boxed if instr.operation.name == "box"]

    def test_find_box_type_raises_for_non_box_instruction(self):
        """find_box_type must raise ValueError when passed a non-box instruction."""
        # Iterate to find an instruction that is NOT a box
        # Use a simple gate circuit without boxing to get a non-box instruction
        plain = QuantumCircuit(2)
        plain.h(0)
        for instr in plain:
            with self.assertRaises(ValueError):
                _find_box_type(instr)
            break  # only test the first non-box instruction

    def test_measurement_box_has_type_measurement(self):
        """A box that contains measurement operations must return 'measurement'."""
        boxes = self._get_all_boxes(self.circuit)
        measurement_boxes = [b for b in boxes if len(b.clbits) > 0]
        self.assertTrue(len(measurement_boxes) > 0, "Expected at least one measurement box")
        for box in measurement_boxes:
            self.assertEqual(_find_box_type(box), "measurement")

    def test_gate_box_has_type_gates(self):
        """A box that contains only standard gates must return 'gates'."""
        boxes = self._get_all_boxes(self.circuit)
        gate_boxes = [b for b in boxes if len(b.clbits) == 0]
        self.assertTrue(len(gate_boxes) > 0, "Expected at least one gate-only box")
        for box in gate_boxes:
            self.assertEqual(_find_box_type(box), "gates")

    def test_pec_gate_box_has_type_gates(self):
        """Gate boxes produced by PEC boxing must also return 'gates'."""
        boxed = PEC._box_circuit(self.circuit, None)
        gate_boxes = [
            instr for instr in boxed if instr.operation.name == "box" and len(instr.clbits) == 0
        ]
        self.assertTrue(len(gate_boxes) > 0, "PEC boxing should produce at least one gate box")
        for box in gate_boxes:
            self.assertEqual(_find_box_type(box), "gates")

    def test_return_value_is_string(self):
        """_find_box_type must always return a string."""
        boxes = self._get_all_boxes(self.circuit)
        for box in boxes:
            result = _find_box_type(box)
            self.assertIsInstance(result, str)

    def test_return_value_is_known_type(self):
        """_find_box_type must return one of the recognised type strings."""
        known_types = {"gates", "measurement"}
        boxes = self._get_all_boxes(self.circuit)
        for box in boxes:
            self.assertIn(_find_box_type(box), known_types)

    def test_empty_body_box_returns_gates(self):
        """A box with zero body instructions must return ``'gates'``."""

        # BoxOp wrapping an empty circuit produces a body with len == 0
        empty_box = BoxOp(QuantumCircuit(1))
        instr = CircuitInstruction(operation=empty_box, qubits=[], clbits=[])
        self.assertEqual(_find_box_type(instr), "gates")

    def test_gate_only_body_returns_gates(self):
        """A box whose body contains only gates (no measurements) must return ``'gates'``."""

        custom = Gate("mygate", 1, [])
        inner = QuantumCircuit(1)
        inner.append(custom, [0])
        box = BoxOp(inner)
        instr = CircuitInstruction(operation=box, qubits=[], clbits=[])
        self.assertEqual(_find_box_type(instr), "gates")


# ---------------------------------------------------------------------------
# find_combined_unique_layers — argument validation
# ---------------------------------------------------------------------------


class TestFindCombinedUniqueLayersValidation(unittest.TestCase):
    def setUp(self):
        self.circuit = _simple_circuit()

    def test_raises_when_mitigation_types_length_mismatch(self):
        """Supplying fewer mitigation_types than circuits must raise ValueError."""
        with self.assertRaises(ValueError):
            find_combined_unique_layers(
                [self.circuit, self.circuit],
                mitigation_types=[MitigationTask()],
            )

    def test_raises_when_custom_boxing_options_list_length_mismatch(self):
        """Supplying more boxing-option dicts than circuits must raise ValueError."""
        with self.assertRaises(ValueError):
            find_combined_unique_layers(
                [self.circuit, self.circuit],
                custom_boxing_options=[{}, {}, {}],
            )

    def test_returns_list(self):
        """The return type must be a list."""
        result = find_combined_unique_layers([self.circuit])
        self.assertIsInstance(result, list)

    def test_each_element_is_circuit_instruction(self):
        """Every returned element must be a CircuitInstruction."""
        layers = find_combined_unique_layers([self.circuit])
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertIsInstance(layer, CircuitInstruction)


# ---------------------------------------------------------------------------
# find_combined_unique_layers — real flow with MitigationTask and PEC
# ---------------------------------------------------------------------------


class TestFindCombinedUniqueLayersWithMitigationTypes(unittest.TestCase):
    """Tests that use real MitigationTask / PEC instances — no mocks."""

    def setUp(self):
        self.circuit = _simple_circuit()

    # --- box_types='all' ---

    def test_all_with_mitigation_task_and_pec(self):
        """box_types='all' with [MitigationTask, PEC] returns all unique box types."""
        task_unmitigated = MitigationTask()
        pec = PEC()
        tasks = [task_unmitigated, pec]
        combined_layers = find_combined_unique_layers(
            [self.circuit, self.circuit], mitigation_types=tasks, box_types="all"
        )
        # PEC boxing adds an extra gate box; combined result has gates + measurement types
        types = {_find_box_type(layer) for layer in combined_layers}
        self.assertIn("gates", types)
        self.assertIn("measurement", types)

    def test_all_without_mitigation_types(self):
        """box_types='all' without mitigation_types defaults to MitigationTask boxing."""
        layers = find_combined_unique_layers([self.circuit, self.circuit], box_types="all")
        types = {_find_box_type(layer) for layer in layers}
        self.assertIn("gates", types)
        self.assertIn("measurement", types)

    def test_all_with_single_mitigation_task(self):
        """box_types='all' with a single MitigationTask returns gate and measurement layers."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[MitigationTask()], box_types="all"
        )
        self.assertTrue(len(layers) > 0)
        types = {_find_box_type(layer) for layer in layers}
        self.assertIn("gates", types)
        self.assertIn("measurement", types)

    def test_all_with_single_pec(self):
        """box_types='all' with a single PEC task returns gate and measurement layers."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[PEC()], box_types="all"
        )
        self.assertTrue(len(layers) > 0)
        types = {_find_box_type(layer) for layer in layers}
        self.assertIn("gates", types)
        self.assertIn("measurement", types)

    # --- box_types='gates' ---

    def test_gates_only_with_mitigation_task_and_pec(self):
        """box_types='gates' must return only gate-type boxes."""
        task_unmitigated = MitigationTask()
        pec = PEC()
        tasks = [task_unmitigated, pec]
        combined_layers = find_combined_unique_layers(
            [self.circuit, self.circuit], mitigation_types=tasks, box_types="gates"
        )
        self.assertTrue(len(combined_layers) > 0, "Expected at least one gate box")
        for layer in combined_layers:
            self.assertEqual(_find_box_type(layer), "gates")

    def test_gates_only_without_mitigation_types(self):
        """box_types='gates' without mitigation_types returns only gate boxes."""
        layers = find_combined_unique_layers([self.circuit, self.circuit], box_types="gates")
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "gates")

    def test_gates_only_with_single_mitigation_task(self):
        """box_types='gates' with a single MitigationTask returns only gate boxes."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[MitigationTask()], box_types="gates"
        )
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "gates")

    def test_gates_only_with_single_pec(self):
        """box_types='gates' with a single PEC task returns only gate boxes."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[PEC()], box_types="gates"
        )
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "gates")

    # --- box_types='measurement' ---

    def test_measurement_only_with_mitigation_task_and_pec(self):
        """box_types='measurement' must return only measurement-type boxes."""
        task_unmitigated = MitigationTask()
        pec = PEC()
        tasks = [task_unmitigated, pec]
        combined_layers = find_combined_unique_layers(
            [self.circuit, self.circuit], mitigation_types=tasks, box_types="measurement"
        )
        self.assertTrue(len(combined_layers) > 0, "Expected at least one measurement box")
        for layer in combined_layers:
            self.assertEqual(_find_box_type(layer), "measurement")

    def test_measurement_only_without_mitigation_types(self):
        """box_types='measurement' without mitigation_types returns only measurement boxes."""
        layers = find_combined_unique_layers([self.circuit, self.circuit], box_types="measurement")
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "measurement")

    def test_measurement_only_with_single_mitigation_task(self):
        """box_types='measurement' with a single MitigationTask returns only measurement boxes."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[MitigationTask()], box_types="measurement"
        )
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "measurement")

    def test_measurement_only_with_single_pec(self):
        """box_types='measurement' with a single PEC task returns only measurement boxes."""
        layers = find_combined_unique_layers(
            [self.circuit], mitigation_types=[PEC()], box_types="measurement"
        )
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertEqual(_find_box_type(layer), "measurement")

    # --- uniqueness and deduplication ---

    def test_duplicate_circuits_produce_same_unique_layers_as_single_circuit(self):
        """Passing the same circuit twice must not duplicate the unique layers."""
        single = find_combined_unique_layers([self.circuit], box_types="all")
        doubled = find_combined_unique_layers([self.circuit, self.circuit], box_types="all")
        self.assertEqual(len(single), len(doubled))

    def test_pec_produces_more_gate_boxes_than_mitigation_task(self):
        """PEC boxing introduces an extra injection box, so combining MitigationTask + PEC
        must yield more gate-type unique layers than MitigationTask alone."""
        layers_task = find_combined_unique_layers(
            [self.circuit], mitigation_types=[MitigationTask()], box_types="gates"
        )
        layers_combined = find_combined_unique_layers(
            [self.circuit, self.circuit],
            mitigation_types=[MitigationTask(), PEC()],
            box_types="gates",
        )
        self.assertGreater(len(layers_combined), len(layers_task))

    # --- custom_boxing_options ---

    def test_single_dict_boxing_options_applies_to_all_circuits(self):
        """A single dict passed as custom_boxing_options must work for multiple circuits."""
        layers = find_combined_unique_layers(
            [self.circuit, self.circuit],
            custom_boxing_options={"enable_measures": True},
            box_types="all",
        )
        self.assertTrue(len(layers) > 0)

    def test_list_boxing_options_per_circuit(self):
        """A list of boxing-option dicts (one per circuit) must be accepted."""
        layers = find_combined_unique_layers(
            [self.circuit, self.circuit],
            custom_boxing_options=[{"enable_measures": True}, {"enable_measures": True}],
            box_types="all",
        )
        self.assertTrue(len(layers) > 0)


# ---------------------------------------------------------------------------
# find_combined_unique_layers — meas_qubits
# ---------------------------------------------------------------------------


def _three_qubit_gate_circuit() -> QuantumCircuit:
    """Return a 3-qubit circuit with gates on every qubit."""
    circuit = QuantumCircuit(3)
    circuit.sx(0)
    circuit.cz(0, 1)
    circuit.sx(1)
    circuit.cz(1, 2)
    circuit.sx(2)
    circuit.sx(0)
    return circuit


class TestFindCombinedUniqueLayersMeasQubits(unittest.TestCase):
    """Tests for the ``meas_qubits`` argument of ``find_combined_unique_layers``."""

    def setUp(self):
        self.circuit = _three_qubit_gate_circuit()

    def test_none_equals_measuring_all_qubits(self):
        """``meas_qubits=None`` gives the same layers as listing every qubit."""
        layers_none = find_combined_unique_layers([self.circuit])
        layers_all = find_combined_unique_layers([self.circuit], meas_qubits=[0, 1, 2])
        self.assertEqual(len(layers_none), len(layers_all))
        self.assertEqual(
            [layer.operation.num_qubits for layer in layers_none],
            [layer.operation.num_qubits for layer in layers_all],
        )

    def test_unsorted_and_duplicate_meas_qubits_are_accepted(self):
        """The order and repetition of ``meas_qubits`` must not matter."""
        layers_sorted = find_combined_unique_layers([self.circuit], meas_qubits=[0, 1, 2])
        layers_shuffled = find_combined_unique_layers([self.circuit], meas_qubits=[2, 0, 1, 1])
        self.assertEqual(len(layers_sorted), len(layers_shuffled))

    def test_insufficient_meas_qubits_raises(self):
        """Not measuring a qubit that has gates on it must raise an explicit ``ValueError``."""
        with self.assertRaisesRegex(ValueError, r"\[2\].*not in ``meas_qubits``"):
            find_combined_unique_layers([self.circuit], meas_qubits=[0, 1])

    def test_out_of_range_meas_qubits_raises(self):
        """Qubits outside the circuit must raise an explicit ``ValueError``."""
        with self.assertRaisesRegex(ValueError, "out of range"):
            find_combined_unique_layers([self.circuit], meas_qubits=[0, 1, 2, 3])

    def test_shared_meas_qubits_are_checked_for_every_circuit(self):
        """The shared ``meas_qubits`` must be sufficient for each of the circuits."""
        wide = QuantumCircuit(3)
        wide.sx(2)
        narrow = QuantumCircuit(3)
        narrow.sx(0)
        with self.assertRaisesRegex(ValueError, r"\[2\]"):
            find_combined_unique_layers([narrow, wide], meas_qubits=[0])

    def test_shared_meas_qubits_give_identical_measurement_layers(self):
        """Circuits that end with the same gate layer and share ``meas_qubits`` share their measurement layer."""
        first = QuantumCircuit(3)
        first.sx(0)
        first.cz(0, 1)
        first.cz(1, 2)
        second = QuantumCircuit(3)
        second.cz(1, 2)
        second.sx(0)
        second.cz(0, 1)
        measurement_layers = find_combined_unique_layers(
            [first, second], meas_qubits=[0, 1, 2], box_types="measurement"
        )
        self.assertEqual(len(measurement_layers), 1)

    def test_measurement_layers_span_exactly_the_measured_qubits(self):
        """Every measurement layer acts on the measured qubits and nothing else."""
        circuit = QuantumCircuit(5)
        circuit.sx(0)
        circuit.cz(0, 1)
        for meas_qubits in ([0, 1], [0, 1, 3], [0, 1, 2, 3, 4]):
            with self.subTest(meas_qubits=meas_qubits):
                layers = find_combined_unique_layers(
                    [circuit], meas_qubits=meas_qubits, box_types="measurement"
                )
                self.assertEqual(len(layers), 1)
                layer = layers[0]
                self.assertEqual(
                    sorted(circuit.find_bit(q).index for q in layer.qubits), meas_qubits
                )
                measured = [
                    op for op in layer.operation.body.data if op.operation.name == "measure"
                ]
                self.assertEqual(len(measured), len(meas_qubits))

    def test_different_meas_qubits_give_different_measurement_layers(self):
        """``meas_qubits`` affects the boxing, so it must match the one given to ``prepare``."""
        circuit = QuantumCircuit(3)
        circuit.sx(0)
        circuit.cz(0, 1)
        layers_01 = find_combined_unique_layers(
            [circuit], meas_qubits=[0, 1], box_types="measurement"
        )
        layers_012 = find_combined_unique_layers(
            [circuit], meas_qubits=[0, 1, 2], box_types="measurement"
        )
        self.assertNotEqual(layers_01[0].qubits, layers_012[0].qubits)

    def test_layers_match_those_of_prepare(self):
        """The layers found with ``meas_qubits`` are the ones used by ``prepare`` with the same ``meas_qubits``."""
        circuit = _three_qubit_gate_circuit()
        observables = [SparsePauliOp("IIZ")]
        meas_qubits = get_measured_qubits([circuit], [observables])
        task = MitigationTask()
        task.prepare(circuit, observables, None, meas_qubits=meas_qubits)
        found = find_combined_unique_layers([circuit], meas_qubits=meas_qubits)
        prepared = find_unique_box_instructions(
            instructions=task.boxed_circuit, normalize_annotations=None, undress_boxes=True
        )
        self.assertEqual(len(found), len(prepared))

    def test_twirling_strategy_all_requires_every_qubit(self):
        """``twirling_strategy="all"`` boxes every qubit, so every qubit must be measured."""
        with self.assertRaisesRegex(ValueError, "twirling_strategy"):
            find_combined_unique_layers(
                [self.circuit],
                custom_boxing_options={"twirling_strategy": "all"},
                meas_qubits=[0, 1],
            )

    def test_unmeasured_idle_qubits_leave_no_gates_after_measurement(self):
        """With the qubits from ``get_measured_qubits`` no box is added after the measurement."""
        circuit = QuantumCircuit(5)
        circuit.sx(0)
        circuit.cz(0, 1)
        circuit.sx(2)
        meas_qubits = get_measured_qubits([circuit], [[SparsePauliOp("ZIIII")]])
        self.assertEqual(meas_qubits, [0, 1, 2, 4])
        boxed = MitigationTask._box_circuit(circuit, None, meas_qubits)
        last_box = boxed.data[-1].operation
        self.assertTrue(
            any(instruction.operation.name == "measure" for instruction in last_box.body)
        )

    def test_meas_qubits_missing_active_qubit_raises_in_box_circuit(self):
        """Leaving an active qubit unmeasured would leave gates after the measurement box.

        ``_box_circuit`` rejects it instead of producing such a circuit.
        """
        circuit = QuantumCircuit(3)
        circuit.sx(0)
        circuit.cz(1, 2)
        with self.assertRaisesRegex(ValueError, r"\[1, 2\].*not in ``meas_qubits``"):
            MitigationTask._box_circuit(circuit, None, [0])


# ---------------------------------------------------------------------------
# get_active_qubits
# ---------------------------------------------------------------------------


class TestGetActiveQubits(unittest.TestCase):
    """Tests for :func:`get_active_qubits`."""

    def test_qubits_with_gates_are_active(self):
        circuit = QuantumCircuit(4)
        circuit.sx(0)
        circuit.cz(0, 2)
        self.assertEqual(get_active_qubits(circuit), [0, 2])

    def test_empty_circuit_has_no_active_qubits(self):
        self.assertEqual(get_active_qubits(QuantumCircuit(3)), [])

    def test_barrier_and_delay_do_not_make_a_qubit_active(self):
        circuit = QuantumCircuit(4)
        circuit.sx(0)
        circuit.barrier(1, 2)
        circuit.delay(10, 3)
        self.assertEqual(get_active_qubits(circuit), [0])

    def test_identity_and_zero_angle_gates_are_active(self):
        """Any non-barrier, non-delay instruction counts, even a trivial one."""
        circuit = QuantumCircuit(3)
        circuit.id(1)
        circuit.rz(0.0, 2)
        self.assertEqual(get_active_qubits(circuit), [1, 2])

    def test_final_measurements_are_ignored(self):
        circuit = QuantumCircuit(3, 3)
        circuit.sx(0)
        circuit.measure([0, 1, 2], [0, 1, 2])
        self.assertEqual(get_active_qubits(circuit), [0])

    def test_does_not_modify_the_circuit(self):
        circuit = QuantumCircuit(2, 2)
        circuit.sx(0)
        circuit.measure([0, 1], [0, 1])
        get_active_qubits(circuit)
        self.assertEqual(circuit.count_ops().get("measure"), 2)


# ---------------------------------------------------------------------------
# get_measured_qubits
# ---------------------------------------------------------------------------


class TestGetMeasuredQubits(unittest.TestCase):
    """Tests for :func:`get_measured_qubits`."""

    def test_union_of_observable_support_and_active_qubits(self):
        circuit = QuantumCircuit(5)
        circuit.sx(0)
        circuit.cz(0, 1)
        result = get_measured_qubits([circuit], [[SparsePauliOp("ZIIII")]])
        self.assertEqual(result, [0, 1, 4])

    def test_observable_support_on_an_idle_qubit_is_included(self):
        circuit = QuantumCircuit(3)
        circuit.sx(0)
        result = get_measured_qubits([circuit], [[SparsePauliOp("IZI")]])
        self.assertEqual(result, [0, 1])

    def test_union_across_circuits(self):
        first = QuantumCircuit(4)
        first.sx(0)
        second = QuantumCircuit(4)
        second.sx(3)
        result = get_measured_qubits(
            [first, second], [[SparsePauliOp("IIZI")], [SparsePauliOp("IIII")]]
        )
        self.assertEqual(result, [0, 1, 3])

    def test_projector_observables_are_included(self):
        """Projector terms (given as dict observables) count towards the observable support."""
        circuit = QuantumCircuit(3)
        observables = ObservablesArray([{"1II": 1.0, "IIr": 0.5}])
        self.assertEqual(get_measured_qubits([circuit], [observables]), [0, 2])

    def test_accepts_observables_arrays(self):
        circuit = QuantumCircuit(3)
        circuit.sx(0)
        observables = ObservablesArray([SparsePauliOp("IIZ"), SparsePauliOp("ZII")])
        self.assertEqual(get_measured_qubits([circuit], [observables]), [0, 2])

    def test_result_is_sorted_without_duplicates(self):
        circuit = QuantumCircuit(4)
        circuit.sx(3)
        circuit.sx(1)
        result = get_measured_qubits([circuit], [[SparsePauliOp("ZIII")]])
        self.assertEqual(result, sorted(set(result)))

    def test_raises_when_observables_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "observable arrays"):
            get_measured_qubits([QuantumCircuit(2), QuantumCircuit(2)], [[SparsePauliOp("ZI")]])

    def test_twirling_strategy_all_returns_every_qubit_of_the_widest_circuit(self):
        circuits = [QuantumCircuit(2), QuantumCircuit(5)]
        observables = [[SparsePauliOp("ZI")], [SparsePauliOp("IIIIZ")]]
        result = get_measured_qubits(
            circuits, observables, custom_boxing_options={"twirling_strategy": "all"}
        )
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_twirling_strategy_all_in_a_list_of_options(self):
        result = get_measured_qubits(
            [QuantumCircuit(3), QuantumCircuit(3)],
            [[SparsePauliOp("ZII")], [SparsePauliOp("ZII")]],
            custom_boxing_options=[None, {"twirling_strategy": "all"}],
        )
        self.assertEqual(result, [0, 1, 2])

    def test_other_twirling_strategies_do_not_measure_everything(self):
        circuit = QuantumCircuit(4)
        circuit.sx(0)
        result = get_measured_qubits(
            [circuit],
            [[SparsePauliOp("IIIZ")]],
            custom_boxing_options={"twirling_strategy": "active"},
        )
        self.assertEqual(result, [0])

    def test_no_circuits_returns_empty_list(self):
        self.assertEqual(get_measured_qubits([], []), [])

    def test_result_is_accepted_by_prepare_and_find_combined_unique_layers(self):
        """The helper's output is always sufficient for both pathways."""
        circuit = _three_qubit_gate_circuit()
        observables = [SparsePauliOp("IIZ")]
        meas_qubits = get_measured_qubits([circuit], [observables])
        task = MitigationTask()
        task.prepare(circuit, observables, None, meas_qubits=meas_qubits)
        self.assertEqual(task.meas_qubits, meas_qubits)
        layers = find_combined_unique_layers([circuit], meas_qubits=meas_qubits)
        self.assertTrue(len(layers) > 0)


# ---------------------------------------------------------------------------
# validate_meas_qubits
# ---------------------------------------------------------------------------


class TestValidateMeasQubits(unittest.TestCase):
    """Tests for :func:`validate_meas_qubits`."""

    def setUp(self):
        self.circuit = QuantumCircuit(4)
        self.circuit.sx(0)
        self.circuit.cz(1, 2)

    def test_returns_sorted_unique_qubits(self):
        self.assertEqual(validate_meas_qubits(self.circuit, [3, 2, 1, 1, 0]), [0, 1, 2, 3])

    def test_accepts_numpy_integers(self):
        self.assertEqual(validate_meas_qubits(self.circuit, np.array([0, 1, 2])), [0, 1, 2])

    def test_unmeasured_active_qubits_raise(self):
        with self.assertRaisesRegex(ValueError, r"Qubits \[1, 2\].*get_measured_qubits"):
            validate_meas_qubits(self.circuit, [0])

    def test_unmeasured_idle_qubits_are_allowed(self):
        self.assertEqual(validate_meas_qubits(self.circuit, [0, 1, 2]), [0, 1, 2])

    def test_negative_index_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[-1\].*out of range"):
            validate_meas_qubits(self.circuit, [-1, 0, 1, 2])

    def test_index_too_large_raises(self):
        with self.assertRaisesRegex(ValueError, r"\[4\].*out of range"):
            validate_meas_qubits(self.circuit, [0, 1, 2, 4])

    def test_twirling_strategy_all_requires_every_qubit(self):
        with self.assertRaisesRegex(ValueError, "3 of the 4"):
            validate_meas_qubits(self.circuit, [0, 1, 2], {"twirling_strategy": "all"})

    def test_twirling_strategy_all_accepts_every_qubit(self):
        result = validate_meas_qubits(self.circuit, [3, 2, 1, 0], {"twirling_strategy": "all"})
        self.assertEqual(result, [0, 1, 2, 3])

    def test_boxing_options_without_strategy_use_the_active_qubits_check(self):
        with self.assertRaisesRegex(ValueError, "not in ``meas_qubits``"):
            validate_meas_qubits(self.circuit, [0], {"enable_measures": True})


# ---------------------------------------------------------------------------
# TREX.create_instance_from_passthrough_data — direct unit test
# ---------------------------------------------------------------------------


class TestTREXCreateInstanceFromPassthroughData(unittest.TestCase):
    """Direct unit tests for :meth:`TREX.create_instance_from_passthrough_data`."""

    def test_happy_path_returns_trex_instance(self):
        """A passthrough dict with ``mitigation='trex'`` must return a TREX instance."""
        passthrough = {"mitigation": "trex", "version": "0.1", "program_item_index": 3}
        trex_task = TREX.create_instance_from_passthrough_data(passthrough)
        self.assertIsInstance(trex_task, TREX)
        self.assertEqual(trex_task._program_item_index, 3)

    def test_raises_when_mitigation_is_not_trex(self):
        """A passthrough dict with a mitigation type other than ``'trex'`` must raise ``ValueError``."""
        with self.assertRaises(ValueError, msg="Should raise for wrong mitigation type"):
            TREX.create_instance_from_passthrough_data({"mitigation": "gate_folding"})

    def test_raises_when_mitigation_key_missing(self):
        """A passthrough dict with no ``'mitigation'`` key must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            TREX.create_instance_from_passthrough_data({})


if __name__ == "__main__":
    unittest.main()
