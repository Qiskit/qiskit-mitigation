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

"""Shared helpers for the ``trim_measurements`` tests of the different task classes.

The behavior that every task class must satisfy is written once in
:class:`TrimMeasurementsTestsMixin`. The test module of each class subclasses it with its own
``task_cls``, so that the tests live next to the other tests of that class.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp
from qiskit_mitigation import PEA, PEC, TREX, GateFolding, MitigationTask
from qiskit_mitigation.utils.utils import find_combined_unique_layers, load_tasks_from_result
from samplomatic import InjectNoise
from samplomatic.quantum_program import QuantumProgramResult
from samplomatic.utils import get_annotation

SHOTS = 8
RANDOMIZATIONS = 2


def identity_noise_maps(circuits, task_cls, observables, trim, custom_boxing_options=None):
    """Return zero-noise maps for the layers that ``prepare`` is going to look for."""
    layers = find_combined_unique_layers(
        circuits,
        [task_cls() for _ in circuits],
        custom_boxing_options=custom_boxing_options,
        observables=observables if trim else None,
    )
    return {
        get_annotation(layer.operation, InjectNoise).ref: PauliLindbladMap.identity(
            len(layer.qubits)
        )
        for layer in layers
        if get_annotation(layer.operation, InjectNoise) is not None
    }


def layer_refs(layers):
    """Return the noise-injection refs of the given layers."""
    return {
        get_annotation(layer.operation, InjectNoise).ref
        for layer in layers
        if get_annotation(layer.operation, InjectNoise) is not None
    }


def prepare_task(task_cls, circuit, observables, trim, parameters=None, **kwargs):
    """Prepare a task of any class with the arguments that class needs."""
    task = task_cls()
    options = {
        "shots_per_randomization": SHOTS,
        "num_randomizations": RANDOMIZATIONS,
        "trim_measurements": trim,
    }
    # ``noise_maps`` can be given by the caller, for the classes that use it
    noise_maps = kwargs.pop("noise_maps", None)
    if task_cls in (PEC, PEA):
        options["noise_maps"] = (
            noise_maps
            if noise_maps is not None
            else identity_noise_maps(
                [circuit], task_cls, [observables], trim, kwargs.get("custom_boxing_options")
            )
        )
    if task_cls in (PEA, GateFolding):
        options.update(noise_factors=(1, 3), extrapolator=["linear"])
    options.update(kwargs)
    program = task.prepare(circuit, observables, parameters, **options)
    return task, program


def simulate_item(item):
    """Sample the item's samplex and run its circuits on a noiseless simulator.

    Returns an item result with ``_meas`` and, if present, ``pauli_signs``.
    """
    args = dict(item.samplex_arguments)
    basis_keys = [key for key in args if key.startswith("basis_changes.")]
    scale_keys = [key for key in args if key.startswith("noise_scales.")]
    rest = {k: v for k, v in args.items() if k not in basis_keys and k not in scale_keys}
    has_factor_axis = bool(scale_keys) and np.ndim(args[scale_keys[0]]) > 0
    num_factors = np.shape(args[scale_keys[0]])[0] if has_factor_axis else 1
    if basis_keys:
        all_bases = np.asarray(args[basis_keys[0]])
        all_bases = all_bases.reshape(-1, all_bases.shape[-1])
    else:
        all_bases = [None]
    num_bits = item.circuit.cregs[0].size
    data = np.zeros((num_factors, RANDOMIZATIONS, len(all_bases), SHOTS, num_bits), dtype=bool)
    signs = np.zeros((num_factors, RANDOMIZATIONS, len(all_bases), 1), dtype=bool)
    has_signs = False
    for factor in range(num_factors):
        for basis_index, bases in enumerate(all_bases):
            sample_args = dict(rest)
            for key in scale_keys:
                sample_args[key] = (
                    np.asarray(args[key])[factor].reshape(()) if has_factor_axis else args[key]
                )
            if bases is not None:
                sample_args[basis_keys[0]] = bases
            outputs = item.samplex.sample(sample_args, num_randomizations=RANDOMIZATIONS)
            if "pauli_signs" in outputs:
                has_signs = True
                signs[factor, :, basis_index] = np.asarray(outputs["pauli_signs"]).reshape(
                    RANDOMIZATIONS, -1
                )
            for randomization in range(RANDOMIZATIONS):
                bound = item.circuit.assign_parameters(outputs["parameter_values"][randomization])
                bitstrings = (
                    StatevectorSampler()
                    .run([bound], shots=SHOTS)
                    .result()[0]
                    .data["_meas"]
                    .get_bitstrings()
                )
                # bitstrings are little-endian: the last character is the first register bit
                data[factor, randomization, basis_index] = [
                    [c == "1" for c in bits[::-1]] for bits in bitstrings
                ]
    result = {"_meas": data if has_factor_axis else data[0]}
    if has_signs:
        result["pauli_signs"] = signs[0]
    return result


def simulated_result(program):
    """Return the :class:`QuantumProgramResult` of running every item of ``program``."""
    return QuantumProgramResult(
        [simulate_item(item) for item in program.items], passthrough_data=program.passthrough_data
    )


def postprocess_evs(task, result):
    """Postprocess ``result`` with ``task`` and return the expectation values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return task.postprocess(result).data.evs


def eigenstate_circuit(num_qubits=4):
    """q0 = |1>, q3 = |1> and q2 = |1> (copied from q3); q1 is idle."""
    circuit = QuantumCircuit(num_qubits)
    circuit.x(0)
    circuit.x(3)
    circuit.cx(3, 2)
    return circuit


class TrimMeasurementsTestsMixin:
    """Tests of ``trim_measurements`` that every task class must pass.

    Subclass it together with :class:`unittest.TestCase` and set ``task_cls``.
    """

    task_cls: ClassVar[type]

    # Pauli labels are little-endian: the last character acts on qubit 0.
    OBSERVABLES: ClassVar[list[SparsePauliOp]] = [
        SparsePauliOp("IIIZ"),
        SparsePauliOp("ZIII"),
        SparsePauliOp("ZZII"),
    ]
    EXPECTED: ClassVar[list[float]] = [-1.0, -1.0, 1.0]

    def _prepare(self, circuit, observables, trim, **kwargs):
        return prepare_task(self.task_cls, circuit, observables, trim, **kwargs)

    # -- default behavior ----------------------------------------------------------------------

    def test_default_measures_all_qubits_and_stores_no_meas_qubits(self):
        task, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=False)
        self.assertIsNone(task.meas_qubits)
        self.assertIsNone(task.circuit_num_qubits)
        for item in program.items:
            self.assertEqual(item.circuit.cregs[0].size, 4)
        for entry in program.passthrough_data["qiskit_mitigation"]:
            self.assertIsNone(entry["meas_qubits"])
            self.assertIsNone(entry["circuit_num_qubits"])

    def test_new_version_is_written_to_the_passthrough_data(self):
        _, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=False)
        self.assertEqual(program.passthrough_data["qiskit_mitigation"][0]["version"], "0.2")

    def test_twirling_strategy_all_is_fine_without_trimming(self):
        task, _ = self._prepare(
            eigenstate_circuit(),
            self.OBSERVABLES,
            trim=False,
            custom_boxing_options={"twirling_strategy": "all"},
        )
        self.assertIsNone(task.meas_qubits)

    # -- trimmed execution ---------------------------------------------------------------------

    def test_measured_qubits_are_active_plus_observed(self):
        task, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=True)
        self.assertEqual(task.meas_qubits, [0, 2, 3])
        self.assertEqual(task.circuit_num_qubits, 4)
        for item in program.items:
            self.assertEqual(item.circuit.cregs[0].size, 3)

    def test_observed_idle_qubit_is_measured(self):
        task, _ = self._prepare(eigenstate_circuit(), [SparsePauliOp("IIZI")], trim=True)
        self.assertEqual(task.meas_qubits, [0, 1, 2, 3])

    def test_trimmed_and_full_width_agree(self):
        circuit = eigenstate_circuit()
        trimmed_task, trimmed_program = self._prepare(circuit, self.OBSERVABLES, trim=True)
        full_task, full_program = self._prepare(circuit, self.OBSERVABLES, trim=False)
        np.testing.assert_allclose(
            postprocess_evs(trimmed_task, simulated_result(trimmed_program)),
            self.EXPECTED,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            postprocess_evs(full_task, simulated_result(full_program)), self.EXPECTED, atol=1e-9
        )

    def test_non_contiguous_qubits_with_different_bases(self):
        """Every measured qubit gets the basis change of its own circuit qubit."""
        circuit = QuantumCircuit(5)
        circuit.x(0)  # Z = -1
        circuit.x(1)
        circuit.h(1)  # |->, X = -1
        circuit.h(3)  # |+>, X = +1
        circuit.cx(0, 4)  # an entangling layer; qubit 4 is |1>. Qubit 2 stays idle
        observables = [SparsePauliOp(label) for label in ["IIIIZ", "IIIXI", "IXIII", "IXIXZ"]]
        task, program = self._prepare(circuit, observables, trim=True)
        self.assertEqual(task.meas_qubits, [0, 1, 3, 4])
        np.testing.assert_allclose(
            postprocess_evs(task, simulated_result(program)), [-1, -1, 1, 1], atol=1e-9
        )

    # -- loading -------------------------------------------------------------------------------

    def test_postprocess_reads_the_stored_meas_qubits_after_loading(self):
        _, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=True)
        result = simulated_result(program)
        (restored,) = load_tasks_from_result(result)
        self.assertIsInstance(restored, self.task_cls)
        self.assertEqual(restored.meas_qubits, [0, 2, 3])
        self.assertEqual(restored.circuit_num_qubits, 4)
        np.testing.assert_allclose(postprocess_evs(restored, result), self.EXPECTED, atol=1e-9)

    def test_version_0_1_without_trimming_keys_loads_and_postprocesses(self):
        _, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=False)
        result = simulated_result(program)
        entry = result.passthrough_data["qiskit_mitigation"][0]
        entry["version"] = "0.1"
        del entry["meas_qubits"]
        del entry["circuit_num_qubits"]
        (restored,) = load_tasks_from_result(result)
        self.assertIsNone(restored.meas_qubits)
        np.testing.assert_allclose(postprocess_evs(restored, result), self.EXPECTED, atol=1e-9)

    def test_unknown_version_raises_on_postprocess(self):
        _, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=False)
        result = simulated_result(program)
        result.passthrough_data["qiskit_mitigation"][0]["version"] = "9.9"
        (restored,) = load_tasks_from_result(result)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            restored.postprocess(result)

    # -- failures ------------------------------------------------------------------------------

    def _check_prepare_fails_cleanly(self, circuit, observables, regex, **kwargs):
        """``prepare`` raises and leaves the TREX instance and the program untouched."""
        trex = TREX()
        program = MitigationTask().prepare(
            eigenstate_circuit(),
            [SparsePauliOp("IIIZ")],
            None,
            shots_per_randomization=SHOTS,
            num_randomizations=RANDOMIZATIONS,
        )
        num_items = len(program.items)
        num_entries = len(program.passthrough_data["qiskit_mitigation"])
        extra = {}
        if self.task_cls in (PEC, PEA):
            extra["noise_maps"] = {}
        if self.task_cls in (PEA, GateFolding):
            extra.update(noise_factors=(1, 3))
        with self.assertRaisesRegex(ValueError, regex):
            self.task_cls().prepare(
                circuit,
                observables,
                None,
                trex=trex,
                quantum_program=program,
                shots_per_randomization=SHOTS,
                trim_measurements=True,
                **extra,
                **kwargs,
            )
        self.assertEqual(trex.tasks, [])
        self.assertEqual(len(program.items), num_items)
        self.assertEqual(len(program.passthrough_data["qiskit_mitigation"]), num_entries)

    def test_empty_measured_set_raises(self):
        self._check_prepare_fails_cleanly(
            QuantumCircuit(3), [SparsePauliOp("III")], "no qubit is active"
        )

    def test_twirling_strategy_all_warns(self):
        """``twirling_strategy="all"`` with trimming now warns instead of raising."""
        with self.assertWarns(UserWarning):
            self._prepare(
                eigenstate_circuit(),
                [SparsePauliOp("IIIZ")],
                trim=True,
                custom_boxing_options={"twirling_strategy": "all"},
            )

    def test_failed_prepare_does_not_widen_the_trex_calibration(self):
        """A task that failed to prepare must not widen the calibration of the others."""
        trex = TREX()
        wide = QuantumCircuit(8)
        wide.x(7)
        # Use an all-identity observable on a circuit with no instructions to force the
        # "no qubit is active or observed" error, which still raises.
        with self.assertRaises(ValueError):
            self._prepare(
                QuantumCircuit(8),
                [SparsePauliOp("I" * 8)],
                trim=True,
                trex=trex,
                noise_maps={},
            )
        self.assertEqual(trex.tasks, [])
        _, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=True, trex=trex)
        trex.prepare(RANDOMIZATIONS, program)
        self.assertEqual(program.items[-1].circuit.cregs[0].size, 4)

    # -- TREX ----------------------------------------------------------------------------------

    def test_trex_calibration_is_the_prefix_up_to_the_highest_measured_qubit(self):
        trex = TREX()
        task, program = self._prepare(eigenstate_circuit(), self.OBSERVABLES, trim=True, trex=trex)
        self.assertEqual(trex.tasks, [task])
        trex.prepare(RANDOMIZATIONS, program)
        # the highest measured qubit is 3, so the calibration measures the qubits 0 to 3
        self.assertEqual(program.items[-1].circuit.cregs[0].size, 4)


class LearnedLayersTestsMixin:
    """Noise learned with a different choice of trimming is not found by ``prepare``.

    For the task classes that consume learned noise (PEC and PEA). Set ``task_cls``.
    """

    task_cls: ClassVar[type]

    def setUp(self):
        self.circuit = eigenstate_circuit()
        self.observables = [SparsePauliOp("IIIZ")]

    def _prepare_with_maps(self, learned_trimmed, prepare_trimmed):
        maps = identity_noise_maps(
            [self.circuit], self.task_cls, [self.observables], learned_trimmed
        )
        return prepare_task(
            self.task_cls, self.circuit, self.observables, prepare_trimmed, noise_maps=maps
        )

    def test_matching_choices_work(self):
        for trimmed in (False, True):
            with self.subTest(trimmed=trimmed):
                self._prepare_with_maps(trimmed, trimmed)

    def test_mismatch_in_both_directions_fails_loudly(self):
        for learned, prepared in ((False, True), (True, False)):
            with (
                self.subTest(learned=learned, prepared=prepared),
                self.assertRaisesRegex(ValueError, "missing"),
            ):
                self._prepare_with_maps(learned, prepared)

    def test_method_and_function_agree(self):
        from_function = find_combined_unique_layers(
            [self.circuit], [self.task_cls()], observables=[self.observables]
        )
        from_method = self.task_cls().find_unique_layers(self.circuit, observables=self.observables)
        self.assertEqual(layer_refs(from_function), layer_refs(from_method))
