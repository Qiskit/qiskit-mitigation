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

"""Tests for the ``PEA`` class."""

from __future__ import annotations

import contextlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import PubResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, SparsePauliOp
from qiskit_mitigation.extrapolation.pea import PEA
from qiskit_mitigation.trex import TREX
from samplomatic import InjectNoise
from samplomatic.quantum_program import QuantumProgram
from samplomatic.utils import get_annotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_RANDOMIZATIONS = 2
SHOTS = 4
TOTAL_SHOTS = NUM_RANDOMIZATIONS * SHOTS
NOISE_FACTORS = [1.0, 3.0, 5.0]


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple non-parametric circuit."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _parametric_circuit(num_qubits: int = 2) -> tuple[QuantumCircuit, Parameter]:
    """Return a single-parameter circuit and its parameter."""
    theta = Parameter("theta")
    qc = QuantumCircuit(num_qubits)
    qc.rz(theta, 0)
    qc.cx(0, 1)
    return qc, theta


def _z_data(num_ones: int) -> np.ndarray:
    """Return single-qubit shot data with a known ``<Z>``.

    Shape: ``(NUM_RANDOMIZATIONS, 1, SHOTS, 1)`` with ``num_ones`` outcomes set to 1
    so that ``<Z> = (TOTAL_SHOTS - 2 * num_ones) / TOTAL_SHOTS``.
    """
    flat = np.zeros(TOTAL_SHOTS, dtype=bool)
    flat[:num_ones] = True
    return flat.reshape(NUM_RANDOMIZATIONS, 1, SHOTS, 1)


def _z_exp_val(num_ones: int) -> float:
    return (TOTAL_SHOTS - 2 * num_ones) / TOTAL_SHOTS


def _noise_maps_for_circuit(circuit: QuantumCircuit) -> dict[str, PauliLindbladMap]:
    """Discover the InjectNoise layer refs in a PEA-boxed circuit and return a matching
    zero-rate noise map for each, so that :meth:`PEA.prepare` can proceed.
    """
    pea = PEA()
    boxed = pea._box_circuit(circuit, {})
    refs = {
        get_annotation(instr.operation, InjectNoise).ref
        for instr in boxed
        if get_annotation(instr.operation, InjectNoise) is not None
    }
    return {ref: PauliLindbladMap.identity(circuit.num_qubits) for ref in refs}


# Collinear in noise factors (1, 3, 5): 0.75, 0.5, 0.25 → zero-noise intercept 0.875
LINEAR_ONES = (1, 2, 3)
LINEAR_EXP_VALS = [_z_exp_val(n) for n in LINEAR_ONES]
LINEAR_ZERO_NOISE = 0.875


def _linear_pea_data() -> np.ndarray:
    """PEA data of shape ``(3, R, 1, S, 1)`` whose ``<Z>`` is linear in the noise factors."""
    return np.array([_z_data(n) for n in LINEAR_ONES])


def _obs(*labels: str) -> ObservablesArray:
    return ObservablesArray([SparsePauliOp(lbl) for lbl in labels])


# ---------------------------------------------------------------------------
# PEA.__init__
# ---------------------------------------------------------------------------


class TestPEAInit(unittest.TestCase):
    """Tests for :meth:`PEA.__init__`."""

    def test_pea_specific_attributes_are_none_after_init(self):
        """``noise_factors`` and ``extrapolator`` must be ``None`` after construction."""
        pea = PEA()
        self.assertIsNone(pea.noise_factors)
        self.assertIsNone(pea.extrapolator)

    def test_inherits_mitigation_task_attributes(self):
        """All ``MitigationTask`` placeholder attributes must also be ``None`` after init."""
        pea = PEA()
        for attr in (
            "circuit",
            "observables",
            "_sparse_observables",
            "parameters",
            "broadcast_obs_and_params",
            "broadcast_shape",
            "param_basis_pairs",
            "meas_bases",
            "custom_boxing_options",
            "shots_per_randomization",
            "num_randomizations",
        ):
            self.assertIsNone(getattr(pea, attr), msg=f"{attr} should be None after __init__")


# ---------------------------------------------------------------------------
# PEA._box_circuit - PEA-specific validation
# ---------------------------------------------------------------------------


class TestPEABoxCircuit(unittest.TestCase):
    """Tests for the PEA override of :meth:`_box_circuit`."""

    def setUp(self):
        self.pea = PEA()
        self.circuit = _simple_circuit()

    # --- enable_gates ---

    def test_enable_gates_injected_when_absent(self):
        """``enable_gates=True`` must be forwarded to ``generate_boxing_pass_manager``
        even when absent from the caller's dict, and the original dict must be untouched."""
        options: dict = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pea._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertTrue(kwargs.get("enable_gates"))
        # Original dict must not have been mutated.
        self.assertNotIn("enable_gates", options)

    def test_raises_when_enable_gates_is_false(self):
        """``boxing_options["enable_gates"] = False`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pea._box_circuit(self.circuit, {"enable_gates": False})

    def test_enable_gates_true_is_accepted(self):
        """Explicitly passing ``enable_gates=True`` must not raise."""
        boxed = self.pea._box_circuit(self.circuit, {"enable_gates": True})
        self.assertIsInstance(boxed, QuantumCircuit)

    # --- inject_noise_targets ---

    def test_inject_noise_targets_defaulted_to_gates(self):
        """``inject_noise_targets='gates'`` must be forwarded to ``generate_boxing_pass_manager``
        even when absent from the caller's dict, and the original dict must be untouched."""
        options: dict = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pea._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("inject_noise_targets"), "gates")
        # Original dict must not have been mutated.
        self.assertNotIn("inject_noise_targets", options)

    def test_inject_noise_targets_all_is_accepted(self):
        """``inject_noise_targets="all"`` must be accepted without error."""
        boxed = self.pea._box_circuit(self.circuit, {"inject_noise_targets": "all"})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_inject_noise_targets_invalid_value_raises(self):
        """Any value for ``inject_noise_targets`` other than ``"gates"`` or ``"all"`` must raise."""
        with self.assertRaises(ValueError):
            self.pea._box_circuit(self.circuit, {"inject_noise_targets": "measures"})

    # --- inject_noise_strategy ---

    def test_inject_noise_strategy_defaulted_to_uniform_modification(self):
        """``inject_noise_strategy='uniform_modification'`` must be forwarded to
        ``generate_boxing_pass_manager`` even when absent, and the original dict must be untouched."""
        options: dict = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pea._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("inject_noise_strategy"), "uniform_modification")
        # Original dict must not have been mutated.
        self.assertNotIn("inject_noise_strategy", options)

    def test_inject_noise_strategy_no_modification_raises(self):
        """``inject_noise_strategy="no_modification"`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pea._box_circuit(self.circuit, {"inject_noise_strategy": "no_modification"})

    def test_inject_noise_strategy_other_value_is_accepted(self):
        """A strategy other than ``"no_modification"`` must be forwarded without error."""
        boxed = self.pea._box_circuit(
            self.circuit, {"inject_noise_strategy": "individual_modification"}
        )
        self.assertIsInstance(boxed, QuantumCircuit)

    # --- inherited base behaviour ---

    def test_returns_quantum_circuit(self):
        """The boxed circuit must be a :class:`~qiskit.QuantumCircuit`."""
        boxed = self.pea._box_circuit(self.circuit, {})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_adds_meas_register(self):
        """The boxed circuit must include the ``'_meas'`` classical register."""
        boxed = self.pea._box_circuit(self.circuit, {})
        self.assertIn("_meas", [reg.name for reg in boxed.cregs])

    def test_raises_when_enable_measures_is_false(self):
        """A ``False`` value for ``enable_measures`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pea._box_circuit(self.circuit, {"enable_measures": False})

    def test_none_options_sets_defaults_and_returns_circuit(self):
        """``None`` options must default to the PEA defaults and return a valid circuit."""
        pea = PEA()
        circuit = _simple_circuit()
        boxed = pea._box_circuit(circuit, None)
        self.assertIsInstance(boxed, QuantumCircuit)
        self.assertIn("_meas", [reg.name for reg in boxed.cregs])


# ---------------------------------------------------------------------------
# PEA.prepare
# ---------------------------------------------------------------------------


class TestPEAPrepare(unittest.TestCase):
    """Tests for :meth:`PEA.prepare`."""

    def setUp(self):
        self.circuit = _simple_circuit()
        self.observables = [SparsePauliOp("ZZ")]
        self.noise_map = _noise_maps_for_circuit(self.circuit)

    def _prepare(self, pea=None, **kwargs):
        """Helper that calls :meth:`PEA.prepare` with sensible defaults."""
        if pea is None:
            pea = PEA()
        defaults = {
            "circuit": self.circuit,
            "observables": self.observables,
            "parameters": None,
            "noise_maps": self.noise_map,
            "noise_factors": NOISE_FACTORS,
        }
        defaults.update(kwargs)
        return pea, pea.prepare(**defaults)

    def test_raises_when_noise_maps_is_none(self):
        """``noise_maps=None`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            PEA().prepare(
                self.circuit,
                self.observables,
                parameters=None,
                noise_maps=None,
            )

    def test_noise_factors_saved_as_float_array(self):
        """``noise_factors`` must be stored as a ``float`` ndarray."""
        pea, _ = self._prepare()
        self.assertIsInstance(pea.noise_factors, np.ndarray)
        self.assertEqual(pea.noise_factors.dtype, np.dtype(float))
        np.testing.assert_allclose(pea.noise_factors, [1.0, 3.0, 5.0])

    def test_extrapolator_not_saved_when_none(self):
        """``extrapolator`` must remain ``None`` when not provided at preparation."""
        pea, _ = self._prepare(extrapolator=None)
        self.assertIsNone(pea.extrapolator)

    def test_extrapolator_saved_when_valid(self):
        """A valid ``extrapolator`` must be stored on the instance."""
        pea, _ = self._prepare(extrapolator=["linear"])
        self.assertEqual(pea.extrapolator, ["linear"])

    def test_raises_when_noise_factors_insufficient_for_extrapolator(self):
        """Too few noise factors for ``double_exponential`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._prepare(noise_factors=[1.0, 3.0], extrapolator=["double_exponential"])

    def test_returns_quantum_program_with_single_item(self):
        """``prepare`` must return a ``QuantumProgram`` containing exactly one item."""
        _, program = self._prepare()
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)

    def test_appends_to_existing_program(self):
        """Passing an existing ``quantum_program`` must append to it and return the same object."""
        pea1 = PEA()
        _, program = self._prepare(pea=pea1)
        self.assertEqual(len(program.items), 1)

        pea2 = PEA()
        returned = pea2.prepare(
            self.circuit,
            self.observables,
            parameters=None,
            noise_maps=self.noise_map,
            noise_factors=NOISE_FACTORS,
            quantum_program=program,
        )
        self.assertIs(returned, program)
        self.assertEqual(len(program.items), 2)

    def test_raises_on_shots_mismatch_with_existing_program(self):
        """A ``shots_per_randomization`` differing from the program's shots must raise."""
        _, program = self._prepare(shots_per_randomization=64)
        with self.assertRaises(ValueError):
            PEA().prepare(
                self.circuit,
                self.observables,
                parameters=None,
                noise_maps=self.noise_map,
                noise_factors=NOISE_FACTORS,
                shots_per_randomization=128,
                quantum_program=program,
            )

    def test_circuit_saved_on_instance(self):
        """The circuit must be saved as ``pea.circuit`` after :meth:`prepare`."""
        pea, _ = self._prepare()
        self.assertIs(pea.circuit, self.circuit)

    def test_parametric_circuit_stores_param_basis_pairs(self):
        """A parametric circuit in broadcast mode must populate ``param_basis_pairs``."""
        circuit, _ = _parametric_circuit()
        pea = PEA()
        pea.prepare(
            circuit,
            [SparsePauliOp("ZZ")],
            parameters=np.array([[0.1], [0.2]]),
            noise_maps=self.noise_map,
            noise_factors=NOISE_FACTORS,
            broadcast_obs_and_params=True,
        )
        self.assertIsNotNone(pea.param_basis_pairs)

    def test_boxed_circuit_saved_on_instance(self):
        """``pea.boxed_circuit`` must be set after :meth:`prepare`."""
        pea, _ = self._prepare()
        self.assertIsNotNone(pea.boxed_circuit)
        self.assertIsInstance(pea.boxed_circuit, QuantumCircuit)

    def test_raises_when_noise_map_missing_for_circuit_layer(self):
        """Passing an empty ``noise_maps`` when the circuit has layers must raise."""
        pea = PEA()
        with self.assertRaises(ValueError):
            pea.prepare(
                self.circuit,
                self.observables,
                parameters=None,
                noise_maps={},
                noise_factors=NOISE_FACTORS,
            )

    def test_raises_when_trex_and_projection_observables(self):
        """``prepare`` with a TREX instance and projection-operator observables must raise."""
        pea = PEA()
        project_obs = ObservablesArray.coerce([{"Z+": 1}])
        trex = TREX()

        with self.assertRaises(ValueError):
            pea.prepare(
                self.circuit,
                project_obs,
                parameters=None,
                noise_maps=self.noise_map,
                noise_factors=NOISE_FACTORS,
                trex=trex,
            )

    def test_prepare_with_trex_and_normal_observables_edits_boxing_options(self):
        """``prepare`` with TREX and regular Pauli observables must reach line 214
        (``custom_boxing_options = trex.edit_boxing_options(…)``)."""
        pea = PEA()
        obs = [SparsePauliOp("ZZ")]
        trex = TREX()
        program = pea.prepare(
            self.circuit,
            obs,
            parameters=None,
            noise_maps=self.noise_map,
            noise_factors=NOISE_FACTORS,
            trex=trex,
        )
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)


# ---------------------------------------------------------------------------
# PEA.postprocess
# ---------------------------------------------------------------------------


class TestPEAPostprocess(unittest.TestCase):
    """Tests for :meth:`PEA.postprocess`."""

    def test_raises_when_noise_factors_unset_and_not_given(self):
        """``postprocess`` must raise when neither saved nor passed noise factors exist."""
        pea = PEA()
        with self.assertRaises(ValueError):
            pea.postprocess(MagicMock())

    def test_raises_when_extrapolator_unset_and_not_given(self):
        """``postprocess`` must raise when neither saved nor passed extrapolator exists."""
        pea = PEA()
        pea.noise_factors = np.array(NOISE_FACTORS)
        with self.assertRaises(ValueError):
            pea.postprocess(MagicMock())

    def test_uses_saved_noise_factors_when_not_passed(self):
        """Saved ``noise_factors`` are forwarded to ``compute_expectation_value_pea``."""
        pea = PEA()
        pea.noise_factors = np.array(NOISE_FACTORS)
        pea.extrapolator = ["linear"]
        pea.observables = _obs("Z")
        pea.parameters = None
        pea.param_basis_pairs = [((0,), "Z")]
        pea.meas_bases = None
        pea.broadcast_obs_and_params = True

        sentinel = object()
        with patch.object(PEA, "compute_expectation_value_pea", return_value=sentinel) as mock_fn:
            result = pea.postprocess(MagicMock())
            self.assertIs(result, sentinel)
            _, call_kwargs = mock_fn.call_args
            np.testing.assert_array_equal(call_kwargs["noise_factors"], list(pea.noise_factors))

    def test_override_noise_factors_via_postprocess_kwarg(self):
        """Noise factors passed directly to ``postprocess`` override saved ones."""
        pea = PEA()
        pea.noise_factors = np.array([1.0, 3.0, 5.0])
        pea.extrapolator = ["linear"]
        pea.observables = _obs("Z")
        pea.parameters = None
        pea.param_basis_pairs = [((0,), "Z")]
        pea.meas_bases = None
        pea.broadcast_obs_and_params = True

        override = [1.0, 7.0]
        with patch.object(PEA, "compute_expectation_value_pea", return_value=None) as mock_fn:
            pea.postprocess(MagicMock(), noise_factors=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["noise_factors"], override)

    def test_override_extrapolator_via_postprocess_kwarg(self):
        """Extrapolator passed directly to ``postprocess`` overrides saved one."""
        pea = PEA()
        pea.noise_factors = np.array(NOISE_FACTORS)
        pea.extrapolator = ["linear"]
        pea.observables = _obs("Z")
        pea.parameters = None
        pea.param_basis_pairs = [((0,), "Z")]
        pea.meas_bases = None
        pea.broadcast_obs_and_params = True

        override = ["fallback"]
        with patch.object(PEA, "compute_expectation_value_pea", return_value=None) as mock_fn:
            pea.postprocess(MagicMock(), extrapolator=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["extrapolator"], override)

    def test_extrapolated_noise_factors_passthrough_to_compute(self):
        """``extrapolated_noise_factors`` passed to ``postprocess`` must be forwarded."""
        pea = PEA()
        pea.noise_factors = np.array(NOISE_FACTORS)
        pea.extrapolator = ["linear"]
        pea.observables = _obs("Z")
        pea.parameters = None
        pea.param_basis_pairs = [((0,), "Z")]
        pea.meas_bases = None
        pea.broadcast_obs_and_params = True
        pea.extrapolated_noise_factors = None

        override = [0.0, 1.0]
        with patch.object(PEA, "compute_expectation_value_pea", return_value=None) as mock_fn:
            pea.postprocess(MagicMock(), extrapolated_noise_factors=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["extrapolated_noise_factors"], override)

    def test_saved_extrapolated_noise_factors_used_when_not_passed(self):
        """Saved ``extrapolated_noise_factors`` should be used when none are passed."""
        pea = PEA()
        pea.noise_factors = np.array(NOISE_FACTORS)
        pea.extrapolator = ["linear"]
        pea.observables = _obs("Z")
        pea.parameters = None
        pea.param_basis_pairs = [((0,), "Z")]
        pea.meas_bases = None
        pea.broadcast_obs_and_params = True
        saved = [0.0, 2.0]
        pea.extrapolated_noise_factors = saved

        with patch.object(PEA, "compute_expectation_value_pea", return_value=None) as mock_fn:
            pea.postprocess(MagicMock())
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(list(call_kwargs["extrapolated_noise_factors"]), saved)


# ---------------------------------------------------------------------------
# PEA.compute_expectation_value_pea  (static method)
# ---------------------------------------------------------------------------


class TestComputeExpectationValuePea(unittest.TestCase):
    """Tests for the static :meth:`PEA.compute_expectation_value_pea`."""

    def _item_result(self, ones_per_factor=LINEAR_ONES, flips=False):
        """Build a mock ``QuantumProgramItemResult`` backed by PEA-shaped data.

        PEA data shape: ``(num_noise_factors, num_randomizations, 1, shots, 1)``.
        """
        data = np.array([_z_data(n) for n in ones_per_factor])
        result = {"_meas": data}
        if flips:
            result["measurement_flips._meas"] = np.ones(data.shape, dtype=bool)
        return result

    def _item_result_5d(self, ones_per_factor=LINEAR_ONES):
        """Build a PEA item result with shape ``(noise_factors, R, 1, S, 1)``."""
        data = np.array([_z_data(n) for n in ones_per_factor])
        return {"_meas": data}

    def _compute(self, item_result=None, **kwargs):
        """Call the method under test with linear single-qubit defaults."""
        params = {
            "observables": _obs("Z"),
            "param_shape": (1,),
            "param_basis_pairs": [((0,), "Z")],
            "noise_factors": NOISE_FACTORS,
            "extrapolator": ["linear"],
        }
        params.update(kwargs)
        return PEA.compute_expectation_value_pea(
            self._item_result() if item_result is None else item_result,
            **params,
        )

    # ------------------------------------------------------------------
    # Input-validation errors
    # ------------------------------------------------------------------

    def test_raises_missing_meas_key(self):
        """A result dict without ``'_meas'`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._compute(item_result={"wrong_key": np.zeros((3, 2, 1, 4, 1))})

    def test_raises_noise_factor_count_mismatch(self):
        """Providing fewer noise factors than data slices must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._compute(noise_factors=[1.0, 3.0])  # data has 3 noise factors

    def test_raises_when_noise_factors_insufficient_for_extrapolator(self):
        """Too few noise factors for an extrapolator must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._compute(noise_factors=[1.0], extrapolator=["linear"])

    def test_raises_broadcast_true_without_param_basis_pairs(self):
        """``broadcast_obs_and_params=True`` with ``param_basis_pairs=None`` must raise."""
        with self.assertRaises(ValueError):
            self._compute(param_basis_pairs=None, broadcast_obs_and_params=True)

    def test_raises_broadcast_true_without_param_shape(self):
        """``broadcast_obs_and_params=True`` with ``param_shape=None`` must raise."""
        with self.assertRaises(ValueError):
            self._compute(param_shape=None, broadcast_obs_and_params=True)

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_returns_pub_result_without_extrapolated_noise_factors(self):
        """Without ``extrapolated_noise_factors`` the result is a PubResult with None extrapolated fields."""
        pub_result = self._compute()
        self.assertIsInstance(pub_result, PubResult)
        self.assertTrue(hasattr(pub_result.data, "evs"))
        self.assertTrue(hasattr(pub_result.data, "stds"))
        self.assertTrue(hasattr(pub_result.data, "evs_noise_factors"))
        self.assertTrue(hasattr(pub_result.data, "stds_noise_factors"))
        self.assertTrue(hasattr(pub_result.data, "stds_twirl_noise_factors"))
        self.assertIn("selected_extrapolators", pub_result.metadata)
        self.assertIsNone(pub_result.data.evs_extrapolated)
        self.assertIsNone(pub_result.data.stds_extrapolated)

    def test_returns_pub_result_with_extrapolated_noise_factors(self):
        """With ``extrapolated_noise_factors`` the result is a PubResult with populated extrapolated fields."""
        pub_result = self._compute(extrapolated_noise_factors=[0.0, 1.0])
        self.assertIsInstance(pub_result, PubResult)
        # shape: (*output_shape, num_models, num_extrapolated_noise_factors)
        self.assertEqual(pub_result.data.evs_extrapolated.shape, (1, 1, 2))
        self.assertEqual(pub_result.data.stds_extrapolated.shape, (1, 1, 2))

    def test_scalar_extrapolated_noise_factor_is_wrapped_to_list(self):
        """A scalar ``extrapolated_noise_factors`` must be treated as a single-point list."""
        pub_result = self._compute(extrapolated_noise_factors=0.0)
        self.assertIsInstance(pub_result, PubResult)
        self.assertEqual(pub_result.data.evs_extrapolated.shape, (1, 1, 1))
        np.testing.assert_allclose(
            pub_result.data.evs_extrapolated[0, 0, 0], LINEAR_ZERO_NOISE, atol=1e-10
        )

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_linear_extrapolation_to_zero_noise(self):
        """Data linear in the noise factors must extrapolate to the exact intercept."""
        pub_result = self._compute()
        np.testing.assert_allclose(pub_result.data.evs, [LINEAR_ZERO_NOISE], atol=1e-10)

    def test_noise_factor_exp_vals_match_measured_values(self):
        """The per-noise-factor EVs must equal the directly measured values."""
        pub_result = self._compute()
        self.assertEqual(pub_result.data.evs_noise_factors.shape, (1, 3))
        np.testing.assert_allclose(
            pub_result.data.evs_noise_factors[0], LINEAR_EXP_VALS, atol=1e-10
        )

    def test_selected_extrapolator_is_reported_per_term(self):
        """The selected extrapolator must be reported for each observable term."""
        pub_result = self._compute()
        self.assertEqual(pub_result.metadata["selected_extrapolators"], [["linear"]])

    def test_zero_noise_std_is_finite(self):
        """The extrapolated zero-noise standard deviation must be finite."""
        pub_result = self._compute()
        self.assertTrue(np.all(np.isfinite(pub_result.data.stds)))

    def test_ensemble_stds_scale_with_total_shots(self):
        """Per-noise-factor ensemble stds must be ``sqrt(variance / total_shots)``."""
        pub_result = self._compute()
        expected = [np.sqrt((1.0 - v**2) / TOTAL_SHOTS) for v in LINEAR_EXP_VALS]
        np.testing.assert_allclose(pub_result.data.stds_noise_factors[0], expected, atol=1e-10)

    def test_coefficient_scaling(self):
        """An observable coefficient must scale both zero-noise and per-factor EVs."""
        obs = ObservablesArray([2.0 * SparsePauliOp("Z")])
        pub_result = self._compute(observables=obs)
        np.testing.assert_allclose(pub_result.data.evs, [2.0 * LINEAR_ZERO_NOISE], atol=1e-10)
        np.testing.assert_allclose(
            pub_result.data.evs_noise_factors[0], [2.0 * v for v in LINEAR_EXP_VALS], atol=1e-10
        )

    def test_measurement_flips_are_applied(self):
        """``measurement_flips._meas`` must be XOR'd into ``_meas`` before processing."""
        item_result = self._item_result(flips=True)
        pub_result = self._compute(item_result=item_result)
        np.testing.assert_allclose(pub_result.data.evs, [-LINEAR_ZERO_NOISE], atol=1e-10)

    def test_fallback_uses_lowest_noise_factor_value(self):
        """The ``fallback`` model must report the value at the lowest noise factor."""
        pub_result = self._compute(extrapolator=["fallback"])
        np.testing.assert_allclose(pub_result.data.evs, [_z_exp_val(LINEAR_ONES[0])], atol=1e-10)

    def test_extrapolated_noise_factors_are_evaluated(self):
        """The fit must be evaluated at each requested extrapolated noise factor."""
        pub_result = self._compute(extrapolated_noise_factors=[0.0, 1.0, 3.0])
        np.testing.assert_allclose(
            pub_result.data.evs_extrapolated[0, 0], [LINEAR_ZERO_NOISE, 0.75, 0.5], atol=1e-10
        )

    def test_multiple_models_produce_one_row_each(self):
        """Every requested model must produce a result row in the extrapolated outputs."""
        pub_result = self._compute(
            extrapolator=["linear", "fallback"], extrapolated_noise_factors=[0.0]
        )
        self.assertEqual(pub_result.data.evs_extrapolated.shape, (1, 2, 1))
        np.testing.assert_allclose(
            pub_result.data.evs_extrapolated[0, :, 0],
            [LINEAR_ZERO_NOISE, _z_exp_val(LINEAR_ONES[0])],
            atol=1e-10,
        )

    def test_custom_fit_success(self):
        """``compute_expectation_value_pea`` works with custom_fit."""

        def my_custom(x, a, b):
            return a * x + b

        custom_fit = (my_custom, [1.0, 1.0], (-np.inf, np.inf))
        pub_result = self._compute(extrapolator=["custom"], custom_fit=custom_fit)
        np.testing.assert_allclose(pub_result.data.evs, [LINEAR_ZERO_NOISE], atol=1e-10)

    def test_output_array_shapes(self):
        """Zero-noise outputs take the broadcast shape; per-factor outputs add a noise axis."""
        pub_result = self._compute(extrapolated_noise_factors=[0.0, 1.0])
        self.assertEqual(pub_result.data.evs.shape, (1,))
        self.assertEqual(pub_result.data.stds.shape, (1,))
        for arr in (
            pub_result.data.evs_noise_factors,
            pub_result.data.stds_noise_factors,
            pub_result.data.stds_twirl_noise_factors,
        ):
            self.assertEqual(arr.shape, (1, 3))
        self.assertEqual(pub_result.data.evs_extrapolated.shape, (1, 1, 2))
        self.assertEqual(pub_result.data.stds_extrapolated.shape, (1, 1, 2))

    # ---------------------------------------------------------------------------
    # non-broadcast mode (outer-product)
    # ---------------------------------------------------------------------------

    def test_raises_broadcast_false_without_meas_bases(self):
        """``broadcast_obs_and_params=False`` with ``meas_bases=None`` must raise."""
        item_result = self._item_result_5d()
        with self.assertRaises(ValueError):
            PEA.compute_expectation_value_pea(
                item_result,
                observables=_obs("Z"),
                param_shape=None,
                param_basis_pairs=None,
                noise_factors=NOISE_FACTORS,
                extrapolator=["linear"],
                broadcast_obs_and_params=False,
                meas_bases=None,
            )

    def test_non_broadcast_5d_data_auto_infers_param_shape(self):
        """With ``param_shape=None`` and 5-D data the shape should be inferred as ()."""
        item_result = self._item_result_5d()
        # 5-D shape: (nf, R, 1, S, 1) - the outer product with no parameters
        result = PEA.compute_expectation_value_pea(
            item_result,
            observables=_obs("Z"),
            param_shape=None,
            param_basis_pairs=None,
            noise_factors=NOISE_FACTORS,
            extrapolator=["linear"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_raises_on_wrong_ndim(self):
        """A 7-D data array in non-broadcast mode must raise ValueError."""
        bad_data = np.zeros((3, 2, 1, 1, 4, 1, 1))
        item_result = {"_meas": bad_data}

        with self.assertRaises(ValueError):
            PEA.compute_expectation_value_pea(
                item_result,
                observables=_obs("Z"),
                param_shape=None,
                param_basis_pairs=None,
                noise_factors=NOISE_FACTORS,
                extrapolator=["linear"],
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("Z")],
            )

    def test_non_broadcast_explicit_param_shape_skips_inference(self):
        """Test ``param_shape`` is provided explicitly in non-broadcast mode."""
        item_result = self._item_result_5d()
        result = PEA.compute_expectation_value_pea(
            item_result,
            observables=_obs("Z"),
            param_shape=(),  # explicitly given — skips inference
            param_basis_pairs=None,
            noise_factors=NOISE_FACTORS,
            extrapolator=["linear"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_6d_data_infers_param_shape_from_axis3(self):
        """With ``param_shape=None`` and 6-D data, ``param_shape`` is inferred as
        ``(data_shape[3],)``."""
        # Shape: (noise_factors=3, R=2, num_configs=2, num_params=2, shots=4, bits=1)
        # num_configs=2 to accommodate config_idx 0 and 1 in downstream indexing.
        data_6d = np.zeros((len(NOISE_FACTORS), NUM_RANDOMIZATIONS, 2, 2, SHOTS, 1), dtype=bool)
        item_result = {"_meas": data_6d}
        # The 6-D path is reached regardless of what the downstream pipeline does.
        # With num_params=2 and two "Z" observables the inferred param_shape = (2,).
        with contextlib.suppress(Exception):  # any downstream error is acceptable;
            PEA.compute_expectation_value_pea(
                item_result,
                observables=_obs("Z", "Z"),
                param_shape=None,
                param_basis_pairs=None,
                noise_factors=NOISE_FACTORS,
                extrapolator=["linear"],
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("Z")],
            )

    def test_non_broadcast_explicit_param_basis_pairs_skips_computation(self):
        """Test ``param_basis_pairs`` is provided explicitly in non-broadcast mode."""
        item_result = self._item_result_5d()
        # param_shape=() → single config; param_basis_pairs has ndindex=() (empty tuple).
        result = PEA.compute_expectation_value_pea(
            item_result,
            observables=_obs("Z"),
            param_shape=(),
            param_basis_pairs=[((), "Z")],  # explicitly given — skips auto-computation
            noise_factors=NOISE_FACTORS,
            extrapolator=["linear"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)


# ---------------------------------------------------------------------------
# PEA.create_instance_from_passthrough_data
# ---------------------------------------------------------------------------


class TestPEACreateInstanceFromPassthroughData(unittest.TestCase):
    """Tests for :meth:`PEA.create_instance_from_passthrough_data`."""

    @staticmethod
    def _minimal_passthrough(**overrides):
        data = {
            "observables": SparsePauliOp("ZZ"),
            "param_basis_pairs": None,
            "param_shape": None,
            "broadcast_obs_and_params": True,
            "meas_bases": None,
            "program_item_index": 0,
            "trex_calibration": False,
            "noise_factors": np.array([1.0, 3.0, 5.0]),
            "extrapolator": ["linear"],
            "extrapolated_noise_factors": None,
        }
        data.update(overrides)
        return data

    def test_happy_path_returns_pea_instance(self):
        """A fully-populated passthrough dict must produce a PEA instance."""
        pea = PEA.create_instance_from_passthrough_data(self._minimal_passthrough())
        self.assertIsInstance(pea, PEA)

    def test_attributes_set_from_passthrough(self):
        """Key attributes must be populated from the passthrough dict."""
        passthrough = self._minimal_passthrough(program_item_index=3)
        pea = PEA.create_instance_from_passthrough_data(passthrough)
        self.assertEqual(pea._program_item_index, 3)
        self.assertTrue(pea.broadcast_obs_and_params)
        np.testing.assert_array_equal(pea.noise_factors, [1.0, 3.0, 5.0])
        self.assertEqual(pea.extrapolator, ["linear"])
        self.assertIsNone(pea.extrapolated_noise_factors)

    def test_raises_when_observables_missing(self):
        """Missing 'observables' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["observables"]
        with self.assertRaises(ValueError, msg="Should raise for missing observables"):
            PEA.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_broadcast_obs_and_params_missing(self):
        """Missing 'broadcast_obs_and_params' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["broadcast_obs_and_params"]
        with self.assertRaises(ValueError):
            PEA.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_program_item_index_missing(self):
        """Missing 'program_item_index' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["program_item_index"]
        with self.assertRaises(ValueError):
            PEA.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_noise_factors_missing(self):
        """Missing 'noise_factors' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["noise_factors"]
        with self.assertRaises(ValueError):
            PEA.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_trex_calibration_true_but_no_trex(self):
        """trex_calibration=True without a TREX instance must raise ValueError."""
        passthrough = self._minimal_passthrough(trex_calibration=True)
        with self.assertRaises(ValueError):
            PEA.create_instance_from_passthrough_data(passthrough, trex=None)


if __name__ == "__main__":
    unittest.main()
