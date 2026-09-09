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

"""Tests for the ``GateFolding`` class."""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import PubResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_mitigation.trex import TREX
from qiskit_mitigation.zne.gate_folding import GateFolding
from samplomatic.quantum_program import QuantumProgram, QuantumProgramResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_RANDOMIZATIONS = 2
SHOTS = 4
TOTAL_SHOTS = NUM_RANDOMIZATIONS * SHOTS


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple non-parametric circuit with a Bell-pair structure."""
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

    The returned array has shape ``(NUM_RANDOMIZATIONS, 1, SHOTS, 1)`` (one config, one
    qubit) with exactly ``num_ones`` of the ``TOTAL_SHOTS`` outcomes set to 1, so that
    ``<Z> = (TOTAL_SHOTS - 2 * num_ones) / TOTAL_SHOTS``.
    """
    flat = np.zeros(TOTAL_SHOTS, dtype=bool)
    flat[:num_ones] = True
    return flat.reshape(NUM_RANDOMIZATIONS, 1, SHOTS, 1)


def _z_exp_val(num_ones: int) -> float:
    """Return the ``<Z>`` produced by :func:`_z_data` for ``num_ones``."""
    return (TOTAL_SHOTS - 2 * num_ones) / TOTAL_SHOTS


# Exactly collinear in the noise factors (1, 3, 5): 0.75, 0.5, 0.25.
# A linear fit therefore has slope -0.125 and intercept 0.875 at zero noise.
LINEAR_ONES = (1, 2, 3)
NOISE_FACTORS = [1.0, 3.0, 5.0]
LINEAR_EXP_VALS = [_z_exp_val(n) for n in LINEAR_ONES]
LINEAR_ZERO_NOISE = 0.875


def _linear_noise_amplified_data() -> np.ndarray:
    """Return data of shape ``(3, R, 1, S, 1)`` whose ``<Z>`` is linear in the noise factors."""
    return np.array([_z_data(n) for n in LINEAR_ONES])


def _obs(*labels: str) -> ObservablesArray:
    """Return an ``ObservablesArray`` built from the given Pauli labels."""
    return ObservablesArray([SparsePauliOp(lbl) for lbl in labels])


# ---------------------------------------------------------------------------
# GateFolding.__init__
# ---------------------------------------------------------------------------


class TestGateFoldingInit(unittest.TestCase):
    """Tests for :meth:`GateFolding.__init__`."""

    def test_gate_folding_attributes_are_none_after_init(self):
        """``noise_factors`` and ``extrapolator`` must be None immediately after construction."""
        gf = GateFolding()
        self.assertIsNone(gf.noise_factors)
        self.assertIsNone(gf.extrapolator)

    def test_inherits_mitigation_task_attributes(self):
        """All MitigationTask placeholder attributes must also be None."""
        gf = GateFolding()
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
            self.assertIsNone(getattr(gf, attr), msg=f"{attr} should be None after __init__")


# ---------------------------------------------------------------------------
# GateFolding._box_circuit
# ---------------------------------------------------------------------------


class TestGateFoldingBoxCircuit(unittest.TestCase):
    """Tests for the GateFolding override of :meth:`_box_circuit`."""

    def setUp(self):
        self.gf = GateFolding()
        self.circuit = _simple_circuit()

    def test_returns_quantum_circuit(self):
        """The boxed circuit must be a :class:`~qiskit.QuantumCircuit`."""
        boxed = self.gf._box_circuit(self.circuit, {})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_adds_meas_register(self):
        """The boxed circuit must include the ``'_meas'`` classical register."""
        boxed = self.gf._box_circuit(self.circuit, {})
        self.assertIn("_meas", [reg.name for reg in boxed.cregs])

    def test_enable_measures_injected_when_absent(self):
        """``enable_measures`` must be set to ``True`` on the caller's options dict."""
        options: dict = {}
        self.gf._box_circuit(self.circuit, options)
        self.assertTrue(options["enable_measures"])
        self.assertEqual(options["measure_annotations"], "change_basis")

    def test_raises_when_enable_measures_is_false(self):
        """A ``False`` value for ``enable_measures`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.gf._box_circuit(self.circuit, {"enable_measures": False})

    def test_none_options_uses_gate_folding_defaults(self):
        """``None`` options must fall back to the GateFolding default of ``enable_gates=False``."""
        boxed = self.gf._box_circuit(self.circuit, None)
        self.assertIsInstance(boxed, QuantumCircuit)
        self.assertIn("_meas", [reg.name for reg in boxed.cregs])


# ---------------------------------------------------------------------------
# GateFolding.prepare
# ---------------------------------------------------------------------------


class TestGateFoldingPrepare(unittest.TestCase):
    """Tests for :meth:`GateFolding.prepare`."""

    def setUp(self):
        self.gf = GateFolding()
        self.circuit = _simple_circuit()
        self.observables = [SparsePauliOp("ZZ")]

    def test_raises_on_invalid_folding_method(self):
        """An unsupported ``folding_method`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.gf.prepare(
                self.circuit, self.observables, parameters=None, folding_method="not_a_method"
            )

    def test_one_item_per_noise_factor(self):
        """The returned program must contain one item per noise factor."""
        program = self.gf.prepare(
            self.circuit, self.observables, parameters=None, noise_factors=(1, 3, 5)
        )
        self.assertEqual(len(program.items), 3)

    def test_noise_factors_saved_as_float_array(self):
        """``noise_factors`` must be stored as a float ndarray."""
        self.gf.prepare(self.circuit, self.observables, parameters=None, noise_factors=(1, 3))
        self.assertIsInstance(self.gf.noise_factors, np.ndarray)
        self.assertEqual(self.gf.noise_factors.dtype, np.dtype(float))
        np.testing.assert_allclose(self.gf.noise_factors, [1.0, 3.0])

    def test_extrapolator_not_saved_when_none(self):
        """``extrapolator`` must stay ``None`` when not requested at preparation."""
        self.gf.prepare(self.circuit, self.observables, parameters=None, extrapolator=None)
        self.assertIsNone(self.gf.extrapolator)

    def test_extrapolator_saved_when_valid(self):
        """A valid ``extrapolator`` must be stored for later post-processing."""
        self.gf.prepare(
            self.circuit,
            self.observables,
            parameters=None,
            noise_factors=(1, 3, 5),
            extrapolator=["linear"],
        )
        self.assertEqual(self.gf.extrapolator, ["linear"])

    def test_raises_when_noise_factors_insufficient_for_extrapolator(self):
        """Too few noise factors for the requested extrapolator must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.gf.prepare(
                self.circuit,
                self.observables,
                parameters=None,
                noise_factors=(1, 3),
                extrapolator=["double_exponential"],
            )

    def test_appends_to_existing_program(self):
        """Passing a ``quantum_program`` must append the new items to it."""
        program = self.gf.prepare(
            self.circuit, self.observables, parameters=None, noise_factors=(1, 3)
        )
        self.assertEqual(len(program.items), 2)
        second = GateFolding()
        returned = second.prepare(
            self.circuit,
            self.observables,
            parameters=None,
            noise_factors=(1, 3, 5),
            quantum_program=program,
        )
        self.assertIs(returned, program)
        self.assertEqual(len(program.items), 5)

    def test_program_item_index_points_at_first_item(self):
        """``_program_item_index`` must be the index of the task's *first* item in the program.

        Regression test: when appending to an existing program with multiple noise factors,
        the index used to be overwritten on every append and ended up pointing at the task's
        last item, so postprocessing sliced the wrong items out of the program results.
        """
        program = self.gf.prepare(
            self.circuit, self.observables, parameters=None, noise_factors=(1, 3)
        )
        self.assertEqual(self.gf._program_item_index, 0)
        second = GateFolding()
        second.prepare(
            self.circuit,
            self.observables,
            parameters=None,
            noise_factors=(1, 3, 5),
            quantum_program=program,
        )
        self.assertEqual(second._program_item_index, 2)

    def test_raises_on_shots_mismatch_with_existing_program(self):
        """A ``shots_per_randomization`` differing from the program's shots must raise."""
        program = self.gf.prepare(
            self.circuit, self.observables, parameters=None, shots_per_randomization=64
        )
        with self.assertRaises(ValueError):
            GateFolding().prepare(
                self.circuit,
                self.observables,
                parameters=None,
                shots_per_randomization=128,
                quantum_program=program,
            )

    def test_parametric_circuit_broadcast_mode(self):
        """A parametric circuit in broadcast mode must record ``param_basis_pairs``."""
        circuit, _theta = _parametric_circuit()
        gf = GateFolding()
        program = gf.prepare(
            circuit,
            [SparsePauliOp("ZZ")],
            parameters=np.array([[0.1], [0.2]]),
            noise_factors=(1, 3),
            broadcast_obs_and_params=True,
        )
        self.assertEqual(len(program.items), 2)
        self.assertTrue(gf.broadcast_obs_and_params)
        self.assertIsNotNone(gf.param_basis_pairs)

    def test_raises_when_trex_and_projection_observables(self):
        """``prepare`` with a TREX instance and projection-operator observables must raise."""
        gf = GateFolding()
        qc = _simple_circuit()
        project_obs = ObservablesArray.coerce([{"Z+": 1}])
        trex = TREX()
        with self.assertRaises(ValueError):
            gf.prepare(qc, project_obs, parameters=None, trex=trex)

    def test_prepare_with_trex_and_normal_observables_edits_boxing_options(self):
        """``prepare`` with TREX and regular Pauli observables must reach line 239
        (``custom_boxing_options = trex.edit_boxing_options(…)``)."""
        gf = GateFolding()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        trex = TREX()
        program = gf.prepare(qc, obs, parameters=None, trex=trex)
        self.assertIsInstance(program, QuantumProgram)
        self.assertGreater(len(program.items), 0)


# ---------------------------------------------------------------------------
# GateFolding._calculate_extrapolated_expectation_values  (static method)
# ---------------------------------------------------------------------------


class TestCalculateExtrapolatedExpectationValues(unittest.TestCase):
    """Tests for :meth:`GateFolding._calculate_extrapolated_expectation_values`."""

    def _calculate(
        self,
        observables=None,
        data=None,
        param_shape=(1,),
        param_basis_pairs=None,
        noise_factors=None,
        extrapolated_noise_factors=None,
        extrapolator=None,
    ):
        """Call the method under test with linear single-qubit defaults."""
        return GateFolding._calculate_extrapolated_expectation_values(
            _linear_noise_amplified_data() if data is None else data,
            _obs("Z") if observables is None else observables,
            param_shape,
            [((0,), "Z")] if param_basis_pairs is None else param_basis_pairs,
            NOISE_FACTORS if noise_factors is None else noise_factors,
            [] if extrapolated_noise_factors is None else extrapolated_noise_factors,
            ["linear"] if extrapolator is None else extrapolator,
            None,
        )

    def test_returns_eight_tuple(self):
        """The method must return an 8-tuple."""
        self.assertEqual(len(self._calculate()), 8)

    def test_linear_extrapolation_to_zero_noise(self):
        """Data linear in the noise factors must extrapolate to the exact intercept."""
        zero_exp_vals = self._calculate()[0]
        np.testing.assert_allclose(zero_exp_vals, [LINEAR_ZERO_NOISE], atol=1e-10)

    def test_noise_factor_exp_vals_are_the_measured_values(self):
        """The per-noise-factor expectation values must match the raw measured values."""
        noise_factors_exp_vals = self._calculate()[2]
        self.assertEqual(noise_factors_exp_vals.shape, (1, 3))
        np.testing.assert_allclose(noise_factors_exp_vals[0], LINEAR_EXP_VALS, atol=1e-10)

    def test_selected_extrapolator_is_reported_per_term(self):
        """The selected extrapolator must be reported for each observable term."""
        selected = self._calculate()[7]
        self.assertEqual(selected, [["linear"]])

    def test_output_shapes(self):
        """Zero-noise outputs take the broadcast shape; per-noise-factor outputs add an axis."""
        (
            zero_exp_vals,
            zero_stds,
            noise_factors_exp_vals,
            noise_factors_ensemble_stds,
            noise_factors_stds,
            extrapolated_exp_vals,
            extrapolated_stds,
            _selected,
        ) = self._calculate(extrapolated_noise_factors=[0.0, 1.0])
        self.assertEqual(zero_exp_vals.shape, (1,))
        self.assertEqual(zero_stds.shape, (1,))
        for arr in (noise_factors_exp_vals, noise_factors_ensemble_stds, noise_factors_stds):
            self.assertEqual(arr.shape, (1, 3))
        # (*output_shape, num_models, num_extrapolated_noise_factors)
        self.assertEqual(extrapolated_exp_vals.shape, (1, 1, 2))
        self.assertEqual(extrapolated_stds.shape, (1, 1, 2))

    def test_extrapolated_noise_factors_are_evaluated(self):
        """The fit must be evaluated at each requested extrapolated noise factor."""
        extrapolated_exp_vals = self._calculate(extrapolated_noise_factors=[0.0, 1.0, 3.0])[5]
        np.testing.assert_allclose(
            extrapolated_exp_vals[0, 0], [LINEAR_ZERO_NOISE, 0.75, 0.5], atol=1e-10
        )

    def test_fallback_uses_lowest_noise_factor_value(self):
        """The ``fallback`` model must report the value measured at the lowest noise factor."""
        zero_exp_vals = self._calculate(extrapolator=["fallback"])[0]
        np.testing.assert_allclose(zero_exp_vals, [_z_exp_val(LINEAR_ONES[0])], atol=1e-10)

    def test_coefficient_scaling(self):
        """An observable coefficient must scale both the zero-noise and per-factor values."""
        obs = ObservablesArray([2.0 * SparsePauliOp("Z")])
        zero_exp_vals, _, noise_factors_exp_vals = self._calculate(observables=obs)[:3]
        np.testing.assert_allclose(zero_exp_vals, [2.0 * LINEAR_ZERO_NOISE], atol=1e-10)
        np.testing.assert_allclose(
            noise_factors_exp_vals[0], [2.0 * v for v in LINEAR_EXP_VALS], atol=1e-10
        )

    def test_ensemble_stds_scale_with_total_shots(self):
        """Per-noise-factor ensemble stds must be ``sqrt(var / total_shots)``."""
        noise_factors_ensemble_stds = self._calculate()[3]
        expected = [np.sqrt((1.0 - v**2) / TOTAL_SHOTS) for v in LINEAR_EXP_VALS]
        np.testing.assert_allclose(noise_factors_ensemble_stds[0], expected, atol=1e-10)

    def test_zero_noise_std_is_finite(self):
        """The extrapolated zero-noise standard deviation must be finite."""
        zero_stds = self._calculate()[1]
        self.assertTrue(np.all(np.isfinite(zero_stds)))

    def test_multiple_observables_broadcast(self):
        """Two observables over two parameter configs must broadcast to shape (2,)."""
        obs = _obs("Z", "X")
        # config 0 measures Z, config 1 measures X
        data = np.array(
            [np.concatenate([_z_data(ones), _z_data(ones)], axis=1) for ones in LINEAR_ONES]
        )
        zero_exp_vals = self._calculate(
            observables=obs,
            data=data,
            param_shape=(2,),
            param_basis_pairs=[((0,), "Z"), ((1,), "X")],
        )[0]
        self.assertEqual(zero_exp_vals.shape, (2,))
        np.testing.assert_allclose(
            zero_exp_vals, [LINEAR_ZERO_NOISE, LINEAR_ZERO_NOISE], atol=1e-10
        )

    def test_raises_incompatible_broadcast_shapes(self):
        """``param_shape`` (2,) vs observables shape (3,) must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._calculate(
                observables=_obs("Z", "Z", "Z"),
                param_shape=(2,),
                param_basis_pairs=[((0,), "Z"), ((1,), "Z")],
            )

    def test_identity_observable_gives_one(self):
        """An identity observable must extrapolate to 1.0 at every noise factor."""
        # Constant data makes every standard error degenerate, which warns before
        # falling back to an unweighted fit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero_exp_vals, _, noise_factors_exp_vals = self._calculate(observables=_obs("I"))[:3]
        np.testing.assert_allclose(zero_exp_vals, [1.0], atol=1e-10)
        np.testing.assert_allclose(noise_factors_exp_vals[0], [1.0, 1.0, 1.0], atol=1e-10)

    def test_multiple_models_are_all_evaluated(self):
        """Every requested model must produce a row in the extrapolated outputs."""
        extrapolated_exp_vals = self._calculate(
            extrapolator=["linear", "fallback"], extrapolated_noise_factors=[0.0]
        )[5]
        self.assertEqual(extrapolated_exp_vals.shape, (1, 2, 1))
        np.testing.assert_allclose(
            extrapolated_exp_vals[0, :, 0],
            [LINEAR_ZERO_NOISE, _z_exp_val(LINEAR_ONES[0])],
            atol=1e-10,
        )

    def test_raises_when_no_config_for_param_index(self):
        """``ValueError`` must be raised when ``param_basis_pairs`` has no entry for an index."""
        obs = _obs("Z", "Z")  # shape (2,)
        data = _linear_noise_amplified_data()  # (3, R, 1, S, 1)
        with self.assertRaises(ValueError):
            GateFolding._calculate_extrapolated_expectation_values(
                data,
                obs,
                param_shape=(2,),
                param_basis_pairs=[((0,), "Z")],  # (1,) is missing
                noise_factors=NOISE_FACTORS,
                extrapolated_noise_factors=[],
                extrapolator=["linear"],
                trex_scale_factors=None,
            )

    def test_custom_fit_success(self):
        """Custom fit extrapolation to zero noise."""

        def my_custom(x, a, b):
            return a * x + b

        custom_fit = (my_custom, [1.0, 1.0], (-np.inf, np.inf))

        zero_exp_vals = GateFolding._calculate_extrapolated_expectation_values(
            _linear_noise_amplified_data(),
            _obs("Z"),
            (1,),
            [((0,), "Z")],
            NOISE_FACTORS,
            [],
            ["custom"],
            None,
            custom_fit=custom_fit,
        )[0]
        np.testing.assert_allclose(zero_exp_vals, [LINEAR_ZERO_NOISE], atol=1e-10)


# ---------------------------------------------------------------------------
# GateFolding.compute_expectation_value_gate_folding  (static method)
# ---------------------------------------------------------------------------


class TestComputeExpectationValueFateFolding(unittest.TestCase):
    """Tests for the static :meth:`GateFolding.compute_expectation_value_gate_folding`."""

    def _item_results(self) -> list[dict]:
        """Return one result dict per noise factor, linear in the noise factors."""
        return [{"_meas": _z_data(ones)} for ones in LINEAR_ONES]

    def _compute(self, item_results=None, **kwargs):
        """Call the method under test with linear single-qubit defaults."""
        params = {
            "observables": _obs("Z"),
            "param_shape": (1,),
            "param_basis_pairs": [((0,), "Z")],
            "noise_factors": NOISE_FACTORS,
            "extrapolator": ["linear"],
        }
        params.update(kwargs)
        return GateFolding.compute_expectation_value_gate_folding(
            self._item_results() if item_results is None else item_results, **params
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_raises_when_noise_factors_insufficient(self):
        """Too few noise factors for the extrapolator must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self._compute(noise_factors=[1.0], extrapolator=["linear"])

    def test_raises_missing_meas_key(self):
        """A result dict without ``'_meas'`` must raise ``ValueError``."""
        item_results = self._item_results()
        item_results[1] = {"wrong_key": _z_data(2)}
        with self.assertRaises(ValueError):
            self._compute(item_results)

    def test_raises_wrong_ndim_in_broadcast_mode(self):
        """A 3-D ``_meas`` array in broadcast mode must raise ``ValueError``."""
        item_results = self._item_results()
        item_results[0] = {"_meas": np.zeros((2, 1, 4), dtype=bool)}
        with self.assertRaises(ValueError):
            self._compute(item_results)

    def test_raises_broadcast_true_without_param_basis_pairs(self):
        """``broadcast_obs_and_params=True`` with ``param_basis_pairs=None`` must raise."""
        with self.assertRaises(ValueError):
            self._compute(param_basis_pairs=None, broadcast_obs_and_params=True)

    def test_raises_broadcast_true_without_param_shape(self):
        """``broadcast_obs_and_params=True`` with ``param_shape=None`` must raise."""
        with self.assertRaises(ValueError):
            self._compute(param_shape=None, broadcast_obs_and_params=True)

    def test_raises_broadcast_false_without_meas_bases(self):
        """``broadcast_obs_and_params=False`` with ``meas_bases=None`` must raise."""
        with self.assertRaises(ValueError):
            self._compute(broadcast_obs_and_params=False, meas_bases=None)

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
        self.assertEqual(pub_result.data.evs_extrapolated.shape, (1, 1, 2))
        self.assertEqual(pub_result.data.stds_extrapolated.shape, (1, 1, 2))

    def test_scalar_extrapolated_noise_factor_is_wrapped(self):
        """A scalar ``extrapolated_noise_factors`` must be treated as a single point."""
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

    def test_measurement_flips_are_applied(self):
        """``measurement_flips._meas`` must be XOR'd into ``_meas`` before processing."""
        # Flipping every bit maps <Z> to -<Z>, so the intercept flips sign too.
        item_results = [
            {
                "_meas": _z_data(ones),
                "measurement_flips._meas": np.ones((NUM_RANDOMIZATIONS, 1, SHOTS, 1), dtype=bool),
            }
            for ones in LINEAR_ONES
        ]
        pub_result = self._compute(item_results)
        np.testing.assert_allclose(pub_result.data.evs, [-LINEAR_ZERO_NOISE], atol=1e-10)

    def test_non_broadcast_5d_data_infers_param_shape(self):
        """5-D data in non-broadcast mode must auto-infer ``param_shape`` from ``data_shape[2]``."""
        # 5-D shape: (R, bases, params, S, bits) - use 1 param set
        # data_shape[2] is the int 1 (num_params); param_basis_pairs must be supplied so
        # the auto-compute branch is skipped after param_shape is inferred
        data_5d = np.zeros((2, 1, 1, 4, 1), dtype=bool)
        item_results = [{"_meas": data_5d} for _ in NOISE_FACTORS]
        result = GateFolding.compute_expectation_value_gate_folding(
            item_results,
            observables=_obs("Z"),
            param_shape=None,
            param_basis_pairs=[((0,), "Z")],
            noise_factors=NOISE_FACTORS,
            extrapolator=["fallback"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_raises_on_wrong_ndim_in_loop(self):
        """A 6-D data array in the result loop in non-broadcast mode must raise ``ValueError``."""
        # First item is fine (4-D), second has wrong shape (6-D) to hit the in-loop check
        good = {"_meas": np.zeros((2, 1, 4, 1), dtype=bool)}
        bad = {"_meas": np.zeros((2, 1, 1, 1, 4, 1))}
        with self.assertRaises(ValueError):
            GateFolding.compute_expectation_value_gate_folding(
                [good, bad, good],
                observables=_obs("Z"),
                param_shape=(),
                param_basis_pairs=[(((),), "Z")],
                noise_factors=NOISE_FACTORS,
                extrapolator=["fallback"],
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("Z")],
            )

    def _item_results(self):
        """One result dict per noise factor, linear in the noise factors."""
        return [{"_meas": _z_data(ones)} for ones in LINEAR_ONES]

    def test_non_broadcast_4d_data_auto_infers_empty_param_shape(self):
        """4-D data in non-broadcast mode must auto-infer ``param_shape=()``."""
        # 4-D shape: (R, bases, S, bits) - use 1 param set
        data_4d = np.zeros((2, 1, 4, 1), dtype=bool)
        item_results = [{"_meas": data_4d} for _ in NOISE_FACTORS]

        result = GateFolding.compute_expectation_value_gate_folding(
            item_results,
            observables=_obs("Z"),
            param_shape=None,
            param_basis_pairs=None,
            noise_factors=NOISE_FACTORS,
            extrapolator=["linear"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_5d_data_auto_infers_param_shape_and_computes_pairs(self):
        """5-D data with ``param_basis_pairs=None`` must infer ``param_shape`` from
        ``data_shape[2]`` and then broadcast the observables array."""
        # 5-D shape: (num_randomizations, num_bases, num_parameters, shots, num_bits)
        data_5d = np.zeros((2, 1, 1, 4, 1), dtype=bool)
        item_results = [{"_meas": data_5d} for _ in NOISE_FACTORS]

        result = GateFolding.compute_expectation_value_gate_folding(
            item_results,
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
        """A 6-D data array in non-broadcast mode must raise ValueError."""
        bad_results = [{"_meas": np.zeros((2, 1, 1, 4, 1, 1))} for _ in LINEAR_ONES]
        with self.assertRaises(ValueError):
            GateFolding.compute_expectation_value_gate_folding(
                bad_results,
                observables=_obs("Z"),
                param_shape=None,
                param_basis_pairs=None,
                noise_factors=NOISE_FACTORS,
                extrapolator=["linear"],
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("Z")],
            )

    # ---------------------------------------------------------------------------
    # non-broadcast mode
    # ---------------------------------------------------------------------------

    def test_non_broadcast_missing_meas_key_raises(self):
        """A result dict without '_meas' in non-broadcast mode must raise ValueError."""
        bad_results = [{"wrong": _z_data(n)} for n in LINEAR_ONES]

        with self.assertRaises(ValueError):
            GateFolding.compute_expectation_value_gate_folding(
                bad_results,
                observables=_obs("Z"),
                param_shape=None,
                param_basis_pairs=None,
                noise_factors=NOISE_FACTORS,
                extrapolator=["linear"],
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("Z")],
            )

    def test_non_broadcast_param_basis_pairs_auto_computed(self):
        """When ``param_basis_pairs=None`` in non-broadcast mode, they must be computed
        from the observables and meas_bases."""
        result = GateFolding.compute_expectation_value_gate_folding(
            self._item_results(),
            observables=_obs("Z"),
            param_shape=(),
            param_basis_pairs=None,
            noise_factors=NOISE_FACTORS,
            extrapolator=["linear"],
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("Z")],
        )
        self.assertIsInstance(result, PubResult)
        np.testing.assert_allclose(result.data.evs, [LINEAR_ZERO_NOISE], atol=1e-10)

    def test_custom_fit_success(self):
        """``compute_expectation_value_gate_folding`` works with custom_fit."""

        def my_custom(x, a, b):
            return a * x + b

        custom_fit = (my_custom, [1.0, 1.0], (-np.inf, np.inf))
        pub_result = self._compute(extrapolator=["custom"], custom_fit=custom_fit)
        np.testing.assert_allclose(pub_result.data.evs, [LINEAR_ZERO_NOISE], atol=1e-10)


# ---------------------------------------------------------------------------
# GateFolding.postprocess
# ---------------------------------------------------------------------------


class TestGateFoldingPostprocess(unittest.TestCase):
    """Tests for :meth:`GateFolding.postprocess`."""

    def _make_minimal_gate_folding(self):
        """Return a GateFolding instance with the minimum attributes set for postprocess."""

        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        gf.extrapolator = ["linear"]
        gf.observables = _obs("Z")
        gf.parameters = None
        gf.param_basis_pairs = [((0,), "Z")]
        gf.meas_bases = None
        gf.broadcast_obs_and_params = True
        gf.extrapolated_noise_factors = None
        return gf

    def test_raises_when_noise_factors_unset_and_not_given(self):
        """``postprocess`` must raise when neither the saved nor the passed factors exist."""
        with self.assertRaises(ValueError):
            GateFolding().postprocess([])

    def test_raises_when_extrapolator_unset_and_not_given(self):
        """``postprocess`` must raise when neither the saved nor the passed extrapolator exists."""
        with self.assertRaises(ValueError):
            GateFolding().postprocess([], noise_factors=NOISE_FACTORS)

    def test_uses_saved_noise_factors_when_not_passed(self):
        """Saved ``noise_factors`` must be forwarded when none are given to postprocess."""
        gf = self._make_minimal_gate_folding()
        with patch.object(
            GateFolding, "compute_expectation_value_gate_folding", return_value=None
        ) as mock_fn:
            gf.postprocess(MagicMock())
            _, call_kwargs = mock_fn.call_args
            np.testing.assert_array_equal(call_kwargs["noise_factors"], list(gf.noise_factors))

    def test_uses_saved_extrapolator_when_not_passed(self):
        """Saved ``extrapolator`` must be forwarded when none is given to postprocess."""
        gf = self._make_minimal_gate_folding()
        with patch.object(
            GateFolding, "compute_expectation_value_gate_folding", return_value=None
        ) as mock_fn:
            gf.postprocess(MagicMock())
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["extrapolator"], list(gf.extrapolator))

    def test_override_noise_factors_via_postprocess_kwarg(self):
        """Noise factors passed to ``postprocess`` must override saved ones."""
        gf = self._make_minimal_gate_folding()
        override = [1.0, 7.0, 9.0]
        with patch.object(
            GateFolding, "compute_expectation_value_gate_folding", return_value=None
        ) as mock_fn:
            gf.postprocess(MagicMock(), noise_factors=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["noise_factors"], override)

    def test_override_extrapolator_via_postprocess_kwarg(self):
        """Extrapolator passed to ``postprocess`` must override saved one."""
        gf = self._make_minimal_gate_folding()
        override = ["fallback"]
        with patch.object(
            GateFolding, "compute_expectation_value_gate_folding", return_value=None
        ) as mock_fn:
            gf.postprocess(MagicMock(), extrapolator=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["extrapolator"], override)

    def test_extrapolated_noise_factors_passthrough(self):
        """``extrapolated_noise_factors`` passed to postprocess must be forwarded."""
        gf = self._make_minimal_gate_folding()
        override = [0.0, 1.0]
        with patch.object(
            GateFolding, "compute_expectation_value_gate_folding", return_value=None
        ) as mock_fn:
            gf.postprocess(MagicMock(), extrapolated_noise_factors=override)
            _, call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs["extrapolated_noise_factors"], override)

    def test_postprocess_with_extrapolated_noise_factors(self):
        qc = QuantumCircuit(1)
        qc.x(0)

        gf = GateFolding()
        gf.prepare(
            circuit=qc,
            observables=[SparsePauliOp("Z")],
            parameters=None,
            noise_factors=[1, 3],
            extrapolator=["linear"],
            extrapolated_noise_factors=[0.0],
        )

        fake_meas = np.zeros((1, 1, 1, 1), dtype=np.uint8)
        result = gf.postprocess([{"_meas": fake_meas}, {"_meas": fake_meas}])
        self.assertIsInstance(result, PubResult)


# ---------------------------------------------------------------------------
# GateFolding._extract_items_list_data - warning paths
# ---------------------------------------------------------------------------


class TestGateFoldingExtractItemsListData(unittest.TestCase):
    """Tests for the TREX warning branches in :meth:`GateFolding._extract_items_list_data`."""

    def test_warns_when_trex_set_but_no_measure_noise_and_item_result_list(self):
        """When results is a list (not QuantumProgramResult) and TREX is set but no measure_noise_data is
        provided, a warning should be emitted."""
        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        mock_trex = TREX()
        gf.trex = mock_trex

        item_results = [{"_meas": _z_data(n)} for n in LINEAR_ONES]

        with self.assertWarns(UserWarning):
            results, noise_data = gf._extract_items_list_data(item_results, None)

        self.assertIs(results, item_results)
        self.assertIsNone(noise_data)

    def test_no_warning_when_trex_is_none(self):
        """No warning must be emitted when TREX is not enabled."""
        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        gf.trex = None

        item_results = [{"_meas": _z_data(n)} for n in LINEAR_ONES]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results, noise_data = gf._extract_items_list_data(item_results, None)

        self.assertIs(results, item_results)
        self.assertIsNone(noise_data)

    def test_warns_when_trex_set_no_calibration_and_full_result(self):
        """When a ``QuantumProgramResult`` is passed and TREX has no calibration,
        a warning must be emitted."""
        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        gf._program_item_index = 0
        trex = TREX()
        gf.trex = trex

        result = QuantumProgramResult(data=[{"_meas": _z_data(n)} for n in LINEAR_ONES])
        with self.assertWarns(UserWarning):
            gf._extract_items_list_data(result, None)

    def test_calls_compute_noise_model_when_trex_has_calibration(self):
        """When a ``QuantumProgramResult`` is passed and TREX has a calibration result,
        ``compute_noise_model`` must be called."""
        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        gf._program_item_index = 0

        trex = TREX()
        # Simulate that TREX.prepare() was called (sets _program_item_index)
        trex._program_item_index = len(NOISE_FACTORS)  # non-None → has_calibration_result=True
        gf.trex = trex

        result = QuantumProgramResult(data=[{"_meas": _z_data(n)} for n in LINEAR_ONES])

        sentinel_noise_data = object()
        with patch.object(TREX, "compute_noise_model", return_value=sentinel_noise_data):
            _, noise_data = gf._extract_items_list_data(result, None)

        self.assertIs(noise_data, sentinel_noise_data)

    def test_no_warning_when_trex_none_and_full_result(self):
        """When a ``QuantumProgramResult`` is passed and ``trex=None``, no warning must be
        emitted and ``measure_noise_data`` stays ``None``."""
        gf = GateFolding()
        gf.noise_factors = np.array(NOISE_FACTORS)
        gf._program_item_index = 0
        gf.trex = None

        result = QuantumProgramResult(data=[{"_meas": _z_data(n)} for n in LINEAR_ONES])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _item_results, noise_data = gf._extract_items_list_data(result, None)

        self.assertIsNone(noise_data)


# ---------------------------------------------------------------------------
# GateFolding.create_instance_from_passthrough_data
# ---------------------------------------------------------------------------


class TestGateFoldingCreateInstanceFromPassthroughData(unittest.TestCase):
    """Tests for :meth:`GateFolding.create_instance_from_passthrough_data`."""

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

    def test_happy_path_returns_gate_folding_instance(self):
        """A fully-populated passthrough dict must produce a GateFolding instance."""

        gf = GateFolding.create_instance_from_passthrough_data(self._minimal_passthrough())
        self.assertIsInstance(gf, GateFolding)

    def test_attributes_set_from_passthrough(self):
        """Key attributes must be populated from the passthrough dict."""
        passthrough = self._minimal_passthrough(program_item_index=5)
        gf = GateFolding.create_instance_from_passthrough_data(passthrough)
        self.assertEqual(gf._program_item_index, 5)
        self.assertTrue(gf.broadcast_obs_and_params)
        np.testing.assert_array_equal(gf.noise_factors, [1.0, 3.0, 5.0])
        self.assertEqual(gf.extrapolator, ["linear"])
        self.assertIsNone(gf.extrapolated_noise_factors)

    def test_raises_when_observables_missing(self):
        """Missing 'observables' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["observables"]
        with self.assertRaises(ValueError):
            GateFolding.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_broadcast_obs_and_params_missing(self):
        """Missing 'broadcast_obs_and_params' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["broadcast_obs_and_params"]
        with self.assertRaises(ValueError):
            GateFolding.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_program_item_index_missing(self):
        """Missing 'program_item_index' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["program_item_index"]
        with self.assertRaises(ValueError):
            GateFolding.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_noise_factors_missing(self):
        """Missing 'noise_factors' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["noise_factors"]
        with self.assertRaises(ValueError):
            GateFolding.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_trex_calibration_true_but_no_trex(self):
        """trex_calibration=True without a TREX instance must raise ValueError."""
        passthrough = self._minimal_passthrough(trex_calibration=True)
        with self.assertRaises(ValueError):
            GateFolding.create_instance_from_passthrough_data(passthrough, trex=None)


if __name__ == "__main__":
    unittest.main()
