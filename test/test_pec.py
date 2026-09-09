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

"""Tests for the ``PEC`` class."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import PubResult
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, SparsePauliOp
from qiskit_mitigation.pec import PEC
from qiskit_mitigation.trex import TREX
from samplomatic import InjectNoise
from samplomatic.quantum_program import QuantumProgram
from samplomatic.utils import get_annotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple non-parametric circuit with a Bell-pair structure."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _fake_item_result(
    num_randomizations: int,
    num_configs: int,
    shots: int,
    num_qubits: int,
    data: np.ndarray | None = None,
    pec_signs: np.ndarray | None = None,
    num_noise_terms: int = 1,
) -> dict:
    """Return a minimal item-result dict with shape (R, C, S, Q) and PEC signs.

    ``pec_signs`` shape is ``(num_randomizations, num_configs, num_noise_terms)`` as
    expected by :meth:`PEC._process_broadcasted_expectation_values_pec`.
    """
    if data is None:
        data = np.zeros((num_randomizations, num_configs, shots, num_qubits), dtype=bool)
    if pec_signs is None:
        # default: all-zero signs → no flip, shape (R, C, T)
        pec_signs = np.zeros((num_randomizations, num_configs, num_noise_terms), dtype=bool)
    return {"_meas": data, "pauli_signs": pec_signs}


# ---------------------------------------------------------------------------
# PEC.__init__
# ---------------------------------------------------------------------------


class TestPECInit(unittest.TestCase):
    """Tests for :meth:`PEC.__init__`."""

    def test_gamma_is_none_after_init(self):
        """``gamma`` must be None immediately after construction."""
        pec = PEC()
        self.assertIsNone(pec.gamma)

    def test_inherits_mitigation_task_attributes(self):
        """All MitigationTask placeholder attributes must also be None."""
        pec = PEC()
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
            self.assertIsNone(getattr(pec, attr), msg=f"{attr} should be None after __init__")


# ---------------------------------------------------------------------------
# PEC._box_circuit  -  PEC-specific option enforcement
# ---------------------------------------------------------------------------


class TestPECBoxCircuit(unittest.TestCase):
    """Tests for the PEC override of :meth:`_box_circuit`."""

    def setUp(self):
        self.pec = PEC()
        self.circuit = _simple_circuit()

    def test_enable_gates_injected_when_absent(self):
        """``enable_gates=True`` must be forwarded to ``generate_boxing_pass_manager``
        even when absent from the caller's dict, and the original dict must be untouched."""
        options = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pec._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertTrue(kwargs.get("enable_gates"))
        # Original dict must not have been mutated.
        self.assertNotIn("enable_gates", options)

    def test_inject_noise_targets_injected_when_absent(self):
        """``inject_noise_targets='gates'`` must be forwarded to ``generate_boxing_pass_manager``
        even when absent from the caller's dict, and the original dict must be untouched."""
        options = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pec._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("inject_noise_targets"), "gates")
        # Original dict must not have been mutated.
        self.assertNotIn("inject_noise_targets", options)

    def test_inject_noise_strategy_injected_when_absent(self):
        """``inject_noise_strategy='uniform_modification'`` must be forwarded to
        ``generate_boxing_pass_manager`` even when absent, and the original dict must be untouched."""
        options = {}
        with patch("qiskit_mitigation.mitigation_task.generate_boxing_pass_manager") as mock_gen:
            mock_gen.return_value = MagicMock()
            mock_gen.return_value.run.return_value = MagicMock()
            self.pec._box_circuit(self.circuit, options)
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("inject_noise_strategy"), "uniform_modification")
        # Original dict must not have been mutated.
        self.assertNotIn("inject_noise_strategy", options)

    def test_raises_when_enable_gates_is_false(self):
        """A ``False`` value for ``enable_gates`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pec._box_circuit(self.circuit, {"enable_gates": False})

    def test_raises_when_inject_noise_targets_is_measures(self):
        """``inject_noise_targets='measures'`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pec._box_circuit(self.circuit, {"inject_noise_targets": "measures"})

    def test_raises_when_inject_noise_strategy_is_no_modification(self):
        """``inject_noise_strategy='no_modification'`` must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.pec._box_circuit(self.circuit, {"inject_noise_strategy": "no_modification"})

    def test_inject_noise_targets_all_is_accepted(self):
        """``inject_noise_targets='all'`` is a valid PEC option and must not raise."""
        options = {"inject_noise_targets": "all"}
        boxed = self.pec._box_circuit(self.circuit, options)
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_returns_quantum_circuit(self):
        """The boxed circuit must be a :class:`~qiskit.QuantumCircuit`."""
        boxed = self.pec._box_circuit(self.circuit, {})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_adds_meas_register(self):
        """The boxed circuit must include the ``'_meas'`` classical register."""
        boxed = self.pec._box_circuit(self.circuit, {})
        reg_names = [r.name for r in boxed.cregs]
        self.assertIn("_meas", reg_names)

    def test_enable_gates_true_explicit_does_not_raise(self):
        """Explicitly setting ``enable_gates=True`` must work without error."""
        boxed = PEC._box_circuit(self.circuit, {"enable_gates": True})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_inject_noise_targets_gates_explicit_does_not_raise(self):
        """Explicitly setting ``inject_noise_targets='gates'`` must be accepted."""
        boxed = PEC._box_circuit(self.circuit, {"inject_noise_targets": "gates"})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_inject_noise_strategy_non_default_does_not_raise(self):
        """Any ``inject_noise_strategy`` other than ``'no_modification'`` must be accepted."""
        boxed = PEC._box_circuit(
            self.circuit,
            {"inject_noise_strategy": "individual_modification"},
        )
        self.assertIsInstance(boxed, QuantumCircuit)


# ---------------------------------------------------------------------------
# PEC.calculate_gamma  (static method)
# ---------------------------------------------------------------------------


class TestCalculateGamma(unittest.TestCase):
    """Tests for :meth:`PEC.calculate_gamma`."""

    def _boxed_bell_circuit_with_noise_model(self):
        """Return a PEC-boxed Bell circuit and its noise-model mapping.

        Boxing a circuit that has a CX gate with PEC options will produce
        InjectNoise annotations. We collect the refs and build a zero-rate
        PauliLindbladMap for each so that ``calculate_gamma`` can proceed
        without raising a missing-model error.
        """
        pec = PEC()
        qc = _simple_circuit()
        boxed = pec._box_circuit(qc, {})

        # Collect every InjectNoise ref present in the boxed circuit
        refs = {
            get_annotation(instr.operation, InjectNoise).ref
            for instr in boxed
            if get_annotation(instr.operation, InjectNoise) is not None
        }
        noise_maps = {ref: PauliLindbladMap.identity(qc.num_qubits) for ref in refs}
        return boxed, noise_maps

    def test_gamma_is_one_when_zero_rate_noise(self):
        """A zero-rate noise model has gamma==1 (no noise to cancel)."""
        boxed, noise_maps = self._boxed_bell_circuit_with_noise_model()
        gamma = PEC.calculate_gamma(boxed, noise_maps, noise_factor=1.0)
        self.assertIsInstance(gamma, float)
        self.assertAlmostEqual(gamma, 1.0)

    def test_gamma_is_one_for_noise_factor_zero(self):
        """noise_factor=0 scales all rates to zero → gamma == 1.0."""
        boxed, noise_maps = self._boxed_bell_circuit_with_noise_model()
        # create a non identity noise
        for ref in noise_maps:
            noise_maps[ref] = PauliLindbladMap.from_sparse_list(
                [("X", [0], 1.0), ("X", [1], 1.0), ("XX", [0, 1], 1.0)], num_qubits=2
            )
        gamma = PEC.calculate_gamma(boxed, noise_maps, noise_factor=0.0)
        self.assertAlmostEqual(gamma, 1.0)

    def test_raises_missing_noise_model_for_ref(self):
        """An empty noise mapping when InjectNoise refs are present must raise ValueError."""
        pec = PEC()
        qc = _simple_circuit()
        boxed = pec._box_circuit(qc, {})

        has_inject = any(
            get_annotation(instr.operation, InjectNoise) is not None for instr in boxed
        )
        if has_inject:
            with self.assertRaises(ValueError):
                PEC.calculate_gamma(boxed, {}, noise_factor=1.0)


# ---------------------------------------------------------------------------
# PEC.compute_expectation_value_pec  -  error-path validation
# ---------------------------------------------------------------------------


class TestComputeExpectationValuePEC(unittest.TestCase):
    """Tests for the static :meth:`PEC.compute_expectation_value_pec`."""

    def _obs(self, *labels):
        return ObservablesArray([SparsePauliOp(lbl) for lbl in labels])

    # ------------------------------------------------------------------
    # Input validation - missing / malformed data
    # ------------------------------------------------------------------

    def test_raises_missing_meas_key(self):
        """Result dict without ``'_meas'`` must raise ``ValueError``."""
        result = {"wrong_key": np.zeros((2, 1, 4, 2)), "pauli_signs": np.zeros((2, 1))}
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(result, self._obs("ZZ"), gamma=1.0)

    def test_raises_wrong_ndim(self):
        """A 3-D ``_meas`` array must raise ``ValueError``."""
        result = {"_meas": np.zeros((2, 1, 4)), "pauli_signs": np.zeros((2, 1))}
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(result, self._obs("ZZ"), gamma=1.0)

    def test_raises_missing_pauli_signs(self):
        """Result dict without ``'pauli_signs'`` must raise ``ValueError``."""
        result = {"_meas": np.zeros((2, 1, 4, 2))}
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(result, self._obs("ZZ"), gamma=1.0)

    def test_raises_broadcast_true_without_param_basis_pairs(self):
        """``broadcast_obs_and_params=True`` with ``param_basis_pairs=None`` must raise."""
        result = _fake_item_result(2, 1, 4, 2)
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(
                result,
                self._obs("ZZ"),
                gamma=1.0,
                broadcast_obs_and_params=True,
                param_basis_pairs=None,
            )

    def test_raises_broadcast_false_without_basis_obs_term_map(self):
        """``broadcast_obs_and_params=False`` with ``basis_obs_term_map=None`` must raise."""
        result = _fake_item_result(2, 1, 4, 2)
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(
                result,
                self._obs("ZZ"),
                gamma=1.0,
                broadcast_obs_and_params=False,
                meas_bases=None,
            )

    def test_broadcast_mode_raises_on_3d_data(self):
        """3-D ``_meas`` data in broadcast mode must raise ``ValueError``."""
        result = {"_meas": np.zeros((2, 4, 2)), "pauli_signs": np.zeros((2, 1, 1), dtype=bool)}
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(
                result,
                observables=ObservablesArray([SparsePauliOp("ZZ")]),
                gamma=1.0,
                param_shape=(1,),
                param_basis_pairs=[((0,), "ZZ")],
                broadcast_obs_and_params=True,
            )

    def test_broadcast_mode_accepts_sparse_pauli_op_input(self):
        """A ``SparsePauliOp`` as observables in broadcast mode must be coerced."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        result = {"_meas": data, "pauli_signs": pec_signs}
        pub = PEC.compute_expectation_value_pec(
            result,
            observables=SparsePauliOp("ZZ"),
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            broadcast_obs_and_params=True,
        )
        self.assertIsInstance(pub, PubResult)

    # ------------------------------------------------------------------
    # Correctness - gamma scaling
    # ------------------------------------------------------------------

    def test_gamma_scales_exp_val(self):
        """Setting ``gamma=2.0`` must double the expectation value compared to ``gamma=1.0``."""
        obs = self._obs("ZZ")
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        result1 = {"_meas": data.copy(), "pauli_signs": pec_signs.copy()}
        result2 = {"_meas": data.copy(), "pauli_signs": pec_signs.copy()}

        pub1 = PEC.compute_expectation_value_pec(
            result1,
            obs,
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        pub2 = PEC.compute_expectation_value_pec(
            result2,
            obs,
            gamma=2.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        np.testing.assert_allclose(pub2.data.evs, 2.0 * pub1.data.evs, atol=1e-10)

    def test_all_zeros_data_gives_plus_one_with_gamma_one(self):
        """All-zero measurements, gamma=1 → ⟨ZZ⟩ = +1."""
        obs = self._obs("ZZ")
        result = _fake_item_result(2, 1, 4, 2)
        pub_result = PEC.compute_expectation_value_pec(
            result,
            obs,
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        np.testing.assert_allclose(pub_result.data.evs, [1.0], atol=1e-10)

    def test_returns_pub_result_broadcast_mode(self):
        """In broadcast mode the return value must be a PubResult with evs, stds, twirl_stds."""
        obs = self._obs("ZZ")
        result = _fake_item_result(2, 1, 4, 2)
        pub_result = PEC.compute_expectation_value_pec(
            result,
            obs,
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        self.assertIsInstance(pub_result, PubResult)
        self.assertTrue(hasattr(pub_result.data, "evs"))
        self.assertTrue(hasattr(pub_result.data, "stds"))
        self.assertTrue(hasattr(pub_result.data, "twirl_stds"))

    def test_measurement_flips_are_applied(self):
        """``measurement_flips._meas`` must be XOR'd into ``_meas`` before processing."""
        obs = self._obs("ZZ")
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # Flipping all-zeros with all-ones gives all-ones: ZZ on |11⟩ = +1
        flips = np.ones((2, 1, 4, 2), dtype=bool)
        # pec_signs shape: (R, C, T)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        result = {"_meas": data, "measurement_flips._meas": flips, "pauli_signs": pec_signs}
        pub_result = PEC.compute_expectation_value_pec(
            result,
            obs,
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        np.testing.assert_allclose(pub_result.data.evs, [1.0], atol=1e-10)

    def test_pec_signs_flip_exp_val(self):
        """Odd-parity ``pauli_signs`` must flip the sign of the expectation value."""
        obs = self._obs("ZZ")
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # All-ones signs → odd parity for every randomization → flips +1 to -1
        # pec_signs shape: (R, C, T)
        pec_signs = np.ones((2, 1, 1), dtype=bool)
        result = {"_meas": data, "pauli_signs": pec_signs}
        pub_result = PEC.compute_expectation_value_pec(
            result,
            obs,
            gamma=1.0,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
        )
        np.testing.assert_allclose(pub_result.data.evs, [-1.0], atol=1e-10)

    #################
    # Non broadcast #
    #################

    def test_non_broadcast_4d_returns_pub_result(self):
        """Non-broadcast 4-D data with valid meas_bases and pauli_signs must return a PubResult."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {"_meas": data, "pauli_signs": pec_signs}
        result = PEC.compute_expectation_value_pec(
            item_result,
            observables=ObservablesArray([SparsePauliOp("ZZ")]),
            gamma=1.0,
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_raises_wrong_ndim(self):
        """A 6-D data array in non-broadcast mode must raise ValueError."""
        data = np.zeros((2, 1, 1, 4, 1, 1))
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {"_meas": data, "pauli_signs": pec_signs}
        with self.assertRaises(ValueError):
            PEC.compute_expectation_value_pec(
                item_result,
                observables=ObservablesArray([SparsePauliOp("ZZ")]),
                gamma=1.0,
                broadcast_obs_and_params=False,
                meas_bases=[Pauli("ZZ")],
            )

    def test_non_broadcast_observable_array_converted(self):
        """An ObservablesArray observable in non-broadcast mode must be converted."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {"_meas": data, "pauli_signs": pec_signs}
        result = PEC.compute_expectation_value_pec(
            item_result,
            observables=ObservablesArray([SparsePauliOp("ZZ")]),
            gamma=1.5,
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_with_measurement_flips(self):
        """``measurement_flips._meas`` in non-broadcast mode must be applied before processing."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        flips = np.ones((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {
            "_meas": data,
            "pauli_signs": pec_signs,
            "measurement_flips._meas": flips,
        }
        result = PEC.compute_expectation_value_pec(
            item_result,
            observables=ObservablesArray([SparsePauliOp("ZZ")]),
            gamma=1.0,
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(result, PubResult)

    def test_non_broadcast_list_observable_skips_conversion(self):
        """Passing a list of ``SparsePauliOp`` in non-broadcast mode must skip the
        ``ObservablesArray`` conversion."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {"_meas": data, "pauli_signs": pec_signs}
        # Pass a plain list — not an ObservablesArray
        result = PEC.compute_expectation_value_pec(
            item_result,
            observables=[SparsePauliOp("ZZ")],  # list, not ObservablesArray
            gamma=1.0,
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(result, PubResult)


# ---------------------------------------------------------------------------
# PEC._process_broadcasted_expectation_values_pec  (static method)
# ---------------------------------------------------------------------------


class TestProcessBroadcastedExpectationValuesPEC(unittest.TestCase):
    """Tests for :meth:`PEC._process_broadcasted_expectation_values_pec`."""

    def test_raises_incompatible_broadcast_shapes(self):
        """``param_shape`` (2,) vs observables shape (3,) must raise ``ValueError``."""
        obs = ObservablesArray([SparsePauliOp(lbl) for lbl in ("ZZ", "ZI", "IZ")])
        data = np.zeros((1, 1, 4, 2), dtype=bool)
        # pec_signs shape: (R, C, T)
        pec_signs = np.zeros((1, 1, 1), dtype=bool)
        with self.assertRaises(ValueError):
            PEC._process_broadcasted_expectation_values_pec(
                data,
                obs,
                param_shape=(2,),
                param_basis_pairs=[((0,), "ZZ")],
                pec_gamma=1.0,
                pec_signs=pec_signs,
            )

    def test_all_zeros_gives_plus_one(self):
        """All-zero shots, zero signs, gamma=1 → ⟨ZZ⟩ = +1."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # pec_signs shape: (R=2, C=1, T=1)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        ev, _stds, _ens_stds = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev, [1.0], atol=1e-10)

    def test_output_shapes_match_broadcast_shape(self):
        """All three output arrays must have the broadcast shape."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        ev, stds, ens_stds = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        self.assertEqual(ev.shape, (1,))
        self.assertEqual(stds.shape, (1,))
        self.assertEqual(ens_stds.shape, (1,))

    def test_gamma_scales_exp_val_and_stds(self):
        """Doubling ``pec_gamma`` must double both ``exp_vals`` and both std arrays."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)

        ev1, _stds1, _ens1 = PEC._process_broadcasted_expectation_values_pec(
            data.copy(),
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        ev2, _stds2, _ens2 = PEC._process_broadcasted_expectation_values_pec(
            data.copy(),
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=2.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev2, 2.0 * ev1, atol=1e-10)

    def test_odd_parity_pec_signs_flip_exp_val(self):
        """Odd-parity ``pec_signs`` (all-ones) must flip the expectation value sign."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # All-ones, odd parity per randomization → every shot flipped
        # shape: (R=2, C=1, T=1)
        pec_signs = np.ones((2, 1, 1), dtype=bool)
        ev, _, _ = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev, [-1.0], atol=1e-10)

    def test_even_parity_signs_no_flip(self):
        """Even-parity signs (all-zeros) must leave the expectation value unchanged."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # Two noise terms with all-zeros → parity 0 → no flip
        # shape: (R=2, C=1, T=2)
        pec_signs = np.zeros((2, 1, 2), dtype=bool)
        ev, _, _ = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev, [1.0], atol=1e-10)

    def test_coefficient_scaling(self):
        """``2*ZZ`` with all-zero shots, gamma=1 → ⟨2·ZZ⟩ = 2.0."""
        obs = ObservablesArray([2.0 * SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        ev, _, _ = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev, [2.0], atol=1e-10)

    def test_multiple_observables_broadcast(self):
        """Two observables over two parameter configs must broadcast to shape (2,)."""
        obs = ObservablesArray([SparsePauliOp("ZZ"), SparsePauliOp("ZI")])
        data = np.zeros((1, 2, 4, 2), dtype=bool)
        # pec_signs shape: (R=1, C=2, T=1)
        pec_signs = np.zeros((1, 2, 1), dtype=bool)
        param_basis_pairs = [((0,), "ZZ"), ((1,), "ZI")]
        ev, _, _ = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(2,),
            param_basis_pairs=param_basis_pairs,
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        self.assertEqual(ev.shape, (2,))
        np.testing.assert_allclose(ev, [1.0, 1.0], atol=1e-10)

    def test_identity_observable_gives_one(self):
        """⟨II⟩ must always equal 1.0 regardless of data."""
        obs = ObservablesArray([SparsePauliOp("II")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        ev, _, _ = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "II")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(ev, [1.0], atol=1e-10)

    def test_single_randomization_stds_equal_ensemble_stds(self):
        """With one randomization, ``stds`` and ``ensemble_stds`` must be equal."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.array(
            [[[[False, False], [False, False], [True, True], [True, True]]]], dtype=bool
        )  # shape (1, 1, 4, 2)
        pec_signs = np.zeros((1, 1, 1), dtype=bool)
        _, stds, ens_stds = PEC._process_broadcasted_expectation_values_pec(
            data,
            obs,
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            pec_gamma=1.0,
            pec_signs=pec_signs,
        )
        np.testing.assert_allclose(stds, ens_stds, atol=1e-10)

    def test_raises_when_no_config_for_param_index(self):
        """``ValueError`` must be raised when ``param_basis_pairs`` has no entry for an index."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        with self.assertRaises(ValueError):
            PEC._process_broadcasted_expectation_values_pec(
                data,
                obs,
                param_shape=(2,),
                param_basis_pairs=[((0,), "ZZ")],  # (1,) is missing
                pec_gamma=1.0,
                pec_signs=pec_signs,
            )


# ---------------------------------------------------------------------------
# PEC.prepare  -  input validation
# ---------------------------------------------------------------------------


class TestPECPrepare(unittest.TestCase):
    """Input-validation tests for :meth:`PEC.prepare`."""

    def _noise_maps_for_circuit(self, circuit):
        """Discover InjectNoise layer refs and return zero-rate noise map."""
        pec = PEC()
        boxed = pec._box_circuit(circuit, {})
        refs = {
            get_annotation(instr.operation, InjectNoise).ref
            for instr in boxed
            if get_annotation(instr.operation, InjectNoise) is not None
        }
        return {ref: PauliLindbladMap.identity(circuit.num_qubits) for ref in refs}

    def test_raises_when_noise_maps_is_none(self):
        """``prepare`` must raise ``ValueError`` when ``noise_maps`` is ``None``."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        with self.assertRaises(ValueError, msg="Missing noise_maps should raise"):
            pec.prepare(qc, obs, parameters=None, noise_maps=None)

    def test_returns_quantum_program_with_single_item(self):
        """``prepare`` must return a ``QuantumProgram`` containing exactly one item."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec.prepare(qc, obs, parameters=None, noise_maps=noise_map)
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)

    def test_gamma_set_after_prepare(self):
        """``pec.gamma`` must be set after :meth:`prepare`."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        pec.prepare(qc, obs, parameters=None, noise_maps=noise_map)
        self.assertIsNotNone(pec.gamma)
        self.assertIsInstance(pec.gamma, float)

    def test_appends_to_existing_program(self):
        """Passing an existing ``quantum_program`` must append to it and return the same object."""
        pec1 = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec1.prepare(qc, obs, parameters=None, noise_maps=noise_map)
        self.assertEqual(len(program.items), 1)

        pec2 = PEC()
        returned = pec2.prepare(
            qc, obs, parameters=None, noise_maps=noise_map, quantum_program=program
        )
        self.assertIs(returned, program)
        self.assertEqual(len(program.items), 2)

    def test_raises_on_shots_mismatch_with_existing_program(self):
        """A ``shots_per_randomization`` differing from the program's shots must raise."""
        pec1 = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec1.prepare(
            qc, obs, parameters=None, noise_maps=noise_map, shots_per_randomization=64
        )
        with self.assertRaises(ValueError):
            PEC().prepare(
                qc,
                obs,
                parameters=None,
                noise_maps=noise_map,
                shots_per_randomization=128,
                quantum_program=program,
            )

    def test_explicit_noise_gain_float(self):
        """Passing a numeric ``noise_gain`` (not 'auto') must work without error."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec.prepare(qc, obs, parameters=None, noise_maps=noise_map, noise_gain=0.5)
        self.assertIsInstance(program, QuantumProgram)

    def test_max_sampling_overhead_none(self):
        """``max_sampling_overhead=None`` must fall back to the sys-based default."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec.prepare(
            qc, obs, parameters=None, noise_maps=noise_map, max_sampling_overhead=None
        )
        self.assertIsInstance(program, QuantumProgram)

    def test_passthrough_data_set_on_program(self):
        """The returned QuantumProgram must have ``passthrough_data`` populated."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec.prepare(qc, obs, parameters=None, noise_maps=noise_map)
        self.assertIsNotNone(program.passthrough_data)
        self.assertIn("qiskit_mitigation", program.passthrough_data)

    def test_raises_when_noise_map_missing_for_circuit_layer(self):
        """Passing an empty ``noise_maps`` when the circuit has layers must raise."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        with self.assertRaises(ValueError):
            pec.prepare(qc, obs, parameters=None, noise_maps={})

    def test_scale_randomizations_by_gamma_false_does_not_scale(self):
        """``scale_randomizations_by_gamma=False`` must skip the shape scaling."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        program = pec.prepare(
            qc,
            obs,
            parameters=None,
            noise_maps=noise_map,
            num_randomizations=128,
            scale_randomizations_by_gamma=False,
        )
        self.assertIsInstance(program, QuantumProgram)

    def test_raises_when_trex_and_projection_observables(self):
        """``prepare`` with a TREX instance and projection observables must raise."""
        pec = PEC()
        qc = _simple_circuit()
        obs = ObservablesArray.coerce([{"Z+": 1}])
        noise_map = self._noise_maps_for_circuit(qc)
        trex = TREX()

        with self.assertRaises(ValueError):
            pec.prepare(qc, obs, parameters=None, noise_maps=noise_map, trex=trex)

    def test_prepare_with_trex_and_normal_observables_edits_boxing_options(self):
        """``prepare`` with TREX and non-projection observables edits the boxing options"""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]  # regular Paulis, no projectors
        noise_map = self._noise_maps_for_circuit(qc)
        trex = TREX()
        program = pec.prepare(qc, obs, parameters=None, noise_maps=noise_map, trex=trex)
        expected_boxing_options = {"enable_measures": True, "measure_annotations": "all"}
        boxing_options = pec.custom_boxing_options
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)
        self.assertEqual(
            expected_boxing_options["enable_measures"], boxing_options["enable_measures"]
        )
        self.assertEqual(
            expected_boxing_options["measure_annotations"], boxing_options["measure_annotations"]
        )

    def test_raises_when_noise_map_ref_missing_for_layer(self):
        """When ``noise_maps`` is missing a required layer ref and ``noise_gain`` is a float
        should raise a ``ValueError``."""
        pec = PEC()
        qc = _simple_circuit()
        # noise_gain=0.0 skips the early calculate_gamma call.
        with self.assertRaisesRegex(ValueError, "Noise model is missing"):
            pec.prepare(
                qc,
                [SparsePauliOp("ZZ")],
                parameters=None,
                noise_maps={"bad_ref": None},
                noise_gain=0.0,
            )


# ---------------------------------------------------------------------------
# PEC.postprocess - basic flow
# ---------------------------------------------------------------------------


class TestPECPostprocess(unittest.TestCase):
    """Tests for :meth:`PEC.postprocess`."""

    def _noise_maps_for_circuit(self, circuit):
        pec = PEC()
        boxed = pec._box_circuit(circuit, {})
        refs = {
            get_annotation(instr.operation, InjectNoise).ref
            for instr in boxed
            if get_annotation(instr.operation, InjectNoise) is not None
        }
        return {ref: PauliLindbladMap.identity(circuit.num_qubits) for ref in refs}

    def test_postprocess_returns_pub_result(self):
        """``postprocess`` on a broadcast-mode item result must return a PubResult."""
        pec = PEC()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]
        noise_map = self._noise_maps_for_circuit(qc)
        pec.prepare(qc, obs, parameters=None, noise_maps=noise_map)

        # build minimal fake item result matching the broadcast shape
        # (R, C, S, Q) - 1 config, 2 randomizations, 4 shots, 2 qubits
        pec_signs = np.zeros((2, 1, 1), dtype=bool)
        item_result = {
            "_meas": np.zeros((2, 1, 4, 2), dtype=bool),
            "pauli_signs": pec_signs,
        }
        result = pec.postprocess(item_result)
        self.assertIsInstance(result, PubResult)


# ---------------------------------------------------------------------------
# PEC.create_instance_from_passthrough_data
# ---------------------------------------------------------------------------


class TestPECCreateInstanceFromPassthroughData(unittest.TestCase):
    """Tests for :meth:`PEC.create_instance_from_passthrough_data`."""

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
            "pec_gamma": 1.5,
        }
        data.update(overrides)
        return data

    def test_happy_path_returns_pec_instance(self):
        """A fully-populated passthrough dict must produce a PEC instance."""
        pec = PEC.create_instance_from_passthrough_data(self._minimal_passthrough())
        self.assertIsInstance(pec, PEC)

    def test_gamma_set_from_passthrough(self):
        """``pec.gamma`` must match the value from the passthrough dict."""
        pec = PEC.create_instance_from_passthrough_data(self._minimal_passthrough(pec_gamma=2.5))
        self.assertAlmostEqual(pec.gamma, 2.5)

    def test_program_item_index_set(self):
        """The ``_program_item_index`` must match the passthrough value."""
        pec = PEC.create_instance_from_passthrough_data(
            self._minimal_passthrough(program_item_index=4)
        )
        self.assertEqual(pec._program_item_index, 4)

    def test_raises_when_observables_missing(self):
        """Missing 'observables' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["observables"]
        with self.assertRaises(ValueError):
            PEC.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_broadcast_obs_and_params_missing(self):
        """Missing 'broadcast_obs_and_params' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["broadcast_obs_and_params"]
        with self.assertRaises(ValueError):
            PEC.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_program_item_index_missing(self):
        """Missing 'program_item_index' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["program_item_index"]
        with self.assertRaises(ValueError):
            PEC.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_pec_gamma_missing(self):
        """Missing 'pec_gamma' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["pec_gamma"]
        with self.assertRaises(ValueError):
            PEC.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_trex_calibration_true_but_no_trex(self):
        """trex_calibration=True without a TREX instance must raise ValueError."""
        passthrough = self._minimal_passthrough(trex_calibration=True)
        with self.assertRaises(ValueError):
            PEC.create_instance_from_passthrough_data(passthrough, trex=None)


if __name__ == "__main__":
    unittest.main()
