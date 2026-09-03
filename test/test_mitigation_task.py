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

"""Tests for the ``MitigationTask`` class."""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, ClassicalRegister, Instruction, Parameter
from qiskit.primitives import PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_mitigation.mitigation_task import MitigationTask
from qiskit_mitigation.trex import TREX
from samplomatic import build
from samplomatic.quantum_program import QuantumProgram, QuantumProgramResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_circuit(num_qubits: int = 2) -> QuantumCircuit:
    """Return a simple non-parametric circuit with a Bell-pair structure."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _parametric_circuit() -> QuantumCircuit:
    """Return a single-parameter circuit."""
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.rz(theta, 0)
    qc.cx(0, 1)
    return qc


def _fake_item_result(
    num_randomizations: int,
    num_configs: int,
    shots: int,
    num_qubits: int,
    data: np.ndarray | None = None,
) -> dict:
    """Return a minimal item-result dict with shape (R, C, S, Q)."""
    if data is None:
        data = np.zeros((num_randomizations, num_configs, shots, num_qubits), dtype=bool)
    return {"_meas": data}


# ---------------------------------------------------------------------------
# MitigationTask.__init__
# ---------------------------------------------------------------------------


class TestMitigationTaskInit(unittest.TestCase):
    def test_all_attributes_are_none(self):
        task = MitigationTask()
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
            self.assertIsNone(getattr(task, attr), msg=f"{attr} should be None after __init__")


# ---------------------------------------------------------------------------
# MitigationTask._save_basic_variables
# ---------------------------------------------------------------------------


class TestSaveBasicVariables(unittest.TestCase):
    def setUp(self):
        self.task = MitigationTask()
        self.circuit = _simple_circuit()
        self.obs = [SparsePauliOp("ZZ")]

    def test_stores_circuit_and_observables(self):
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 64, 128, True, None, None
        )
        self.assertIs(self.task.circuit, self.circuit)
        self.assertIsInstance(self.task.observables, ObservablesArray)

    def test_sparse_observables_created(self):
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 64, 128, True, None, None
        )
        self.assertIsNotNone(self.task._sparse_observables)
        self.assertEqual(len(self.task._sparse_observables), 1)

    def test_shots_and_randomizations_stored(self):
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 32, 64, True, None, None
        )
        self.assertEqual(self.task.shots_per_randomization, 32)
        self.assertEqual(self.task.num_randomizations, 64)

    def test_broadcast_shape_no_parameters(self):
        """With no parameters, broadcast_shape equals observables.shape."""
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 64, 128, True, None, None
        )
        self.assertEqual(self.task.broadcast_shape, (1,))

    def test_broadcast_shape_with_matching_parameters(self):
        """Matching-length observables and parameters broadcast to that length."""
        qc = _parametric_circuit()
        obs = [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")]
        params = np.array([[0.1], [0.2], [0.3]])
        self.task._save_basic_variables(qc, obs, params, None, 64, 128, True, None, None)
        self.assertEqual(self.task.broadcast_shape, (3,))

    def test_outer_product_shape(self):
        """With broadcast_obs_and_params=False, shape is the outer product."""
        qc = _parametric_circuit()
        obs = [SparsePauliOp("ZZ"), SparsePauliOp("IZ")]  # shape (2,)
        params = np.array([[0.1], [0.2], [0.3]])  # shape (3,)
        self.task._save_basic_variables(qc, obs, params, None, 64, 128, False, None, None)
        self.assertEqual(self.task.broadcast_shape, (2, 3))

    def test_ndarray_parameters_converted_to_bindings_array(self):
        """A bare np.ndarray is coerced into a BindingsArray."""
        qc = _parametric_circuit()
        params = np.array([[0.1], [0.2]])
        self.task._save_basic_variables(
            qc, [SparsePauliOp("ZZ"), SparsePauliOp("ZI")], params, None, 64, 128, True, None, None
        )
        self.assertIsInstance(self.task.parameters, BindingsArray)

    def test_custom_boxing_options_deepcopied(self):
        """Mutating the original dict after _save_basic_variables must not affect the task."""
        original = {"key": "value"}
        self.task._save_basic_variables(
            self.circuit, self.obs, None, original, 64, 128, True, None, None
        )
        original["key"] = "mutated"
        self.assertEqual(self.task.custom_boxing_options["key"], "value")

    def test_none_boxing_options_becomes_empty_dict(self):
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 64, 128, True, None, None
        )
        self.assertEqual(self.task.custom_boxing_options, {})

    def test_raises_on_incompatible_broadcast_shapes(self):
        """Observables shape (3,) and parameters shape (2,) must raise ValueError."""
        qc = _parametric_circuit()
        obs = [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")]
        params = np.array([[0.1], [0.2]])
        with self.assertRaises(ValueError):
            self.task._save_basic_variables(qc, obs, params, None, 64, 128, True, None, None)

    def test_raises_when_quantum_program_shots_mismatch(self):
        """If an existing QuantumProgram has different shots, a ValueError is raised."""
        mock_qp = MagicMock()
        mock_qp.shots = 32
        with self.assertRaises(ValueError):
            self.task._save_basic_variables(
                self.circuit, self.obs, None, None, 64, 128, True, None, mock_qp
            )

    def test_matching_quantum_program_shots_accepted(self):
        """A QuantumProgram whose shots match shots_per_randomization must not raise."""
        mock_qp = MagicMock()
        mock_qp.shots = 64
        # Should not raise
        self.task._save_basic_variables(
            self.circuit, self.obs, None, None, 64, 128, True, None, mock_qp
        )


# ---------------------------------------------------------------------------
# MitigationTask._box_circuit
# ---------------------------------------------------------------------------


class TestBoxCircuit(unittest.TestCase):
    def setUp(self):
        self.task = MitigationTask()

    def test_returns_quantum_circuit(self):
        qc = _simple_circuit()
        boxed = self.task._box_circuit(qc, {})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_adds_meas_register(self):
        qc = _simple_circuit()
        boxed = self.task._box_circuit(qc, {})
        reg_names = [r.name for r in boxed.cregs]
        self.assertIn("_meas", reg_names)

    def test_enable_measures_injected_when_absent(self):
        """enable_measures should be set to True automatically."""
        options = {}
        qc = _simple_circuit()
        self.task._box_circuit(qc, options)
        self.assertTrue(options["enable_measures"])

    def test_measure_annotations_set_to_change_basis_when_absent(self):
        options = {}
        self.task._box_circuit(_simple_circuit(), options)
        self.assertEqual(options["measure_annotations"], "change_basis")

    def test_raises_when_enable_measures_is_false(self):
        with self.assertRaises(ValueError):
            self.task._box_circuit(_simple_circuit(), {"enable_measures": False})

    def test_existing_measurements_removed_before_boxing(self):
        """A circuit with pre-existing measurements should not cause double-measure issues."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()  # adds existing measurements
        boxed = self.task._box_circuit(qc, {})
        self.assertIsInstance(boxed, QuantumCircuit)

    def test_raises_when_circuit_already_has_meas_register(self):
        """A circuit that already has a ``'_meas'`` classical register must raise ``ValueError``."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # Manually add a '_meas' register to trigger the collision path
        qc.add_register(ClassicalRegister(2, "_meas"))
        with self.assertRaises(ValueError, msg="Name collision on '_meas' should raise"):
            MitigationTask._box_circuit(qc, {})

    def test_raises_on_bad_boxing_options(self):
        """Passing boxing options that cause ``generate_boxing_pass_manager`` to fail must raise."""
        with self.assertRaises(ValueError):
            MitigationTask._box_circuit(_simple_circuit(), {"invalid_option_xyz": True})

    def test_measure_annotations_twirl_is_rewritten_to_all(self):
        """``measure_annotations='twirl'`` must be rewritten to ``'all'``."""
        # Must also supply enable_measures=True, otherwise the "absent" branch
        # overwrites measure_annotations to "change_basis" before the twirl check.
        options = {"enable_measures": True, "measure_annotations": "twirl"}
        qc = _simple_circuit()
        MitigationTask._box_circuit(qc, options)
        self.assertEqual(options["measure_annotations"], "all")

    def test_invalid_boxing_options_raise_value_error(self):
        """Passing unknown options that fail boxing should raise a ``ValueError``."""
        with self.assertRaises(ValueError):
            # Passing `enable_measures=False` will raise from the base guard clause
            MitigationTask._box_circuit(_simple_circuit(), {"enable_measures": False})


# ---------------------------------------------------------------------------
# MitigationTask.compute_expectation_value  (static method)
# ---------------------------------------------------------------------------


class TestComputeExpectationValue(unittest.TestCase):
    def _obs(self, *labels):
        return ObservablesArray([SparsePauliOp(lbl) for lbl in labels])

    def test_raises_missing_meas_key(self):
        result = {"wrong_key": np.zeros((2, 1, 4, 2))}
        with self.assertRaises(ValueError, msg="Should raise for missing '_meas'"):
            MitigationTask.compute_expectation_value(result, self._obs("ZZ"))

    def test_raises_wrong_ndim(self):
        result = {"_meas": np.zeros((2, 1, 4))}  # 3-D, not 4-D
        with self.assertRaises(ValueError):
            MitigationTask.compute_expectation_value(result, self._obs("ZZ"))

    def test_raises_broadcast_true_without_param_basis_pairs(self):
        result = _fake_item_result(2, 1, 4, 2)
        with self.assertRaises(ValueError):
            MitigationTask.compute_expectation_value(
                result, self._obs("ZZ"), broadcast_obs_and_params=True, param_basis_pairs=None
            )

    def test_raises_broadcast_false_without_basis_obs_term_map(self):
        result = _fake_item_result(2, 1, 4, 2)
        with self.assertRaises(ValueError):
            MitigationTask.compute_expectation_value(
                result, self._obs("ZZ"), broadcast_obs_and_params=False, meas_bases=None
            )

    def test_all_zeros_gives_exp_val_plus_one(self):
        """All-zero shots → every qubit measured |0⟩ → ZZ eigenvalue = (+1)(+1) = +1."""
        obs = self._obs("ZZ")
        result = _fake_item_result(2, 1, 4, 2, data=np.zeros((2, 1, 4, 2), dtype=bool))
        pub_result = MitigationTask.compute_expectation_value(
            result, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(pub_result.data.evs, [1.0], atol=1e-10)

    def test_all_ones_gives_exp_val_plus_one_for_zz(self):
        """All-one shots → every qubit |1⟩ → ZZ eigenvalue = (-1)(-1) = +1."""
        obs = self._obs("ZZ")
        data = np.ones((2, 1, 4, 2), dtype=bool)
        result = _fake_item_result(2, 1, 4, 2, data=data)
        pub_result = MitigationTask.compute_expectation_value(
            result, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(pub_result.data.evs, [1.0], atol=1e-10)

    def test_half_ones_gives_exp_val_minus_one_for_zi(self):
        """For ZI (Z on qubit 1, I on qubit 0): qubit 1 all-ones → ⟨ZI⟩ = -1."""
        obs = self._obs("ZI")
        # data shape (R, C, S, Q): qubit index 0 is rightmost in label → last array axis
        # ZI: Z on qubit index 1 (second from right in 2-qubit label)
        # We want qubit 1 always |1⟩ → column index 1 all True
        data = np.zeros((1, 1, 4, 2), dtype=bool)
        data[:, :, :, 1] = True  # qubit 1 = |1⟩
        result = _fake_item_result(1, 1, 4, 2, data=data)
        pub_result = MitigationTask.compute_expectation_value(
            result, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZI")]
        )
        np.testing.assert_allclose(pub_result.data.evs, [-1.0], atol=1e-10)

    def test_measurement_flips_are_applied(self):
        """Flipping all-zeros bool data with all-ones mask gives all-ones → ZZ = +1."""
        obs = self._obs("ZZ")
        # Use bool dtype: False XOR True = True, as in real shot data
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        flips = np.ones((2, 1, 4, 2), dtype=bool)
        result = {"_meas": data, "measurement_flips._meas": flips}
        pub_result = MitigationTask.compute_expectation_value(
            result, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(pub_result.data.evs, [1.0], atol=1e-10)

    def test_returns_pub_result(self):
        """Broadcast mode returns a PubResult with evs, stds, and twirl_stds fields."""
        obs = self._obs("ZZ")
        result = _fake_item_result(2, 1, 4, 2)
        pub_result = MitigationTask.compute_expectation_value(
            result, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        self.assertIsInstance(pub_result, PubResult)
        self.assertTrue(hasattr(pub_result.data, "evs"))
        self.assertTrue(hasattr(pub_result.data, "stds"))
        self.assertTrue(hasattr(pub_result.data, "twirl_stds"))

    def test_broadcast_mode_raises_on_3d_data(self):
        """3-D data in broadcast mode must raise ``ValueError``."""
        result = {"_meas": np.zeros((2, 4, 2))}  # 3-D
        with self.assertRaises(ValueError):
            MitigationTask.compute_expectation_value(
                result,
                ObservablesArray([SparsePauliOp("ZZ")]),
                param_shape=(1,),
                param_basis_pairs=[((0,), "ZZ")],
                broadcast_obs_and_params=True,
            )

    # --------------------
    # non-broadcast mode
    # --------------------
    def test_non_broadcast_4d_returns_pub_result(self):
        """Non-broadcast 4-D data with a provided meas_bases must return a PubResult."""
        result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        pub_result = MitigationTask.compute_expectation_value(
            result,
            self._obs("ZZ"),
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(pub_result, PubResult)

    def test_non_broadcast_raises_wrong_ndim(self):
        """A 3-D data array in non-broadcast mode must raise ValueError."""
        result = {"_meas": np.zeros((2, 1, 4))}
        with self.assertRaises(ValueError):
            MitigationTask.compute_expectation_value(
                result,
                self._obs("ZZ"),
                broadcast_obs_and_params=False,
                meas_bases=["ZZ"],
            )

    def test_non_broadcast_sparse_pauli_op_observable(self):
        """A ``SparsePauliOp`` observable in non-broadcast mode must be handled."""
        result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        pub_result = MitigationTask.compute_expectation_value(
            result,
            SparsePauliOp("ZZ"),
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(pub_result, PubResult)

    def test_broadcast_mode_sparse_pauli_op_coerced(self):
        """A ``SparsePauliOp`` passed to broadcast mode must be coerced to ObservablesArray."""
        result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        pub_result = MitigationTask.compute_expectation_value(
            result,
            SparsePauliOp("ZZ"),
            param_shape=(1,),
            param_basis_pairs=[((0,), "ZZ")],
            broadcast_obs_and_params=True,
        )
        self.assertIsInstance(pub_result, PubResult)

    def test_non_broadcast_measurement_flips_applied(self):
        """``measurement_flips._meas`` in non-broadcast mode must be applied before processing."""
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        flips = np.ones((2, 1, 4, 2), dtype=bool)
        result = {"_meas": data, "measurement_flips._meas": flips}
        pub_result = MitigationTask.compute_expectation_value(
            result,
            self._obs("ZZ"),
            broadcast_obs_and_params=False,
            meas_bases=[Pauli("ZZ")],
        )
        self.assertIsInstance(pub_result, PubResult)


# ---------------------------------------------------------------------------
# MitigationTask._process_broadcasted_expectation_values  (static method)
# ---------------------------------------------------------------------------


class TestProcessBroadcastedExpectationValues(unittest.TestCase):
    def test_raises_on_incompatible_shapes(self):
        """param_shape (2,) and observables shape (3,) cannot broadcast → ValueError."""
        obs = ObservablesArray([SparsePauliOp(lbl) for lbl in ("ZZ", "ZI", "IZ")])
        data = np.zeros((1, 1, 4, 2), dtype=bool)
        with self.assertRaises(ValueError):
            MitigationTask._process_broadcasted_expectation_values(
                data, obs, param_shape=(2,), param_basis_pairs=[((0,), "ZZ")]
            )

    def test_scalar_observable_all_zeros(self):
        """All-zero shots → ⟨ZZ⟩ = +1."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        exp_vals, stds, ens_stds = MitigationTask._process_broadcasted_expectation_values(
            data, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(exp_vals, [1.0], atol=1e-10)
        self.assertEqual(exp_vals.shape, (1,))
        self.assertEqual(stds.shape, (1,))
        self.assertEqual(ens_stds.shape, (1,))

    def test_num_randomizations_1_stds_equal_ensemble_stds(self):
        """With a single randomization, stds and ensemble_stds must be equal."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        # 2 shots |00⟩, 2 shots |11⟩ → 50/50 split → ⟨ZZ⟩ = +1 (both eigenvalues +1)
        data = np.array(
            [[[[False, False], [False, False], [True, True], [True, True]]]], dtype=bool
        )  # shape (1, 1, 4, 2)
        _exp_vals, stds, ens_stds = MitigationTask._process_broadcasted_expectation_values(
            data, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(stds, ens_stds, atol=1e-10)

    def test_identity_observable_exp_val_is_one(self):
        """⟨II⟩ must always equal 1.0 regardless of measurement data."""
        obs = ObservablesArray([SparsePauliOp("II")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        exp_vals, _, _ = MitigationTask._process_broadcasted_expectation_values(
            data, obs, param_shape=(1,), param_basis_pairs=[((0,), "II")]
        )
        np.testing.assert_allclose(exp_vals, [1.0], atol=1e-10)

    def test_multiple_observables_broadcast(self):
        """A (2,) observable array broadcasts against a (2,) param_shape correctly."""
        obs = ObservablesArray([SparsePauliOp("ZZ"), SparsePauliOp("ZI")])
        # 2 configs: config 0 for ZZ basis, config 1 for ZI basis (both all-zeros)
        data = np.zeros((1, 2, 4, 2), dtype=bool)
        param_basis_pairs = [((0,), "ZZ"), ((1,), "ZI")]
        exp_vals, _stds, _ens_stds = MitigationTask._process_broadcasted_expectation_values(
            data, obs, param_shape=(2,), param_basis_pairs=param_basis_pairs
        )
        self.assertEqual(exp_vals.shape, (2,))
        np.testing.assert_allclose(exp_vals, [1.0, 1.0], atol=1e-10)

    def test_coefficient_scaling(self):
        """2*ZZ with all-zero shots → ⟨2·ZZ⟩ = 2.0."""
        obs = ObservablesArray([2.0 * SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        exp_vals, _, _ = MitigationTask._process_broadcasted_expectation_values(
            data, obs, param_shape=(1,), param_basis_pairs=[((0,), "ZZ")]
        )
        np.testing.assert_allclose(exp_vals, [2.0], atol=1e-10)

    def test_raises_when_no_config_for_param_index(self):
        """``ValueError`` must be raised when param_basis_pairs has no entry for a given index."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        data = np.zeros((2, 1, 4, 2), dtype=bool)
        # param_basis_pairs covers index (0,) but broadcast_shape is (2,) → (1,) will miss
        with self.assertRaises(ValueError):
            MitigationTask._process_broadcasted_expectation_values(
                data,
                obs,
                param_shape=(2,),
                param_basis_pairs=[((0,), "ZZ")],  # only (0,) — (1,) is missing
            )


class TestUnbroadcastIndex(unittest.TestCase):
    """Tests for :func:`unbroadcast_index`."""

    def test_same_ndim_identity(self):
        """When shape and bc_index have the same number of dims, index is returned unchanged."""
        result = MitigationTask._unbroadcast_index((2, 3), (5, 5))
        self.assertEqual(result, (2, 3))

    def test_scalar_shape_returns_empty_tuple(self):
        """A shape of () (0-dimensional) always returns ()."""
        self.assertEqual(MitigationTask._unbroadcast_index((0, 1), ()), ())

    def test_broadcast_leading_dims_clamped_to_zero(self):
        """Dimensions of size 1 in the array shape must map their index to 0."""
        # shape (1, 5): bc_index (3, 2) → dim-0 has size 1 → index must be 0
        result = MitigationTask._unbroadcast_index((3, 2), (1, 5))
        self.assertEqual(result, (0, 2))

    def test_padded_shape_drops_leading_dims(self):
        """bc_index longer than shape → leading dims are dropped."""
        # array shape (5,), bc_index (7, 3) → only last element matters
        result = MitigationTask._unbroadcast_index((7, 3), (5,))
        self.assertEqual(result, (3,))

    def test_all_broadcast_dims_clamp_to_zero(self):
        """A (1, 1) shape with any bc_index always returns (0, 0)."""
        result = MitigationTask._unbroadcast_index((4, 9), (1, 1))
        self.assertEqual(result, (0, 0))

    def test_single_dim_no_broadcast(self):
        result = MitigationTask._unbroadcast_index((4,), (10,))
        self.assertEqual(result, (4,))

    def test_returns_tuple(self):
        result = MitigationTask._unbroadcast_index((1, 2), (3, 4))
        self.assertIsInstance(result, tuple)

    def test_3d_bc_index_against_1d_shape(self):
        """bc_index (2, 5, 3) against shape (7,) → only last element: (3,)."""
        result = MitigationTask._unbroadcast_index((2, 5, 3), (7,))
        self.assertEqual(result, (3,))


# ---------------------------------------------------------------------------
# MitigationTask._compute_param_basis_pairs  (static method)
# ---------------------------------------------------------------------------


class TestComputeParamBasisPairs(unittest.TestCase):
    """Tests for :meth:`MitigationTask._compute_param_basis_pairs`."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs(*labels):
        """Build an ObservablesArray from Pauli label strings."""
        return ObservablesArray([SparsePauliOp(lbl) for lbl in labels])

    @staticmethod
    def _bases(*labels):
        """Build a PauliList from Pauli label strings."""
        return PauliList([Pauli(lbl) for lbl in labels])

    # ------------------------------------------------------------------
    # return-type / structural tests
    # ------------------------------------------------------------------

    def test_returns_list_of_tuples(self):
        """Return value must be a list of (tuple, str) pairs."""
        obs = self._obs("ZZ")
        bases = self._bases("ZZ")
        result = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            ndindex, basis_label = item
            self.assertIsInstance(ndindex, tuple)
            self.assertIsInstance(basis_label, str)

    # ------------------------------------------------------------------
    # single observable / single parameter
    # ------------------------------------------------------------------

    def test_single_obs_single_param_zz(self):
        """A single ZZ observable with a ZZ basis produces exactly one pair."""
        obs = self._obs("ZZ")
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertEqual(len(pairs), 1)
        ndindex, label = pairs[0]
        self.assertEqual(ndindex, (0,))
        self.assertEqual(label, "ZZ")

    def test_single_obs_single_param_zi_measured_by_zz(self):
        """ZI is qubit-wise commuting with ZZ: ZZ basis should cover it."""
        obs = self._obs("ZI")
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], "ZZ")

    def test_single_obs_single_param_xx_not_covered_by_zz(self):
        """XX is NOT qubit-wise commuting with ZZ: no pair should be produced."""
        obs = self._obs("XX")
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertEqual(len(pairs), 0)

    def test_identity_observable_matches_any_basis(self):
        """II has no non-identity support → it matches every basis; one basis entry expected."""
        obs = self._obs("II")
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        # II observable produces no terms with non-trivial support — the loop over terms
        # yields the identity; pauli_support is all-False so any basis matches.
        self.assertEqual(len(pairs), 1)

    # ------------------------------------------------------------------
    # multiple parameters
    # ------------------------------------------------------------------

    def test_two_params_two_separate_observables(self):
        """Two distinct parameter indices each require one basis entry."""
        obs = self._obs("ZZ", "XX")
        bases = self._bases("ZZ", "XX")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (2,), (2,), bases)
        self.assertEqual(len(pairs), 2)
        ndindices = {p[0] for p in pairs}
        self.assertIn((0,), ndindices)
        self.assertIn((1,), ndindices)

    def test_two_params_same_observable_broadcast(self):
        """observables shape (1,) broadcast against parameters shape (2,) → two param
        indices both map to the same single basis."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])  # shape (1,)
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (2,), (2,), bases)
        # Both param indices (0,) and (1,) map to ZZ
        self.assertEqual(len(pairs), 2)
        labels = {p[1] for p in pairs}
        self.assertEqual(labels, {"ZZ"})

    # ------------------------------------------------------------------
    # broadcasting
    # ------------------------------------------------------------------

    def test_broadcast_shape_obs_1_params_3(self):
        """observables (1,) broadcast to parameters (3,) → 3 pairs, one per param index."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (3,), (3,), bases)
        ndindices = [p[0] for p in pairs]
        self.assertEqual(sorted(ndindices), [(0,), (1,), (2,)])

    def test_2d_broadcast_shape(self):
        """2-D broadcast shape: each cell maps to its (row, col) param index."""
        obs = ObservablesArray([[SparsePauliOp("ZZ"), SparsePauliOp("ZZ")]])
        # obs shape (1, 2), params shape (2, 2), broadcast_shape (2, 2)
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (2, 2), (2, 2), bases)
        ndindices = {p[0] for p in pairs}
        self.assertIn((0, 0), ndindices)
        self.assertIn((0, 1), ndindices)
        self.assertIn((1, 0), ndindices)
        self.assertIn((1, 1), ndindices)

    # ------------------------------------------------------------------
    # multiple bases per parameter index
    # ------------------------------------------------------------------

    def test_observable_with_two_noncommuting_terms_requires_two_bases(self):
        """An observable ZZ + XX requires two separate bases (ZZ and XX)."""
        obs = ObservablesArray([SparsePauliOp(["ZZ", "XX"], [1.0, 1.0])])
        bases = self._bases("ZZ", "XX")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertEqual(len(pairs), 2)
        labels = {p[1] for p in pairs}
        self.assertIn("ZZ", labels)
        self.assertIn("XX", labels)

    # ------------------------------------------------------------------
    # ordering preserved
    # ------------------------------------------------------------------

    def test_basis_order_matches_measure_bases_order(self):
        """Bases in the output must appear in the same order as in ``measure_bases``."""
        obs = ObservablesArray([SparsePauliOp(["ZZ", "XX", "YY"], [1.0, 1.0, 1.0])])
        # Provide bases in a deliberate order: XX first, then YY, then ZZ
        bases = self._bases("XX", "YY", "ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (1,), bases)
        self.assertEqual(len(pairs), 3)
        labels = [p[1] for p in pairs]
        self.assertEqual(labels, ["XX", "YY", "ZZ"])

    # ------------------------------------------------------------------
    # no duplicate param indices accumulated across broadcast iterations
    # ------------------------------------------------------------------

    def test_no_duplicate_basis_per_param_index(self):
        """The same basis should not be listed twice for a single param index."""
        # obs shape (1,), params shape (3,): same param index can be reached multiple
        # times when broadcast_shape > params_shape; deduplication must occur.
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        bases = self._bases("ZZ")
        pairs = MitigationTask._compute_param_basis_pairs(obs, (1,), (3,), bases)
        # param_shape (1,) → only one param index (0,)
        ndindices = [p[0] for p in pairs]
        self.assertEqual(ndindices.count((0,)), 1)


# ---------------------------------------------------------------------------
# MitigationTask._extract_item_data - TREX warning paths
# ---------------------------------------------------------------------------


class TestExtractItemData(unittest.TestCase):
    """Tests for warning paths in :meth:`MitigationTask._extract_item_data`."""

    def test_warns_when_trex_set_but_no_calibration_and_item_result(self):
        """When results is a bare item-result and TREX is set but no measure_noise_data,
        a warning must be emitted."""
        task = MitigationTask()
        trex = TREX()
        task.trex = trex
        task._program_item_index = 0

        item_result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}

        with self.assertWarns(UserWarning):
            result, noise_data = task._extract_item_data(item_result, None)

        self.assertIs(result, item_result)
        self.assertIsNone(noise_data)

    def test_no_warning_when_trex_is_none(self):
        """No warning should be emitted when TREX is not enabled."""
        task = MitigationTask()
        task.trex = None
        task._program_item_index = 0

        item_result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, _measure_noise_data = task._extract_item_data(item_result, None)

        self.assertIs(result, item_result)

    def test_warns_when_trex_set_no_calibration_and_full_result(self):
        """When a ``QuantumProgramResult`` is passed and TREX has no calibration,
        a warning must be emitted."""
        task = MitigationTask()
        task._program_item_index = 0
        trex = TREX()
        task.trex = trex

        data = {"_meas": np.zeros((5, 4, 64, 2), dtype=bool)}
        result = QuantumProgramResult(data)

        with self.assertWarns(UserWarning):
            task._extract_item_data(result, None)

    def test_calls_compute_noise_model_when_trex_has_calibration(self):
        """When a ``QuantumProgramResult`` is passed and TREX has a calibration result,
        ``compute_noise_model`` must be called."""
        task = MitigationTask()
        task._program_item_index = 0

        trex = TREX()
        # Simulate that TREX.prepare() was called by setting the index
        trex._program_item_index = 1
        task.trex = trex

        data = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        result = QuantumProgramResult(data)

        sentinel_noise_data = object()
        with patch.object(TREX, "compute_noise_model", return_value=sentinel_noise_data):
            _, noise_data = task._extract_item_data(result, None)

        self.assertIs(noise_data, sentinel_noise_data)

    def test_no_warning_when_trex_none_and_full_result(self):
        """When a ``QuantumProgramResult`` is passed and ``trex=None``, no warning must be
        emitted and ``measure_noise_data`` must stay ``None``."""
        task = MitigationTask()
        task.trex = None
        task._program_item_index = 0

        data = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        result = QuantumProgramResult(data)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _item_result, noise_data = task._extract_item_data(result, None)

        self.assertIsNone(noise_data)


# ---------------------------------------------------------------------------
# MitigationTask._add_data_to_passthrough_data - all branches
# ---------------------------------------------------------------------------


class TestAddDataToPassthroughData(unittest.TestCase):
    """Tests for :meth:`MitigationTask._add_data_to_passthrough_data`."""

    def test_initializes_passthrough_data_when_none(self):
        """When passthrough_data is None, it must be initialised as a dict."""
        qp = MagicMock()
        qp.passthrough_data = None

        data = {"key": "value"}
        MitigationTask._add_data_to_passthrough_data(data, qp)

        self.assertEqual(qp.passthrough_data, {"qiskit_mitigation": [data]})

    def test_appends_when_qiskit_mitigation_key_absent(self):
        """If passthrough_data exists but has no 'qiskit_mitigation' key, it must be added."""
        qp = MagicMock()
        qp.passthrough_data = {"other": "stuff"}

        data = {"key": "value"}
        MitigationTask._add_data_to_passthrough_data(data, qp)

        self.assertEqual(qp.passthrough_data["qiskit_mitigation"], [data])

    def test_appends_when_qiskit_mitigation_key_present(self):
        """If passthrough_data already has 'qiskit_mitigation', the new data must be appended."""
        first = {"first": "item"}
        second = {"second": "item"}

        qp = MagicMock()
        qp.passthrough_data = {"qiskit_mitigation": [first]}

        MitigationTask._add_data_to_passthrough_data(second, qp)

        self.assertEqual(qp.passthrough_data["qiskit_mitigation"], [first, second])


# ---------------------------------------------------------------------------
# MitigationTask.find_unique_layers
# ---------------------------------------------------------------------------


class TestFindUniqueLayers(unittest.TestCase):
    """Tests for :meth:`MitigationTask.find_unique_layers`."""

    def test_returns_list_of_circuit_instructions(self):
        """The return value must be a non-empty list."""
        task = MitigationTask()
        qc = _simple_circuit()
        layers = task.find_unique_layers(qc)
        self.assertIsInstance(layers, list)
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertIsInstance(layer, CircuitInstruction)

    def test_returns_list_with_none_options(self):
        """``custom_boxing_options=None`` must also return a valid list."""

        task = MitigationTask()
        qc = _simple_circuit()
        layers = task.find_unique_layers(qc, custom_boxing_options=None)
        self.assertIsInstance(layers, list)
        self.assertTrue(len(layers) > 0)


class TestMakeSamplexArguments(unittest.TestCase):
    """Tests for :meth:`MitigationTask._make_samplex_arguments`."""

    def test_make_samplex_arguments_non_box_instruction(self):
        """Test handling of a non-box instruction before the ChangeBasis box"""
        task = MitigationTask()
        qc = _simple_circuit()
        boxed = task._box_circuit(qc, {})
        _template, samplex = build(boxed)
        change_basis = np.zeros((1, 2), dtype=np.uint8)

        # Collect real reverse ops.
        real_reverse_ops = list(boxed.reverse_ops())

        # Build a fake instruction that is NOT a box (name="barrier").
        dummy_instr = CircuitInstruction(Instruction("barrier", 2, 0, []), [0, 1])

        def fake_reverse_ops():
            # Yield the dummy (non-matching) instruction first, then the real reversed ops.
            yield dummy_instr
            yield from real_reverse_ops

        with patch.object(boxed, "reverse_ops", fake_reverse_ops):
            result = MitigationTask._make_samplex_arguments(
                samplex, boxed, np.empty(0), change_basis
            )

        # The dummy instruction caused one extra loop iteration before the real
        # ChangeBasis box was found, and the function returned normally.
        self.assertIsInstance(result, dict)

    def test_raises_when_no_change_basis_box_found(self):
        """When boxed_circuit does not contain a ChangeBasis, should raise ``ValueError``."""
        task = MitigationTask()
        qc = _simple_circuit()
        boxed = task._box_circuit(qc, {})
        _template, samplex = build(boxed)

        # Replace boxed_circuit with an empty circuit.
        empty_qc = QuantumCircuit(2)
        change_basis = np.zeros((1, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            MitigationTask._make_samplex_arguments(samplex, empty_qc, np.empty(0), change_basis)

    def test_raises_when_flat_parameter_values_empty_for_parametric_samplex(self):
        """When the samplex has ``parameter_values`` inputs but ``flat_parameter_values``
        is empty, should raise ``ValueError``."""
        task = MitigationTask()
        qc = _parametric_circuit()
        boxed = task._box_circuit(qc, {})
        _template, samplex = build(boxed)
        change_basis = np.zeros((1, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            MitigationTask._make_samplex_arguments(samplex, boxed, np.empty(0), change_basis)


class TestMitigationTaskPrepare(unittest.TestCase):
    """End-to-end tests for :meth:`MitigationTask.prepare`."""

    def setUp(self):
        self.circuit = _simple_circuit()
        self.obs = [SparsePauliOp("ZZ")]

    def test_returns_quantum_program(self):
        """``prepare`` must return a ``QuantumProgram`` with one item."""
        task = MitigationTask()
        program = task.prepare(self.circuit, self.obs, parameters=None)
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)

    def test_appends_to_existing_program(self):
        """Passing a ``quantum_program`` must append to it and return the same object."""

        task1 = MitigationTask()
        program = task1.prepare(self.circuit, self.obs, parameters=None)
        self.assertEqual(len(program.items), 1)

        task2 = MitigationTask()
        returned = task2.prepare(self.circuit, self.obs, parameters=None, quantum_program=program)
        self.assertIs(returned, program)
        self.assertEqual(len(program.items), 2)

    def test_circuit_saved_on_instance(self):
        """The circuit must be saved as ``task.circuit`` after :meth:`prepare`."""
        task = MitigationTask()
        task.prepare(self.circuit, self.obs, parameters=None)
        self.assertIs(task.circuit, self.circuit)

    def test_boxed_circuit_saved_on_instance(self):
        """``task.boxed_circuit`` must be set after :meth:`prepare`."""
        task = MitigationTask()
        task.prepare(self.circuit, self.obs, parameters=None)
        self.assertIsNotNone(task.boxed_circuit)
        self.assertIsInstance(task.boxed_circuit, QuantumCircuit)

    def test_passthrough_data_set_on_program(self):
        """The returned QuantumProgram must have ``passthrough_data`` populated."""
        task = MitigationTask()
        program = task.prepare(self.circuit, self.obs, parameters=None)
        self.assertIsNotNone(program.passthrough_data)
        self.assertIn("qiskit_mitigation", program.passthrough_data)

    def test_raises_when_trex_and_projection_observables(self):
        """``prepare`` with a TREX instance and projection-operator observables must raise."""
        task = MitigationTask()
        qc = _simple_circuit()
        obs = ObservablesArray.coerce([{"Z+": 1}])
        trex = TREX()

        with self.assertRaises(ValueError):
            task.prepare(qc, obs, parameters=None, trex=trex)

    def test_prepare_with_trex_and_normal_observables_edits_boxing_options(self):
        """``prepare`` with TREX and non-projection observables edits the boxing options."""
        task = MitigationTask()
        qc = _simple_circuit()
        obs = [SparsePauliOp("ZZ")]  # regular Pauli, no projectors
        trex = TREX()
        # Should NOT raise; TREX applies its boxing-option edits and returns a program.
        program = task.prepare(qc, obs, parameters=None, trex=trex)
        expected_boxing_options = {"enable_measures": True, "measure_annotations": "all"}
        boxing_options = task.custom_boxing_options
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)
        self.assertEqual(expected_boxing_options, boxing_options)

    def test_prepare_non_broadcast_with_parameters(self):
        """Test non-broadcast ``prepare`` with a parametric circuit."""
        qc = _parametric_circuit()
        obs = [SparsePauliOp("ZZ")]
        task = MitigationTask()
        program = task.prepare(
            qc,
            obs,
            parameters=np.array([[0.1], [0.2]]),
            broadcast_obs_and_params=False,
        )
        self.assertIsInstance(program, QuantumProgram)
        self.assertEqual(len(program.items), 1)


# ---------------------------------------------------------------------------
# MitigationTask.postprocess - basic flow
# ---------------------------------------------------------------------------


class TestMitigationTaskPostprocess(unittest.TestCase):
    """Tests for :meth:`MitigationTask.postprocess`."""

    def setUp(self):
        self.circuit = _simple_circuit()
        self.obs = [SparsePauliOp("ZZ")]

    def _prepared_task(self):
        """Return a task that has been through prepare so postprocess attributes are set."""
        task = MitigationTask()
        task.prepare(self.circuit, self.obs, parameters=None)
        return task

    def test_postprocess_returns_pub_result(self):
        """``postprocess`` must return a ``PubResult``."""
        task = self._prepared_task()
        item_result = {"_meas": np.zeros((2, 1, 4, 2), dtype=bool)}
        pub_result = task.postprocess(item_result)
        self.assertIsInstance(pub_result, PubResult)


# ---------------------------------------------------------------------------
# MitigationTask.create_instance_from_passthrough_data
# ---------------------------------------------------------------------------


class TestCreateInstanceFromPassthroughData(unittest.TestCase):
    """Tests for :meth:`MitigationTask.create_instance_from_passthrough_data`."""

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
        }
        data.update(overrides)
        return data

    def test_happy_path_returns_mitigation_task(self):
        """A minimal valid passthrough dict must produce a MitigationTask."""
        task = MitigationTask.create_instance_from_passthrough_data(self._minimal_passthrough())
        self.assertIsInstance(task, MitigationTask)

    def test_program_item_index_set(self):
        """The ``_program_item_index`` must match the passthrough value."""
        passthrough = self._minimal_passthrough(program_item_index=7)
        task = MitigationTask.create_instance_from_passthrough_data(passthrough)
        self.assertEqual(task._program_item_index, 7)

    def test_raises_when_observables_missing(self):
        """Missing 'observables' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["observables"]
        with self.assertRaises(ValueError):
            MitigationTask.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_broadcast_obs_and_params_missing(self):
        """Missing 'broadcast_obs_and_params' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["broadcast_obs_and_params"]
        with self.assertRaises(ValueError):
            MitigationTask.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_program_item_index_missing(self):
        """Missing 'program_item_index' must raise ValueError."""
        passthrough = self._minimal_passthrough()
        del passthrough["program_item_index"]
        with self.assertRaises(ValueError):
            MitigationTask.create_instance_from_passthrough_data(passthrough)

    def test_raises_when_trex_calibration_true_but_no_trex(self):
        """trex_calibration=True without a TREX instance must raise ValueError."""
        passthrough = self._minimal_passthrough(trex_calibration=True)
        with self.assertRaises(ValueError):
            MitigationTask.create_instance_from_passthrough_data(passthrough, trex=None)


# ---------------------------------------------------------------------------
# MitigationTask._has_projection_operators
# ---------------------------------------------------------------------------


class TestHasProjectionOperators(unittest.TestCase):
    """Tests for :meth:`MitigationTask._has_projection_operators`."""

    def test_standard_pauli_has_no_projection_operators(self):
        """Standard Pauli observables (Z, X, Y) must not be flagged."""
        obs = ObservablesArray([SparsePauliOp("ZZ")])
        self.assertFalse(MitigationTask._has_projection_operators(obs))

    def test_identity_has_no_projection_operators(self):
        """Pure identity observable must not be flagged."""
        obs = ObservablesArray([SparsePauliOp("II")])
        self.assertFalse(MitigationTask._has_projection_operators(obs))

    def test_list_of_sparse_pauli_ops(self):
        """A plain list of ``SparsePauliOp`` must also work as input."""
        obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX")]
        self.assertFalse(MitigationTask._has_projection_operators(obs))

    def test_multiple_standard_observables_no_projection(self):
        """A mix of standard Pauli observables must return ``False``."""
        obs = ObservablesArray([SparsePauliOp("ZI"), SparsePauliOp("IZ"), SparsePauliOp("XX")])
        self.assertFalse(MitigationTask._has_projection_operators(obs))


if __name__ == "__main__":
    unittest.main()
