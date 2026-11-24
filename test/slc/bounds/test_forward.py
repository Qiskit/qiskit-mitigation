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

"""Tests for the forward bounds."""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pauli_prop.propagation import RotationGates
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList, SparseObservable, SparsePauliOp
from qiskit_mitigation.slc.bounds import compute_forward_bounds, merge_bounds
from qiskit_mitigation.slc.bounds.forward import time_evolved_norm_forward
from qiskit_mitigation.slc.bounds.trivial import trivial_bounds
from qiskit_mitigation.slc.utils import generate_noise_model_paulis
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import find_unique_box_instructions

from .. import construct_trotter_circuit

RANDOM_SEED = 42


def test_max_num_boxes():
    """Test limiting the maximum number of BoxOps for which to compute forward bounds."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    num_qubits = 50
    circuit = construct_trotter_circuit(
        num_qubits=num_qubits,
        num_trotter_steps=4,
        rx_angle=np.pi / 16,
        rzz_angle=-np.pi / 2,
        use_clifford=False,
    )

    boxes_pm = generate_boxing_pass_manager(
        enable_gates=True,
        enable_measures=True,
        twirling_strategy="active",
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        measure_annotations="all",
        remove_barriers=False,
    )
    boxed_circuit = boxes_pm.run(circuit)

    instructions = find_unique_box_instructions(boxed_circuit)

    noise_model_paulis = generate_noise_model_paulis(instructions)

    obs_pauli = Pauli("I" * num_qubits).compose("XYZ", [12, 24, 36])

    max_num_boxes = 2
    forward_bounds = compute_forward_bounds(
        boxed_circuit,
        noise_model_paulis,
        obs_pauli,
        eigval_max_qubits=20,
        evolution_max_terms=1000,
        max_num_boxes=max_num_boxes,
    )

    trivial = trivial_bounds(boxed_circuit, noise_model_paulis)

    # NOTE: here one would get the the noise model rates from the NoiseLearner
    noise_model_rates = {noise_id: None for noise_id in noise_model_paulis}

    merged_bounds = merge_bounds(
        boxed_circuit,
        forward_bounds,
        trivial,
        noise_model_rates,
        is_clifford_circuit=False,
    )

    with open(Path(__file__).parent.parent / "expected_fwd_bounds.pickle", "rb") as file:
        actual_fwd_bounds = {box: bounds for box, bounds in merged_bounds.items()}
        expected_fwd_bounds = pickle.load(file)
        for idx, key in enumerate(expected_fwd_bounds):
            assert key in actual_fwd_bounds
            expected = (
                np.full(len(expected_fwd_bounds[key]), 2.0)
                if idx >= max_num_boxes
                else expected_fwd_bounds[key]
            )
            assert np.allclose(actual_fwd_bounds[key].rates, expected)


def test_truncation_bias_ge_2():
    """Test truncation bias >= 2.0 returns NaN bound."""
    # Test propagation truncation >= 2.0 (line 113)
    qc_prop = QuantumCircuit(2)
    for _ in range(5):
        qc_prop.rx(np.pi / 4, 0)
        qc_prop.rzz(np.pi / 4, 0, 1)
        qc_prop.rx(np.pi / 4, 1)
    gates_prop = RotationGates([], [], [])
    for inst in qc_prop.data:
        qargs = [qc_prop.find_bit(q).index for q in inst.qubits]
        gates_prop.append_circuit_instruction(inst, qargs, 2)
    bound_prop = time_evolved_norm_forward(
        Pauli("ZI"), gates_prop, Pauli("XI"), evolution_max_terms=1
    )
    assert np.isnan(bound_prop.commutator_bound)
    assert bound_prop.truncation_bias >= 2.0

    # Test commutator simplify truncation >= 2.0 (line 148)
    qc_simp = QuantumCircuit(2)
    qc_simp.rx(np.pi / 4, 0)
    gates_simp = RotationGates([], [], [])
    for inst in qc_simp.data:
        qargs = [qc_simp.find_bit(q).index for q in inst.qubits]
        gates_simp.append_circuit_instruction(inst, qargs, 2)
    bound_simp = time_evolved_norm_forward(
        Pauli("IZ"), gates_simp, Pauli("IX"), atol_simplify=100.0
    )
    assert np.isnan(bound_simp.commutator_bound)
    assert bound_simp.truncation_bias >= 2.0


def test_eigensolver_warning_and_failure(caplog):
    """Test eigensolver exceeding 2.0 and failing solver fallback."""
    qc = QuantumCircuit(5)
    for i in range(5):
        qc.rx(0.5, i)
    gates = RotationGates([], [], [])
    for inst in qc.data:
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        gates.append_circuit_instruction(inst, qargs, 5)
    pauli = Pauli("ZZZZZ")
    observable = Pauli("XXXXX")

    with (
        caplog.at_level(logging.DEBUG, logger="qiskit_mitigation.slc.bounds.forward"),
        patch(
            "qiskit_mitigation.slc.bounds.forward.get_extremal_eigenvalue", return_value=(True, 3.5)
        ),
    ):
        bound = time_evolved_norm_forward(pauli, gates, observable)
        assert bound.commutator_bound == 3.5
        assert any("Solver found comm norm" in r.getMessage() for r in caplog.records)

    with (
        caplog.at_level(logging.DEBUG, logger="qiskit_mitigation.slc.bounds.forward"),
        patch(
            "qiskit_mitigation.slc.bounds.forward.get_extremal_eigenvalue",
            return_value=(False, 0.0),
        ),
    ):
        bound = time_evolved_norm_forward(pauli, gates, observable)
        assert bound.fallback_to_tri_ineq
        assert any(
            "Eigensolver failed, reverting to triangle inequality" in r.getMessage()
            for r in caplog.records
        )


def test_observable_types_and_validation():
    """Test various observable input types and multi-term rejection."""
    circuit = construct_trotter_circuit(
        num_qubits=2,
        num_trotter_steps=1,
        rx_angle=np.pi / 16,
        rzz_angle=-np.pi / 2,
        use_clifford=False,
    )
    boxed = generate_boxing_pass_manager(
        inject_noise_targets="all",
        inject_noise_strategy="individual_modification",
        twirling_strategy="active",
        remove_barriers=False,
    ).run(circuit)
    noise_model_paulis = generate_noise_model_paulis(find_unique_box_instructions(boxed))

    with pytest.raises(
        NotImplementedError,
        match="Cannot compute the bounds for an observable with more than 1 Pauli",
    ):
        compute_forward_bounds(boxed, noise_model_paulis, PauliList(["XX", "ZZ"]))

    with pytest.raises(TypeError, match="Unsupported observable type"):
        compute_forward_bounds(boxed, noise_model_paulis, [123])

    # PauliList with single term
    bounds_pl = compute_forward_bounds(boxed, noise_model_paulis, PauliList(["ZZ"]))
    assert len(bounds_pl) > 0

    # SparsePauliOp with single term
    bounds_spo = compute_forward_bounds(boxed, noise_model_paulis, SparsePauliOp(["ZZ"]))
    assert len(bounds_spo) > 0

    # SparseObservable with single term
    sparse_obs = SparseObservable.from_sparse_list([("ZZ", [1, 0], 1.0)], num_qubits=2)
    bounds_so = compute_forward_bounds(boxed, noise_model_paulis, sparse_obs)
    assert len(bounds_so) > 0


if __name__ == "__main__":
    test_max_num_boxes()
